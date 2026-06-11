"""Priced adaptive speculative decoding policy.

Chooses ``speculative_num_steps`` (g) once per decode round by maximizing
expected committed tokens per unit step cost (goodput):

    g* = argmax_{g in candidate_steps} (sum_i E_i[num_correct_drafts | g] + B) / C(B, g)

where ``B`` is the batch size, ``E_i[num_correct_drafts | g]`` is the
per-request expected number of correct drafts (no bonus), and ``C(B, g)``
is the wall-clock cost of one decode round at that configuration.

Cost model ``C(B, g)``:
  - Seeded from an optional CSV cost table (config key ``"cost_table"``,
    header ``batch_size,num_draft_tokens,step_seconds`` with
    ``num_draft_tokens = g + 1``). Lookup is nearest-neighbor in B using
    log-space distance among rows with an exact g match; when no row has
    that g, we take the nearest available g at the log-nearest B and scale
    linearly in draft tokens, ``cost * (g + 1) / (g_near + 1)``. Without a
    table the cost defaults to a flat 1.0, so the policy maximizes expected
    accept tokens until online correction kicks in.
  - Online-corrected by an EMA of realized wall gaps between consecutive
    decode rounds, bucketed by (cuda-graph BS the round pads to, g) and
    attributed to the previous round's configuration. A gap is only used
    when the two policy invocations were consecutive worker rounds (no
    prefill round in between); idle gaps with consecutive indices still
    leak in — see the campaign worklog. EMA overrides the table once a
    bucket has at least one sample.

Acceptance model ``E_i[num_correct_drafts | g] = sum_{d=1..g} prod_{j<=d} a_hat[i][j]``
with ``a_hat[i][j]`` sourced in priority order from:
  1. the request's most recent per-position draft-token confidences
     (positions past the recorded length reuse the last position's value),
  2. a per-request EMA of the position-averaged accept rate observed from
     realized ``num_correct_drafts`` (a scalar alpha_i, ``E = sum_d alpha_i^d``),
  3. a global prior (config key ``"prior_accept_rate"``).

This module intentionally imports only the standard library at module
scope so pure-CPU unit tests can import it without torch/GPU deps.
"""

from __future__ import annotations

import bisect
import csv
import json
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# After the first few switches, only log every Nth one.
_SWITCH_LOG_FIRST = 10
_SWITCH_LOG_EVERY = 50

_COST_TABLE_COLUMNS = ("batch_size", "num_draft_tokens", "step_seconds")


def _log_info_on_rank0(msg: str) -> None:
    """Rank-0 logging with a plain-logger fallback.

    The lazy import keeps this module importable without torch
    (``sglang.srt.utils`` pulls in the full GPU stack).
    """
    try:
        from sglang.srt.utils import log_info_on_rank0
    except ImportError:
        logger.info(msg)
        return
    log_info_on_rank0(logger, msg)


@dataclass
class PricedPolicyConfig:
    candidate_steps: list[int]
    cost_table: str | None = None
    prior_accept_rate: float = 0.7
    switch_margin: float = 0.03
    ema_alpha: float = 0.2
    warmup_rounds: int = 20
    max_tracked_requests: int = 8192


def load_priced_config(cfg: dict) -> PricedPolicyConfig:
    """Validate a ``"policy": "priced"`` adaptive config dict."""
    steps = cfg.get("candidate_steps")
    if (
        not isinstance(steps, list)
        or not steps
        or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 1 for s in steps)
    ):
        raise ValueError(
            "priced policy: candidate_steps is required and must be a non-empty "
            f"list of ints >= 1 (g=0 is not representable at runtime), got {steps!r}"
        )

    out = PricedPolicyConfig(candidate_steps=sorted(set(steps)))

    cost_table = cfg.get("cost_table")
    if cost_table is not None and not isinstance(cost_table, str):
        raise ValueError(
            f"priced policy: cost_table must be a CSV file path string, got {cost_table!r}"
        )
    out.cost_table = cost_table

    prior = cfg.get("prior_accept_rate", out.prior_accept_rate)
    if not isinstance(prior, (int, float)) or not (0.0 < prior <= 1.0):
        raise ValueError(
            f"priced policy: prior_accept_rate must be in (0, 1], got {prior!r}"
        )
    out.prior_accept_rate = float(prior)

    margin = cfg.get("switch_margin", out.switch_margin)
    if not isinstance(margin, (int, float)) or margin < 0.0:
        raise ValueError(
            f"priced policy: switch_margin must be a fraction >= 0, got {margin!r}"
        )
    out.switch_margin = float(margin)

    ema_alpha = cfg.get("ema_alpha", out.ema_alpha)
    if not isinstance(ema_alpha, (int, float)) or not (0.0 < ema_alpha <= 1.0):
        raise ValueError(
            f"priced policy: ema_alpha must be in (0, 1], got {ema_alpha!r}"
        )
    out.ema_alpha = float(ema_alpha)

    warmup_rounds = cfg.get("warmup_rounds", out.warmup_rounds)
    if not isinstance(warmup_rounds, int) or warmup_rounds < 0:
        raise ValueError(
            f"priced policy: warmup_rounds must be an int >= 0, got {warmup_rounds!r}"
        )
    out.warmup_rounds = warmup_rounds

    max_tracked = cfg.get("max_tracked_requests", out.max_tracked_requests)
    if not isinstance(max_tracked, int) or max_tracked < 1:
        raise ValueError(
            f"priced policy: max_tracked_requests must be an int >= 1, got {max_tracked!r}"
        )
    out.max_tracked_requests = max_tracked

    return out


def load_cost_table(path: str) -> dict[tuple[int, int], float]:
    """Load a step-cost CSV into ``{(batch_size, num_steps): step_seconds}``.

    Header: ``batch_size,num_draft_tokens,step_seconds`` where
    ``num_draft_tokens = g + 1``. Rows with ``num_draft_tokens <= 1``
    (g=0 baselines) are skipped — g=0 is not representable at runtime.
    """
    table: dict[tuple[int, int], float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [
            c for c in _COST_TABLE_COLUMNS if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"cost_table {path}: missing columns {missing}; "
                f"expected header {','.join(_COST_TABLE_COLUMNS)}"
            )
        for line_num, row in enumerate(reader, start=2):
            try:
                batch_size = int(row["batch_size"])
                num_steps = int(row["num_draft_tokens"]) - 1
                step_seconds = float(row["step_seconds"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"cost_table {path} line {line_num}: {e}") from e
            if batch_size < 1 or step_seconds <= 0.0:
                raise ValueError(
                    f"cost_table {path} line {line_num}: need batch_size >= 1 "
                    f"and step_seconds > 0, got {row!r}"
                )
            if num_steps < 1:
                continue
            table[(batch_size, num_steps)] = step_seconds
    if not table:
        raise ValueError(f"cost_table {path}: no usable rows (num_draft_tokens >= 2)")
    return table


class _RequestState:
    """Per-request acceptance evidence, LRU-bounded by the owning policy."""

    __slots__ = ("accept_rate_ema", "last_confidences")

    def __init__(self):
        # Position-averaged per-draft accept rate (paper alpha, no bonus).
        self.accept_rate_ema: float | None = None
        # Per-position draft-token confidences from the most recent round.
        self.last_confidences: list[float] | None = None


class PricedSpeculativeParams:
    """Goodput-maximizing step policy (see module docstring).

    Satisfies the same step-policy interface as ``AdaptiveSpeculativeParams``
    (``candidate_steps``, ``set_cuda_graph_bs``, ``get_steps_for_batch``,
    ``on_verify_complete``, ``cuda_graph_bs_for_step``) plus the richer
    priced-only inputs: request ids, a worker round index for cost
    attribution, and draft-confidence observations.
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
        cfg: dict | None = None,
    ):
        if cfg is None:
            if cfg_path is None:
                raise ValueError(
                    "priced policy requires a config file (candidate_steps is mandatory)"
                )
            with open(cfg_path) as f:
                cfg = json.load(f)
        self._cfg = load_priced_config(cfg)
        # All configured candidates; init_states builds a runtime state per
        # candidate, then set_available_steps narrows to what actually exists.
        self._candidate_steps: list[int] = list(self._cfg.candidate_steps)
        self._available_steps: list[int] = list(self._candidate_steps)

        if initial_steps in self._candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self._candidate_steps[len(self._candidate_steps) // 2]
        self._initial_steps = self.current_steps

        self._cost_table: dict[tuple[int, int], float] = (
            load_cost_table(self._cfg.cost_table) if self._cfg.cost_table else {}
        )
        # Online-corrected costs: (B-bucket, num_steps) -> EMA of realized
        # wall gaps between consecutive decode rounds.
        self._cost_ema: dict[tuple[int, int], float] = {}

        self._cuda_graph_bs: list[int] | None = None
        self._reqs: OrderedDict[str, _RequestState] = OrderedDict()

        # Decision bookkeeping.
        self._round_ct = 0
        self._switch_ct = 0
        self._last_round_idx: int | None = None
        self._last_round_t: float | None = None
        self._last_round_cost_key: tuple[int, int] | None = None

        _log_info_on_rank0(
            "PricedSpeculativeParams initialized: "
            f"steps={self.current_steps}, candidate_steps={self._candidate_steps}, "
            f"cost_table={self._cfg.cost_table!r} ({len(self._cost_table)} rows), "
            f"prior_accept_rate={self._cfg.prior_accept_rate}, "
            f"switch_margin={self._cfg.switch_margin}, "
            f"ema_alpha={self._cfg.ema_alpha}, "
            f"warmup_rounds={self._cfg.warmup_rounds}, "
            f"max_tracked_requests={self._cfg.max_tracked_requests}"
        )

    # -- Step-policy interface (shared with AdaptiveSpeculativeParams) --

    @property
    def candidate_steps(self) -> list[int]:
        return list(self._candidate_steps)

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """Priced uses one candidate set for all batch sizes, so every graph
        BS can reach every candidate step. ``None`` when graphs are disabled.
        """
        if self._cuda_graph_bs is None:
            return None
        return list(self._cuda_graph_bs) if step in self._candidate_steps else []

    def get_steps_for_batch(
        self,
        batch_size: int,
        rids: list[str] | None = None,
        round_idx: int | None = None,
    ) -> int:
        """Per-decode-round policy invocation.

        Folds the wall gap since the previous invocation into the cost EMA
        (when the rounds were consecutive), then re-optimizes goodput.
        Idle rounds (``batch_size <= 0``) are ignored entirely.
        """
        if batch_size <= 0:
            return self.current_steps

        now = time.perf_counter()
        self._observe_round_gap(now, round_idx)

        self._round_ct += 1
        if self._round_ct > self._cfg.warmup_rounds:
            self._maybe_switch(batch_size, rids)

        self._last_round_idx = round_idx
        self._last_round_t = now
        self._last_round_cost_key = (
            self._cost_bucket(batch_size),
            self.current_steps,
        )
        return self.current_steps

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        rids: list[str] | None = None,
        num_steps: int | None = None,
    ) -> int | None:
        """Fold realized correct-draft counts into per-request accept-rate EMAs.

        Never requests an immediate switch (always returns ``None``); step
        decisions happen at the next ``get_steps_for_batch``.
        """
        if not rids or not num_steps or num_steps <= 0:
            return None
        ema_alpha = self._cfg.ema_alpha
        for rid, num_correct_drafts in zip(rids, num_correct_drafts_per_req):
            state = self._touch(rid)
            rate = min(max(num_correct_drafts / num_steps, 0.0), 1.0)
            if state.accept_rate_ema is None:
                state.accept_rate_ema = rate
            else:
                state.accept_rate_ema = (
                    1.0 - ema_alpha
                ) * state.accept_rate_ema + ema_alpha * rate
        return None

    # -- Priced-only inputs --

    def set_available_steps(self, steps: list[int]) -> None:
        """Restrict decisions to candidates whose runtime states exist."""
        available = sorted(set(steps) & set(self._candidate_steps))
        if not available:
            raise ValueError(
                f"priced policy: no runtime state matches candidate_steps="
                f"{self._candidate_steps}; built states: {sorted(steps)}"
            )
        self._available_steps = available
        if self.current_steps not in available:
            self.current_steps = min(
                available, key=lambda s: abs(s - self.current_steps)
            )
            self._initial_steps = self.current_steps

    def observe_draft_confidences(
        self, rids: list[str], confidences: list[list[float]]
    ) -> None:
        """Record the just-finished round's per-position draft confidences.

        They become ``a_hat`` for each request in the NEXT round's decision.
        """
        for rid, conf in zip(rids, confidences):
            self._touch(rid).last_confidences = conf

    # -- Decision internals --

    def _maybe_switch(self, batch_size: int, rids: list[str] | None) -> None:
        goodputs = {
            steps: self._goodput(batch_size, rids, steps)
            for steps in self._available_steps
        }
        best = max(self._available_steps, key=lambda s: goodputs[s])
        current = self.current_steps
        if best == current or current not in goodputs:
            return
        # Hysteresis: runtime state swaps have cost, so only move when the
        # predicted goodput gain clears the margin.
        if goodputs[best] <= goodputs[current] * (1.0 + self._cfg.switch_margin):
            return
        self.current_steps = best
        self._switch_ct += 1
        if (
            self._switch_ct <= _SWITCH_LOG_FIRST
            or self._switch_ct % _SWITCH_LOG_EVERY == 0
        ):
            _log_info_on_rank0(
                f"Priced spec params switch #{self._switch_ct}: "
                f"steps {current} -> {best} at B={batch_size} "
                f"(predicted goodput {goodputs[current]:.3f} -> "
                f"{goodputs[best]:.3f} tokens/s)"
            )

    def _goodput(
        self, batch_size: int, rids: list[str] | None, steps: int
    ) -> float:
        """Expected committed tokens per second at *steps* for this batch."""
        if rids:
            expected_correct_drafts = sum(
                self._expected_correct_drafts(rid, steps) for rid in rids
            )
        else:
            expected_correct_drafts = batch_size * _geometric_expected(
                self._cfg.prior_accept_rate, steps
            )
        # Every request always commits the bonus token on top of its drafts.
        expected_accept_tokens = expected_correct_drafts + batch_size
        return expected_accept_tokens / self._step_cost(batch_size, steps)

    def _expected_correct_drafts(self, rid: str, steps: int) -> float:
        """E[num_correct_drafts | steps] for one request (drafts only, no bonus)."""
        state = self._reqs.get(rid)
        if state is None:
            return _geometric_expected(self._cfg.prior_accept_rate, steps)
        if state.last_confidences:
            conf = state.last_confidences
            total = 0.0
            chain_prob = 1.0
            for d in range(steps):
                a_hat = conf[d] if d < len(conf) else conf[-1]
                chain_prob *= min(max(a_hat, 0.0), 1.0)
                total += chain_prob
            return total
        if state.accept_rate_ema is not None:
            return _geometric_expected(state.accept_rate_ema, steps)
        return _geometric_expected(self._cfg.prior_accept_rate, steps)

    # -- Cost internals --

    def _observe_round_gap(self, now: float, round_idx: int | None) -> None:
        """Attribute the wall gap since the previous policy invocation to the
        previous round's (B-bucket, num_steps), but only when the two
        invocations were consecutive worker rounds (both decode)."""
        if (
            round_idx is None
            or self._last_round_idx is None
            or round_idx != self._last_round_idx + 1
            or self._last_round_t is None
            or self._last_round_cost_key is None
        ):
            return
        gap = now - self._last_round_t
        if gap <= 0.0:
            return
        key = self._last_round_cost_key
        prev = self._cost_ema.get(key)
        ema_alpha = self._cfg.ema_alpha
        self._cost_ema[key] = (
            gap if prev is None else (1.0 - ema_alpha) * prev + ema_alpha * gap
        )

    def _step_cost(self, batch_size: int, steps: int) -> float:
        """Corrected cost: realized-gap EMA when available, else table."""
        ema = self._cost_ema.get((self._cost_bucket(batch_size), steps))
        if ema is not None:
            return ema
        return self._table_cost(batch_size, steps)

    def _cost_bucket(self, batch_size: int) -> int:
        """The cuda-graph BS this batch actually pads up to (the executed
        graph), falling back to the raw batch size past the largest graph
        or when graphs are disabled."""
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _table_cost(self, batch_size: int, steps: int) -> float:
        if not self._cost_table:
            # No seed: flat cost — maximize expected accept tokens until the
            # online gap EMA populates (which takes one round per config).
            return 1.0
        log_b = math.log(batch_size)
        rows_with_steps = [
            b for (b, table_steps) in self._cost_table if table_steps == steps
        ]
        if rows_with_steps:
            nearest_b = min(rows_with_steps, key=lambda b: abs(math.log(b) - log_b))
            return self._cost_table[(nearest_b, steps)]
        # No row has this g: take the log-nearest B overall, then the nearest
        # available g there, and scale linearly in draft tokens (g + 1).
        nearest_b = min(
            {b for (b, _) in self._cost_table},
            key=lambda b: abs(math.log(b) - log_b),
        )
        steps_at_b = [
            table_steps
            for (b, table_steps) in self._cost_table
            if b == nearest_b
        ]
        nearest_steps = min(steps_at_b, key=lambda s: (abs(s - steps), s))
        return (
            self._cost_table[(nearest_b, nearest_steps)]
            * (steps + 1)
            / (nearest_steps + 1)
        )

    # -- Request tracking --

    def _touch(self, rid: str) -> _RequestState:
        state = self._reqs.get(rid)
        if state is None:
            state = _RequestState()
            self._reqs[rid] = state
            while len(self._reqs) > self._cfg.max_tracked_requests:
                self._reqs.popitem(last=False)
        else:
            self._reqs.move_to_end(rid)
        return state


def _geometric_expected(accept_rate: float, steps: int) -> float:
    """sum_{d=1..steps} accept_rate**d (chain expectation under constant alpha)."""
    accept_rate = min(max(accept_rate, 0.0), 1.0)
    if accept_rate >= 1.0:
        return float(steps)
    return accept_rate * (1.0 - accept_rate**steps) / (1.0 - accept_rate)
