"""Priced adaptive speculative decoding policy.

Chooses ``speculative_num_steps`` (g) once per decode round by maximizing
expected committed tokens per unit step cost (goodput):

    g* = argmax_{g in candidate_steps} (sum_i E_i[num_correct_drafts | g] + B) / C(B, g)

where ``B`` is the batch size, ``E_i[num_correct_drafts | g]`` is the
per-request expected number of correct drafts (no bonus), and ``C(B, g)``
is the wall-clock cost of one decode round at that configuration.

``g = 0`` is speculation OFF for the round: the drafter is not invoked and
the target runs a plain 1-token decode, so ``E[num_correct_drafts | 0] = 0``
and ``goodput(0) = B / C(B, 0)``. Its cost lives in the table's
``num_draft_tokens = 1`` rows.

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

Acceptance model. Two layers:

  1. A *global realized anchor*: an EMA per g of the mean realized
     ``num_correct_drafts`` per round (updated in ``on_verify_complete``
     for the g that actually ran). For a g with no direct sample, the
     anchor is derived by chain interpolation: invert the nearest
     sampled g's anchor into a constant per-position accept rate, then
     extend that chain to the queried g. Raw per-position confidence
     products systematically overstate deep-chain returns (measured on
     GH200: E[correct|8] - E[correct|4] = +0.15 realized, while a
     0.9^d chain predicts far more), so confidences never set the level.
  2. A *per-request chain signal* ``chain_i(g) = sum_{d=1..g} prod_{j<=d}
     a_hat[i][j]`` with ``a_hat[i][j]`` sourced in priority order from:
       a. the request's most recent per-position draft-token confidences
          (positions past the recorded length reuse the last position's value),
       b. a per-request EMA of the position-averaged accept rate observed from
          realized ``num_correct_drafts`` (a scalar alpha_i, ``E = sum_d alpha_i^d``),
       c. a global prior (config key ``"prior_accept_rate"``).

When the anchor has any realized data, the per-request expectation is
``E_global(g) * clamp(chain_i(g) / chain_baseline(g), 1/MOD, MOD)`` with
``chain_baseline`` a per-g rolling EMA (``"ema_alpha"``) of round-mean
chain values maintained ACROSS rounds and MOD the config key
``"confidence_modulation"``: per-request signal modulates *around* the
realized level instead of free-running. The baseline must be
cross-round: a same-round batch mean equals the lone request's own
chain at batch size 1, forcing modulation to exactly 1.0 and killing
per-request gating where it is worth most (observed at concurrency 1
on GH200). Before the rolling baseline has data at a g, the same-round
batch mean is used as the fallback baseline. Before any realized data
the chain signal is used directly (layer 2 alone).

Evidence staleness: g=0 rounds produce no confidences and no accepts, so
per-request evidence would freeze (frozen pessimistic evidence makes g=0
absorbing — observed at concurrency 4 on the first GH200 grid). Each
request's evidence therefore decays toward the global prior by
``"evidence_decay"`` per policy round without a fresh update, applied
lazily at read time via a per-request last-updated round stamp.

Switching is *laddered*: each switch moves at most
``"max_switch_distance"`` indices through the sorted candidate list, so
e.g. 0 -> 8 in {0,1,2,3,4,8} takes successive cooldown windows via
0 -> 2 -> 4 -> 8. This bounds the g=0 catch-up gap drained per switch and
prevents cliff-edge state swaps.

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

# Cost is attributed as a window mean over this many uniform-config policy
# invocations (see _observe_round_gap).
_COST_WINDOW = 8

_COST_TABLE_COLUMNS = ("batch_size", "num_draft_tokens", "step_seconds")

# Flat early-exit stop price (expected correct drafts per drafter step) used
# when the policy has no cost data to derive one from.
_EARLY_EXIT_FALLBACK_STOP_PRICE = 0.2


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
    # Minimum decode rounds between runtime-state swaps. Swaps are not free
    # and batch size flickers at low load; without a cooldown the policy
    # thrashes between the per-B optima every round.
    switch_cooldown_rounds: int = 16
    # g=0 (speculation-off) only: max tokens buffered per request for the
    # drafter catch-up before the worker force-flushes a catch-up extend.
    g0_max_gap: int = 1024
    # Offline-calibrated E[num_correct_drafts | g] for g = 1..len (the trace
    # bank's e-curve). Online estimators DECAY TOWARD calibrated truth
    # (this curve; the cost table) with per-round factor estimator_decay:
    # a transient that poisons an estimator the policy then avoids would
    # otherwise freeze forever and self-confirm (observed: realized E at
    # g=4 stuck at 0.039 vs 0.91 measured; cost EMAs physically backwards).
    acceptance_prior_curve: list[float] | None = None
    estimator_decay: float = 0.995
    # g=0 only: max gap tokens per drafter catch-up extend. The drain is
    # partitioned into sequential extends of at most this many tokens so a
    # large-batch re-entry cannot OOM on activations (observed at
    # concurrency 256 on GH200: one eager extend over every running
    # request's accumulated gap crashed the server).
    g0_catch_up_chunk_tokens: int = 4096
    # Per-round multiplicative decay of per-request evidence toward the
    # global prior while the request receives no fresh confidence/accept
    # update (g=0 rounds generate neither). 0.97 ~= a 32-round time
    # constant; 1.0 disables decay.
    evidence_decay: float = 0.97
    # Allow the pre-draft depth-0 skip inside early-exit rounds. Measured
    # on GH200: per-round skips thrash (each skip->draft re-entry pays a
    # drafter catch-up extend, ~10 ms/round vs 5.5 parked, and skipping
    # starves the p1 signal that would end the streak) — phase-level g=0
    # switching is the right granularity for spec on/off, so this defaults
    # OFF; in-draft stopping (depths >= 1) is unaffected.
    early_exit_skip: bool = False
    # Minimum in-draft stop depth. Measured (GH200, ShareGPT, B=1):
    # stopping at depth 1 is dilutive (7.9-8.8 ms for ~1.26 commits, worse
    # than not speculating), while stop@2 already beats the best static
    # rate — doomed rounds should pay two passes, not one.
    early_exit_min_stop_depth: int = 1
    # Clamp on how far the per-request chain signal may modulate the
    # global realized-acceptance anchor: ratio in [1/MOD, MOD].
    confidence_modulation: float = 2.0
    # Max candidate-index distance a single switch may move through the
    # sorted candidate list (ladder switching).
    max_switch_distance: int = 2
    # In-round early-exit drafting: at small batch the worker drafts eagerly
    # at the deepest candidate and may stop the loop at any intermediate
    # candidate depth when the marginal step no longer pays — including
    # depth 0 (skip drafting entirely for the round) when 0 is a candidate
    # with a built runtime state and the pre-draft signal (the first draft
    # token's confidence, computed by the previous round's draft extend)
    # says even the first kept draft won't pay. Off by default (opt-in: it
    # changes the engine's draft execution path).
    early_exit: bool = False
    # Largest batch size early exit applies to. Past it the per-step host
    # sync dominates and the captured draft graph wins.
    early_exit_max_batch: int = 4
    # Explicit flat stop price (expected correct drafts one more drafter
    # step must yield). None derives it each round from the cost surface
    # and current goodput (see early_exit_stop_price).
    early_exit_stop_price: float | None = None


def load_priced_config(cfg: dict) -> PricedPolicyConfig:
    """Validate a ``"policy": "priced"`` adaptive config dict."""
    steps = cfg.get("candidate_steps")
    if (
        not isinstance(steps, list)
        or not steps
        or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in steps)
    ):
        raise ValueError(
            "priced policy: candidate_steps is required and must be a non-empty "
            f"list of ints >= 0 (g=0 means speculation off for the round), got {steps!r}"
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

    cooldown = cfg.get("switch_cooldown_rounds", out.switch_cooldown_rounds)
    if not isinstance(cooldown, int) or cooldown < 0:
        raise ValueError(
            f"priced policy: switch_cooldown_rounds must be an int >= 0, got {cooldown!r}"
        )
    out.switch_cooldown_rounds = cooldown

    max_tracked = cfg.get("max_tracked_requests", out.max_tracked_requests)
    if not isinstance(max_tracked, int) or max_tracked < 1:
        raise ValueError(
            f"priced policy: max_tracked_requests must be an int >= 1, got {max_tracked!r}"
        )
    out.max_tracked_requests = max_tracked

    g0_max_gap = cfg.get("g0_max_gap", out.g0_max_gap)
    if not isinstance(g0_max_gap, int) or g0_max_gap < 1:
        raise ValueError(
            f"priced policy: g0_max_gap must be an int >= 1, got {g0_max_gap!r}"
        )
    out.g0_max_gap = g0_max_gap

    curve = cfg.get("acceptance_prior_curve", out.acceptance_prior_curve)
    if curve is not None:
        if (
            not isinstance(curve, list)
            or not curve
            or not all(isinstance(v, (int, float)) and v >= 0.0 for v in curve)
        ):
            raise ValueError(
                "priced policy: acceptance_prior_curve must be a non-empty list "
                f"of E[num_correct_drafts | g] floats >= 0 for g=1.., got {curve!r}"
            )
        out.acceptance_prior_curve = [float(v) for v in curve]

    ee_min_stop = cfg.get("early_exit_min_stop_depth", out.early_exit_min_stop_depth)
    if not isinstance(ee_min_stop, int) or isinstance(ee_min_stop, bool) or ee_min_stop < 1:
        raise ValueError(
            f"priced policy: early_exit_min_stop_depth must be an int >= 1, got {ee_min_stop!r}"
        )
    out.early_exit_min_stop_depth = ee_min_stop

    ee_skip = cfg.get("early_exit_skip", out.early_exit_skip)
    if not isinstance(ee_skip, bool):
        raise ValueError(
            f"priced policy: early_exit_skip must be a bool, got {ee_skip!r}"
        )
    out.early_exit_skip = ee_skip

    est_decay = cfg.get("estimator_decay", out.estimator_decay)
    if not isinstance(est_decay, (int, float)) or not (0.0 < est_decay <= 1.0):
        raise ValueError(
            f"priced policy: estimator_decay must be in (0, 1], got {est_decay!r}"
        )
    out.estimator_decay = float(est_decay)

    chunk_tokens = cfg.get("g0_catch_up_chunk_tokens", out.g0_catch_up_chunk_tokens)
    if not isinstance(chunk_tokens, int) or chunk_tokens < 1:
        raise ValueError(
            "priced policy: g0_catch_up_chunk_tokens must be an int >= 1, "
            f"got {chunk_tokens!r}"
        )
    out.g0_catch_up_chunk_tokens = chunk_tokens

    evidence_decay = cfg.get("evidence_decay", out.evidence_decay)
    if not isinstance(evidence_decay, (int, float)) or not (
        0.0 < evidence_decay <= 1.0
    ):
        raise ValueError(
            f"priced policy: evidence_decay must be in (0, 1], got {evidence_decay!r}"
        )
    out.evidence_decay = float(evidence_decay)

    modulation = cfg.get("confidence_modulation", out.confidence_modulation)
    if not isinstance(modulation, (int, float)) or modulation < 1.0:
        raise ValueError(
            f"priced policy: confidence_modulation must be >= 1, got {modulation!r}"
        )
    out.confidence_modulation = float(modulation)

    max_switch_distance = cfg.get("max_switch_distance", out.max_switch_distance)
    if not isinstance(max_switch_distance, int) or max_switch_distance < 1:
        raise ValueError(
            "priced policy: max_switch_distance must be an int >= 1, "
            f"got {max_switch_distance!r}"
        )
    out.max_switch_distance = max_switch_distance

    early_exit = cfg.get("early_exit", out.early_exit)
    if not isinstance(early_exit, bool):
        raise ValueError(
            f"priced policy: early_exit must be a bool, got {early_exit!r}"
        )
    out.early_exit = early_exit

    early_exit_max_batch = cfg.get("early_exit_max_batch", out.early_exit_max_batch)
    if (
        not isinstance(early_exit_max_batch, int)
        or isinstance(early_exit_max_batch, bool)
        or early_exit_max_batch < 1
    ):
        raise ValueError(
            "priced policy: early_exit_max_batch must be an int >= 1, "
            f"got {early_exit_max_batch!r}"
        )
    out.early_exit_max_batch = early_exit_max_batch

    early_exit_stop_price = cfg.get(
        "early_exit_stop_price", out.early_exit_stop_price
    )
    if early_exit_stop_price is not None:
        if (
            not isinstance(early_exit_stop_price, (int, float))
            or isinstance(early_exit_stop_price, bool)
            or early_exit_stop_price <= 0.0
        ):
            raise ValueError(
                "priced policy: early_exit_stop_price must be a float > 0 "
                f"or null (derive per round), got {early_exit_stop_price!r}"
            )
        out.early_exit_stop_price = float(early_exit_stop_price)

    return out


def load_cost_table(path: str) -> dict[tuple[int, int], float]:
    """Load a step-cost CSV into ``{(batch_size, num_steps): step_seconds}``.

    Header: ``batch_size,num_draft_tokens,step_seconds`` where
    ``num_draft_tokens = g + 1``. Rows with ``num_draft_tokens == 1`` are the
    g=0 (speculation-off) plain-decode rounds; rows with
    ``num_draft_tokens < 1`` are malformed and skipped.
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
            if num_steps < 0:
                continue
            table[(batch_size, num_steps)] = step_seconds
    if not table:
        raise ValueError(f"cost_table {path}: no usable rows (num_draft_tokens >= 1)")
    return table


class _RequestState:
    """Per-request acceptance evidence, LRU-bounded by the owning policy."""

    __slots__ = ("accept_rate_ema", "last_confidences", "last_update_round")

    def __init__(self):
        # Position-averaged per-draft accept rate (paper alpha, no bonus).
        self.accept_rate_ema: float | None = None
        # Per-position draft-token confidences from the most recent round.
        self.last_confidences: list[float] | None = None
        # Policy round of the last fresh confidence/accept update; staleness
        # decay toward the prior is applied lazily at read time from this.
        self.last_update_round: int = 0


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
        # Global realized accept curve: num_steps -> EMA of the batch-mean
        # realized num_correct_drafts per round at that g. Anchors the
        # acceptance model (see module docstring).
        self._realized_correct_ema: dict[int, float] = {}
        # Last-update round stamps: reads decay estimators toward calibrated
        # truth (curve / table) by estimator_decay^age.
        self._realized_round: dict[int, int] = {}
        self._cost_ema_round: dict[tuple[int, int], int] = {}
        # (round_idx, t, prev cost key) ring for window-mean attribution.
        self._cost_window: list = []
        # Rolling modulation baseline: num_steps -> EMA of round-mean chain
        # signals, maintained across rounds. Normalizing against the
        # same-round batch mean instead makes modulation identically 1.0 at
        # batch size 1 (see module docstring).
        self._chain_mean_ema: dict[int, float] = {}

        self._cuda_graph_bs: list[int] | None = None
        self._reqs: OrderedDict[str, _RequestState] = OrderedDict()

        # Decision bookkeeping.
        self._round_ct = 0
        self._switch_ct = 0
        self._last_switch_round = -(10**9)
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
            f"max_tracked_requests={self._cfg.max_tracked_requests}, "
            f"evidence_decay={self._cfg.evidence_decay}, "
            f"confidence_modulation={self._cfg.confidence_modulation}, "
            f"max_switch_distance={self._cfg.max_switch_distance}, "
            f"g0_catch_up_chunk_tokens={self._cfg.g0_catch_up_chunk_tokens}"
        )

    # -- Step-policy interface (shared with AdaptiveSpeculativeParams) --

    @property
    def candidate_steps(self) -> list[int]:
        return list(self._candidate_steps)

    @property
    def g0_max_gap(self) -> int:
        """Per-request drafter-gap cap during g=0 (speculation-off) phases."""
        return self._cfg.g0_max_gap

    @property
    def g0_catch_up_chunk_tokens(self) -> int:
        """Max gap tokens per drafter catch-up extend after a g=0 phase."""
        return self._cfg.g0_catch_up_chunk_tokens

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
            steps_before = self.current_steps
            self._maybe_switch(batch_size, rids)
            if self.current_steps != steps_before:
                # The next wall gap spans the runtime-state swap and the new
                # graphs' warmup — charging it to either config poisons that
                # config's cost EMA (probes self-poison their destination;
                # observed parking at conc 1). Skip attribution for it.
                self._last_round_idx = round_idx
                self._last_round_t = now
                self._last_round_cost_key = None
                return self.current_steps

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
        """Fold realized correct-draft counts into per-request accept-rate EMAs
        and into the global realized accept curve at the g that ran.

        Never requests an immediate switch (always returns ``None``); step
        decisions happen at the next ``get_steps_for_batch``.
        """
        if not rids or not num_steps or num_steps <= 0:
            return None
        ema_alpha = self._cfg.ema_alpha
        if num_correct_drafts_per_req:
            mean_correct_drafts = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
            prev = self._realized_correct_ema.get(num_steps)
            self._realized_correct_ema[num_steps] = (
                mean_correct_drafts
                if prev is None
                else (1.0 - ema_alpha) * prev + ema_alpha * mean_correct_drafts
            )
            self._realized_round[num_steps] = self._round_ct
        for rid, num_correct_drafts in zip(rids, num_correct_drafts_per_req):
            state = self._touch(rid)
            rate = min(max(num_correct_drafts / num_steps, 0.0), 1.0)
            if state.accept_rate_ema is None:
                state.accept_rate_ema = rate
            else:
                state.accept_rate_ema = (
                    1.0 - ema_alpha
                ) * state.accept_rate_ema + ema_alpha * rate
            state.last_update_round = self._round_ct
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
            state = self._touch(rid)
            state.last_confidences = conf
            state.last_update_round = self._round_ct

    def observe_round_executed_steps(self, executed_steps: int) -> None:
        """Re-key the just-decided round's cost attribution to the depth the
        round actually executed.

        ``get_steps_for_batch`` keys the round's wall-gap attribution at
        decision time from ``current_steps`` — the PARKED configuration. An
        early-exit round executes a different depth (an in-draft stop at
        g_v < k_max, or a pre-draft depth-0 skip with no drafter at all),
        and the gap folded at the next invocation must price the
        configuration that ran, not the parked one (a depth-1 stop billed
        to k_max makes the deep config look cheap and the shallow one
        unpriced). The worker calls this as soon as the executed depth is
        known; a pending key from a same-round switch (already cleared) or
        no prior decision is a no-op.
        """
        if self._last_round_cost_key is None:
            return
        self._last_round_cost_key = (
            self._last_round_cost_key[0],
            executed_steps,
        )

    # -- In-round early-exit drafting --

    @property
    def available_steps(self) -> list[int]:
        """Candidates whose runtime states exist (sorted ascending)."""
        return list(self._available_steps)

    def early_exit_active(self, batch_size: int) -> bool:
        """Whether in-round early-exit drafting governs this batch size.

        When True, intermediate depths are reached by stopping inside the
        round; the step decision only parks between speculation-off and the
        deepest candidate (see ``_maybe_switch``).
        """
        return (
            self._cfg.early_exit
            and batch_size <= self._cfg.early_exit_max_batch
            and max(self._available_steps) >= 2
        )

    def early_exit_stop_price(self, batch_size: int) -> float:
        """Price (expected correct drafts) one more drafter step must beat.

        Derivation: one extra drafter step costs ``slope`` seconds — the
        per-step slope of the round-cost surface at this batch size,
        measured between the cheapest and deepest available configurations
        (EMA-corrected, table-seeded ``_step_cost``). At the current
        goodput the same wall time is worth ``slope * goodput`` committed
        tokens, so the step pays iff its expected additional correct
        drafts reach that. An explicit ``early_exit_stop_price`` config
        value short-circuits the derivation; without any cost data the
        flat default applies.
        """
        if self._cfg.early_exit_stop_price is not None:
            return self._cfg.early_exit_stop_price
        if not self._cost_table and not self._cost_ema:
            return _EARLY_EXIT_FALLBACK_STOP_PRICE
        g_hi = max(self._available_steps)
        g_lo = min(self._available_steps)
        if g_hi <= g_lo:
            return _EARLY_EXIT_FALLBACK_STOP_PRICE
        slope = (
            self._step_cost(batch_size, g_hi) - self._step_cost(batch_size, g_lo)
        ) / (g_hi - g_lo)
        if slope <= 0.0:
            return _EARLY_EXIT_FALLBACK_STOP_PRICE
        goodput = self._goodput(batch_size, None, self.current_steps)
        if goodput <= 0.0:
            return _EARLY_EXIT_FALLBACK_STOP_PRICE
        return slope * goodput

    # -- Decision internals --

    def _maybe_switch(self, batch_size: int, rids: list[str] | None) -> None:
        if (
            self._round_ct - self._last_switch_round
            < self._cfg.switch_cooldown_rounds
        ):
            return
        current = self.current_steps
        # Ladder switching clamps the MOVE, not the comparison: the goodput
        # argmax runs over ALL candidates (a distant winner must stay
        # visible — restricting the argmax to the neighborhood parks the
        # policy behind marginal intermediates that don't individually clear
        # the hysteresis margin; observed at conc 4, k=0 vs k=4 +26%), then
        # the actual step moves at most max_switch_distance indices toward
        # it. Distant configs are reached over successive cooldown windows
        # (0 -> 2 -> 4 -> 8), keeping every runtime-state swap incremental
        # and bounding the g=0 catch-up gap a single re-entry must drain.
        available = self._available_steps
        if self.early_exit_active(batch_size):
            # In-round early exit reaches intermediate depths by stopping
            # the draft loop, not by runtime-state swaps: the swap decision
            # only weighs speculation-off against the deepest candidate.
            # `current` stays comparable so the policy can still leave an
            # intermediate state it parked at before early exit applied.
            deepest = max(self._available_steps)
            available = sorted(
                {s for s in self._available_steps if s in (0, deepest)}
                | {current}
            )
        try:
            current_idx = available.index(current)
        except ValueError:  # defensive; current is kept inside available
            current_idx = min(
                range(len(available)),
                key=lambda i: abs(available[i] - current),
            )
        goodputs = {
            steps: self._goodput(batch_size, rids, steps)
            for steps in available
        }
        best = max(available, key=lambda s: goodputs[s])
        self._debug_decision(batch_size, rids, goodputs, best)
        if best == current or current not in goodputs:
            return
        # Hysteresis: runtime state swaps have cost, so only move when the
        # predicted goodput gain clears the margin.
        if goodputs[best] <= goodputs[current] * (1.0 + self._cfg.switch_margin):
            return
        best_idx = available.index(best)
        max_distance = self._cfg.max_switch_distance
        step_idx = current_idx + max(
            -max_distance, min(max_distance, best_idx - current_idx)
        )
        best = available[step_idx]
        if best == current:
            return
        self.current_steps = best
        self._switch_ct += 1
        self._last_switch_round = self._round_ct
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

    def _debug_decision(
        self,
        batch_size: int,
        rids: list[str] | None,
        goodputs: dict[int, float],
        best: int,
    ) -> None:
        """Every 16th decision, log the estimator internals the argmax used
        (side="d" records in the spec telemetry stream) — the only way to
        attribute mispricing to a specific estimator on hardware."""
        self._debug_ct = getattr(self, "_debug_ct", 0) + 1
        if self._debug_ct % 16 != 1:
            return
        try:
            from sglang.srt.speculative.spec_telemetry import get_spec_telemetry

            telem = get_spec_telemetry()
        except ImportError:
            telem = None
        if telem is None:
            return
        telem._write(
            {
                "side": "d",
                "B": batch_size,
                "cur": self.current_steps,
                "best": best,
                "gp": {str(g): round(v, 3) for g, v in goodputs.items()},
                "cost_ms": {
                    str(g): round(1e3 * self._step_cost(batch_size, g), 3)
                    for g in self._available_steps
                },
                "E": {
                    str(g): round(
                        self._expected_correct_drafts_batch(batch_size, rids, g), 3
                    )
                    for g in self._available_steps
                },
                "anchor": {
                    str(g): round(v, 3)
                    for g, v in self._realized_correct_ema.items()
                },
                "target_rate": round(self._decay_target_rate(), 3),
            }
        )

    def _goodput(
        self, batch_size: int, rids: list[str] | None, steps: int
    ) -> float:
        """Expected committed tokens per second at *steps* for this batch."""
        expected_correct_drafts = self._expected_correct_drafts_batch(
            batch_size, rids, steps
        )
        # Every request always commits the bonus token on top of its drafts.
        expected_accept_tokens = expected_correct_drafts + batch_size
        return expected_accept_tokens / self._step_cost(batch_size, steps)

    def _expected_correct_drafts_batch(
        self, batch_size: int, rids: list[str] | None, steps: int
    ) -> float:
        """Batch E[num_correct_drafts | steps] (drafts only, no bonus).

        With realized data anywhere on the accept curve, each request gets
        ``E_global(steps) * clamp(chain_i / baseline, 1/MOD, MOD)``: the
        realized anchor sets the level and the per-request chain signal only
        modulates around it (raw confidence chains overstate deep-g returns).
        The baseline is the per-g rolling EMA of round-mean chains — see
        ``_observe_chain_mean`` — so a confident round modulates > 1 even at
        batch size 1. Without realized data the chain signal is used directly.
        """
        if steps <= 0:
            return 0.0
        anchor = self._anchored_correct_drafts(steps)
        if not rids:
            if anchor is not None:
                return batch_size * anchor
            return batch_size * _geometric_expected(
                self._cfg.prior_accept_rate, steps
            )
        chains = [self._chain_correct_drafts(rid, steps) for rid in rids]
        baseline = self._observe_chain_mean(steps, sum(chains) / len(chains))
        if anchor is None:
            return sum(chains)
        if baseline <= 0.0:
            return anchor * len(chains)
        modulation = self._cfg.confidence_modulation
        return sum(
            anchor
            * min(max(chain / baseline, 1.0 / modulation), modulation)
            for chain in chains
        )

    def _observe_chain_mean(self, steps: int, batch_chain_mean: float) -> float:
        """Fold this round's batch-mean chain signal at *steps* into the
        rolling baseline and return the PRE-update baseline (the recent
        typical chain level at this g, excluding the current observation).

        Before the rolling baseline has data at this g, falls back to the
        batch mean itself — the original same-round behavior, where the
        modulation ratio is 1.0 for a lone request. Once cross-round data
        exists, a confident round at batch size 1 yields a ratio > 1 (deeper
        g can be justified) and an unconfident one < 1.
        """
        prev = self._chain_mean_ema.get(steps)
        ema_alpha = self._cfg.ema_alpha
        self._chain_mean_ema[steps] = (
            batch_chain_mean
            if prev is None
            else (1.0 - ema_alpha) * prev + ema_alpha * batch_chain_mean
        )
        return batch_chain_mean if prev is None else prev

    def _anchored_correct_drafts(self, steps: int) -> float | None:
        """E_global[num_correct_drafts | steps] from the realized accept curve.

        Exact EMA when this g has run; otherwise chain-interpolated from the
        nearest sampled g (invert its EMA into a constant per-position accept
        rate, extend that chain to *steps*). ``None`` before any realized data.
        """
        curve = self._curve_expected(steps)
        if not self._realized_correct_ema:
            return curve  # calibrated truth (None without a curve)
        exact = self._anchor_read(steps)
        if exact is not None:
            return max(exact, 0.0)
        nearest_steps = min(
            self._realized_correct_ema, key=lambda g: (abs(g - steps), g)
        )
        implied_rate = _implied_accept_rate(
            self._anchor_read(nearest_steps), nearest_steps
        )
        interpolated = _geometric_expected(implied_rate, steps)
        # With a calibrated curve, cross-g interpolation only modulates it.
        if curve is not None:
            base = self._curve_expected(nearest_steps)
            if base and base > 0:
                ratio = min(max(interpolated / _geometric_expected(
                    _implied_accept_rate(base, nearest_steps), steps
                ), 0.25), 4.0)
                return curve * ratio
        return interpolated

    def _curve_expected(self, steps: int) -> float | None:
        """E[num_correct_drafts | steps] from the calibrated prior curve;
        chain-extends past the curve's depth from its last point."""
        curve = self._cfg.acceptance_prior_curve
        if not curve or steps <= 0:
            return None if not curve else 0.0
        if steps <= len(curve):
            return curve[steps - 1]
        rate = _implied_accept_rate(curve[-1], len(curve))
        return _geometric_expected(rate, steps)

    def _anchor_read(self, steps: int) -> float | None:
        """Realized-accept EMA at *steps*, decayed toward the calibrated
        curve by estimator_decay^age (frozen garbage must wash out)."""
        ema = self._realized_correct_ema.get(steps)
        if ema is None:
            return None
        base = self._curve_expected(steps)
        decay = self._cfg.estimator_decay
        if base is None or decay >= 1.0:
            return ema
        age = max(self._round_ct - self._realized_round.get(steps, self._round_ct), 0)
        w = decay**age
        return base + (ema - base) * w

    def _chain_correct_drafts(self, rid: str, steps: int) -> float:
        """Per-request chain E[num_correct_drafts | steps] (no bonus).

        Evidence priority: latest per-position confidences, then the
        accept-rate EMA, then the global prior. Stale evidence (no fresh
        update since ``last_update_round``) is decayed toward the prior by
        ``evidence_decay`` per unupdated round, so g=0 phases — which
        generate no updates — cannot freeze pessimistic evidence forever.
        """
        # Stale/absent evidence reverts to the realized-anchor-implied rate
        # when realized data exists, NOT the config prior: an optimistic
        # prior re-convinces the policy every decay time-constant that
        # speculation pays, probe-loops, and burns throughput (observed at
        # conc 256: 5.7%/round switch rate, half the no-spec throughput).
        target = self._decay_target_rate()
        state = self._reqs.get(rid)
        if state is None:
            return _geometric_expected(target, steps)
        decay = self._staleness_decay(state)
        if state.last_confidences:
            conf = state.last_confidences
            total = 0.0
            chain_prob = 1.0
            for d in range(steps):
                a_hat = conf[d] if d < len(conf) else conf[-1]
                a_hat = min(max(a_hat, 0.0), 1.0)
                chain_prob *= target + (a_hat - target) * decay
                total += chain_prob
            return total
        if state.accept_rate_ema is not None:
            return _geometric_expected(
                target + (state.accept_rate_ema - target) * decay, steps
            )
        return _geometric_expected(target, steps)

    def _decay_target_rate(self) -> float:
        """Per-position rate that stale evidence reverts to: implied by the
        calibrated curve when present, else the deepest realized g, else
        the prior."""
        curve = self._cfg.acceptance_prior_curve
        if curve:
            return _implied_accept_rate(curve[-1], len(curve))
        if self._realized_correct_ema:
            g = max(self._realized_correct_ema)
            return _implied_accept_rate(self._anchor_read(g), g)
        return self._cfg.prior_accept_rate

    def _staleness_decay(self, state: _RequestState) -> float:
        """Multiplier in (0, 1] pulling stale evidence toward the prior.

        A request updated for round N is fresh for the round N+1 decision;
        each further round without an update compounds ``evidence_decay``.
        """
        evidence_decay = self._cfg.evidence_decay
        if evidence_decay >= 1.0:
            return 1.0
        stale_rounds = self._round_ct - state.last_update_round - 1
        if stale_rounds <= 0:
            return 1.0
        return evidence_decay**stale_rounds

    # -- Cost internals --

    def _observe_round_gap(self, now: float, round_idx: int | None) -> None:
        """Fold WINDOW-MEAN effective round cost into the cost EMA.

        Per-pair wall gaps are dishonest in both tails: under overlap
        run-ahead at high batch the host dispatches rounds back-to-back, so
        consecutive-pair gaps measure dispatch rate, not execution (observed
        online at B=62: a k=4 cost EMA of 15.6 ms vs ~40 ms real, while
        RUNNING k=4 — the same artifact the offline fitter hit). Instead:
        when the last ``_COST_WINDOW`` policy invocations all executed the
        same (bucket, steps) key, attribute mean cost = (t_now − t_first) /
        (worker rounds spanned). The denominator counts prefill rounds, so a
        config's k-dependent prefill surcharge (draft-extend) is priced in —
        the throughput-relevant effective cost. Switch rounds clear the key
        (None breaks window uniformity), excluding swap transients by
        construction.
        """
        if round_idx is None:
            self._cost_window.clear()
            return
        # The PREVIOUS invocation's key describes the round whose cost the
        # gap to ``now`` reflects; executed-depth corrections (early exit)
        # were applied to it via observe_round_executed_steps.
        self._cost_window.append((round_idx, now, self._last_round_cost_key))
        if len(self._cost_window) > _COST_WINDOW + 1:
            self._cost_window.pop(0)
        if len(self._cost_window) < _COST_WINDOW + 1:
            return
        first = self._cost_window[0]
        keys = {e[2] for e in self._cost_window[:-1]}
        if len(keys) != 1 or None in keys:
            return
        ri_span = round_idx - first[0]
        if ri_span < _COST_WINDOW:
            return
        mean_cost = (now - first[1]) / ri_span
        if mean_cost <= 0.0:
            return
        key = next(iter(keys))
        prev = self._cost_ema.get(key)
        ema_alpha = self._cfg.ema_alpha
        self._cost_ema[key] = (
            mean_cost
            if prev is None
            else (1.0 - ema_alpha) * prev + ema_alpha * mean_cost
        )
        self._cost_ema_round[key] = self._round_ct
        # Slide by half a window so successive updates aren't fully
        # correlated samples of the same span.
        del self._cost_window[: _COST_WINDOW // 2]

    def _cost_ema_read(self, key: tuple[int, int]) -> float | None:
        """Cost EMA decayed toward the calibrated table by estimator_decay^age.

        A transient-poisoned EMA at a config the policy then avoids would
        otherwise freeze and self-confirm (observed: cost(B=23, g=2) stuck
        at 59 ms vs 21 ms measured, while evidence-beats-optimism blocked
        the honest table).
        """
        ema = self._cost_ema.get(key)
        if ema is None:
            return None
        decay = self._cfg.estimator_decay
        if decay >= 1.0 or not self._cost_table:
            return ema
        base = self._table_cost(key[0], key[1])
        if base is None or base <= 0:
            return ema
        age = max(self._round_ct - self._cost_ema_round.get(key, self._round_ct), 0)
        return base + (ema - base) * decay**age

    def _step_cost(self, batch_size: int, steps: int) -> float:
        """Corrected cost: realized-gap EMA when available, else derived
        from observed EMA buckets, else table.

        Candidates must be priced in commensurate units: comparing one
        configuration at a realized ~10 ms against another at the flat 1.0 s
        default makes the argmax meaningless (observed on first GPU run).
        """
        bucket = self._cost_bucket(batch_size)
        ema = self._cost_ema_read((bucket, steps))
        if ema is not None:
            return ema

        # Unseen bucket: derive an estimate from observed EMAs. Same-g
        # evidence at another bucket comes FIRST — it is config-honest.
        # Token-linear scaling within the bucket is only a fallback: at
        # small batch the verify width is nearly free (latency-bound), so
        # scaling k=0 cost by (g+1) wildly overprices wide configs
        # (observed at conc 4: derived 30 ms vs 9.6 ms measured for k=4,
        # parking the policy at k=0 against a +26% optimum).
        derived: float | None = None
        if self._cost_ema:
            same_g = [
                (b, self._cost_ema_read((b, g)))
                for (b, g) in self._cost_ema
                if g == steps
            ]
            if same_g:
                log_b = math.log(max(bucket, 1))
                b_near, c_near = min(
                    same_g, key=lambda bc: abs(math.log(max(bc[0], 1)) - log_b)
                )
                # Transfer across buckets via the TABLE's measured B-shape
                # at this g (rank-1 cost model): raw transfer applied bucket-1
                # evidence (~9 ms) at bucket 256 (real ~100 ms) and priced a
                # probe festival. Clamp the ratio for off-table sanity.
                ratio = 1.0
                if self._cost_table:
                    t_here = self._table_cost(bucket, steps)
                    t_near = self._table_cost(b_near, steps)
                    if t_here > 0 and t_near > 0:
                        ratio = min(max(t_here / t_near, 0.25), 32.0)
                derived = c_near * ratio
            else:
                same_bucket = [
                    (g, self._cost_ema_read((b, g)))
                    for (b, g) in self._cost_ema
                    if b == bucket
                ]
                if same_bucket:
                    g_near, c_near = min(
                        same_bucket, key=lambda gc: abs(gc[0] - steps)
                    )
                    derived = c_near * (steps + 1) / (g_near + 1)

        # Evidence beats optimism: once this g has a realized-gap EMA at ANY
        # bucket, the derived estimate stands alone — re-applying the table's
        # optimistic claim would resurrect configs that nearby-bucket
        # evidence already refuted (observed at concurrency 64 on GH200:
        # periodic k>0 probe excursions at ~3x step cost taxed a correct
        # k=0 steady state below the no-spec bar).
        if derived is not None and any(g == steps for (_, g) in self._cost_ema):
            return derived

        # Optimism under uncertainty, but only for never-visited g: a
        # never-observed configuration must stay reachable — pricing it
        # only from the incumbent's scaled EMA can freeze the policy on the
        # incumbent forever (no exploration); the table's optimistic claim
        # triggers the probe whose realized gap then corrects the EMA.
        table = self._table_cost(batch_size, steps) if self._cost_table else None
        estimates = [c for c in (derived, table) if c is not None]
        if estimates:
            return min(estimates)
        # Nothing known at all: flat cost, maximize expected accept tokens.
        return 1.0

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
            state.last_update_round = self._round_ct
            self._reqs[rid] = state
            while len(self._reqs) > self._cfg.max_tracked_requests:
                self._reqs.popitem(last=False)
        else:
            self._reqs.move_to_end(rid)
        return state


def early_exit_should_stop(marginal_gain: float, stop_price: float) -> bool:
    """In-round early-exit stop rule at one allowed depth: stop when the
    expected marginal correct drafts of one more drafter step fall short of
    the price. Ties continue (the step exactly pays for itself)."""
    return marginal_gain < stop_price


def early_exit_should_skip(
    first_draft_probs: "list[float] | tuple[float, ...]", stop_price: float
) -> bool:
    """Pre-draft DEPTH-0 decision: skip drafting entirely for the round.

    ``first_draft_probs`` are the per-request confidences of the FIRST
    draft token (``spec_info.topk_p[:, 0]``) — computed by the previous
    round's draft extend, so they are available before any drafter pass
    this round. Keeping the chain to depth >= 1 commits request *i*'s
    first draft with probability ``p1_i``, so the batch expected marginal
    correct drafts of drafting at all is ``sum_i p1_i``; it is priced
    against the same per-step stop price as the in-draft checks (the
    depth-0 analog of the d-step rule — same shape, evaluated one step
    earlier). Ties proceed, mirroring ``early_exit_should_stop``.
    """
    return early_exit_should_stop(sum(first_draft_probs), stop_price)


def early_exit_stop_depths(
    available_steps: "list[int] | tuple[int, ...]",
    allow_skip: bool = False,
    min_stop_depth: int = 1,
) -> list[int]:
    """Allowed early-exit stop depths given the candidates with built
    runtime states: every candidate strictly below the deepest (k_max).

    Includes 0 — the pre-draft depth-0 skip (no drafter at all for the
    round) — only when 0 itself has a runtime state; without a g=0 state
    the minimum depth stays 1. Depths >= 1 are the in-draft stop checks
    (``decide_early_exit_depth``); k_max is the run-to-end default, never
    a stop.
    """
    if not available_steps:
        return []
    k_max = max(available_steps)
    min_depth = 0 if allow_skip else max(1, min_stop_depth)
    return sorted(g for g in set(available_steps) if min_depth <= g < k_max)


def decide_early_exit_depth(
    marginal_gains: "list[float] | tuple[float, ...]",
    stop_depths: "list[int] | tuple[int, ...] | set[int]",
    k_max: int,
    stop_price: float,
) -> int:
    """Depth the early-exit draft loop stops at (the worker loop's pure twin).

    ``marginal_gains[d - 1]`` is the expected marginal correct drafts of
    running one more drafter forward when ``d`` drafts are already
    determined: the batch sum of ``chain_prob(d) * a_next``, where
    ``chain_prob(d)`` is the probability the first ``d`` drafts all commit
    and ``a_next`` is the proxy for the not-yet-drafted position (the
    ``d``-th draft's own probability — for ``d == 1`` that makes the gain
    the chain probability squared, since no drafter forward has run yet).

    The rule is evaluated only at depths in *stop_depths* (>= 1 and
    < ``k_max`` — the candidate depths with a built verify state), in
    increasing order, so a raw stop point between candidates lands on the
    nearest allowed depth at or past it. Returns the first depth whose
    gain fails ``stop_price``, else ``k_max``; the result is always >= 1
    and <= ``k_max``.
    """
    for depth in sorted(set(stop_depths)):
        if depth < 1 or depth >= k_max:
            continue
        if depth <= len(marginal_gains) and early_exit_should_stop(
            marginal_gains[depth - 1], stop_price
        ):
            return depth
    return k_max


def _geometric_expected(accept_rate: float, steps: int) -> float:
    """sum_{d=1..steps} accept_rate**d (chain expectation under constant alpha)."""
    accept_rate = min(max(accept_rate, 0.0), 1.0)
    if accept_rate >= 1.0:
        return float(steps)
    return accept_rate * (1.0 - accept_rate**steps) / (1.0 - accept_rate)


def _implied_accept_rate(expected_correct_drafts: float, steps: int) -> float:
    """Invert ``_geometric_expected``: the constant per-position accept rate
    whose *steps*-deep chain yields *expected_correct_drafts*.

    ``_geometric_expected(a, steps)`` is strictly increasing in ``a`` on
    [0, 1] with range [0, steps], so a short bisection suffices.
    """
    target = min(max(expected_correct_drafts, 0.0), float(steps))
    if target <= 0.0:
        return 0.0
    if target >= float(steps):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _geometric_expected(mid, steps) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
