"""Per-category device-token-pool lock accounting (lane prefill-oom-1328).

The invariant checker reports ``available / evictable / protected /
session_held / uncached`` but collapses every locked page into ``protected``.
In the 2026-09-03 13:28Z incident the prefill pool reached ``available=0``
AND ``evictable=0`` and no instrument could say WHO held the 1.8M locked
tokens. This module splits the locked population by holder class, with owner
identity and age, so that

  * an allocation failure can be ATTRIBUTED (class + owner + age) instead of
    guessed at from an aggregate, and
  * a budgeted, preemptible transition class (lane ``transition-page-budget``)
    has the per-owner identity it needs for deadlines and preemption.

Shared taxonomy — adopted by transition-page-budget; do not fork it. If a
holder does not fit one of these classes, extend the taxonomy here first
(settle it between the lanes before either ships):

  Each locked category names WHAT HOLDS THE PAGE and WHY IT CANNOT BE
  RECLAIMED, never which engine phase the holder is in (the same holder
  class exists on the prefill and decode arms):

  free               pages in the allocator's free set; nothing holds them.
  evictable          tree rows with zero lock refs; reclaimable by eviction.
  locked.transfer    held by an IN-FLIGHT INTER-RANK KV TRANSFER (prefill
                     done, handover to the decode peer not yet complete:
                     ``disagg_prefill_inflight_queue``). Unreclaimable while
                     the transfer reads the pages. Owner: rid / bootstrap
                     room. PREEMPTIBLE: re-send or pull from the peer.
  locked.forward     held because THE OWNER REQUEST STILL HAS A FORWARD TO
                     RUN (chunked prefill, pipeline micro-batches, decode
                     steps): the pages are attention input for the owner's
                     next step. Unreclaimable without aborting the owner.
                     Owner: rid.
  locked.admission   held by an ADMITTED-BUT-NOT-RUNNING request (waiting +
                     bootstrap queues hold matched-prefix locks). Reclaimable
                     only by un-admitting the request. Owner: rid.
  locked.store       held by an IN-FLIGHT DEVICE->HOST STORE TRANSITION
                     (hicache write-through: ``ongoing_write_through``
                     acks). PREEMPTIBLE: the store can be abandoned with its
                     pages still in the pool. Owner: the write batch.
  locked.migration   held by a PARK / MIGRATION / HANDOVER TRANSITION in
                     progress (session-parking, rank migration). PREEMPTIBLE:
                     re-driveable. Owner: the migration target.
  session_held       session-controller held (as the invariant checker
                     already reports it).
  uncached           allocated outside the tree (in-flight extend rows).

``free + evictable + locked.* + session_held + uncached == total`` is the
same invariant the checker enforces; this module only re-partitions it.

The ``locked.migration`` category is wired through a provider hook
(``migration_held`` on the scheduler, if present) because those holders live
in other lanes' components (session-parking, handover, migration agents);
those lanes wire their holders into THIS taxonomy rather than inventing a
parallel set.

Wiring today: the disagg-PREFILL arm (log line at a cadence + load-snapshot
fields + attributed OOM message). The decode arm reuses the same taxonomy
via ``LoadInquirerAdapter`` when its holders are wired.

All access is defensive: a diagnostic must never kill the engine. Any
failure inside the breakdown degrades to omitting the offending category.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Categories that describe pages locked by a *transition* (the budgeted,
# preemptible class of the transition-page-budget design).
TRANSITION_CATEGORIES = ("transfer", "store", "migration")


@dataclass
class _Cat:
    n: int = 0
    tokens: int = 0
    oldest_age_s: float = -1.0
    # up to 3 largest owners: (rid, room, tokens, age_s)
    owners: List[Any] = field(default_factory=list)


class LoadInquirerAdapter:
    """Duck-type a SchedulerLoadInquirer into the attribute surface that
    ``compute_pool_lock_breakdown`` reads (the inquirer exposes getters, not
    attributes). Only used for the load-snapshot path; the scheduler itself
    is passed directly."""

    def __init__(self, inquirer):
        self._inq = inquirer

    @property
    def disagg_prefill_inflight_queue(self):
        try:
            return self._inq.get_disagg_prefill_inflight_queue()
        except Exception:
            return None

    @property
    def waiting_queue(self):
        try:
            return self._inq.get_waiting_queue()
        except Exception:
            return None

    @property
    def running_batch(self):
        try:
            return self._inq.get_running_batch()
        except Exception:
            return None

    @property
    def chunked_req(self):
        try:
            return self._inq.get_chunked_req()
        except Exception:
            return None

    @property
    def disagg_prefill_bootstrap_queue(self):
        try:
            return self._inq.get_disagg_prefill_bootstrap_queue()
        except Exception:
            return None

    @property
    def tree_cache(self):
        try:
            return self._inq.get_tree_cache()
        except Exception:
            return None

    @property
    def pool_stats_observer(self):
        return self._inq.pool_stats_observer

    @property
    def max_total_num_tokens(self):
        return self._inq.max_total_num_tokens


def _req_locked_tokens(req: "Req") -> int:
    try:
        pi = getattr(req, "prefix_indices", None)
        if pi is None:
            return 0
        return int(len(pi))
    except Exception:
        return 0


def _age(ts: float, now: float) -> float:
    try:
        return round(now - ts, 1) if ts and ts > 0 else -1.0
    except Exception:
        return -1.0


def _add_req(cat: _Cat, req: "Req", now: float, age_attr: str) -> None:
    toks = _req_locked_tokens(req)
    if toks <= 0:
        return
    cat.n += 1
    cat.tokens += toks
    ts = getattr(req, "time_stats", None)
    age = _age(getattr(ts, age_attr, 0.0) if ts is not None else 0.0, now)
    if age > cat.oldest_age_s:
        cat.oldest_age_s = age
    cat.owners.append(
        (getattr(req, "rid", "?"), getattr(req, "bootstrap_room", None), toks, age)
    )


def _iter_running_reqs(source: Any):
    for attr in ("running_batch", "last_batch", "chunked_req"):
        b = getattr(source, attr, None)
        if b is None:
            continue
        if hasattr(b, "reqs"):
            yield from b.reqs
        else:
            yield b  # chunked_req is a bare Req
    for attr in ("mbs", "running_mbs"):
        for b in getattr(source, attr, None) or []:
            if b is not None and hasattr(b, "reqs"):
                yield from b.reqs


def compute_pool_lock_breakdown(source: Any) -> Dict[str, Any]:
    """Partition the device token pool by holder category.

    ``source`` is the Scheduler (log/raise path) or a LoadInquirerAdapter
    (load-snapshot path); every access is a getattr, so both work.
    Returns a dict of category -> _Cat (for locked.*) plus scalar fields.
    Never raises.
    """
    now = time.monotonic()
    out: Dict[str, Any] = {}

    # --- locked.transfer: inter-rank KV transfer in flight ------------------
    cat = _Cat()
    for req in getattr(source, "disagg_prefill_inflight_queue", None) or []:
        _add_req(cat, req, now, "prefill_transfer_queue_entry_time")
    out["transfer"] = cat

    # --- locked.forward: the owner still has a forward to run ----------------
    cat = _Cat()
    seen = set()
    for req in _iter_running_reqs(source):
        rid = getattr(req, "rid", None)
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        _add_req(cat, req, now, "forward_entry_time")
    out["forward"] = cat

    # --- locked.admission: admitted, not running, holding match locks ------
    cat = _Cat()
    for req in getattr(source, "waiting_queue", None) or []:
        _add_req(cat, req, now, "wait_queue_entry_time")
    bq = getattr(source, "disagg_prefill_bootstrap_queue", None)
    for req in getattr(bq, "queue", None) or []:
        _add_req(cat, req, now, "prefill_bootstrap_queue_entry_time")
    out["admission"] = cat

    # --- locked.store: in-flight write-through D->H transitions -----------
    cat = _Cat()
    tree_cache = getattr(source, "tree_cache", None)
    ongoing = getattr(tree_cache, "ongoing_write_through", None)
    if ongoing:
        # Tokens are not cheaply recoverable from the ack bookkeeping; the
        # ack count bounds the store class and the sibling budget lane can
        # extend this with per-node sizes if it needs them.
        cat.n = len(ongoing)
        cat.tokens = -1
    out["store"] = cat

    # --- locked.migration: park/handover transitions (provider hook) -------
    cat = _Cat()
    prov = getattr(source, "migration_held", None)
    if callable(prov):
        try:
            held = prov()
            if held:
                cat.n = int(held.get("reqs", 0))
                cat.tokens = int(held.get("tokens", 0))
                cat.oldest_age_s = float(held.get("oldest_age_s", -1.0))
                cat.owners = list(held.get("owners", []))
        except Exception:
            pass
    out["migration"] = cat

    # --- pool-level scalars (same source as the invariant checker) ---------
    try:
        pso = getattr(source, "pool_stats_observer", None)
        ps = pso.get_pool_stats()
        out["free"] = int(ps.full_available_size)
        out["evictable"] = int(ps.full_evictable_size)
        out["session_held"] = int(pso.session_held_tokens())
        out["total"] = int(getattr(source, "max_total_num_tokens", 0))
    except Exception:
        pass
    return out


def _fmt_owners(cat: _Cat, k: int = 3) -> str:
    if not cat.owners:
        return ""
    top = sorted(cat.owners, key=lambda o: -int(o[2]))[:k]
    return ";".join(
        f"{str(rid)[:10]}:r{room}:t{tok}:a{age}" for rid, room, tok, age in top
    )


def format_pool_lock_breakdown(b: Dict[str, Any]) -> str:
    parts = []
    for name in ("free", "evictable", "session_held", "total"):
        if name in b:
            parts.append(f"{name}={b[name]}")
    locked_total = 0
    for name in ("transfer", "forward", "admission", "store", "migration"):
        cat = b.get(name)
        if cat is None:
            continue
        if name in TRANSITION_CATEGORIES and cat.tokens > 0:
            locked_total += cat.tokens
        parts.append(
            f"locked.{name}(n={cat.n},tok={cat.tokens},age={cat.oldest_age_s}"
            + (f"[{_fmt_owners(cat)}]" if cat.owners else "")
            + ")"
        )
    parts.append(f"locked_sum={locked_total}")
    return "[pool-locks] " + " ".join(parts)


def _get_pool_lock_breakdown_str(source: Any) -> str:
    try:
        return format_pool_lock_breakdown(compute_pool_lock_breakdown(source))
    except Exception:
        logger.debug("pool lock breakdown failed", exc_info=True)
        return "[pool-locks] unavailable"


_last_log_ts: Dict[int, float] = {}

# prefill-oom-1328 hotfix: fallback registry for caches that forbid ad-hoc
# attributes (setattr on the live tree cache raised silently on green g11
# and swallowed every [pool-locks] line for the whole arm).
_HOOKS: Dict[int, Any] = {}
_warned = False


def maybe_log_pool_locks(scheduler: "Scheduler") -> None:
    """Cadence-gated [pool-locks] log + lazy wiring of the failure-attribution
    hook onto the tree cache. Call once per scheduler-loop iteration; the cost
    is one clock read when the cadence has not elapsed. Never raises."""
    try:
        from sglang.srt.environ import envs

        interval = envs.SGLANG_POOL_LOCK_LOG_SECS.get()
        if not interval or interval <= 0:
            return
        tc = getattr(scheduler, "tree_cache", None)
        if tc is not None and getattr(tc, "pool_lock_breakdown_str", None) is None:
            # Wire the attribution hook: allocation.py's OOM raise sites call
            # available_and_evictable_str(tree_cache), which appends this.
            # hotfix: the wiring itself must never be able to kill the
            # cadence log. Fall back to the module-level registry when the
            # cache class forbids ad-hoc attributes.
            try:
                tc.pool_lock_breakdown_str = (
                    lambda s=scheduler: _get_pool_lock_breakdown_str(s)
                )
            except Exception:
                _HOOKS[id(tc)] = lambda s=scheduler: _get_pool_lock_breakdown_str(s)
        rank = id(scheduler)
        now = time.monotonic()
        if now - _last_log_ts.get(rank, 0.0) < interval:
            return
        _last_log_ts[rank] = now
        logger.info("%s", _get_pool_lock_breakdown_str(scheduler))
    except Exception:
        # hotfix: a silent except is how the g11 failure hid for a whole
        # arm. Warn once with the traceback so the next boot self-
        # diagnoses; keep retrying silently afterwards.
        global _warned
        if not _warned:
            _warned = True
            logger.warning("maybe_log_pool_locks failed; will keep retrying", exc_info=True)
