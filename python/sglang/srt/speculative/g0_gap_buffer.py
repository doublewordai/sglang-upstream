"""Per-request (token, hidden_state) gap bookkeeping for g=0 rounds.

During g=0 (speculation-off) decode rounds the drafter is never invoked, so
its KV cache falls behind the target's by one position per round. To re-enter
g >= 1 the drafter must catch up over the gap, which needs the target hidden
state and the next token at every missed position. The worker buffers that
pair per request each g=0 round and drains the buffer into a catch-up
draft-extend when speculation turns back on (or when a request's gap hits
``max_gap``, to bound memory).

With deferred drafter prefill (``g0_defer_prefill``) a request admitted while
the worker is parked at g=0 skips the drafter prefill-extend entirely; the
tail of its prompt's (token, hidden_state) pairs is seeded here instead
(``seed_gap``) and the same catch-up rebuilds the drafter KV on re-entry. In
that mode memory is bounded by dropping the OLDEST entries past the cap
(``append_round(drop_oldest=True)``) rather than force-flushing a drafter
extend mid-phase — running the drafter during a parked g=0 phase is exactly
the tax deferral avoids.

Invariant either way: a request's buffered entries are a CONTIGUOUS TAIL of
its drafter positions ending at its current position. Entry ``p`` (in
drafter terms) pairs the input token of position ``p + 1`` with the target's
output hidden state of position ``p``, so a catch-up over ``gap`` entries of
a request at length ``seq_len`` always rebuilds drafter KV for positions
``[seq_len - gap, seq_len)`` — dropping old entries (or never seeding the
full prompt) only moves the rebuilt range's start forward, never makes it
non-contiguous. Drafter KV below the kept range is simply never rebuilt;
that degrades draft quality for those requests, never output correctness
(target verification rejects bad drafts).

Entries are opaque to this class (the worker stores GPU tensor rows); it is
stdlib-only so pure-CPU unit tests can import it without torch.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# (token, hidden_state) pair for one gap position. The token is the drafter's
# input token at that position (the token sampled by the round, i.e. position
# + 1 in target terms); hidden_state is the target's output hidden state at
# the position (None for draft architectures that don't consume it).
GapEntry = Tuple[Any, Optional[Any]]

# One per-request slice of a catch-up chunk: (rid, start, length) into that
# request's buffered gap entries.
GapSlice = Tuple[str, int, int]


def partition_catch_up_chunks(
    gap_lens: Sequence[Tuple[str, int]], chunk_tokens: int
) -> List[List[GapSlice]]:
    """Partition drained gaps into catch-up extends of bounded token count.

    Each returned chunk is a list of ``(rid, start, length)`` slices whose
    lengths sum to at most *chunk_tokens*; the caller runs one drafter extend
    per chunk, in order. Drafter KV writes are sequential per request, so a
    request whose gap exceeds the budget is split across consecutive chunks
    with its slices in position order (slice ``start`` is the offset into the
    request's gap). A rid never appears twice within one chunk, and requests
    keep their input order. Zero-length gaps are skipped.
    """
    if chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be >= 1, got {chunk_tokens}")
    chunks: List[List[GapSlice]] = []
    current: List[GapSlice] = []
    room = chunk_tokens
    for rid, gap_len in gap_lens:
        start = 0
        remaining = gap_len
        while remaining > 0:
            if room == 0:
                chunks.append(current)
                current = []
                room = chunk_tokens
            take = min(remaining, room)
            current.append((rid, start, take))
            start += take
            remaining -= take
            room -= take
    if current:
        chunks.append(current)
    return chunks


class G0GapBuffer:
    """Bounded per-rid append/drain buffer of drafter catch-up pairs."""

    def __init__(self, max_gap: int = 1024):
        if max_gap < 1:
            raise ValueError(f"g0_max_gap must be >= 1, got {max_gap}")
        self.g0_max_gap = max_gap
        self._entries: Dict[str, List[GapEntry]] = {}

    def is_empty(self) -> bool:
        return not self._entries

    def num_buffered(self, rid: str) -> int:
        entries = self._entries.get(rid)
        return len(entries) if entries else 0

    def append_round(
        self,
        rids: Sequence[str],
        tokens: Sequence[Any],
        hidden_states: Optional[Sequence[Any]],
        drop_oldest: bool = False,
    ) -> bool:
        """Record one g=0 round's (token, hidden_state) per request.

        Returns True when any request reached ``g0_max_gap`` — the caller
        must then run a catch-up flush (``drain``) before the next append.

        With ``drop_oldest=True`` (deferred-prefill mode) the cap is enforced
        by dropping each request's oldest entries instead — the kept entries
        stay a contiguous tail ending at the request's current position, so
        the catch-up anchors at the kept range — and the return value is
        always False (no mid-phase flush: flushing runs the drafter, which is
        the tax deferral avoids).
        """
        needs_flush = False
        for i, rid in enumerate(rids):
            entries = self._entries.setdefault(rid, [])
            entries.append(
                (tokens[i], hidden_states[i] if hidden_states is not None else None)
            )
            if len(entries) >= self.g0_max_gap:
                if drop_oldest:
                    del entries[: len(entries) - self.g0_max_gap]
                else:
                    needs_flush = True
        return needs_flush

    def seed_gap(self, rid: str, entries: Sequence[GapEntry], tail_cap: int) -> None:
        """Replace *rid*'s buffer with the trailing entries of a deferred
        drafter prefill.

        *entries* are the request's prompt-derived (token, hidden_state)
        pairs in position order, ending at the request's current position;
        at most ``min(tail_cap, g0_max_gap)`` trailing entries are kept
        (older prompt positions matter least for drafter KV and are the
        memory bound). Any previously buffered entries for *rid* are stale
        (the target re-prefilled) and are discarded.
        """
        if tail_cap < 1:
            raise ValueError(f"tail_cap must be >= 1, got {tail_cap}")
        kept = list(entries)[-min(tail_cap, self.g0_max_gap) :]
        if kept:
            self._entries[rid] = kept
        else:
            self._entries.pop(rid, None)

    def retain(self, rids: Iterable[str]) -> None:
        """Drop buffers for requests no longer running (finished/retracted)."""
        keep = set(rids)
        self._entries = {
            rid: entries for rid, entries in self._entries.items() if rid in keep
        }

    def drop(self, rids: Iterable[str]) -> None:
        """Drop buffers for requests that just (re-)prefilled: the prefill
        draft-extend resyncs the drafter, so any buffered gap is stale."""
        for rid in rids:
            self._entries.pop(rid, None)

    def drain(self, rids: Sequence[str]) -> Dict[str, List[GapEntry]]:
        """Return the buffered gaps for *rids* (only non-empty ones) and clear
        the whole buffer. Entries for rids not in *rids* belong to requests
        that left the running batch and are discarded."""
        out = {
            rid: self._entries[rid]
            for rid in rids
            if self._entries.get(rid)
        }
        self._entries = {}
        return out
