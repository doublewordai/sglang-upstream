"""Pure-Python unit tests for G0GapBuffer (g=0 drafter-gap bookkeeping).

The buffer module is stdlib-only by design, so these tests also run on
machines without torch/GPU deps: when the sglang package is not importable
we load the module straight from the source tree. Entries are opaque to the
buffer; plain ints stand in for the GPU tensor rows the worker stores.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # local dep-light runs; CI parses this call via AST anyway

    def register_cpu_ci(*args, **kwargs):
        return None


register_cpu_ci(est_time=1, suite="base-a-test-cpu")

try:
    from sglang.srt.speculative.g0_gap_buffer import (
        G0GapBuffer,
        partition_catch_up_chunks,
    )
except ImportError:
    _MODULE_PATH = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
        / "g0_gap_buffer.py"
    )
    _spec = importlib.util.spec_from_file_location("g0_gap_buffer", _MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)
    G0GapBuffer = _mod.G0GapBuffer
    partition_catch_up_chunks = _mod.partition_catch_up_chunks


class TestAppend(unittest.TestCase):
    def test_append_accumulates_per_rid_in_order(self):
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a", "b"], [10, 20], [100, 200])
        buf.append_round(["a", "b"], [11, 21], [101, 201])
        self.assertEqual(buf.num_buffered("a"), 2)
        self.assertEqual(buf.num_buffered("b"), 2)
        drained = buf.drain(["a", "b"])
        self.assertEqual(drained["a"], [(10, 100), (11, 101)])
        self.assertEqual(drained["b"], [(20, 200), (21, 201)])

    def test_append_without_hidden_states_stores_none(self):
        # STANDALONE drafts consume tokens only.
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a"], [10], None)
        self.assertEqual(buf.drain(["a"]), {"a": [(10, None)]})

    def test_num_buffered_unknown_rid_is_zero(self):
        buf = G0GapBuffer(max_gap=8)
        self.assertEqual(buf.num_buffered("nope"), 0)


class TestCap(unittest.TestCase):
    def test_append_signals_flush_at_max_gap(self):
        buf = G0GapBuffer(max_gap=3)
        self.assertFalse(buf.append_round(["a"], [1], [1]))
        self.assertFalse(buf.append_round(["a"], [2], [2]))
        self.assertTrue(buf.append_round(["a"], [3], [3]))

    def test_any_request_at_cap_signals_flush(self):
        buf = G0GapBuffer(max_gap=2)
        buf.append_round(["a"], [1], [1])
        # "b" joins later; "a" hits the cap first.
        self.assertTrue(buf.append_round(["a", "b"], [2, 9], [2, 9]))

    def test_max_gap_must_be_positive(self):
        with self.assertRaises(ValueError):
            G0GapBuffer(max_gap=0)


class TestDrain(unittest.TestCase):
    def test_drain_returns_only_requested_nonempty_rids(self):
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a", "b"], [1, 2], [1, 2])
        drained = buf.drain(["a", "c"])
        self.assertEqual(set(drained), {"a"})

    def test_drain_clears_everything_including_unrequested(self):
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a", "b"], [1, 2], [1, 2])
        buf.drain(["a"])
        self.assertTrue(buf.is_empty())
        self.assertEqual(buf.num_buffered("b"), 0)

    def test_drain_empty_buffer_returns_empty(self):
        buf = G0GapBuffer(max_gap=8)
        self.assertEqual(buf.drain(["a"]), {})


class TestRetainDrop(unittest.TestCase):
    def test_retain_keeps_only_running_rids(self):
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a", "b", "c"], [1, 2, 3], [1, 2, 3])
        buf.retain(["a", "c"])
        self.assertEqual(buf.num_buffered("a"), 1)
        self.assertEqual(buf.num_buffered("b"), 0)
        self.assertEqual(buf.num_buffered("c"), 1)

    def test_drop_clears_prefilled_rids(self):
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a", "b"], [1, 2], [1, 2])
        buf.drop(["a", "never_seen"])
        self.assertEqual(buf.num_buffered("a"), 0)
        self.assertEqual(buf.num_buffered("b"), 1)

    def test_append_after_drain_restarts_clean(self):
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a"], [1], [1])
        buf.drain(["a"])
        buf.append_round(["a"], [2], [2])
        self.assertEqual(buf.drain(["a"]), {"a": [(2, 2)]})


class TestSeedGap(unittest.TestCase):
    """Deferred drafter prefill: the prompt tail is seeded as the request's
    initial gap, capped at the trailing tail_cap entries."""

    def test_seed_under_cap_keeps_everything(self):
        buf = G0GapBuffer(max_gap=8)
        buf.seed_gap("a", [(1, 10), (2, 20)], tail_cap=4)
        self.assertEqual(buf.drain(["a"]), {"a": [(1, 10), (2, 20)]})

    def test_seed_keeps_only_the_trailing_tail_cap_entries(self):
        buf = G0GapBuffer(max_gap=8)
        buf.seed_gap("a", [(i, i * 10) for i in range(6)], tail_cap=3)
        self.assertEqual(buf.drain(["a"]), {"a": [(3, 30), (4, 40), (5, 50)]})

    def test_seed_is_also_capped_by_max_gap(self):
        buf = G0GapBuffer(max_gap=2)
        buf.seed_gap("a", [(i, None) for i in range(5)], tail_cap=4)
        self.assertEqual(buf.drain(["a"]), {"a": [(3, None), (4, None)]})

    def test_seed_replaces_previous_buffer(self):
        # A re-prefill makes any previous gap stale.
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a"], [99], [99])
        buf.seed_gap("a", [(1, 10)], tail_cap=4)
        self.assertEqual(buf.drain(["a"]), {"a": [(1, 10)]})

    def test_seed_with_no_entries_clears_the_rid(self):
        buf = G0GapBuffer(max_gap=8)
        buf.append_round(["a"], [99], [99])
        buf.seed_gap("a", [], tail_cap=4)
        self.assertEqual(buf.num_buffered("a"), 0)
        self.assertTrue(buf.is_empty())

    def test_tail_cap_must_be_positive(self):
        buf = G0GapBuffer(max_gap=8)
        with self.assertRaises(ValueError):
            buf.seed_gap("a", [(1, 10)], tail_cap=0)


class TestAppendDropOldest(unittest.TestCase):
    """Deferred-prefill mode: the per-request cap is enforced by dropping
    the OLDEST entries (the kept entries stay a contiguous tail), never by
    signaling a mid-phase flush (flushing runs the drafter — the tax
    deferral avoids)."""

    def test_drop_oldest_never_signals_flush(self):
        buf = G0GapBuffer(max_gap=2)
        for token in range(5):
            self.assertFalse(
                buf.append_round(["a"], [token], [token], drop_oldest=True)
            )

    def test_drop_oldest_keeps_the_trailing_max_gap_entries(self):
        buf = G0GapBuffer(max_gap=3)
        for token in range(6):
            buf.append_round(["a"], [token], [token * 10], drop_oldest=True)
        self.assertEqual(buf.drain(["a"]), {"a": [(3, 30), (4, 40), (5, 50)]})

    def test_drop_oldest_after_seed_trims_the_seed_first(self):
        buf = G0GapBuffer(max_gap=4)
        buf.seed_gap("a", [(i, None) for i in range(3)], tail_cap=3)
        buf.append_round(["a"], [10], [None], drop_oldest=True)
        buf.append_round(["a"], [11], [None], drop_oldest=True)
        # 3 seed + 2 decode = 5 > 4: the oldest seed entry is dropped.
        self.assertEqual(
            buf.drain(["a"]),
            {"a": [(1, None), (2, None), (10, None), (11, None)]},
        )

    def test_drop_oldest_is_per_request(self):
        buf = G0GapBuffer(max_gap=2)
        buf.append_round(["a"], [1], [None], drop_oldest=True)
        for token in (2, 3):
            buf.append_round(["a", "b"], [token, token], [None, None], drop_oldest=True)
        self.assertEqual(buf.num_buffered("a"), 2)
        self.assertEqual(buf.num_buffered("b"), 2)
        drained = buf.drain(["a", "b"])
        self.assertEqual(drained["a"], [(2, None), (3, None)])
        self.assertEqual(drained["b"], [(2, None), (3, None)])


class TestCatchUpAnchorArithmetic(unittest.TestCase):
    """The catch-up anchor arithmetic against a tail-only (deferred-prefill)
    buffer.

    The worker anchors each catch-up slice of a request at drafter positions
    ``[seq_len - tail - gap, seq_len - tail)``, where ``gap`` is the slice
    length and ``tail`` the entries remaining after it
    (``_draft_extend_for_g0_catch_up``). These tests simulate a request's
    life — deferred prefill seed, g=0 decode appends, drop-oldest trims —
    tracking each entry's true drafter position as its payload, and assert
    the anchor formula recovers exactly the kept positions: a tail-only
    rebuild only moves the anchored range's START forward, it never breaks
    the contiguity the per-request sequential drafter KV writes need.
    """

    def _drain_positions(self, buf, rid, seq_len, chunk_tokens):
        """Drain and partition like the worker; return the anchored position
        ranges and the entries' true positions per slice."""
        drained = buf.drain([rid])
        entries = drained.get(rid, [])
        num_entries = len(entries)
        chunks = partition_catch_up_chunks([(rid, num_entries)], chunk_tokens)
        anchored = []
        for chunk in chunks:
            for _, start, length in chunk:
                tail = num_entries - (start + length)
                # _draft_extend_for_g0_catch_up: prefix = seq_len - tail - gap
                anchor_start = seq_len - tail - length
                anchor_end = seq_len - tail
                true_positions = [
                    pos for pos, _ in entries[start : start + length]
                ]
                anchored.append(((anchor_start, anchor_end), true_positions))
        return anchored

    def _check(self, anchored, expect_start, seq_len):
        # Each slice's anchored range matches its entries' true positions,
        # and the union is exactly the contiguous kept tail.
        covered = []
        for (anchor_start, anchor_end), true_positions in anchored:
            self.assertEqual(
                list(range(anchor_start, anchor_end)), true_positions
            )
            covered.extend(true_positions)
        self.assertEqual(covered, list(range(expect_start, seq_len)))

    def test_seed_only_reentry_anchors_at_the_prompt_tail(self):
        # Prompt length 10, tail cap 4, no decode rounds: the catch-up must
        # rebuild drafter KV for positions [6, 10) anchored from seq_len 10.
        buf = G0GapBuffer(max_gap=64)
        buf.seed_gap("a", [(p, f"h{p}") for p in range(10)], tail_cap=4)
        anchored = self._drain_positions(buf, "a", seq_len=10, chunk_tokens=100)
        self._check(anchored, expect_start=6, seq_len=10)

    def test_seed_plus_decode_rounds_anchor_contiguously(self):
        # Prompt 10 / tail 4, then 3 g=0 decode rounds: entries cover
        # [6, 13) and seq_len is 13.
        buf = G0GapBuffer(max_gap=64)
        buf.seed_gap("a", [(p, f"h{p}") for p in range(10)], tail_cap=4)
        for p in (10, 11, 12):
            buf.append_round(["a"], [p], [f"h{p}"], drop_oldest=True)
        anchored = self._drain_positions(buf, "a", seq_len=13, chunk_tokens=100)
        self._check(anchored, expect_start=6, seq_len=13)

    def test_drop_oldest_moves_the_anchor_start_forward(self):
        # max_gap 5 forces drop-oldest during the parked phase; the anchor
        # must follow the kept tail, not the original seed start.
        buf = G0GapBuffer(max_gap=5)
        buf.seed_gap("a", [(p, None) for p in range(10)], tail_cap=4)
        for p in range(10, 14):  # 4 decode rounds: 8 entries -> keep last 5
            buf.append_round(["a"], [p], [None], drop_oldest=True)
        anchored = self._drain_positions(buf, "a", seq_len=14, chunk_tokens=100)
        self._check(anchored, expect_start=9, seq_len=14)

    def test_chunked_drain_keeps_slice_anchors_in_position_order(self):
        # A chunk budget below the gap splits the rebuild across extends;
        # every slice must still anchor exactly its own positions, in order.
        buf = G0GapBuffer(max_gap=64)
        buf.seed_gap("a", [(p, None) for p in range(10)], tail_cap=6)
        for p in (10, 11):
            buf.append_round(["a"], [p], [None], drop_oldest=True)
        anchored = self._drain_positions(buf, "a", seq_len=12, chunk_tokens=3)
        self.assertEqual(len(anchored), 3)  # 8 entries / 3-token budget
        self._check(anchored, expect_start=4, seq_len=12)


class TestPartitionCatchUpChunks(unittest.TestCase):
    """Token-budgeted partition of drained gaps into sequential catch-up
    extends (one unbounded extend over a 256-request batch's accumulated
    gaps OOMed on activations — the catch-up must be chunked)."""

    def _check_invariants(self, gap_lens, chunks, budget):
        # Every chunk respects the budget.
        for chunk in chunks:
            self.assertLessEqual(sum(length for _, _, length in chunk), budget)
            # A rid appears at most once per chunk (ForwardBatch rows are
            # per-request).
            rids = [rid for rid, _, _ in chunk]
            self.assertEqual(len(rids), len(set(rids)))
        # Per rid: slices are contiguous, in order across chunks, and cover
        # the whole gap exactly once.
        flat = [s for chunk in chunks for s in chunk]
        for rid, gap_len in gap_lens:
            slices = [(start, length) for r, start, length in flat if r == rid]
            expect_start = 0
            for start, length in slices:
                self.assertEqual(start, expect_start)
                self.assertGreater(length, 0)
                expect_start += length
            self.assertEqual(expect_start, gap_len)

    def test_everything_fits_in_one_chunk(self):
        gap_lens = [("a", 3), ("b", 2)]
        chunks = partition_catch_up_chunks(gap_lens, 8)
        self.assertEqual(chunks, [[("a", 0, 3), ("b", 0, 2)]])

    def test_requests_split_across_chunks_at_the_budget(self):
        gap_lens = [("a", 3), ("b", 2)]
        chunks = partition_catch_up_chunks(gap_lens, 4)
        self.assertEqual(
            chunks, [[("a", 0, 3), ("b", 0, 1)], [("b", 1, 1)]]
        )
        self._check_invariants(gap_lens, chunks, 4)

    def test_single_gap_larger_than_budget_is_split_in_order(self):
        gap_lens = [("a", 10)]
        chunks = partition_catch_up_chunks(gap_lens, 4)
        self.assertEqual(
            chunks, [[("a", 0, 4)], [("a", 4, 4)], [("a", 8, 2)]]
        )

    def test_many_requests_keep_per_rid_ordering(self):
        gap_lens = [("a", 5), ("b", 1), ("c", 7), ("d", 3)]
        chunks = partition_catch_up_chunks(gap_lens, 4)
        self._check_invariants(gap_lens, chunks, 4)

    def test_zero_length_gaps_are_skipped(self):
        chunks = partition_catch_up_chunks([("a", 0), ("b", 2)], 4)
        self.assertEqual(chunks, [[("b", 0, 2)]])

    def test_empty_input_yields_no_chunks(self):
        self.assertEqual(partition_catch_up_chunks([], 4), [])

    def test_budget_must_be_positive(self):
        with self.assertRaises(ValueError):
            partition_catch_up_chunks([("a", 1)], 0)


if __name__ == "__main__":
    unittest.main()
