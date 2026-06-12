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
    from sglang.srt.speculative.g0_gap_buffer import G0GapBuffer
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


if __name__ == "__main__":
    unittest.main()
