"""Pure-Python unit tests for PricedSpeculativeParams.

The policy module is stdlib-only by design, so these tests also run on
machines without torch/GPU deps: when the sglang package is not importable
we load the module straight from the source tree.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # local dep-light runs; CI parses this call via AST anyway

    def register_cpu_ci(*args, **kwargs):
        return None


register_cpu_ci(est_time=4, suite="base-a-test-cpu")

try:
    from sglang.srt.speculative.priced_spec_params import (
        PricedSpeculativeParams,
        load_cost_table,
        load_priced_config,
    )
except ImportError:
    _MODULE_PATH = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
        / "priced_spec_params.py"
    )
    _spec = importlib.util.spec_from_file_location("priced_spec_params", _MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod  # dataclasses resolves cls.__module__
    _spec.loader.exec_module(_mod)
    PricedSpeculativeParams = _mod.PricedSpeculativeParams
    load_cost_table = _mod.load_cost_table
    load_priced_config = _mod.load_priced_config


class PricedSpecParamsTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)

    def _write_cost_table(self, rows, name="cost.csv"):
        """rows: iterable of (batch_size, num_draft_tokens, step_seconds)."""
        path = str(Path(self._tmp_dir.name) / name)
        with open(path, "w") as f:
            f.write("batch_size,num_draft_tokens,step_seconds\n")
            for batch_size, num_draft_tokens, step_seconds in rows:
                f.write(f"{batch_size},{num_draft_tokens},{step_seconds!r}\n")
        return path

    def _cfg(self, candidate_steps, **overrides):
        cfg = {
            "policy": "priced",
            "candidate_steps": candidate_steps,
            "prior_accept_rate": 1.0,
            "switch_margin": 0.0,
            "ema_alpha": 1.0,
            "warmup_rounds": 0,
        }
        cfg.update(overrides)
        return cfg


class TestGoodputDecision(PricedSpecParamsTestBase):
    def test_flat_costs_pick_largest_steps_with_perfect_acceptance(self):
        # (a) Flat cost surface: every extra correct draft is free, so the
        # goodput argmax is the largest candidate when acceptance is perfect.
        cost_table = self._write_cost_table(
            [(1, g + 1, 0.01) for g in (1, 2, 4, 8)]
        )
        policy = PricedSpeculativeParams(
            initial_steps=1, cfg=self._cfg([1, 2, 4, 8], cost_table=cost_table)
        )
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 8)

    def test_cost_cliff_caps_steps_below_the_cliff(self):
        # (b) g=1..3 cost 1.0, g=4 costs 10.0: even with perfect acceptance
        # the marginal commit (4+1 vs 3+1 tokens) cannot pay for the cliff.
        cost_table = self._write_cost_table(
            [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 10.0)]
        )
        policy = PricedSpeculativeParams(
            initial_steps=1, cfg=self._cfg([1, 2, 3, 4], cost_table=cost_table)
        )
        chosen = policy.get_steps_for_batch(1, rids=["r0"])
        self.assertLessEqual(chosen, 3)
        self.assertEqual(chosen, 3)

    def test_hysteresis_suppresses_marginal_gains(self):
        # (c) goodput(3) is only ~1.5% above goodput(2); a 3% switch margin
        # must hold the current step, a 0 margin must take the gain.
        cost_3 = 4.0 / (3.0 * 1.015)  # goodput(3) = 1.015 * goodput(2)
        cost_table = self._write_cost_table([(1, 3, 1.0), (1, 4, cost_3)])

        held = PricedSpeculativeParams(
            initial_steps=2,
            cfg=self._cfg([2, 3], cost_table=cost_table, switch_margin=0.03),
        )
        self.assertEqual(held.get_steps_for_batch(1, rids=["r0"]), 2)

        eager = PricedSpeculativeParams(
            initial_steps=2,
            cfg=self._cfg([2, 3], cost_table=cost_table, switch_margin=0.0),
        )
        self.assertEqual(eager.get_steps_for_batch(1, rids=["r0"]), 3)

    def test_per_request_evidence_shifts_the_decision(self):
        # (d) Same policy, same costs: a low-acceptance request pins g=1, a
        # high-confidence request justifies g=4.
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 2.0)])
        policy = PricedSpeculativeParams(
            initial_steps=1,
            cfg=self._cfg([1, 4], cost_table=cost_table, switch_cooldown_rounds=0),
        )

        # Realized rejections drive the low request's accept-rate EMA to 0.
        policy.on_verify_complete([0], batch_size=1, rids=["lo"], num_steps=4)
        self.assertEqual(policy.get_steps_for_batch(1, rids=["lo"]), 1)

        # High per-position confidences make the deep chain worth its cost.
        policy.observe_draft_confidences(["hi"], [[0.99, 0.99, 0.99, 0.99]])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["hi"]), 4)

        # Back to the low request: the policy drops down again.
        self.assertEqual(policy.get_steps_for_batch(1, rids=["lo"]), 1)

    def test_confidences_take_priority_over_accept_rate_ema(self):
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 2.0)])
        policy = PricedSpeculativeParams(
            initial_steps=1, cfg=self._cfg([1, 4], cost_table=cost_table)
        )
        # Bad realized history, then fresh high confidences: source priority
        # says the most recent confidences win.
        policy.on_verify_complete([0], batch_size=1, rids=["r0"], num_steps=4)
        policy.observe_draft_confidences(["r0"], [[0.99, 0.99, 0.99, 0.99]])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 4)

    def test_unknown_rids_fall_back_to_the_global_prior(self):
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 2.0)])
        # prior 1.0: goodput(4) = 5/2 > goodput(1) = 2/1.
        policy = PricedSpeculativeParams(
            initial_steps=1, cfg=self._cfg([1, 4], cost_table=cost_table)
        )
        self.assertEqual(policy.get_steps_for_batch(1, rids=["never_seen"]), 4)
        # prior 0.1: E(4) ~ 0.111 -> goodput(4) ~ 0.56 < goodput(1) = 1.1.
        policy = PricedSpeculativeParams(
            initial_steps=4,
            cfg=self._cfg([1, 4], cost_table=cost_table, prior_accept_rate=0.1),
        )
        self.assertEqual(policy.get_steps_for_batch(1, rids=["never_seen"]), 1)

    def test_warmup_holds_the_initial_step(self):
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 9, 1.0)])
        policy = PricedSpeculativeParams(
            initial_steps=1,
            cfg=self._cfg([1, 8], cost_table=cost_table, warmup_rounds=3),
        )
        for _ in range(3):
            self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 1)
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 8)

    def test_idle_rounds_do_not_decide_or_consume_warmup(self):
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 9, 1.0)])
        policy = PricedSpeculativeParams(
            initial_steps=1,
            cfg=self._cfg([1, 8], cost_table=cost_table, warmup_rounds=0),
        )
        self.assertEqual(policy.get_steps_for_batch(0), 1)
        self.assertEqual(policy._round_ct, 0)


class TestOnlineCostCorrection(PricedSpecParamsTestBase):
    def test_realized_gaps_override_a_wrong_table(self):
        # (e) The table claims g=4 costs the same as g=1, so the policy jumps
        # to 4 — then the realized wall gaps (10s at g=4, 1s at g=1) correct
        # the cost surface and pull it back to 1 for good.
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 1.0)])
        cfg = self._cfg(
            [1, 4], cost_table=cost_table, ema_alpha=0.5, switch_cooldown_rounds=0
        )

        clock = {"t": 0.0}
        with mock.patch("time.perf_counter", new=lambda: clock["t"]):
            policy = PricedSpeculativeParams(initial_steps=1, cfg=cfg)
            chosen = []
            for round_idx in range(1, 13):
                steps = policy.get_steps_for_batch(
                    1, rids=["r0"], round_idx=round_idx
                )
                chosen.append(steps)
                clock["t"] += 10.0 if steps == 4 else 1.0

        self.assertEqual(chosen[0], 4)  # trusted the wrong table
        self.assertTrue(
            all(steps == 1 for steps in chosen[2:]),
            f"online correction did not stick: {chosen}",
        )

    def test_switch_cooldown_suppresses_thrash(self):
        # Alternating per-request evidence flips the optimum every round;
        # the cooldown must hold the line between swaps.
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 2.0)])
        policy = PricedSpeculativeParams(
            initial_steps=1,
            cfg=self._cfg([1, 4], cost_table=cost_table, switch_cooldown_rounds=8),
        )
        policy.on_verify_complete([0], batch_size=1, rids=["lo"], num_steps=4)
        policy.observe_draft_confidences(["hi"], [[0.99] * 4])
        switches = 0
        last = policy.current_steps
        for i in range(8):
            steps = policy.get_steps_for_batch(1, rids=["hi" if i % 2 else "lo"])
            if steps != last:
                switches += 1
                last = steps
        self.assertLessEqual(switches, 1, "cooldown failed to suppress thrash")

    def test_unseen_buckets_price_from_observed_emas(self):
        # No table: once one (bucket, g) has a realized gap, other g at the
        # same bucket must be priced commensurately (token-ratio scaled),
        # not at the flat 1.0 default (observed thrashing on first GPU run).
        # warmup pins steps=1 so the realized gap lands at (bucket 1, g=1).
        cfg = self._cfg([1, 4], switch_cooldown_rounds=0, warmup_rounds=100)
        clock = {"t": 0.0}
        with mock.patch("time.perf_counter", new=lambda: clock["t"]):
            policy = PricedSpeculativeParams(initial_steps=1, cfg=cfg)
            policy.get_steps_for_batch(1, rids=["r0"], round_idx=1)
            clock["t"] += 0.010  # realized 10 ms at (1, g=1)
            policy.get_steps_for_batch(1, rids=["r0"], round_idx=2)
        # (1, g=4) unseen: expect 10 ms * (4+1)/(1+1) = 25 ms, not 1.0 s.
        self.assertAlmostEqual(policy._step_cost(1, 4), 0.025, places=3)

    def test_non_consecutive_rounds_do_not_feed_the_cost_ema(self):
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 1.0)])
        cfg = self._cfg([1, 4], cost_table=cost_table)

        clock = {"t": 0.0}
        with mock.patch("time.perf_counter", new=lambda: clock["t"]):
            policy = PricedSpeculativeParams(initial_steps=1, cfg=cfg)
            policy.get_steps_for_batch(1, rids=["r0"], round_idx=1)
            clock["t"] += 100.0  # a prefill round ran in between
            policy.get_steps_for_batch(1, rids=["r0"], round_idx=3)

        self.assertEqual(policy._cost_ema, {})


class TestCostTable(PricedSpecParamsTestBase):
    def test_nearest_batch_size_in_log_space_with_exact_steps_match(self):
        cost_table = self._write_cost_table([(1, 3, 1.0), (64, 3, 2.0)])
        policy = PricedSpeculativeParams(
            initial_steps=2, cfg=self._cfg([2], cost_table=cost_table)
        )
        # log(7/1) < log(64/7) -> snap down to B=1; log(9/1) > log(64/9) -> up.
        self.assertEqual(policy._table_cost(7, 2), 1.0)
        self.assertEqual(policy._table_cost(9, 2), 2.0)

    def test_missing_steps_scale_from_nearest_available_steps(self):
        cost_table = self._write_cost_table([(1, 3, 1.0)])  # only g=2
        policy = PricedSpeculativeParams(
            initial_steps=2, cfg=self._cfg([2, 4], cost_table=cost_table)
        )
        # g=4 falls back to g=2 scaled by draft tokens: 1.0 * 5 / 3.
        self.assertAlmostEqual(policy._table_cost(1, 4), 5.0 / 3.0)

    def test_no_table_means_flat_cost(self):
        policy = PricedSpeculativeParams(initial_steps=1, cfg=self._cfg([1, 8]))
        self.assertEqual(policy._table_cost(1, 1), 1.0)
        self.assertEqual(policy._table_cost(32, 8), 1.0)
        # Flat costs + perfect prior: maximizes expected accept tokens.
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 8)

    def test_bad_header_raises(self):
        path = str(Path(self._tmp_dir.name) / "bad.csv")
        with open(path, "w") as f:
            f.write("bs,k,seconds\n1,2,0.5\n")
        with self.assertRaises(ValueError):
            load_cost_table(path)

    def test_g0_rows_are_skipped(self):
        cost_table = self._write_cost_table([(1, 1, 1.0), (1, 2, 1.5)])
        table = load_cost_table(cost_table)
        self.assertEqual(table, {(1, 1): 1.5})


class TestConfigValidation(PricedSpecParamsTestBase):
    def test_candidate_steps_required(self):
        with self.assertRaises(ValueError):
            load_priced_config({"policy": "priced"})

    def test_candidate_steps_must_be_positive_ints(self):
        for bad in ([], [0], [1, "2"], "1,2", [True]):
            with self.assertRaises(ValueError):
                load_priced_config({"candidate_steps": bad})

    def test_scalar_knobs_validated(self):
        base = {"candidate_steps": [1, 3]}
        for key, bad in (
            ("prior_accept_rate", 0.0),
            ("prior_accept_rate", 1.5),
            ("switch_margin", -0.1),
            ("ema_alpha", 0.0),
            ("warmup_rounds", -1),
            ("max_tracked_requests", 0),
            ("cost_table", 7),
        ):
            with self.assertRaises(ValueError):
                load_priced_config({**base, key: bad})

    def test_defaults(self):
        cfg = load_priced_config({"candidate_steps": [3, 1]})
        self.assertEqual(cfg.candidate_steps, [1, 3])
        self.assertEqual(cfg.prior_accept_rate, 0.7)
        self.assertEqual(cfg.switch_margin, 0.03)
        self.assertEqual(cfg.ema_alpha, 0.2)
        self.assertEqual(cfg.warmup_rounds, 20)
        self.assertEqual(cfg.max_tracked_requests, 8192)

    def test_config_file_round_trip(self):
        import json

        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 1.0)])
        cfg_path = str(Path(self._tmp_dir.name) / "adaptive.json")
        with open(cfg_path, "w") as f:
            json.dump(
                {
                    "policy": "priced",
                    "candidate_steps": [1, 4],
                    "cost_table": cost_table,
                    "prior_accept_rate": 1.0,
                    "switch_margin": 0.0,
                    "warmup_rounds": 0,
                },
                f,
            )
        policy = PricedSpeculativeParams(initial_steps=1, cfg_path=cfg_path)
        self.assertEqual(policy.candidate_steps, [1, 4])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 4)


class TestRequestTrackingAndStates(PricedSpecParamsTestBase):
    def test_lru_bound_evicts_oldest(self):
        policy = PricedSpeculativeParams(
            initial_steps=1, cfg=self._cfg([1, 4], max_tracked_requests=2)
        )
        for rid in ("a", "b", "c"):
            policy.on_verify_complete([1], batch_size=1, rids=[rid], num_steps=4)
        self.assertEqual(list(policy._reqs), ["b", "c"])

    def test_set_available_steps_restricts_decisions(self):
        policy = PricedSpeculativeParams(initial_steps=1, cfg=self._cfg([1, 4, 8]))
        policy.set_available_steps([1, 4])  # the g=8 state failed to build
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 4)
        with self.assertRaises(ValueError):
            policy.set_available_steps([16])

    def test_cuda_graph_bs_for_step(self):
        policy = PricedSpeculativeParams(initial_steps=1, cfg=self._cfg([1, 4]))
        self.assertIsNone(policy.cuda_graph_bs_for_step(4))
        policy.set_cuda_graph_bs([4, 8, 16])
        self.assertEqual(policy.cuda_graph_bs_for_step(4), [4, 8, 16])
        self.assertEqual(policy.cuda_graph_bs_for_step(5), [])


if __name__ == "__main__":
    unittest.main()
