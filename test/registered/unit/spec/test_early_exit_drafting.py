"""Pure-Python unit tests for in-round early-exit drafting.

Covers the standalone stop rule (``decide_early_exit_depth`` /
``early_exit_should_stop``) and the priced-policy coupling (stop-price hook,
small-batch switch restriction, config validation). The policy module is
stdlib-only by design, so these tests also run on machines without torch/GPU
deps: when the sglang package is not importable we load the module straight
from the source tree.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # local dep-light runs; CI parses this call via AST anyway

    def register_cpu_ci(*args, **kwargs):
        return None


register_cpu_ci(est_time=4, suite="base-a-test-cpu")

try:
    from sglang.srt.speculative.priced_spec_params import (
        _EARLY_EXIT_FALLBACK_STOP_PRICE,
        PricedSpeculativeParams,
        decide_early_exit_depth,
        early_exit_should_skip,
        early_exit_should_stop,
        early_exit_stop_depths,
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
    decide_early_exit_depth = _mod.decide_early_exit_depth
    early_exit_should_skip = _mod.early_exit_should_skip
    early_exit_should_stop = _mod.early_exit_should_stop
    early_exit_stop_depths = _mod.early_exit_stop_depths
    load_priced_config = _mod.load_priced_config
    _EARLY_EXIT_FALLBACK_STOP_PRICE = _mod._EARLY_EXIT_FALLBACK_STOP_PRICE


class TestEarlyExitShouldStop(unittest.TestCase):
    def test_stops_strictly_below_the_price(self):
        self.assertTrue(early_exit_should_stop(0.19, 0.2))
        self.assertFalse(early_exit_should_stop(0.21, 0.2))

    def test_tie_continues(self):
        # The marginal step exactly pays for itself: keep drafting.
        self.assertFalse(early_exit_should_stop(0.2, 0.2))


class TestDecideEarlyExitDepth(unittest.TestCase):
    def test_stops_at_the_first_failing_candidate_depth(self):
        # Gains per determined depth d=1..: drop below the price at d=2.
        gains = [0.9, 0.1, 0.05]
        self.assertEqual(decide_early_exit_depth(gains, [1, 2, 3], 4, 0.2), 2)

    def test_runs_to_k_max_when_every_step_pays(self):
        gains = [0.9, 0.8, 0.7]
        self.assertEqual(decide_early_exit_depth(gains, [1, 2, 3], 4, 0.2), 4)

    def test_raw_stop_point_rounds_up_to_the_nearest_candidate(self):
        # Gains fail from depth 2 on, but 2 is not a candidate: the loop
        # keeps drafting to the nearest allowed depth at or past the raw
        # stop point (3), where the rule still fails for non-increasing
        # gains.
        gains = [0.9, 0.1, 0.08, 0.05]
        self.assertEqual(decide_early_exit_depth(gains, [1, 3], 5, 0.2), 3)

    def test_always_at_least_one(self):
        # Depth 0 (or negatives) can never be a stop depth: the first draft
        # token is free (it comes from draft extend), so g_v >= 1 always.
        gains = [0.0, 0.0, 0.0]
        self.assertEqual(decide_early_exit_depth(gains, [0, 1], 4, 0.2), 1)
        self.assertEqual(decide_early_exit_depth(gains, [-1, 0], 4, 0.2), 4)

    def test_depths_at_or_past_k_max_are_ignored(self):
        # k_max itself is not a "stop": it is the run-to-end default.
        gains = [0.9, 0.9, 0.0]
        self.assertEqual(decide_early_exit_depth(gains, [4, 7], 4, 0.2), 4)

    def test_no_stop_depths_means_k_max(self):
        self.assertEqual(decide_early_exit_depth([0.0, 0.0], [], 3, 0.2), 3)

    def test_unsorted_and_duplicate_stop_depths(self):
        gains = [0.9, 0.1, 0.05]
        self.assertEqual(
            decide_early_exit_depth(gains, [3, 2, 2, 1], 4, 0.2), 2
        )

    def test_missing_gain_entries_continue(self):
        # A depth whose gain was never produced cannot trigger a stop.
        self.assertEqual(decide_early_exit_depth([0.9], [1, 2, 3], 4, 0.2), 4)

    def test_zero_price_never_stops_and_huge_price_stops_first(self):
        gains = [0.9, 0.5, 0.2]
        self.assertEqual(decide_early_exit_depth(gains, [1, 2, 3], 4, 0.0), 4)
        self.assertEqual(decide_early_exit_depth(gains, [1, 2, 3], 4, 10.0), 1)


class TestEarlyExitShouldSkip(unittest.TestCase):
    def test_low_first_probs_skip(self):
        # A doomed batch (every first draft near-certain to be rejected)
        # is identified before the drafter pass is paid.
        self.assertTrue(early_exit_should_skip([0.05, 0.08], 0.2))

    def test_high_first_prob_proceeds(self):
        self.assertFalse(early_exit_should_skip([0.9], 0.2))

    def test_batch_gain_is_the_sum(self):
        # Each request alone falls short of the price, but the drafter pass
        # is shared across the batch: the summed expected commits clear it.
        self.assertTrue(early_exit_should_skip([0.15], 0.2))
        self.assertTrue(early_exit_should_skip([0.1], 0.2))
        self.assertFalse(early_exit_should_skip([0.15, 0.1], 0.2))

    def test_tie_proceeds(self):
        # Mirrors early_exit_should_stop: the round exactly pays for itself.
        self.assertFalse(early_exit_should_skip([0.1, 0.1], 0.2))


class TestEarlyExitStopDepths(unittest.TestCase):
    def test_zero_included_only_when_zero_has_a_state(self):
        self.assertEqual(early_exit_stop_depths([0, 2, 4]), [0, 2])
        # No candidate 0 -> no depth-0 skip: the minimum depth stays 1.
        self.assertEqual(early_exit_stop_depths([1, 2, 4]), [1, 2])

    def test_zero_can_be_the_only_stop(self):
        # Candidates {0, k_max}: no in-draft stop exists, but the pre-draft
        # skip does.
        self.assertEqual(early_exit_stop_depths([0, 4]), [0])

    def test_k_max_is_never_a_stop(self):
        self.assertEqual(early_exit_stop_depths([4]), [])
        self.assertEqual(early_exit_stop_depths([]), [])


class EarlyExitPolicyTestBase(unittest.TestCase):
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
            "switch_cooldown_rounds": 0,
            "early_exit": True,
        }
        cfg.update(overrides)
        return cfg

    def _params(self, candidate_steps, initial_steps=None, **overrides):
        if initial_steps is None:
            initial_steps = max(candidate_steps)
        return PricedSpeculativeParams(
            initial_steps=initial_steps, cfg=self._cfg(candidate_steps, **overrides)
        )


class TestEarlyExitStopPrice(EarlyExitPolicyTestBase):
    def test_explicit_config_value_is_used_flat(self):
        params = self._params([0, 4], early_exit_stop_price=0.35)
        self.assertEqual(params.early_exit_stop_price(1), 0.35)
        self.assertEqual(params.early_exit_stop_price(4), 0.35)

    def test_no_cost_data_falls_back_flat(self):
        params = self._params([0, 4])
        self.assertEqual(
            params.early_exit_stop_price(1), _EARLY_EXIT_FALLBACK_STOP_PRICE
        )

    def test_seeded_table_yields_slope_times_goodput(self):
        # B=1: C(g=0) = 5 ms, C(g=4) = 9 ms -> slope = 1 ms per draft step.
        # prior_accept_rate = 1.0 -> E[correct | 4] = 4, so goodput at the
        # parked g=4 is (4 + 1) / 0.009 tokens/s and the stop price is
        # slope * goodput = 0.001 * 5 / 0.009.
        table = self._write_cost_table([(1, 1, 0.005), (1, 5, 0.009)])
        params = self._params([0, 4], cost_table=table)
        self.assertEqual(params.current_steps, 4)
        self.assertAlmostEqual(
            params.early_exit_stop_price(1), 0.001 * 5 / 0.009, places=9
        )

    def test_flat_cost_surface_falls_back(self):
        # Zero slope: an extra draft step is free per the table, so the
        # derivation degenerates and the flat default applies.
        table = self._write_cost_table([(1, 1, 0.005), (1, 5, 0.005)])
        params = self._params([0, 4], cost_table=table)
        self.assertEqual(
            params.early_exit_stop_price(1), _EARLY_EXIT_FALLBACK_STOP_PRICE
        )


class TestEarlyExitActivation(EarlyExitPolicyTestBase):
    def test_active_within_batch_cap_only(self):
        params = self._params([0, 1, 4], early_exit_max_batch=4)
        self.assertTrue(params.early_exit_active(1))
        self.assertTrue(params.early_exit_active(4))
        self.assertFalse(params.early_exit_active(5))

    def test_inactive_without_the_knob(self):
        params = self._params([0, 1, 4], early_exit=False)
        self.assertFalse(params.early_exit_active(1))

    def test_inactive_when_deepest_candidate_below_two(self):
        params = self._params([0, 1])
        self.assertFalse(params.early_exit_active(1))


class TestEarlyExitSwitchRestriction(EarlyExitPolicyTestBase):
    # Cost surface making g=2 the unrestricted goodput argmax at B=1
    # (prior 1.0 -> E[correct | g] = g):
    #   goodput(0) = 1/1.0 = 1.0, goodput(2) = 3/0.6 = 5.0,
    #   goodput(4) = 5/2.0 = 2.5.
    _ROWS = [(1, 1, 1.0), (1, 3, 0.6), (1, 5, 2.0)]

    def test_small_batch_parks_at_k_max_instead_of_the_intermediate(self):
        table = self._write_cost_table(self._ROWS)
        params = self._params([0, 2, 4], cost_table=table)
        for _ in range(8):
            steps = params.get_steps_for_batch(1)
        # Intermediate depths are reached by in-round stopping, never by a
        # state swap: the policy must not park at 2.
        self.assertEqual(steps, 4)

    def test_without_early_exit_the_intermediate_wins(self):
        table = self._write_cost_table(self._ROWS)
        params = self._params([0, 2, 4], cost_table=table, early_exit=False)
        for _ in range(8):
            steps = params.get_steps_for_batch(1)
        self.assertEqual(steps, 2)

    def test_past_the_batch_cap_the_restriction_lifts(self):
        table = self._write_cost_table(self._ROWS)
        params = self._params([0, 2, 4], cost_table=table, early_exit_max_batch=4)
        for _ in range(8):
            steps = params.get_steps_for_batch(8)
        self.assertEqual(steps, 2)


class TestExecutedDepthCostAttribution(EarlyExitPolicyTestBase):
    """Early-exit rounds run shallower than the parked configuration the
    policy keyed at decision time; observe_round_executed_steps re-keys the
    round's wall-gap attribution to the depth that actually ran."""

    def _run_two_rounds(self, params, executed_steps=None):
        # Cost attribution is a WINDOW mean: it takes a full window of
        # uniform-key consecutive rounds before the EMA receives a sample
        # (see _observe_round_gap), so drive window+1 rounds, applying the
        # same executed-depth correction each round.
        from priced_spec_params import _COST_WINDOW

        for i in range(_COST_WINDOW + 2):
            params.get_steps_for_batch(1, round_idx=10 + i)
            if executed_steps is not None:
                params.observe_round_executed_steps(executed_steps)

    def test_depth0_skip_keys_the_gap_at_steps_zero(self):
        params = self._params([0, 4])
        self.assertEqual(params.current_steps, 4)
        self._run_two_rounds(params, executed_steps=0)
        self.assertIn((1, 0), params._cost_ema)
        self.assertNotIn((1, 4), params._cost_ema)

    def test_in_draft_stop_keys_the_gap_at_the_stop_depth(self):
        params = self._params([0, 2, 4])
        self.assertEqual(params.current_steps, 4)
        self._run_two_rounds(params, executed_steps=2)
        self.assertIn((1, 2), params._cost_ema)
        self.assertNotIn((1, 4), params._cost_ema)

    def test_full_depth_round_keeps_the_parked_key(self):
        params = self._params([0, 4])
        self._run_two_rounds(params)
        self.assertIn((1, 4), params._cost_ema)
        self.assertNotIn((1, 0), params._cost_ema)

    def test_observe_without_a_pending_round_is_a_no_op(self):
        params = self._params([0, 4])
        params.observe_round_executed_steps(0)  # no decision keyed yet
        params.get_steps_for_batch(1, round_idx=10)
        self.assertEqual(params._cost_ema, {})


class TestEarlyExitConfig(EarlyExitPolicyTestBase):
    def test_defaults(self):
        cfg = load_priced_config({"candidate_steps": [0, 4]})
        self.assertFalse(cfg.early_exit)
        self.assertEqual(cfg.early_exit_max_batch, 4)
        self.assertIsNone(cfg.early_exit_stop_price)

    def test_validation(self):
        with self.assertRaises(ValueError):
            load_priced_config({"candidate_steps": [0, 4], "early_exit": 1})
        with self.assertRaises(ValueError):
            load_priced_config(
                {"candidate_steps": [0, 4], "early_exit_max_batch": 0}
            )
        with self.assertRaises(ValueError):
            load_priced_config(
                {"candidate_steps": [0, 4], "early_exit_max_batch": True}
            )
        with self.assertRaises(ValueError):
            load_priced_config(
                {"candidate_steps": [0, 4], "early_exit_stop_price": 0.0}
            )
        cfg = load_priced_config(
            {
                "candidate_steps": [0, 4],
                "early_exit": True,
                "early_exit_max_batch": 2,
                "early_exit_stop_price": 0.3,
            }
        )
        self.assertTrue(cfg.early_exit)
        self.assertEqual(cfg.early_exit_max_batch, 2)
        self.assertEqual(cfg.early_exit_stop_price, 0.3)


if __name__ == "__main__":
    unittest.main()
