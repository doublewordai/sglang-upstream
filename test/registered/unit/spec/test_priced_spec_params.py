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
        # ADJUSTED for ladder switching (old behavior: one direct 1 -> 8
        # jump): each switch moves at most max_switch_distance=2 candidate
        # indices, so 1 -> 8 in [1, 2, 4, 8] now takes 1 -> 4 -> 8.
        cost_table = self._write_cost_table(
            [(1, g + 1, 0.01) for g in (1, 2, 4, 8)]
        )
        policy = PricedSpeculativeParams(
            initial_steps=1,
            cfg=self._cfg(
                [1, 2, 4, 8], cost_table=cost_table, switch_cooldown_rounds=0
            ),
        )
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 4)
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 8)
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

    def _set_accept_rate_ema(self, policy, rid, rate):
        """Seed a per-request accept-rate EMA WITHOUT touching the global
        realized accept curve. ADJUSTED: on_verify_complete now also anchors
        the acceptance model globally (its own tests live in
        TestRealizedAnchor); these tests target the per-request chain path,
        which only exists un-anchored."""
        state = policy._touch(rid)
        state.accept_rate_ema = rate
        state.last_update_round = policy._round_ct

    def test_per_request_evidence_shifts_the_decision(self):
        # (d) Same policy, same costs: a low-acceptance request pins g=1, a
        # high-confidence request justifies g=4.
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 2.0)])
        policy = PricedSpeculativeParams(
            initial_steps=1,
            cfg=self._cfg([1, 4], cost_table=cost_table, switch_cooldown_rounds=0),
        )

        # A rejection-heavy history pins the low request's accept rate at 0.
        self._set_accept_rate_ema(policy, "lo", 0.0)
        self.assertEqual(policy.get_steps_for_batch(1, rids=["lo"]), 1)

        # High per-position confidences make the deep chain worth its cost.
        policy.observe_draft_confidences(["hi"], [[0.99, 0.99, 0.99, 0.99]])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["hi"]), 4)

        # Back to the low request: the policy drops down again.
        self._set_accept_rate_ema(policy, "lo", 0.0)  # refresh the stamp
        self.assertEqual(policy.get_steps_for_batch(1, rids=["lo"]), 1)

    def test_confidences_take_priority_over_accept_rate_ema(self):
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 2.0)])
        policy = PricedSpeculativeParams(
            initial_steps=1, cfg=self._cfg([1, 4], cost_table=cost_table)
        )
        # Bad realized history, then fresh high confidences: source priority
        # inside the chain says the most recent confidences win.
        self._set_accept_rate_ema(policy, "r0", 0.0)
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


class TestG0Decision(PricedSpecParamsTestBase):
    """g=0 (speculation-off) as a selectable candidate.

    Cost table mirrors the measured Llama-3.1-8B + EAGLE3 / GH200 shape:
    at high batch a plain decode round (ndt=1) is much cheaper than even
    the shallowest spec round, while at B=1 the two are nearly equal.
    """

    def _g0_policy(self, **overrides):
        cost_table = self._write_cost_table(
            [
                (1, 1, 0.016),
                (1, 2, 0.017),
                (64, 1, 0.016),
                (64, 2, 0.039),
            ]
        )
        cfg = self._cfg(
            [0, 1],
            cost_table=cost_table,
            prior_accept_rate=0.01,
            switch_cooldown_rounds=0,
        )
        cfg.update(overrides)
        return PricedSpeculativeParams(initial_steps=1, cfg=cfg)

    def test_high_batch_low_acceptance_picks_g0(self):
        # goodput(0) = 64 / 0.016 = 4000 tok/s;
        # goodput(1) = (64 * 0.01 + 64) / 0.039 ~= 1657 tok/s.
        policy = self._g0_policy()
        self.assertEqual(policy.get_steps_for_batch(64), 0)

    def test_low_batch_high_confidence_picks_spec(self):
        # goodput(1) = (0.99 + 1) / 0.017 ~= 117 > goodput(0) = 62.5.
        policy = self._g0_policy()
        policy.observe_draft_confidences(["hi"], [[0.99]])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["hi"]), 1)

    def test_switch_into_and_out_of_g0_with_zero_cooldown(self):
        policy = self._g0_policy()
        policy.observe_draft_confidences(["hi"], [[0.99]])
        self.assertEqual(policy.get_steps_for_batch(64), 0)
        self.assertEqual(policy.get_steps_for_batch(1, rids=["hi"]), 1)
        self.assertEqual(policy.get_steps_for_batch(64), 0)

    def test_on_verify_complete_with_zero_steps_is_ignored(self):
        # g=0 rounds have no drafts; they must not feed the accept-rate EMA.
        policy = self._g0_policy()
        policy.on_verify_complete([0], batch_size=1, rids=["r0"], num_steps=0)
        self.assertEqual(len(policy._reqs), 0)

    def test_goodput_at_g0_is_batch_over_cost(self):
        policy = self._g0_policy()
        self.assertAlmostEqual(policy._goodput(64, None, 0), 64 / 0.016, places=6)

    def test_runtime_state_availability_handles_zero(self):
        policy = self._g0_policy()
        policy.set_cuda_graph_bs([4, 8])
        self.assertEqual(policy.cuda_graph_bs_for_step(0), [4, 8])
        policy.set_available_steps([0])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["hi"]), 0)


class TestOnlineCostCorrection(PricedSpecParamsTestBase):
    def test_realized_gaps_override_a_wrong_table(self):
        # (e) The table claims g=4 costs the same as g=1, so the policy jumps
        # to 4 — then the realized wall gaps (10s at g=4, 1s at g=1) correct
        # the cost surface and pull it back to 1 for good. The g=4 table row
        # is 2.0 so the 10s reality sits inside the idle-gap sanity band
        # (8x): a CALIBRATED table is never 10x wrong, so samples beyond the
        # band are treated as idle pollution, not correction.
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 2.0)])
        cfg = self._cfg(
            [1, 4], cost_table=cost_table, ema_alpha=0.5, switch_cooldown_rounds=0
        )

        clock = {"t": 0.0}
        with mock.patch("time.perf_counter", new=lambda: clock["t"]):
            policy = PricedSpeculativeParams(initial_steps=1, cfg=cfg)
            chosen = []
            # Window-mean attribution needs a full uniform window before the
            # EMA sees its first sample, so the wrong-table phase lasts about
            # one window rather than one round.
            for round_idx in range(1, 40):
                steps = policy.get_steps_for_batch(
                    1, rids=["r0"], round_idx=round_idx
                )
                chosen.append(steps)
                clock["t"] += 10.0 if steps == 4 else 1.0

        self.assertEqual(chosen[0], 4)  # trusted the wrong table
        self.assertTrue(
            all(steps == 1 for steps in chosen[-10:]),
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
        # ADJUSTED: seed the low request's EMA directly — on_verify_complete
        # now also anchors the acceptance model globally, which would remove
        # the per-request flip pressure this test needs.
        state = policy._touch("lo")
        state.accept_rate_ema = 0.0
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
            # Window-mean attribution: a full uniform window of 10 ms rounds.
            from priced_spec_params import _COST_WINDOW

            for i in range(_COST_WINDOW + 2):
                policy.get_steps_for_batch(1, rids=["r0"], round_idx=1 + i)
                clock["t"] += 0.010  # realized 10 ms at (1, g=1)
        # (1, g=4) unseen: expect 10 ms * (4+1)/(1+1) = 25 ms, not 1.0 s.
        self.assertAlmostEqual(policy._step_cost(1, 4), 0.025, places=3)

    def test_same_g_evidence_ignores_a_cheaper_table_claim(self):
        # Evidence beats optimism: once g=4 has a realized-gap EMA at ANY
        # bucket (here 64), the table's cheaper claim at a nearby bucket
        # (70) must not resurrect the config (the conc-64 probe tax: ~4% of
        # rounds re-probed an already-refuted config at ~3x step cost).
        cost_table = self._write_cost_table([(70, 5, 0.001)])
        policy = PricedSpeculativeParams(
            initial_steps=4, cfg=self._cfg([1, 4], cost_table=cost_table)
        )
        policy._cost_ema[(64, 4)] = 0.05
        self.assertEqual(policy._step_cost(70, 4), 0.05)

    def test_optimism_remains_for_a_never_visited_g(self):
        # g=1 has no EMA evidence at any bucket: the table's optimistic
        # claim still prices it, so exploration stays possible.
        cost_table = self._write_cost_table([(70, 2, 0.001)])
        policy = PricedSpeculativeParams(
            initial_steps=4, cfg=self._cfg([1, 4], cost_table=cost_table)
        )
        policy._cost_ema[(64, 4)] = 0.05
        self.assertEqual(policy._step_cost(70, 1), 0.001)

    def test_idle_polluted_windows_are_rejected(self):
        # A window mean 50x the calibrated table (idle stretch, rung drain)
        # must not poison the EMA.
        cost_table = self._write_cost_table([(1, 2, 0.01)])
        cfg = self._cfg([1], cost_table=cost_table)
        clock = {"t": 0.0}
        with mock.patch("time.perf_counter", new=lambda: clock["t"]):
            policy = PricedSpeculativeParams(initial_steps=1, cfg=cfg)
            from priced_spec_params import _COST_WINDOW

            for i in range(_COST_WINDOW + 2):
                policy.get_steps_for_batch(1, rids=["r0"], round_idx=1 + i)
                clock["t"] += 0.5  # 500 ms "rounds": idle-polluted
        self.assertEqual(policy._cost_ema, {})

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


class TestEvidenceDecay(PricedSpecParamsTestBase):
    """Stale per-request evidence decays toward the prior (g=0 rounds
    generate no confidences and no accepts, so frozen pessimistic evidence
    otherwise makes g=0 absorbing — observed at concurrency 4 on GH200)."""

    def _decay_policy(self, **overrides):
        cfg = self._cfg([1, 4], prior_accept_rate=0.7, evidence_decay=0.97)
        cfg.update(overrides)
        return PricedSpeculativeParams(initial_steps=1, cfg=cfg)

    def test_stale_pessimistic_evidence_reverts_to_prior_level(self):
        policy = self._decay_policy()
        state = policy._touch("r0")
        state.accept_rate_ema = 0.0
        state.last_update_round = policy._round_ct
        prior_e = policy._chain_correct_drafts("never_seen", 4)
        # Fresh evidence reads undecayed.
        self.assertAlmostEqual(policy._chain_correct_drafts("r0", 4), 0.0)
        # ~3 time constants unupdated (tau = -1/ln(0.97) ~= 32.8 rounds).
        policy._round_ct += 100
        e_stale = policy._chain_correct_drafts("r0", 4)
        self.assertGreater(e_stale, 0.85 * prior_e)
        self.assertLess(e_stale, prior_e)

    def test_confidence_evidence_also_decays(self):
        policy = self._decay_policy()
        policy.observe_draft_confidences(["r0"], [[0.0, 0.0, 0.0, 0.0]])
        prior_e = policy._chain_correct_drafts("never_seen", 4)
        self.assertAlmostEqual(policy._chain_correct_drafts("r0", 4), 0.0)
        policy._round_ct += 100
        self.assertGreater(
            policy._chain_correct_drafts("r0", 4), 0.85 * prior_e
        )

    def test_fresh_update_resets_the_stamp(self):
        policy = self._decay_policy()
        state = policy._touch("r0")
        state.accept_rate_ema = 0.0
        state.last_update_round = policy._round_ct
        policy._round_ct += 100
        self.assertGreater(policy._chain_correct_drafts("r0", 4), 1.0)
        # A fresh rejection re-stamps (helper ema_alpha=1.0: EMA = latest).
        policy.on_verify_complete([0], batch_size=1, rids=["r0"], num_steps=4)
        self.assertAlmostEqual(policy._chain_correct_drafts("r0", 4), 0.0)

    def test_g0_is_not_absorbing_under_decay(self):
        # The conc-4 failure shape: pessimistic evidence parks the policy at
        # g=0, where no fresh evidence is ever generated. Decay must revert
        # the aggregate expectation so speculation gets retried.
        cost_table = self._write_cost_table([(4, 1, 0.01), (4, 5, 0.012)])
        cfg = self._cfg(
            [0, 4],
            cost_table=cost_table,
            prior_accept_rate=0.7,
            evidence_decay=0.9,
            switch_cooldown_rounds=0,
        )
        policy = PricedSpeculativeParams(initial_steps=0, cfg=cfg)
        rids = ["a", "b", "c", "d"]
        for rid in rids:
            state = policy._touch(rid)
            state.accept_rate_ema = 0.0
            state.last_update_round = policy._round_ct
        chosen = [policy.get_steps_for_batch(4, rids=rids) for _ in range(40)]
        # Fresh pessimism holds g=0...
        self.assertEqual(chosen[0], 0)
        # ...but unrefreshed evidence reverts and speculation is retried.
        self.assertEqual(chosen[-1], 4)

    def test_evidence_decay_of_one_disables_decay(self):
        policy = self._decay_policy(evidence_decay=1.0)
        state = policy._touch("r0")
        state.accept_rate_ema = 0.0
        state.last_update_round = policy._round_ct
        policy._round_ct += 10**6
        self.assertAlmostEqual(policy._chain_correct_drafts("r0", 4), 0.0)


class TestRealizedAnchor(PricedSpecParamsTestBase):
    """Confidence chains anchored to realized acceptance.

    The conc-1 GH200 failure: raw per-position confidence products
    overstate deep-chain returns (measured E[correct|8] - E[correct|4] was
    +0.15 while a 0.9^d chain predicts much more), saturating the policy at
    k=8 when the measured optimum was k=4 (k=8 measured -9% vs k=4)."""

    def _anchor_cfg(self, **overrides):
        # Measured GH200 conc-1 step costs: cost(B=1, g=4) = 9.4e-3 s
        # (ndt 5), cost(B=1, g=8) = 10.65e-3 s (ndt 9).
        cost_table = self._write_cost_table([(1, 5, 9.4e-3), (1, 9, 10.65e-3)])
        return self._cfg(
            [4, 8], cost_table=cost_table, switch_cooldown_rounds=0, **overrides
        )

    def _feed_realized(self, policy):
        # Helper ema_alpha=1.0: the EMA equals the latest batch mean.
        # E_global(8) = 1.06 and E_global(4) = 0.91 (the measured curve).
        policy.on_verify_complete(
            [2] * 3 + [1] * 47,
            batch_size=50,
            rids=[f"g8-{i}" for i in range(50)],
            num_steps=8,
        )
        policy.on_verify_complete(
            [1] * 91 + [0] * 9,
            batch_size=100,
            rids=[f"g4-{i}" for i in range(100)],
            num_steps=4,
        )

    def test_unanchored_chain_free_runs_to_deep_g(self):
        # Fallback before any realized data: the raw 0.95^d chain says
        # E(8) ~= 6.4, so g=8 wins — exactly the bias being fixed.
        policy = PricedSpeculativeParams(initial_steps=4, cfg=self._anchor_cfg())
        policy.observe_draft_confidences(["r0"], [[0.95] * 8])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 8)

    def test_anchored_chain_no_longer_wins_deep_g(self):
        # With the realized curve in place: goodput(4) = (0.91 + 1)/9.4e-3
        # ~= 203 tok/s > goodput(8) = (1.06 + 1)/10.65e-3 ~= 193 tok/s, and
        # a 0.95^8 confidence chain may only modulate around that anchor.
        policy = PricedSpeculativeParams(initial_steps=4, cfg=self._anchor_cfg())
        self._feed_realized(policy)
        policy.observe_draft_confidences(["r0"], [[0.95] * 8])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["r0"]), 4)

    def test_single_request_expectation_equals_the_anchor(self):
        # A lone request IS the batch mean: its chain ratio is exactly 1, so
        # the expectation is the anchor itself however confident the chain.
        policy = PricedSpeculativeParams(initial_steps=4, cfg=self._cfg([4]))
        policy.on_verify_complete([1], batch_size=1, rids=["a"], num_steps=4)
        policy.observe_draft_confidences(["r0"], [[0.99] * 4])
        self.assertAlmostEqual(
            policy._expected_correct_drafts_batch(1, ["r0"], 4), 1.0, places=9
        )

    def test_modulation_is_clamped_around_the_anchor(self):
        policy = PricedSpeculativeParams(
            initial_steps=4, cfg=self._cfg([4], confidence_modulation=2.0)
        )
        policy.on_verify_complete([1, 1], batch_size=2, rids=["a", "b"], num_steps=4)
        # chain("hi") ~= 3.9, chain("lo") = 0 -> mean ~= 1.95: hi's ratio
        # hits the +clamp (2.0), lo's ratio 0 hits the -clamp (0.5).
        policy.observe_draft_confidences(
            ["hi", "lo"], [[0.99] * 4, [0.0] * 4]
        )
        self.assertAlmostEqual(
            policy._expected_correct_drafts_batch(2, ["hi", "lo"], 4),
            1.0 * (2.0 + 0.5),
            places=9,
        )

    def test_anchor_interpolates_to_unsampled_g(self):
        # Only g=4 has run. g=8's anchor must come from the implied
        # per-position rate of the realized g=4 chain (~0.49), giving a
        # modest deep-chain increment — not a free-running product.
        policy = PricedSpeculativeParams(initial_steps=4, cfg=self._anchor_cfg())
        policy.on_verify_complete(
            [1] * 91 + [0] * 9,
            batch_size=100,
            rids=[f"g4-{i}" for i in range(100)],
            num_steps=4,
        )
        e4 = policy._anchored_correct_drafts(4)
        e8 = policy._anchored_correct_drafts(8)
        self.assertAlmostEqual(e4, 0.91, places=9)
        self.assertGreater(e8, e4)
        self.assertLess(e8, 1.1)

    def test_no_realized_data_means_no_anchor(self):
        policy = PricedSpeculativeParams(initial_steps=4, cfg=self._cfg([4]))
        self.assertIsNone(policy._anchored_correct_drafts(4))


def _chain(conf: float, steps: int) -> float:
    """sum_{d=1..steps} conf**d — the per-request chain signal under a
    constant per-position confidence."""
    total, chain_prob = 0.0, 1.0
    for _ in range(steps):
        chain_prob *= conf
        total += chain_prob
    return total


class TestRollingChainBaseline(PricedSpecParamsTestBase):
    """The modulation baseline is a per-g rolling EMA of round-mean chain
    signals maintained across rounds, not the same-round batch mean. The
    same-round mean equals the lone request's own chain at batch size 1,
    forcing modulation to exactly 1.0 and killing per-request gating where
    it's worth most (conc-1 GH200: policy sat at k=1/k=3, -29% vs static
    k=4, with +22% predicted from working per-request gating)."""

    def _anchored_policy(self, initial_steps, **overrides):
        # Realized anchors E_global(1) = 0.5 and E_global(4) = 1.5 (one
        # ema_alpha=1.0 feed each); costs chosen so at batch size 1 the
        # neutral-modulation goodputs are
        #   goodput(1 | m=1) = (0.5 + 1) / 1.0  = 1.50   (g=1 wins)
        #   goodput(4 | m=1) = (1.5 + 1) / 1.8 ~= 1.39
        # while a +clamp modulation at g=4 flips it:
        #   goodput(4 | m=2) = (3.0 + 1) / 1.8 ~= 2.22   (g=4 wins)
        cost_table = self._write_cost_table([(1, 2, 1.0), (1, 5, 1.8)])
        cfg = self._cfg([1, 4], cost_table=cost_table, switch_cooldown_rounds=0)
        cfg.update(overrides)
        policy = PricedSpeculativeParams(initial_steps=initial_steps, cfg=cfg)
        policy.on_verify_complete(
            [1, 0], batch_size=2, rids=["s0", "s1"], num_steps=1
        )
        policy.on_verify_complete(
            [2, 1], batch_size=2, rids=["s2", "s3"], num_steps=4
        )
        return policy

    def _warm_baseline(self, policy, conf=0.6):
        """Seed the per-g rolling baselines with a typical-confidence round
        (helper ema_alpha=1.0: baseline = the latest round mean)."""
        policy.observe_draft_confidences(["typ"], [[conf] * 4])
        for steps in (1, 4):
            policy._expected_correct_drafts_batch(1, ["typ"], steps)

    def test_bs1_confident_round_modulates_up_and_goes_deeper(self):
        # At batch size 1 a request more confident than the rolling typical
        # level must yield modulation > 1 and justify deeper g. Under the
        # old same-round batch mean the ratio was identically 1.0 and the
        # policy stayed at g=1.
        policy = self._anchored_policy(initial_steps=1)
        self._warm_baseline(policy)
        policy.observe_draft_confidences(["hi"], [[0.99] * 4])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["hi"]), 4)

    def test_bs1_unconfident_round_modulates_down_and_goes_shallower(self):
        policy = self._anchored_policy(initial_steps=4)
        self._warm_baseline(policy)
        policy.observe_draft_confidences(["lo"], [[0.1] * 4])
        self.assertEqual(policy.get_steps_for_batch(1, rids=["lo"]), 1)

    def test_modulation_clamp_respected_against_the_rolling_baseline(self):
        # chain(0.99, 4) / chain(0.6, 4) ~= 2.99 > MOD=2.0: the up-clamp
        # binds, so the expectation is exactly anchor * MOD.
        policy = self._anchored_policy(initial_steps=1)
        self._warm_baseline(policy)
        policy.observe_draft_confidences(["hi"], [[0.99] * 4])
        self.assertAlmostEqual(
            policy._expected_correct_drafts_batch(1, ["hi"], 4),
            1.5 * 2.0,
            places=9,
        )
        # Zero chain hits the down-clamp: anchor / MOD.
        policy2 = self._anchored_policy(initial_steps=1)
        self._warm_baseline(policy2)
        policy2.observe_draft_confidences(["lo"], [[0.0] * 4])
        self.assertAlmostEqual(
            policy2._expected_correct_drafts_batch(1, ["lo"], 4),
            1.5 * 0.5,
            places=9,
        )

    def test_no_baseline_data_falls_back_to_batch_mean(self):
        # Before the rolling baseline has data at a g, the original
        # same-round behavior holds: a lone request IS the mean, ratio 1,
        # expectation = anchor regardless of confidence.
        policy = self._anchored_policy(initial_steps=1)
        policy.observe_draft_confidences(["hi"], [[0.99] * 4])
        self.assertAlmostEqual(
            policy._expected_correct_drafts_batch(1, ["hi"], 4), 1.5, places=9
        )

    def test_baseline_is_a_rolling_ema_across_rounds(self):
        policy = self._anchored_policy(initial_steps=1, ema_alpha=0.5)
        policy.observe_draft_confidences(["a"], [[0.6] * 4])
        policy._expected_correct_drafts_batch(1, ["a"], 4)
        self.assertAlmostEqual(
            policy._chain_mean_ema[4], _chain(0.6, 4), places=9
        )
        policy.observe_draft_confidences(["b"], [[0.99] * 4])
        policy._expected_correct_drafts_batch(1, ["b"], 4)
        self.assertAlmostEqual(
            policy._chain_mean_ema[4],
            0.5 * _chain(0.6, 4) + 0.5 * _chain(0.99, 4),
            places=9,
        )

    def test_baseline_read_excludes_the_current_round(self):
        # The decision uses the PRE-update baseline: the current round's
        # chain must not normalize itself back toward ratio 1. With helper
        # ema_alpha=1.0 a post-update read would collapse the ratio to
        # exactly 1 (baseline = own chain); conf 0.8 keeps the true ratio
        # ~1.81, inside the clamp, so the two are distinguishable.
        policy = self._anchored_policy(initial_steps=1)
        self._warm_baseline(policy)
        policy.observe_draft_confidences(["hi"], [[0.8] * 4])
        expected_ratio = _chain(0.8, 4) / _chain(0.6, 4)
        self.assertLess(expected_ratio, 2.0)
        self.assertAlmostEqual(
            policy._expected_correct_drafts_batch(1, ["hi"], 4),
            1.5 * expected_ratio,
            places=9,
        )


class TestLadderSwitching(PricedSpecParamsTestBase):
    def test_distant_optimum_is_reached_via_the_ladder(self):
        # From g=0 in [0, 1, 2, 3, 4, 8] a huge goodput gap to 8 moves at
        # most max_switch_distance=2 indices per switch: 0 -> 2 -> 4 -> 8.
        # Makes the 0 -> 8 cliff (one catch-up extend over every request's
        # whole gap, the conc-256 OOM) structurally impossible.
        cost_table = self._write_cost_table(
            [(1, g + 1, 0.01) for g in (0, 1, 2, 3, 4, 8)]
        )
        policy = PricedSpeculativeParams(
            initial_steps=0,
            cfg=self._cfg(
                [0, 1, 2, 3, 4, 8],
                cost_table=cost_table,
                switch_cooldown_rounds=0,
            ),
        )
        seq = [policy.get_steps_for_batch(1, rids=["r0"]) for _ in range(4)]
        self.assertEqual(seq, [2, 4, 8, 8])

    def test_cooldown_gates_each_rung(self):
        cost_table = self._write_cost_table(
            [(1, g + 1, 0.01) for g in (0, 1, 2, 3, 4, 8)]
        )
        policy = PricedSpeculativeParams(
            initial_steps=0,
            cfg=self._cfg(
                [0, 1, 2, 3, 4, 8],
                cost_table=cost_table,
                switch_cooldown_rounds=2,
            ),
        )
        seq = [policy.get_steps_for_batch(1, rids=["r0"]) for _ in range(6)]
        self.assertEqual(seq, [2, 2, 4, 4, 8, 8])

    def test_max_switch_distance_one_walks_adjacent_rungs(self):
        cost_table = self._write_cost_table(
            [(1, g + 1, 0.01) for g in (0, 1, 2)]
        )
        policy = PricedSpeculativeParams(
            initial_steps=0,
            cfg=self._cfg(
                [0, 1, 2],
                cost_table=cost_table,
                switch_cooldown_rounds=0,
                max_switch_distance=1,
            ),
        )
        seq = [policy.get_steps_for_batch(1, rids=["r0"]) for _ in range(3)]
        self.assertEqual(seq, [1, 2, 2])


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

    def test_g0_rows_are_loaded(self):
        # ndt=1 rows price the g=0 (speculation-off) plain-decode rounds.
        cost_table = self._write_cost_table([(1, 1, 1.0), (1, 2, 1.5)])
        table = load_cost_table(cost_table)
        self.assertEqual(table, {(1, 0): 1.0, (1, 1): 1.5})

    def test_malformed_zero_ndt_rows_are_skipped(self):
        cost_table = self._write_cost_table([(1, 0, 1.0), (1, 2, 1.5)])
        table = load_cost_table(cost_table)
        self.assertEqual(table, {(1, 1): 1.5})


class TestConfigValidation(PricedSpecParamsTestBase):
    def test_candidate_steps_required(self):
        with self.assertRaises(ValueError):
            load_priced_config({"policy": "priced"})

    def test_candidate_steps_must_be_nonnegative_ints(self):
        for bad in ([], [-1], [1, "2"], "1,2", [True]):
            with self.assertRaises(ValueError):
                load_priced_config({"candidate_steps": bad})

    def test_candidate_steps_accepts_zero(self):
        # g=0 means speculation off for the round; it is a valid candidate.
        self.assertEqual(load_priced_config({"candidate_steps": [0]}).candidate_steps, [0])
        self.assertEqual(
            load_priced_config({"candidate_steps": [3, 0, 1]}).candidate_steps,
            [0, 1, 3],
        )

    def test_g0_max_gap_validated_with_default(self):
        self.assertEqual(load_priced_config({"candidate_steps": [0, 1]}).g0_max_gap, 1024)
        self.assertEqual(
            load_priced_config({"candidate_steps": [0, 1], "g0_max_gap": 64}).g0_max_gap,
            64,
        )
        for bad in (0, -1, 1.5, "64"):
            with self.assertRaises(ValueError):
                load_priced_config({"candidate_steps": [0, 1], "g0_max_gap": bad})

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
            ("g0_catch_up_chunk_tokens", 0),
            ("g0_catch_up_chunk_tokens", 1.5),
            ("g0_defer_prefill", 1),
            ("g0_defer_prefill", "true"),
            ("g0_prompt_tail_tokens", 0),
            ("g0_prompt_tail_tokens", -1),
            ("g0_prompt_tail_tokens", 1.5),
            ("g0_prompt_tail_tokens", True),
            ("evidence_decay", 0.0),
            ("evidence_decay", 1.5),
            ("confidence_modulation", 0.5),
            ("max_switch_distance", 0),
            ("max_switch_distance", 1.5),
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
        self.assertEqual(cfg.g0_catch_up_chunk_tokens, 4096)
        # Deferred drafter prefill is opt-in.
        self.assertFalse(cfg.g0_defer_prefill)
        self.assertEqual(cfg.g0_prompt_tail_tokens, 512)
        self.assertEqual(cfg.evidence_decay, 0.97)
        self.assertEqual(cfg.confidence_modulation, 2.0)
        self.assertEqual(cfg.max_switch_distance, 2)

    def test_g0_defer_prefill_round_trip(self):
        cfg = load_priced_config(
            {
                "candidate_steps": [0, 3],
                "g0_defer_prefill": True,
                "g0_prompt_tail_tokens": 256,
            }
        )
        self.assertTrue(cfg.g0_defer_prefill)
        self.assertEqual(cfg.g0_prompt_tail_tokens, 256)
        policy = PricedSpeculativeParams(
            initial_steps=3,
            cfg=self._cfg(
                [0, 3], g0_defer_prefill=True, g0_prompt_tail_tokens=256
            ),
        )
        self.assertTrue(policy.g0_defer_prefill)
        self.assertEqual(policy.g0_prompt_tail_tokens, 256)

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
