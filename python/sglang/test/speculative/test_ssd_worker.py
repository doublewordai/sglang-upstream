import unittest
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List
from types import SimpleNamespace


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2] / "srt" / "speculative"
spec_info = _load_module(str(ROOT / "spec_info.py"), "test_spec_info")
ssd_utils = _load_module(str(ROOT / "ssd_utils.py"), "test_ssd_utils")

SpeculativeAlgorithm = spec_info.SpeculativeAlgorithm
build_ssd_verify_payload = ssd_utils.build_ssd_verify_payload
extract_ssd_bonus_tokens = ssd_utils.extract_ssd_bonus_tokens


@dataclass
class PendingAcceptance:
    """Mirror of ssd_worker.PendingAcceptance for test isolation."""
    accepted_tokens: List[int]
    bonus_token: int


def _compute_current_token(pending: PendingAcceptance, default: int) -> int:
    """Replicate the fallback logic from SSDWorker._query_remote_drafts."""
    if pending.accepted_tokens:
        return pending.accepted_tokens[-1]
    elif pending.bonus_token != 0:
        return pending.bonus_token
    else:
        return default


def _compute_suppress_mask(
    accept_lengths: List[int], finished: List[bool]
) -> List[bool]:
    """Replicate the suppress mask logic from SSDWorker._suppress_bonus_tokens."""
    return [j > 0 and not f for j, f in zip(accept_lengths, finished)]


def _compute_ssd_accept_lengths(accept_lengths: List[int]) -> List[int]:
    """Replicate the adjusted accept length logic from SSDWorker._suppress_bonus_tokens."""
    return [max(j - 1, 0) for j in accept_lengths]


class TestSSDWorkerHelpers(unittest.TestCase):
    def test_build_ssd_verify_payload_falls_back_to_linear_tree(self):
        result = SimpleNamespace(
            draft_tokens=[10, 11],
            positions=[7, 8],
            retrieve_index=[0, 1],
            retrieve_next_token=[1, -1],
            retrieve_next_sibling=[-1, -1],
            tree_mask=[[True, False], [True, True]],
            num_tokens=2,
        )

        payload = build_ssd_verify_payload(
            current_token=99,
            prefix_len=5,
            num_draft_tokens=4,
            draft_result=result,
        )

        self.assertEqual(payload["draft_tokens"], [99, 99, 99, 99])
        self.assertEqual(payload["positions"], [5, 6, 7, 8])
        self.assertEqual(payload["retrieve_index"], [0, 1, 2, 3])
        self.assertEqual(payload["retrieve_next_token"], [1, 2, 3, -1])
        self.assertEqual(payload["retrieve_next_sibling"], [-1, -1, -1, -1])
        self.assertEqual(len(payload["custom_mask"]), 4 * (5 + 4))

    def test_build_ssd_verify_payload_keeps_exact_remote_tree(self):
        result = SimpleNamespace(
            draft_tokens=[10, 11, 12],
            positions=[6, 7, 8],
            retrieve_index=[0, 1, 2],
            retrieve_next_token=[1, 2, -1],
            retrieve_next_sibling=[-1, -1, -1],
            tree_mask=[
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            num_tokens=3,
        )

        payload = build_ssd_verify_payload(
            current_token=99,
            prefix_len=4,
            num_draft_tokens=3,
            draft_result=result,
        )

        self.assertEqual(payload["draft_tokens"], [10, 11, 12])
        self.assertEqual(payload["positions"], [6, 7, 8])
        self.assertEqual(payload["retrieve_index"], [0, 1, 2])
        self.assertEqual(payload["retrieve_next_token"], [1, 2, -1])
        self.assertEqual(payload["retrieve_next_sibling"], [-1, -1, -1])
        expected_mask = [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        self.assertEqual(payload["custom_mask"], expected_mask)

    def test_extract_ssd_bonus_tokens_uses_per_request_offsets(self):
        bonus_tokens = extract_ssd_bonus_tokens(
            verified_id=[101, 102, 201, 301, 302, 303],
            accept_lengths=[1, 0, 2],
        )

        self.assertEqual(bonus_tokens, [102, 201, 303])

    def test_extract_ssd_bonus_tokens_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            extract_ssd_bonus_tokens(
                verified_id=[101, 102, 201],
                accept_lengths=[1, 1],
            )

    def test_speculative_algorithm_routes_ssd(self):
        self.assertTrue(SpeculativeAlgorithm.from_string("ssd").is_ssd())
        self.assertFalse(SpeculativeAlgorithm.SSD.supports_spec_v2())


class TestBonusSuppression(unittest.TestCase):
    """Tests for the SSD bonus token suppression logic."""

    def test_adjusted_accept_lengths_j_gt_0(self):
        """For j > 0: adjusted accept_length = j - 1, preserving accept + 1 = total."""
        adjusted = _compute_ssd_accept_lengths([2, 3, 4])
        self.assertEqual(adjusted, [1, 2, 3])
        # Verify convention: accept_length + 1 = total tokens
        for j, adj in zip([2, 3, 4], adjusted):
            self.assertEqual(adj + 1, j)  # j tokens produced (no bonus)

    def test_adjusted_accept_lengths_j_eq_0(self):
        """For j = 0: adjusted accept_length = 0, total = 1 (bonus only)."""
        adjusted = _compute_ssd_accept_lengths([0, 0])
        self.assertEqual(adjusted, [0, 0])
        for adj in adjusted:
            self.assertEqual(adj + 1, 1)  # 1 token produced (bonus)

    def test_adjusted_accept_lengths_mixed(self):
        """Mixed batch: some j > 0, some j = 0."""
        adjusted = _compute_ssd_accept_lengths([2, 0, 4, 0, 1])
        self.assertEqual(adjusted, [1, 0, 3, 0, 0])
        # Total tokens: 2 + 1 + 4 + 1 + 1 = 9
        total = sum(adj + 1 for adj in adjusted)
        self.assertEqual(total, 9)

    def test_adjusted_accept_lengths_all_accepted(self):
        """j = K (all drafts accepted): adjusted = K - 1."""
        K = 4
        adjusted = _compute_ssd_accept_lengths([K])
        self.assertEqual(adjusted, [K - 1])

    def test_suppress_mask_j_gt_0_not_finished(self):
        """Bonus suppressed when j > 0 and request not finished."""
        mask = _compute_suppress_mask([2, 0, 3], [False, False, False])
        self.assertEqual(mask, [True, False, True])

    def test_suppress_mask_j_gt_0_finished(self):
        """Bonus NOT suppressed when request is finished (bonus may be EOS)."""
        mask = _compute_suppress_mask([2, 3], [False, True])
        self.assertEqual(mask, [True, False])

    def test_suppress_mask_j_eq_0_always_false(self):
        """Bonus never suppressed when j = 0 regardless of finished state."""
        mask = _compute_suppress_mask([0, 0], [False, True])
        self.assertEqual(mask, [False, False])

    def test_pending_acceptance_j_gt_0_suppressed(self):
        """When bonus is suppressed: bonus_token=0, accepted_tokens present."""
        pending = PendingAcceptance(accepted_tokens=[10, 11], bonus_token=0)
        self.assertEqual(pending.bonus_token, 0)
        self.assertEqual(pending.accepted_tokens, [10, 11])

    def test_pending_acceptance_j_eq_0_kept(self):
        """When bonus is kept: bonus_token=<value>, accepted_tokens empty."""
        pending = PendingAcceptance(accepted_tokens=[], bonus_token=42)
        self.assertEqual(pending.bonus_token, 42)
        self.assertEqual(pending.accepted_tokens, [])

    def test_current_token_fallback_with_accepted_tokens(self):
        """When previous round had j > 0: use last accepted token as fallback."""
        pending = PendingAcceptance(accepted_tokens=[10, 11, 12], bonus_token=0)
        self.assertEqual(_compute_current_token(pending, default=-1), 12)

    def test_current_token_fallback_with_bonus(self):
        """When previous round had j = 0: use bonus token as fallback."""
        pending = PendingAcceptance(accepted_tokens=[], bonus_token=42)
        self.assertEqual(_compute_current_token(pending, default=-1), 42)

    def test_current_token_fallback_default(self):
        """When no accepted tokens and bonus = 0: fall through to default."""
        pending = PendingAcceptance(accepted_tokens=[], bonus_token=0)
        self.assertEqual(_compute_current_token(pending, default=-1), -1)

    def test_num_accepted_tokens_metric(self):
        """num_accepted_tokens and spec metrics should be correct."""
        # Batch: req0 j=2 (suppress), req1 j=0 (keep), req2 j=4 (suppress)
        ssd_adjusted = _compute_ssd_accept_lengths([2, 0, 4])
        num_accepted = sum(ssd_adjusted)

        # spec_num_accepted_tokens += num_accepted + bs
        bs = 3
        spec_total = num_accepted + bs  # = 1 + 0 + 3 + 3 = 7
        # Actual tokens: 2 + 1 + 4 = 7  ✓
        self.assertEqual(spec_total, 7)

        # num_generated_tokens += len(batch.reqs) + num_accepted
        gen_total = bs + num_accepted  # 3 + 4 = 7
        self.assertEqual(gen_total, 7)

    def test_orchestrator_receives_correct_state_j_gt_0(self):
        """After bonus suppression, orchestrator gets accepted=[D1..Dj], bonus=0."""
        # Simulate: j=2 accepted, bonus suppressed
        pending = PendingAcceptance(accepted_tokens=[100, 200], bonus_token=0)
        # Orchestrator condition: !accepted.is_empty() || bonus != 0
        self.assertTrue(len(pending.accepted_tokens) > 0 or pending.bonus_token != 0)
        # Orchestrator will advance_and_prune by [100, 200]
        self.assertEqual(pending.accepted_tokens, [100, 200])

    def test_orchestrator_receives_correct_state_j_eq_0(self):
        """After j=0, orchestrator gets accepted=[], bonus=B."""
        pending = PendingAcceptance(accepted_tokens=[], bonus_token=42)
        self.assertTrue(len(pending.accepted_tokens) > 0 or pending.bonus_token != 0)
        # Orchestrator: accepted is empty, bonus != 0, pushes bonus → advance by [42]


if __name__ == "__main__":
    unittest.main()
