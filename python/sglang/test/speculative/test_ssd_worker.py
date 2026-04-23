import unittest
import importlib.util
from pathlib import Path
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

    def test_speculative_algorithm_routes_ssd(self):
        self.assertTrue(SpeculativeAlgorithm.from_string("ssd").is_ssd())
        self.assertFalse(SpeculativeAlgorithm.SSD.supports_spec_v2())


if __name__ == "__main__":
    unittest.main()
