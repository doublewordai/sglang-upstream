import unittest

from sglang.srt.disaggregation.nixl.conn import dense_layer_params
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDenseLayerParams(CustomTestCase):
    def test_skips_elided_layers(self):
        """Layers with a 0-byte item or a null pointer on either side carry nothing."""
        src = [10, 0, 30, 40]
        dst = [50, 60, 0, 80]
        lens = [4, 4, 4, 0]
        self.assertEqual(dense_layer_params(src, dst, lens, 4), [(10, 50, 4)])

    def test_pp_stage_prefix(self):
        src = [10, 20, 30]
        dst = [50, 60, 70]
        lens = [4, 4, 4]
        self.assertEqual(
            dense_layer_params(src, dst, lens, 2), [(10, 50, 4), (20, 60, 4)]
        )


if __name__ == "__main__":
    unittest.main()
