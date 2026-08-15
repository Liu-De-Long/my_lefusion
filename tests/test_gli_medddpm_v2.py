import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "gli_medddpm_v2.py"
SPEC = importlib.util.spec_from_file_location("gli_medddpm_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MedDDPMV2Test(unittest.TestCase):
    def test_xyz_to_cdhw_preserves_voxel_order(self):
        xyz = np.arange(64 * 64 * 32, dtype=np.float32).reshape(64, 64, 32)
        cdhw = MODULE.xyz_to_cdhw(xyz)
        self.assertEqual(cdhw.shape, (1, 32, 64, 64))
        self.assertEqual(cdhw[0, 7, 11, 13], xyz[11, 13, 7])

    def test_state_dict_prefix_normalization(self):
        value = MODULE.torch.ones(1)
        single = MODULE.normalize_state_dict({"denoise_fn.module.x": value}, multi_gpu=False)
        self.assertEqual(set(single), {"denoise_fn.x"})
        multi = MODULE.normalize_state_dict({"denoise_fn.x": value}, multi_gpu=True)
        self.assertEqual(set(multi), {"denoise_fn.module.x"})


if __name__ == "__main__":
    unittest.main()
