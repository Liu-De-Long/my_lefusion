from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "scripts" / "gli_exp020_filtered_two_scope_metrics.py"
SPEC = importlib.util.spec_from_file_location("gli_exp020_filtered_two_scope_metrics", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FilteredTwoScopeTests(unittest.TestCase):
    def test_full_fov_slices_preserve_complete_plane(self) -> None:
        image = np.arange(8 * 10 * 6, dtype=np.float32).reshape(8, 10, 6)
        mask = np.zeros_like(image, dtype=bool)
        mask[3, 2:8, 1:5] = True
        slices = MODULE._full_fov_slices(image, mask)
        self.assertEqual([value.shape for value in slices], [(10, 6), (8, 6), (8, 10)])

    def test_reference_biased_rbf_mmd(self) -> None:
        values = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        self.assertAlmostEqual(MODULE.rad_mmd2(values, values, gamma=0.5), 0.0)
        shifted = values + 2.0
        self.assertGreater(MODULE.rad_mmd2(values, shifted, gamma=0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
