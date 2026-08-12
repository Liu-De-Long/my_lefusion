from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gli_exp020_test200_metrics import _region_rows, _summarize_region


class Exp020ConditionedTest200Tests(unittest.TestCase):
    def test_region_metrics_split_generated_rejected_and_safe_outside(self) -> None:
        shape = (16, 16, 16)
        reference = np.zeros(shape, dtype=np.float32)
        generated = reference.copy()
        truth = np.zeros(shape, dtype=np.uint8)
        truth[5:11, 5:11, 5:11] = 3
        retained = np.zeros(shape, dtype=bool)
        retained[5:8, 5:11, 5:11] = True
        generated[retained] = 0.5
        support = np.ones(shape, dtype=bool)
        rows = _region_rows(
            model="filtered", relative="patches/a.npz", subject="subject",
            reference=reference, generated=generated, truth_seg=truth,
            retained=retained, support=support,
        )
        indexed = {row["region"]: row for row in rows}
        self.assertAlmostEqual(indexed["filtered_retained_union"]["mse"], 0.25)
        self.assertAlmostEqual(indexed["filtered_retained_union"]["psnr_db"], 12.041199826559248)
        self.assertEqual(indexed["filtered_rejected_gt_union"]["mse"], 0.0)
        self.assertTrue(math.isinf(indexed["filtered_rejected_gt_union"]["psnr_db"]))
        self.assertTrue(indexed["brain_outside_gt_union_safe"]["exact_invariance"])
        summary = _summarize_region([indexed["filtered_rejected_gt_union"]])
        self.assertTrue(summary["pooled"]["psnr_infinite"])
        self.assertIsNone(summary["pooled"]["psnr_db"])

    def test_exp020_configs_use_corresponding_overlay_roots(self) -> None:
        try:
            from hydra import compose, initialize_config_dir
        except ModuleNotFoundError:
            self.skipTest("Hydra is unavailable")
        config_dir = ROOT / "LeFusion/inference/confs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            direct = compose(config_name="gli_exp020_direct_test200_shard0")
            filtered = compose(config_name="gli_exp020_filtered_test200_shard0")
        self.assertIn("exp020_test200_pseudomasks/direct", direct.dataset.mask_overlay_root)
        self.assertIn("exp020_test200_pseudomasks/filtered", filtered.dataset.mask_overlay_root)
        self.assertEqual(int(direct.dataset.batch_size), 8)
        self.assertFalse(bool(direct.model.amp))
        self.assertEqual(int(direct.repaint.schedule_jump_params.t_T), 300)


if __name__ == "__main__":
    unittest.main()
