from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import linalg


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gli_exp019_test200_metrics.py"
SPEC = importlib.util.spec_from_file_location("gli_exp019_test200_metrics", SCRIPT)
METRICS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(METRICS)


class Exp019MetricTests(unittest.TestCase):
    def test_psnr_ssim_and_orthogonal_crops(self) -> None:
        reference = np.zeros((12, 13, 10), dtype=np.float32)
        generated = reference.copy()
        seg = np.zeros(reference.shape, dtype=np.uint8)
        seg[3:8, 4:10, 2:7] = 3
        generated[seg > 0] = 0.5
        self.assertAlmostEqual(METRICS.psnr(reference, generated, seg > 0), 12.041199826559248)
        self.assertTrue(math.isinf(METRICS.psnr(reference, reference)))
        crops = METRICS.orthogonal_lesion_crops(reference, seg, margin=2)
        self.assertEqual(len(crops), 3)
        self.assertTrue(all(crop.ndim == 2 and min(crop.shape) >= 5 for crop in crops))
        self.assertAlmostEqual(METRICS.lesion_crop_ssim(reference, reference, seg), 1.0)

    def test_frechet_low_rank_matches_covariance_formula(self) -> None:
        rng = np.random.default_rng(17)
        left = rng.normal(size=(9, 5))
        right = rng.normal(loc=0.4, scale=1.2, size=(8, 5))
        cov_left = np.cov(left, rowvar=False)
        cov_right = np.cov(right, rowvar=False)
        covariance_root = linalg.sqrtm(cov_left @ cov_right)
        expected = np.square(left.mean(0) - right.mean(0)).sum()
        expected += np.trace(cov_left + cov_right - 2 * covariance_root).real
        self.assertAlmostEqual(METRICS.frechet_distance(left, right), float(expected), places=8)
        self.assertAlmostEqual(METRICS.frechet_distance(left, left), 0.0, places=9)

    def test_kid_and_histogram_wasserstein_are_deterministic(self) -> None:
        left = np.arange(48, dtype=np.float64).reshape(12, 4) / 48
        right = left + 0.25
        first = METRICS.kid_repeated(left, right, repeats=10, subset_size=8, seed=3)
        second = METRICS.kid_repeated(left, right, repeats=10, subset_size=8, seed=3)
        self.assertEqual(first, second)
        self.assertTrue(np.isfinite(first["mean"]))
        real = np.full((4, 4, 4), -0.5, dtype=np.float32)
        fake = np.full((4, 4, 4), 0.5, dtype=np.float32)
        self.assertAlmostEqual(METRICS._hist_w1(real, fake, np.ones_like(real, dtype=bool)), 1.0)

    def test_patient_bootstrap_is_clustered_and_finite(self) -> None:
        rows = [
            {"subject_id": "a", "metric": 1.0},
            {"subject_id": "a", "metric": 3.0},
            {"subject_id": "b", "metric": 5.0},
        ]
        interval = METRICS.patient_bootstrap_ci(rows, "metric", repeats=100)
        self.assertEqual(len(interval), 2)
        self.assertLessEqual(interval[0], interval[1])

    def test_exp019_hydra_configs_share_one_contract(self) -> None:
        try:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf
        except ModuleNotFoundError:
            self.skipTest("Hydra is unavailable")
        names = [
            "gli_exp019_exp010_test200_shard0",
            "gli_exp019_exp010_test200_shard1",
            "gli_exp019_direct_test200_shard0",
            "gli_exp019_direct_test200_shard1",
            "gli_exp019_filtered_test200_shard0",
            "gli_exp019_filtered_test200_shard1",
        ]
        configs = []
        with initialize_config_dir(
            version_base=None,
            config_dir=str(ROOT / "LeFusion" / "inference" / "confs"),
        ):
            for name in names:
                configs.append(OmegaConf.to_container(compose(config_name=name), resolve=True))
        allowed = (("variant",), ("device",), ("checkpoint", "path"), ("selection", "shard_index"), ("output", "root"))
        def remove_allowed(payload: dict) -> dict:
            import copy
            result = copy.deepcopy(payload)
            for path in allowed:
                node = result
                for part in path[:-1]:
                    node = node[part]
                node.pop(path[-1])
            return result
        baseline = remove_allowed(configs[0])
        self.assertTrue(all(remove_allowed(config) == baseline for config in configs[1:]))
        self.assertTrue(all(config["model"]["amp"] is False for config in configs))
        self.assertTrue(all(config["dataset"]["batch_size"] == 4 for config in configs))
        self.assertTrue(all(config["conditioning"]["source"] == "real" for config in configs))
        self.assertTrue(all(config["output"]["save_nifti"] is False for config in configs))
        self.assertTrue(all(config["output"]["save_qa"] is False for config in configs))


if __name__ == "__main__":
    unittest.main()
