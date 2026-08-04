from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "brats_gli_crop_local_patches.py"
SPEC = importlib.util.spec_from_file_location("brats_gli_crop_local_patches", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CropUtilityTests(unittest.TestCase):
    def test_robust_normalization_preserves_background_and_range(self) -> None:
        image = np.zeros((5, 5, 5), dtype=np.float32)
        image[1:4, 1:4, 1:4] = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1
        normalized, low, high, degenerate = MODULE.robust_normalize_t1c(image)
        self.assertFalse(degenerate)
        self.assertLess(low, high)
        self.assertTrue(np.all(normalized[image == 0] == 0))
        self.assertGreaterEqual(float(normalized.min()), -1.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_constant_and_empty_foregrounds_are_degenerate(self) -> None:
        empty = np.zeros((3, 3, 3), dtype=np.float32)
        normalized, _, _, degenerate = MODULE.robust_normalize_t1c(empty)
        self.assertTrue(degenerate)
        self.assertFalse(normalized.any())

        constant = np.full((3, 3, 3), 7.0, dtype=np.float32)
        normalized, low, high, degenerate = MODULE.robust_normalize_t1c(constant)
        self.assertTrue(degenerate)
        self.assertEqual(low, high)
        self.assertFalse(normalized.any())

    def test_extract_patch_pads_at_volume_boundary(self) -> None:
        array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        patch, before, after = MODULE.extract_patch(array, (-2, 2, 3), (5, 4, 4), -1)
        self.assertEqual(patch.shape, (5, 4, 4))
        self.assertEqual(before, (2, 0, 0))
        self.assertEqual(after, (0, 1, 1))
        self.assertTrue(np.all(patch[:2] == -1))
        np.testing.assert_array_equal(patch[2:, :3, :3], array[:3, 2:5, 3:6])

    def test_histograms_are_normalized_per_present_label(self) -> None:
        t1c = np.linspace(-1, 1, 64, dtype=np.float32).reshape(4, 4, 4)
        seg = np.zeros((4, 4, 4), dtype=np.uint8)
        seg[:2] = 1
        seg[2:] = 4
        hist = MODULE.lesion_histograms(t1c, seg)
        self.assertEqual(hist.shape, (4, 16))
        self.assertAlmostEqual(float(hist[0].sum()), 1.0)
        self.assertAlmostEqual(float(hist[3].sum()), 1.0)
        self.assertEqual(float(hist[1].sum()), 0.0)
        self.assertEqual(float(hist[2].sum()), 0.0)

    def test_subject_split_has_no_leakage(self) -> None:
        cases = [
            MODULE.CaseInfo("BraTS-GLI-00001-100", "s1", "a", (1, 2), ""),
            MODULE.CaseInfo("BraTS-GLI-00001-101", "s1", "b", (2, 3), ""),
            MODULE.CaseInfo("BraTS-GLI-00002-100", "s2", "c", (1, 2), ""),
            MODULE.CaseInfo("BraTS-GLI-00003-100", "s3", "d", (1, 2), ""),
            MODULE.CaseInfo("BraTS-GLI-00004-100", "s4", "e", (1, 2), ""),
        ]
        assigned, split_by_subject = MODULE.stratified_subject_split(cases, 0.2, 20260804)
        self.assertEqual(len({case.split for case in assigned if case.subject_id == "s1"}), 1)
        self.assertEqual(set(split_by_subject), {"s1", "s2", "s3", "s4"})

    def test_parallel_discovery_preserves_sorted_case_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train = root / "train"
            for case_id, label in (("BraTS-GLI-00002-100", 4), ("BraTS-GLI-00001-100", 1)):
                case_dir = train / case_id
                case_dir.mkdir(parents=True)
                image = np.ones((3, 3, 3), dtype=np.float32)
                seg = np.zeros((3, 3, 3), dtype=np.uint8)
                seg[1, 1, 1] = label
                nib.save(nib.Nifti1Image(image, np.eye(4)), case_dir / f"{case_id}-t1c.nii.gz")
                nib.save(nib.Nifti1Image(seg, np.eye(4)), case_dir / f"{case_id}-seg.nii.gz")

            cases = MODULE.discover_cases(root, "train", 0, {}, workers=2)
            self.assertEqual(
                [case.case_id for case in cases],
                ["BraTS-GLI-00001-100", "BraTS-GLI-00002-100"],
            )
            self.assertEqual([case.labels_present for case in cases], [(1,), (4,)])


class ProcessCaseIntegrationTests(unittest.TestCase):
    def test_process_case_writes_two_samples_per_present_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            case_id = "BraTS-GLI-00001-100"
            case_dir = root / case_id
            case_dir.mkdir()
            image = np.zeros((16, 18, 12), dtype=np.float32)
            image[1:-1, 1:-1, 1:-1] = 10.0
            image[4:12, 4:14, 3:9] += np.linspace(0, 5, 8, dtype=np.float32)[:, None, None]
            seg = np.zeros(image.shape, dtype=np.uint8)
            seg[4:8, 5:10, 4:8] = 1
            seg[8:13, 9:15, 5:10] = 4
            affine = np.array(
                [[1.0, 0.0, 0.0, 5.0], [0.0, 2.0, 0.0, 6.0], [0.0, 0.0, 3.0, 7.0], [0, 0, 0, 1]],
                dtype=np.float64,
            )
            nib.save(nib.Nifti1Image(image, affine), case_dir / f"{case_id}-t1c.nii.gz")
            nib.save(nib.Nifti1Image(seg, affine), case_dir / f"{case_id}-seg.nii.gz")
            staging = root / "staging"
            case = MODULE.CaseInfo(case_id, "BraTS-GLI-00001", str(case_dir), (1, 4), "train")
            result = MODULE.process_case(case, [(8, 8, 4)], str(staging), 20260804, 64)
            rows = result["rows_by_size"]["patch_8x8x4"]
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["sample_role"] for row in rows}, {"interior", "boundary"})
            for row in rows:
                path = staging / "patch_8x8x4" / row["relative_path"]
                self.assertTrue(path.is_file())
                with np.load(path) as arrays:
                    self.assertEqual(arrays["t1c"].shape, (8, 8, 4))
                    self.assertEqual(arrays["seg"].dtype, np.uint8)
                    self.assertEqual(arrays["hist"].shape, (4, 16))
                    self.assertTrue(np.any(arrays["seg"] == int(row["anchor_label"])))


if __name__ == "__main__":
    unittest.main()
