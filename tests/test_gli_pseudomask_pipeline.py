from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LEFUSION_ROOT = PROJECT_ROOT / "LeFusion"
if str(LEFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(LEFUSION_ROOT))

from dataset.gli_hist import GLIDataset
from scripts.gli_pseudomask_pipeline import (
    filter_components,
    lesion_histograms,
    select_intersection,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    overlay_root = tmp_path / "overlay"
    size_root = source_root / "patch_64x64x32"
    overlay_size_root = overlay_root / "patch_64x64x32"
    relative_path = "patches/case.npz"
    (size_root / "patches").mkdir(parents=True)
    (overlay_size_root / "patches").mkdir(parents=True)
    shape = (64, 64, 32)
    t1c = np.linspace(-1, 1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    source_seg = np.zeros(shape, dtype=np.uint8)
    source_seg[4:10, 4:10, 4:8] = 4
    np.savez_compressed(
        size_root / relative_path,
        t1c=t1c,
        seg=source_seg,
        hist=lesion_histograms(t1c, source_seg),
        affine=np.eye(4, dtype=np.float64),
    )
    overlay_seg = np.zeros(shape, dtype=np.uint8)
    overlay_seg[20:24, 20:24, 8:12] = 1
    overlay_seg[28:34, 28:34, 10:14] = 3
    overlay_mask = np.stack([(overlay_seg == value) for value in (1, 2, 3, 4)]).astype(np.uint8)
    confidence = np.zeros(shape, dtype=np.float16)
    confidence[overlay_seg > 0] = np.float16(0.875)
    np.savez_compressed(
        overlay_size_root / relative_path,
        seg_xyz=overlay_seg,
        lesion_mask_xyz=overlay_mask,
        confidence_xyz=confidence,
        hist=lesion_histograms(t1c, overlay_seg),
    )
    (overlay_root / "contract.json").write_text("{}\n", encoding="utf-8")
    fields = [
        "relative_path", "case_id", "subject_id", "split", "patch_size_xyz",
        "anchor_label", "anchor_name", "sample_role", "origin_x", "origin_y", "origin_z",
        "pad_before_x", "pad_before_y", "pad_before_z", "pad_after_x", "pad_after_y", "pad_after_z",
        "normalization_p005", "normalization_p995", "normalization_degenerate",
    ]
    row = {
        "relative_path": relative_path, "case_id": "case", "subject_id": "subject",
        "split": "train", "patch_size_xyz": "64x64x32", "anchor_label": "1",
        "anchor_name": "netc", "sample_role": "interior", "origin_x": "0", "origin_y": "0",
        "origin_z": "0", "pad_before_x": "0", "pad_before_y": "0", "pad_before_z": "0",
        "pad_after_x": "0", "pad_after_y": "0", "pad_after_z": "0",
        "normalization_p005": "0", "normalization_p995": "1", "normalization_degenerate": "0",
    }
    with (size_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps({"subject_split": {"subject": "train"}}), encoding="utf-8")
    return source_root, overlay_root, split_file


class PseudoMaskPipelineTests(unittest.TestCase):
    def test_inference_dataset_uses_overlay_conditioning(self) -> None:
        try:
            import nibabel as nib
            from dataset.gli_hist_in import GLIInferenceDataset
        except ModuleNotFoundError:
            self.skipTest("nibabel is unavailable in the local CPU test environment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, overlay_root, split_file = _write_fixture(root)
            raw_case = root / "raw/train/case"
            raw_case.mkdir(parents=True)
            shape = (64, 64, 32)
            image = np.ones(shape, dtype=np.float32)
            affine = np.eye(4)
            for modality in ("t1c", "t1n", "t2f", "t2w"):
                nib.save(nib.Nifti1Image(image, affine), raw_case / f"case-{modality}.nii.gz")
            nib.save(
                nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine),
                raw_case / "case-seg.nii.gz",
            )
            dataset = GLIInferenceDataset(
                source_root,
                root / "raw",
                (64, 64, 32),
                split="train",
                split_file=split_file,
                mask_overlay_root=overlay_root,
            )
            sample = dataset[0]
            self.assertEqual(sample["mask_source"], "overlay")
            self.assertTrue(torch.equal(sample["conditioning_seg"], sample["label"]))
            self.assertEqual(int(torch.count_nonzero(sample["conditioning_seg"] == 1)), 64)
            self.assertEqual(int(torch.count_nonzero(sample["conditioning_seg"] == 3)), 144)

    def test_overlay_loader_ignores_source_seg_and_hist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_root, overlay_root, split_file = _write_fixture(Path(directory))
            dataset = GLIDataset(
                source_root,
                (64, 64, 32),
                split="train",
                split_file=split_file,
                mask_overlay_root=overlay_root,
            )
            first = dataset[0]
            source_path = source_root / "patch_64x64x32" / "patches/case.npz"
            with np.load(source_path, allow_pickle=False) as arrays:
                t1c = np.asarray(arrays["t1c"])
                affine = np.asarray(arrays["affine"])
            replacement_seg = np.full((64, 64, 32), 2, dtype=np.uint8)
            replacement_hist = np.full((4, 16), 123.0, dtype=np.float32)
            np.savez_compressed(
                source_path,
                t1c=t1c,
                seg=replacement_seg,
                hist=replacement_hist,
                affine=affine,
            )
            second = dataset[0]
            for key in ("label", "lesion_mask", "masked_context", "hist"):
                self.assertTrue(torch.equal(first[key], second[key]))
            self.assertEqual(first["mask_source"], "overlay")
            self.assertEqual(tuple(first["lesion_mask"].shape), (4, 32, 64, 64))
            self.assertEqual(int(torch.count_nonzero(first["lesion_mask"][0])), 64)
            self.assertEqual(int(torch.count_nonzero(first["lesion_mask"][2])), 144)
            union = first["lesion_mask"].bool().any(dim=0, keepdim=True)
            self.assertEqual(int(torch.count_nonzero(first["masked_context"][union])), 0)

    def test_component_filter_keeps_whole_components_and_falls_back(self) -> None:
        seg = np.zeros((8, 8, 8), dtype=np.uint8)
        seg[1:3, 1:3, 1:3] = 1
        seg[5:7, 5:7, 5:7] = 3
        confidence = np.zeros_like(seg, dtype=np.float16)
        confidence[seg == 1] = np.float16(0.8)
        confidence[seg == 3] = np.float16(0.6)
        filtered, stats = filter_components(seg, confidence, 0.7)
        self.assertTrue(np.array_equal(filtered == 1, seg == 1))
        self.assertFalse(np.any(filtered == 3))
        self.assertFalse(stats["fallback_kept_best_component"])
        fallback, fallback_stats = filter_components(seg, confidence, 0.95)
        self.assertTrue(np.array_equal(fallback == 1, seg == 1))
        self.assertFalse(np.any(fallback == 3))
        self.assertTrue(fallback_stats["fallback_kept_best_component"])

    def test_intersection_tie_break_prefers_balanced_mean_then_lower_threshold(self) -> None:
        curve = [
            {"threshold": 0.4, "absolute_gap": 0.01, "balanced_mean": 0.7},
            {"threshold": 0.6, "absolute_gap": 0.01, "balanced_mean": 0.8},
            {"threshold": 0.5, "absolute_gap": 0.01, "balanced_mean": 0.8},
        ]
        self.assertEqual(select_intersection(curve)["threshold"], 0.5)

    def test_training_configs_only_differ_by_variant_outputs_and_overlay(self) -> None:
        config_dir = PROJECT_ROOT / "LeFusion/train/config/experiment"
        direct = yaml.safe_load((config_dir / "gli_exp018_direct_mask_fp32_50k.yaml").read_text(encoding="utf-8"))
        filtered = yaml.safe_load((config_dir / "gli_exp018_filtered_mask_fp32_50k.yaml").read_text(encoding="utf-8"))
        direct["variant"] = filtered["variant"]
        direct["dataset"]["mask_overlay"]["root"] = filtered["dataset"]["mask_overlay"]["root"]
        direct["model"]["results_folder"] = filtered["model"]["results_folder"]
        direct["wandb"] = filtered["wandb"]
        direct["preflight"] = filtered["preflight"]
        self.assertEqual(direct, filtered)


if __name__ == "__main__":
    unittest.main()
