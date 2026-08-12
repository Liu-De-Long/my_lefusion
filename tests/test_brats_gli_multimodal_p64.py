from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from LeFusion.classifier.data import GLIClassifierPatchDataset, sha256_file


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "brats_gli_rebuild_multimodal_p64.py"
SPEC = importlib.util.spec_from_file_location("brats_gli_rebuild_multimodal_p64", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MultimodalReplayTests(unittest.TestCase):
    def test_test_materialization_requires_and_validates_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "selection.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split": "test",
                        "patch_size_xyz": [64, 64, 32],
                        "selected_count": 2,
                        "selected_relative_paths": ["a.npz", "b.npz"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE._selection_paths(manifest, include_splits=("test",)),
                {"a.npz", "b.npz"},
            )
            with self.assertRaisesRegex(ValueError, "requires selection_manifest"):
                MODULE.build_multimodal_p64(
                    raw_root=root,
                    raw_split="train",
                    source_dataset_root=root,
                    split_file=root / "missing.json",
                    output_root=root / "output",
                    include_splits=("test",),
                )

    def test_replay_preserves_manifest_and_omits_histogram(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            source_root = root / "source"
            output_root = root / "output"
            case_id = "BraTS-GLI-00001-100"
            subject_id = "BraTS-GLI-00001"
            case_root = raw_root / "train" / case_id
            case_root.mkdir(parents=True)
            shape = (72, 70, 40)
            grid = np.indices(shape, dtype=np.float32)
            foreground = np.zeros(shape, dtype=bool)
            foreground[2:-2, 2:-2, 2:-2] = True
            base = np.zeros(shape, dtype=np.float32)
            base[foreground] = 10.0 + grid[0][foreground] + 0.1 * grid[1][foreground]
            affine = np.array(
                [[1.0, 0.0, 0.0, 5.0], [0.0, 1.5, 0.0, 6.0], [0.0, 0.0, 2.0, 7.0], [0, 0, 0, 1]],
                dtype=np.float64,
            )
            for index, modality in enumerate(MODULE.MODALITIES, start=1):
                image = base * float(index)
                nib.save(
                    nib.Nifti1Image(image, affine),
                    case_root / f"{case_id}-{modality}.nii.gz",
                )
            seg = np.zeros(shape, dtype=np.uint8)
            seg[20:30, 22:32, 10:18] = 3
            nib.save(nib.Nifti1Image(seg, affine), case_root / f"{case_id}-seg.nii.gz")

            source_size_root = source_root / "patch_64x64x32"
            relative_path = f"patches/{case_id}/sample.npz"
            source_path = source_size_root / relative_path
            source_path.parent.mkdir(parents=True)
            origin = (4, 3, 2)
            t1c, _, _, _ = MODULE.robust_normalize_t1c(base)
            t1c_patch, before, after = MODULE.extract_patch(
                t1c, origin, MODULE.PATCH_SIZE_XYZ, 0.0
            )
            seg_patch, _, _ = MODULE.extract_patch(
                seg, origin, MODULE.PATCH_SIZE_XYZ, 0
            )
            patch_affine = MODULE.translated_affine(affine, origin)
            np.savez_compressed(
                source_path,
                t1c=t1c_patch.astype(np.float32),
                seg=seg_patch.astype(np.uint8),
                hist=np.zeros((4, 16), dtype=np.float32),
                affine=patch_affine,
            )
            row = {
                "relative_path": relative_path,
                "case_id": case_id,
                "subject_id": subject_id,
                "patch_size_xyz": "64x64x32",
                "origin_x": str(origin[0]),
                "origin_y": str(origin[1]),
                "origin_z": str(origin[2]),
                "pad_before_x": str(before[0]),
                "pad_before_y": str(before[1]),
                "pad_before_z": str(before[2]),
                "pad_after_x": str(after[0]),
                "pad_after_y": str(after[1]),
                "pad_after_z": str(after[2]),
            }
            manifest = source_size_root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            split_file = root / "splits.json"
            split_file.write_text(
                json.dumps({"subject_split": {subject_id: "train"}}), encoding="utf-8"
            )

            audit = MODULE.build_multimodal_p64(
                raw_root=raw_root,
                raw_split="train",
                source_dataset_root=source_root,
                split_file=split_file,
                output_root=output_root,
                include_splits=("train", "val"),
                workers=1,
            )
            output_manifest = output_root / "patch_64x64x32" / "manifest.csv"
            self.assertEqual(sha256_file(manifest), sha256_file(output_manifest))
            self.assertEqual(audit["patch_count"], 1)
            self.assertFalse(audit["test_materialized"])
            with np.load(output_root / "patch_64x64x32" / relative_path) as arrays:
                self.assertEqual(set(arrays.files), set(MODULE.OUTPUT_KEYS))
                self.assertNotIn("hist", arrays.files)
                np.testing.assert_allclose(arrays["t1c"], t1c_patch, atol=1e-6)
            dataset = GLIClassifierPatchDataset(
                output_root,
                [{"relative_path": relative_path, "case_id": case_id, "subject_id": subject_id}],
                modalities=MODULE.MODALITIES,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (4, 32, 64, 64))
            self.assertEqual(tuple(sample["target"].shape), (32, 64, 64))


if __name__ == "__main__":
    unittest.main()
