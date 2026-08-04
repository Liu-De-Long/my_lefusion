from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "LeFusion"))

from dataset.gli_hist import GLIDataset  # noqa: E402


MANIFEST_FIELDS = (
    "relative_path",
    "case_id",
    "subject_id",
    "split",
    "patch_size_xyz",
    "anchor_label",
    "anchor_name",
    "sample_role",
    "origin_x",
    "origin_y",
    "origin_z",
    "pad_before_x",
    "pad_before_y",
    "pad_before_z",
    "pad_after_x",
    "pad_after_y",
    "pad_after_z",
    "normalization_p005",
    "normalization_p995",
    "normalization_degenerate",
)


class _DatasetConfig(SimpleNamespace):
    def get(self, name: str, default=None):
        return getattr(self, name, default)


def _write_fixture(root: Path, patch_size: tuple[int, int, int]) -> None:
    size_name = "patch_" + "x".join(map(str, patch_size))
    size_root = root / size_name
    patch_root = size_root / "patches"
    patch_root.mkdir(parents=True)
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        relative_path = f"patches/sample_{index}.npz"
        t1c = np.arange(np.prod(patch_size), dtype=np.float32).reshape(patch_size)
        seg = np.zeros(patch_size, dtype=np.uint8)
        seg[0, 0, 0] = 1
        seg[-1, -1, -1] = 4
        hist = np.arange(64, dtype=np.float32).reshape(4, 16)
        affine = np.eye(4, dtype=np.float64)
        np.savez_compressed(
            size_root / relative_path,
            t1c=t1c,
            seg=seg,
            hist=hist,
            affine=affine,
        )
        pad_before = (1, 2, 0) if index == 0 else (0, 0, 0)
        pad_after = (0, 1, 0) if index == 0 else (0, 0, 0)
        rows.append(
            {
                "relative_path": relative_path,
                "case_id": f"case-{index}",
                "subject_id": f"subject-{index}",
                "split": split,
                "patch_size_xyz": "x".join(map(str, patch_size)),
                "anchor_label": "1",
                "anchor_name": "netc",
                "sample_role": "interior",
                "origin_x": "0",
                "origin_y": "0",
                "origin_z": "0",
                "pad_before_x": str(pad_before[0]),
                "pad_before_y": str(pad_before[1]),
                "pad_before_z": str(pad_before[2]),
                "pad_after_x": str(pad_after[0]),
                "pad_after_y": str(pad_after[1]),
                "pad_after_z": str(pad_after[2]),
                "normalization_p005": "0.0",
                "normalization_p995": "1.0",
                "normalization_degenerate": "0",
            }
        )
    with (size_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class GLILoaderTests(unittest.TestCase):
    def test_axis_label_mask_hist_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            patch_size = (4, 5, 3)
            _write_fixture(root, patch_size)
            sample = GLIDataset(root, patch_size, split="train")[0]

            self.assertEqual(tuple(sample["data"].shape), (4, 3, 4, 5))
            self.assertEqual(tuple(sample["label"].shape), (1, 3, 4, 5))
            self.assertEqual(tuple(sample["lesion_mask"].shape), (4, 3, 4, 5))
            self.assertEqual(tuple(sample["hist"].shape), (64,))
            self.assertEqual(int(sample["label"][0, 0, 0, 0]), 1)
            self.assertEqual(int(sample["label"][0, 2, 3, 4]), 4)
            self.assertTrue(bool(sample["lesion_mask"][0, 0, 0, 0]))
            self.assertTrue(bool(sample["lesion_mask"][3, 2, 3, 4]))
            self.assertFalse(bool(sample["lesion_mask"][:, 1, 1, 1].any()))
            self.assertEqual(float(sample["hist"][0]), 0.0)
            self.assertEqual(float(sample["hist"][16]), 16.0)
            self.assertEqual(float(sample["hist"][48]), 48.0)

            patch_voxels = np.prod(patch_size)
            valid_voxels = (patch_size[0] - 1) * (patch_size[1] - 3) * patch_size[2]
            expected_padding = 1.0 - valid_voxels / patch_voxels
            self.assertTrue(sample["is_padded"])
            self.assertAlmostEqual(sample["padding_fraction"], expected_padding)
            self.assertAlmostEqual(sample["valid_volume_fraction"], 1.0 - expected_padding)

    def test_both_patch_sizes_and_batch_shapes(self) -> None:
        expected = {
            (64, 64, 32): (4, 32, 64, 64),
            (80, 96, 80): (4, 80, 80, 96),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for patch_size, shape in expected.items():
                _write_fixture(root, patch_size)
                dataset = GLIDataset(root, patch_size, split="train")
                batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
                self.assertEqual(tuple(batch["data"].shape), (2, *shape))
                self.assertEqual(tuple(batch["label"].shape), (2, 1, shape[1], shape[2], shape[3]))
                self.assertEqual(tuple(batch["lesion_mask"].shape), (2, 4, shape[1], shape[2], shape[3]))
                self.assertEqual(tuple(batch["hist"].shape), (2, 64))

    def test_factory_registers_gli(self) -> None:
        from get_dataset.get_dataset import get_train_dataset

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _write_fixture(root, (4, 5, 3))
            cfg = SimpleNamespace(
                dataset=_DatasetConfig(
                    data_type="gli",
                    root_dir=str(root),
                    patch_size_xyz=[4, 5, 3],
                    split="train",
                )
            )
            dataset, sampler = get_train_dataset(cfg)
            self.assertIsInstance(dataset, GLIDataset)
            self.assertIsNone(sampler)
            self.assertEqual(len(dataset), 2)

    @unittest.skipUnless(
        os.environ.get("GLI_PATCH_ROOT"),
        "set GLI_PATCH_ROOT to run against the published remote patch dataset",
    )
    def test_published_patch_sizes_iterate(self) -> None:
        root = Path(os.environ["GLI_PATCH_ROOT"])
        for patch_size in ((64, 64, 32), (80, 96, 80)):
            dataset = GLIDataset(root, patch_size, split="train")
            sample = dataset[0]
            self.assertEqual(sample["data"].shape[0], 4)
            self.assertEqual(sample["lesion_mask"].shape[0], 4)
            batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
            self.assertEqual(batch["hist"].shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
