"""BraTS2024 GLI T1c local-patch dataset for LeFusion.

The published NPZ files keep a scalar segmentation map.  This loader keeps
that map as ``label`` for compatibility with the original LeFusion dataset
contract and exposes a separate four-channel ``lesion_mask`` for GLI-aware
losses and inference.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_NAMES = ("netc", "snfh", "et", "rc")
LABEL_VALUES = (1, 2, 3, 4)
HIST_BINS = 16
COND_DIM = len(LABEL_VALUES) * HIST_BINS
REQUIRED_NPZ_KEYS = {"t1c", "seg", "hist", "affine"}


def _parse_patch_size(value: Sequence[int] | str) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = value.lower().replace("×", "x").split("x")
        value = tuple(int(part) for part in parts if part)
    result = tuple(int(part) for part in value)
    if len(result) != 3 or any(part <= 0 for part in result):
        raise ValueError(f"patch_size_xyz must contain three positive integers: {value!r}")
    return result  # type: ignore[return-value]


def _patch_dir_name(patch_size_xyz: Sequence[int]) -> str:
    return "patch_" + "x".join(str(int(value)) for value in patch_size_xyz)


def _as_int_tuple(value: Iterable[int], field_name: str) -> tuple[int, int, int]:
    result = tuple(int(part) for part in value)
    if len(result) != 3 or any(part < 0 for part in result):
        raise ValueError(f"invalid {field_name}: {value!r}")
    return result  # type: ignore[return-value]


class GLIDataset(Dataset):
    """Load T1c patches and GLI lesion conditions.

    ``data`` repeats the single T1c modality over four lesion-aligned model
    channels.  ``label`` is the scalar segmentation map with one channel;
    ``lesion_mask`` is the NETC/SNFH/ET/RC one-hot representation.
    """

    modality = "t1c"
    image_modality_channels = 1
    lesion_channels = LABEL_NAMES
    hist_bins = HIST_BINS
    cond_dim = COND_DIM

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        patch_size_xyz: Sequence[int] | str,
        split: str | None = "train",
        manifest_name: str = "manifest.csv",
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir).expanduser()
        self.patch_size_xyz = _parse_patch_size(patch_size_xyz)
        self.split = split
        self.strict = strict
        self.size_root = self.root_dir / _patch_dir_name(self.patch_size_xyz)
        self.manifest_path = self.size_root / manifest_name
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"GLI manifest not found: {self.manifest_path}")
        self.records = self._read_manifest()
        if not self.records:
            raise RuntimeError(
                f"no GLI patches matched split={self.split!r} in {self.manifest_path}"
            )

    def _read_manifest(self) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
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
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"manifest missing fields: {sorted(missing)}")
            seen: set[str] = set()
            for row in reader:
                if self.split is not None and row["split"] != self.split:
                    continue
                relative_path = row["relative_path"]
                if relative_path in seen:
                    raise ValueError(f"duplicate manifest path: {relative_path}")
                seen.add(relative_path)
                if row["patch_size_xyz"] != "x".join(map(str, self.patch_size_xyz)):
                    raise ValueError(
                        f"manifest patch size mismatch: {row['patch_size_xyz']} "
                        f"!= {self.patch_size_xyz}"
                    )
                path = (self.size_root / relative_path).resolve()
                if self.size_root.resolve() not in path.parents:
                    raise ValueError(f"manifest path escapes patch root: {relative_path}")
                records.append(row)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_npz(self, record: dict[str, str]) -> dict[str, torch.Tensor | str | int | float]:
        relative_path = record["relative_path"]
        path = self.size_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as arrays:
            keys = set(arrays.files)
            if keys != REQUIRED_NPZ_KEYS:
                raise ValueError(f"invalid NPZ keys in {path}: {sorted(keys)}")
            t1c = np.asarray(arrays["t1c"])
            seg = np.asarray(arrays["seg"])
            hist = np.asarray(arrays["hist"])
            affine = np.asarray(arrays["affine"])

        if t1c.shape != self.patch_size_xyz or seg.shape != self.patch_size_xyz:
            raise ValueError(
                f"invalid patch shape in {path}: t1c={t1c.shape}, seg={seg.shape}, "
                f"expected={self.patch_size_xyz}"
            )
        if t1c.dtype != np.float32 or seg.dtype != np.uint8:
            raise ValueError(f"invalid dtypes in {path}: {t1c.dtype}, {seg.dtype}")
        if hist.shape != (len(LABEL_VALUES), HIST_BINS) or hist.dtype != np.float32:
            raise ValueError(f"invalid histogram in {path}: {hist.shape}, {hist.dtype}")
        if affine.shape != (4, 4):
            raise ValueError(f"invalid affine in {path}: {affine.shape}")
        if not np.isfinite(t1c).all() or not np.isfinite(hist).all() or not np.isfinite(affine).all():
            raise ValueError(f"non-finite array in {path}")
        if not np.isfinite(seg).all() or not set(np.unique(seg).tolist()).issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"invalid segmentation labels in {path}")

        origin_xyz = _as_int_tuple(
            (record["origin_x"], record["origin_y"], record["origin_z"]), "origin_xyz"
        )
        pad_before_xyz = _as_int_tuple(
            (
                record["pad_before_x"],
                record["pad_before_y"],
                record["pad_before_z"],
            ),
            "pad_before_xyz",
        )
        pad_after_xyz = _as_int_tuple(
            (record["pad_after_x"], record["pad_after_y"], record["pad_after_z"]),
            "pad_after_xyz",
        )
        valid_xyz = tuple(
            size - before - after
            for size, before, after in zip(self.patch_size_xyz, pad_before_xyz, pad_after_xyz)
        )
        if any(value < 0 for value in valid_xyz):
            raise ValueError(f"padding exceeds patch size in {path}: {valid_xyz}")
        patch_voxels = int(np.prod(self.patch_size_xyz))
        valid_voxels = int(np.prod(valid_xyz))
        valid_volume_fraction = valid_voxels / patch_voxels
        padding_fraction = 1.0 - valid_volume_fraction

        # NIfTI/NPZ storage is XYZ; LeFusion uses channel, depth, height, width.
        image_dhw = np.transpose(t1c, (2, 0, 1)).copy()
        seg_dhw = np.transpose(seg, (2, 0, 1)).copy()
        data = np.repeat(image_dhw[None, ...], len(LABEL_VALUES), axis=0)
        scalar_label = seg_dhw[None, ...]
        lesion_mask = np.stack(
            [(seg_dhw == label_value) for label_value in LABEL_VALUES], axis=0
        ).astype(np.float32, copy=False)

        return {
            "data": torch.from_numpy(data.astype(np.float32, copy=False)),
            "label": torch.from_numpy(scalar_label.astype(np.int64, copy=False)),
            "lesion_mask": torch.from_numpy(lesion_mask),
            "hist": torch.from_numpy(hist.reshape(-1).astype(np.float32, copy=False)),
            "affine": torch.from_numpy(affine.astype(np.float32, copy=False)),
            "case_id": record["case_id"],
            "subject_id": record["subject_id"],
            "split": record["split"],
            "relative_path": relative_path,
            "patch_size_xyz": torch.tensor(self.patch_size_xyz, dtype=torch.int64),
            "anchor_label": int(record["anchor_label"]),
            "anchor_name": record["anchor_name"],
            "sample_role": record["sample_role"],
            "origin_xyz": torch.tensor(origin_xyz, dtype=torch.int64),
            "pad_before_xyz": torch.tensor(pad_before_xyz, dtype=torch.int64),
            "pad_after_xyz": torch.tensor(pad_after_xyz, dtype=torch.int64),
            "is_padded": bool(padding_fraction > 0.0),
            "padding_fraction": float(padding_fraction),
            "valid_volume_fraction": float(valid_volume_fraction),
            "normalization_p005": float(record["normalization_p005"]),
            "normalization_p995": float(record["normalization_p995"]),
            "normalization_degenerate": bool(int(record["normalization_degenerate"])),
            "cond_dim": self.cond_dim,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int | float]:
        return self._load_npz(self.records[index])
