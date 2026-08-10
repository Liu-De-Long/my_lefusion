"""Inference adapter and explicit multimodal support mask for GLI patches."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from dataset.gli_hist import GLIDataset


MODALITIES = ("t1c", "t1n", "t2f", "t2w")


def build_explicit_brain_support(
    modalities_xyz: Sequence[np.ndarray],
    segmentation_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Build a reproducible support mask, not a manually annotated brain mask."""
    if len(modalities_xyz) != len(MODALITIES):
        raise ValueError(f"expected four modalities, got {len(modalities_xyz)}")
    shape = modalities_xyz[0].shape
    if any(array.shape != shape for array in modalities_xyz):
        raise ValueError("multimodal shape mismatch")
    votes = np.stack([np.asarray(array) != 0 for array in modalities_xyz], axis=0).sum(axis=0)
    consensus = votes >= 2
    components, count = ndimage.label(consensus)
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        consensus = components == int(np.argmax(sizes))
        consensus = ndimage.binary_fill_holes(consensus)
    consensus = np.asarray(consensus, dtype=bool)
    outside_voxels = 0
    if segmentation_xyz is not None:
        lesion = np.asarray(segmentation_xyz) > 0
        if lesion.shape != shape:
            raise ValueError("segmentation/support shape mismatch")
        outside_voxels = int(np.count_nonzero(lesion & ~consensus))
        consensus |= lesion
    return consensus, outside_voxels


def extract_xyz_patch(
    array: np.ndarray,
    origin_xyz: Sequence[int],
    patch_size_xyz: Sequence[int],
    *,
    fill_value: int | float | bool = 0,
) -> np.ndarray:
    origin = tuple(int(value) for value in origin_xyz)
    size = tuple(int(value) for value in patch_size_xyz)
    patch = np.full(size, fill_value, dtype=array.dtype)
    source_start = tuple(max(0, value) for value in origin)
    source_end = tuple(min(array.shape[axis], origin[axis] + size[axis]) for axis in range(3))
    target_start = tuple(max(0, -origin[axis]) for axis in range(3))
    target_end = tuple(
        target_start[axis] + max(0, source_end[axis] - source_start[axis]) for axis in range(3)
    )
    if all(source_end[axis] > source_start[axis] for axis in range(3)):
        patch[
            target_start[0] : target_end[0],
            target_start[1] : target_end[1],
            target_start[2] : target_end[2],
        ] = array[
            source_start[0] : source_end[0],
            source_start[1] : source_end[1],
            source_start[2] : source_end[2],
        ]
    return patch


class GLIInferenceDataset(GLIDataset):
    """Map GLI training patches to the RePaint inference contract."""

    def __init__(
        self,
        root_dir: str | Path,
        raw_root_dir: str | Path,
        patch_size_xyz: Sequence[int] | str,
        split: str = "test",
        split_file: str | Path | None = None,
        raw_source_split: str = "train",
        selected_relative_paths: Sequence[str] | None = None,
        mask_overlay_root: str | Path | None = None,
    ) -> None:
        self.raw_root_dir = Path(raw_root_dir).expanduser()
        self.raw_source_split = raw_source_split
        super().__init__(
            root_dir=root_dir,
            patch_size_xyz=patch_size_xyz,
            split=split,
            split_file=split_file,
            mask_overlay_root=mask_overlay_root,
        )
        if selected_relative_paths is not None:
            by_path = {str(record["relative_path"]): record for record in self.records}
            requested = [str(path) for path in selected_relative_paths]
            if len(requested) != len(set(requested)):
                raise ValueError("selected_relative_paths contains duplicates")
            missing = [path for path in requested if path not in by_path]
            if missing:
                raise ValueError(
                    f"selection contains {len(missing)} paths outside split={split!r}: "
                    f"{missing[:3]}"
                )
            self.records = [by_path[path] for path in requested]

    @lru_cache(maxsize=4)
    def _case_support(self, case_id: str) -> tuple[np.ndarray, int]:
        case_dir = self.raw_root_dir / self.raw_source_split / case_id
        arrays = []
        reference_shape = None
        for modality in MODALITIES:
            path = case_dir / f"{case_id}-{modality}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            array = np.asanyarray(nib.load(str(path)).dataobj)
            reference_shape = reference_shape or array.shape
            if array.shape != reference_shape:
                raise ValueError(f"multimodal shape mismatch for {case_id}")
            arrays.append(array)
        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        if not seg_path.is_file():
            raise FileNotFoundError(seg_path)
        segmentation = np.asanyarray(nib.load(str(seg_path)).dataobj)
        return build_explicit_brain_support(arrays, segmentation)

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        case_id = str(sample["case_id"])
        support_xyz, outside_voxels = self._case_support(case_id)
        support_patch_xyz = extract_xyz_patch(
            support_xyz,
            sample["origin_xyz"].tolist(),
            self.patch_size_xyz,
            fill_value=False,
        )
        support_dhw = np.transpose(support_patch_xyz, (2, 0, 1)).copy()
        support = torch.from_numpy(support_dhw[None, ...].astype(np.bool_, copy=False))
        scalar_label = sample["label"]
        lesion_union = scalar_label > 0
        if bool((lesion_union & ~support).any()):
            raise ValueError(f"lesion lies outside explicit support in patch {sample['relative_path']}")
        healthy = support & ~lesion_union
        outside = ~support
        input_t1c = sample["data"][:1]
        result = dict(sample)
        result.update(
            {
                "GT": sample["data"],
                "GT_name": Path(str(sample["relative_path"])).stem,
                "gt_keep_mask": scalar_label,
                "input_t1c": input_t1c,
                "conditioning_seg": scalar_label,
                "explicit_brain_support_mask": support,
                "healthy_brain_mask": healthy,
                "outside_mask": outside,
                "lesion_outside_consensus_voxels_full_volume": outside_voxels,
            }
        )
        return result
