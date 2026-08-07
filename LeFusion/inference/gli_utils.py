"""Pure GLI conditioning and spatial conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


LABEL_VALUES = (1, 2, 3, 4)
HIST_BINS = 16


def load_cluster_centers(path: str | Path, patch_size_xyz) -> list[torch.Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported GLI cluster schema")
    if tuple(payload.get("patch_size_xyz", ())) != tuple(int(v) for v in patch_size_xyz):
        raise ValueError("cluster/patch shape mismatch")
    if payload.get("source_split") != "train" or int(payload.get("condition_dim", -1)) != 64:
        raise ValueError("cluster asset must be a train-only 64-D condition")
    by_value = {int(item["value"]): item for item in payload["labels"]}
    result = []
    for label_value in LABEL_VALUES:
        centers = torch.tensor(by_value[label_value]["centers"], dtype=torch.float32)
        if centers.ndim != 2 or centers.shape[1] != HIST_BINS:
            raise ValueError(f"invalid centers for label {label_value}: {tuple(centers.shape)}")
        if not bool(torch.isfinite(centers).all()):
            raise ValueError(f"non-finite centers for label {label_value}")
        result.append(centers)
    return result


def nearest_cluster_condition(
    source_hist: torch.Tensor,
    scalar_seg: torch.Tensor,
    centers: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_hist.ndim != 2 or source_hist.shape[1] != 64:
        raise ValueError(f"source_hist must be [B,64], got {tuple(source_hist.shape)}")
    if scalar_seg.ndim != 5 or scalar_seg.shape[1] != 1:
        raise ValueError(f"scalar_seg must be [B,1,D,H,W], got {tuple(scalar_seg.shape)}")
    batch = source_hist.shape[0]
    condition = torch.zeros_like(source_hist, dtype=torch.float32)
    cluster_ids = torch.full((batch, 4), -1, dtype=torch.int64, device=source_hist.device)
    for channel, label_value in enumerate(LABEL_VALUES):
        block = source_hist[:, channel * HIST_BINS : (channel + 1) * HIST_BINS].float()
        current_centers = centers[channel].to(source_hist.device)
        present = (scalar_seg == label_value).flatten(1).any(dim=1)
        if bool(present.any()):
            distances = torch.cdist(block[present], current_centers)
            selected = distances.argmin(dim=1)
            condition[present, channel * HIST_BINS : (channel + 1) * HIST_BINS] = current_centers[selected]
            cluster_ids[present, channel] = selected
    return condition, cluster_ids


def indexed_cluster_condition(
    source_hist: torch.Tensor,
    scalar_seg: torch.Tensor,
    centers: list[torch.Tensor],
    *,
    index_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the first or last train-only center for every present lesion class."""
    if index_mode not in {"first", "last"}:
        raise ValueError(f"unsupported cluster index mode: {index_mode}")
    if source_hist.ndim != 2 or source_hist.shape[1] != 64:
        raise ValueError(f"source_hist must be [B,64], got {tuple(source_hist.shape)}")
    condition = torch.zeros_like(source_hist, dtype=torch.float32)
    cluster_ids = torch.full(
        (source_hist.shape[0], 4), -1, dtype=torch.int64, device=source_hist.device
    )
    for channel, label_value in enumerate(LABEL_VALUES):
        present = (scalar_seg == label_value).flatten(1).any(dim=1)
        if not bool(present.any()):
            continue
        current_centers = centers[channel].to(source_hist.device)
        selected_index = 0 if index_mode == "first" else current_centers.shape[0] - 1
        start = channel * HIST_BINS
        condition[present, start:start + HIST_BINS] = current_centers[selected_index]
        cluster_ids[present, channel] = selected_index
    return condition, cluster_ids


def union_label_cluster_condition(
    source_hist: torch.Tensor,
    scalar_seg: torch.Tensor,
    target_labels: torch.Tensor,
    centers: list[torch.Tensor],
    *,
    index_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign the full lesion union to an explicit target class and center."""
    if index_mode not in {"first", "last"}:
        raise ValueError(f"unsupported cluster index mode: {index_mode}")
    target_labels = target_labels.to(
        device=source_hist.device, dtype=torch.long
    ).reshape(-1)
    if target_labels.shape[0] != source_hist.shape[0]:
        raise ValueError("target_labels batch size does not match source_hist")
    if not bool(
        torch.isin(
            target_labels, torch.tensor(LABEL_VALUES, device=target_labels.device)
        ).all()
    ):
        raise ValueError(f"target_labels must be in {LABEL_VALUES}")
    lesion_union = scalar_seg > 0
    if not bool(lesion_union.flatten(1).any(dim=1).all()):
        raise ValueError("every union-as-single sample must contain lesion voxels")
    target_seg = torch.where(
        lesion_union,
        target_labels.view(-1, 1, 1, 1, 1),
        torch.zeros((), device=scalar_seg.device, dtype=torch.long),
    )
    target_mask = torch.cat(
        [(target_seg == value) for value in LABEL_VALUES], dim=1
    ).to(dtype=torch.float32)
    condition = torch.zeros_like(source_hist, dtype=torch.float32)
    cluster_ids = torch.full(
        (source_hist.shape[0], 4), -1, dtype=torch.int64, device=source_hist.device
    )
    for channel, label_value in enumerate(LABEL_VALUES):
        selected_samples = target_labels == label_value
        if not bool(selected_samples.any()):
            continue
        current_centers = centers[channel].to(source_hist.device)
        selected_index = 0 if index_mode == "first" else current_centers.shape[0] - 1
        start = channel * HIST_BINS
        condition[selected_samples, start:start + HIST_BINS] = current_centers[selected_index]
        cluster_ids[selected_samples, channel] = selected_index
    return target_seg, target_mask, condition, cluster_ids


def mask_input_inside_lesion(
    input_t1c: torch.Tensor,
    scalar_seg: torch.Tensor,
    *,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Explicitly remove every labelled lesion voxel from the model input."""
    if input_t1c.ndim != 5 or input_t1c.shape[1] != 1:
        raise ValueError(f"input_t1c must be [B,1,D,H,W], got {tuple(input_t1c.shape)}")
    if scalar_seg.shape != input_t1c.shape:
        raise ValueError(
            "scalar_seg must have the same [B,1,D,H,W] shape as input_t1c: "
            f"{tuple(scalar_seg.shape)} != {tuple(input_t1c.shape)}"
        )
    lesion_union = scalar_seg > 0
    if not bool(lesion_union.flatten(1).any(dim=1).all()):
        raise ValueError("every masked-input QA sample must contain lesion voxels")
    return torch.where(
        lesion_union,
        torch.as_tensor(fill_value, device=input_t1c.device, dtype=input_t1c.dtype),
        input_t1c,
    )


def anchor_union_cluster_condition(
    source_hist: torch.Tensor,
    scalar_seg: torch.Tensor,
    anchor_labels: torch.Tensor,
    centers: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map the complete lesion union to one anchor-label channel and condition."""
    if source_hist.ndim != 2 or source_hist.shape[1] != len(LABEL_VALUES) * HIST_BINS:
        raise ValueError(f"source_hist must be [B,64], got {tuple(source_hist.shape)}")
    if scalar_seg.ndim != 5 or scalar_seg.shape[1] != 1:
        raise ValueError(f"scalar_seg must be [B,1,D,H,W], got {tuple(scalar_seg.shape)}")
    anchor_labels = anchor_labels.to(device=source_hist.device, dtype=torch.long).reshape(-1)
    if anchor_labels.shape[0] != source_hist.shape[0]:
        raise ValueError("anchor_labels batch size does not match source_hist")
    if not bool(torch.isin(anchor_labels, torch.tensor(LABEL_VALUES, device=anchor_labels.device)).all()):
        raise ValueError(f"anchor_labels must be in {LABEL_VALUES}")

    lesion_union = scalar_seg > 0
    if not bool(lesion_union.flatten(1).any(dim=1).all()):
        raise ValueError("every anchor-union QA sample must contain lesion voxels")
    target_seg = torch.where(
        lesion_union,
        anchor_labels.view(-1, 1, 1, 1, 1),
        torch.zeros((), device=scalar_seg.device, dtype=torch.long),
    ).to(dtype=torch.long)
    target_mask = torch.cat(
        [(target_seg == label_value) for label_value in LABEL_VALUES], dim=1
    ).to(dtype=torch.float32)
    condition = torch.zeros_like(source_hist, dtype=torch.float32)
    cluster_ids = torch.full(
        (source_hist.shape[0], len(LABEL_VALUES)),
        -1,
        dtype=torch.int64,
        device=source_hist.device,
    )
    for channel, label_value in enumerate(LABEL_VALUES):
        selected_samples = anchor_labels == label_value
        if not bool(selected_samples.any()):
            continue
        block_start = channel * HIST_BINS
        block_end = (channel + 1) * HIST_BINS
        source_block = source_hist[selected_samples, block_start:block_end].float()
        current_centers = centers[channel].to(source_hist.device)
        selected_centers = torch.cdist(source_block, current_centers).argmin(dim=1)
        condition[selected_samples, block_start:block_end] = current_centers[selected_centers]
        cluster_ids[selected_samples, channel] = selected_centers
    return target_seg, target_mask, condition, cluster_ids


def dhw_to_xyz(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        if tensor_or_array.ndim != 3:
            raise ValueError(f"DHW tensor must be 3-D, got {tensor_or_array.shape}")
        return tensor_or_array.permute(1, 2, 0)
    array = np.asarray(tensor_or_array)
    if array.ndim != 3:
        raise ValueError(f"DHW array must be 3-D, got {array.shape}")
    return np.transpose(array, (1, 2, 0))


def xyz_to_dhw(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        if tensor_or_array.ndim != 3:
            raise ValueError(f"XYZ tensor must be 3-D, got {tensor_or_array.shape}")
        return tensor_or_array.permute(2, 0, 1)
    array = np.asarray(tensor_or_array)
    if array.ndim != 3:
        raise ValueError(f"XYZ array must be 3-D, got {array.shape}")
    return np.transpose(array, (2, 0, 1))
