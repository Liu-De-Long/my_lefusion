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
