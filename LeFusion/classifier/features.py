"""Leak-safe feature construction from T1c and total lesion mask only."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


FeatureKind = Literal["m0", "m1", "m2"]
FEATURE_CHANNELS = {"m0": 1, "m1": 17, "m2": 54}


def _validate(image: torch.Tensor, total_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if total_mask.ndim == 4 and total_mask.shape[0] == 1:
        total_mask = total_mask[0]
    if image.ndim != 3 or total_mask.ndim != 3 or image.shape != total_mask.shape:
        raise ValueError(
            f"expected matching [D,H,W] tensors, got {tuple(image.shape)}, {tuple(total_mask.shape)}"
        )
    image = image.to(dtype=torch.float32, device="cpu").contiguous()
    total_mask = total_mask.to(dtype=torch.bool, device="cpu").contiguous()
    if not torch.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    if not torch.any(total_mask):
        raise ValueError("total lesion mask is empty")
    return image, total_mask


def _coordinate_features(mask: torch.Tensor) -> list[torch.Tensor]:
    depth, height, width = mask.shape
    z = torch.linspace(-1.0, 1.0, depth).view(depth, 1, 1).expand_as(mask)
    y = torch.linspace(-1.0, 1.0, height).view(1, height, 1).expand_as(mask)
    x = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand_as(mask)
    coords = torch.stack((x, y, z), dim=0)

    lesion_indices = torch.nonzero(mask, as_tuple=False).to(torch.float32)
    mins_zyx = lesion_indices.min(dim=0).values
    maxs_zyx = lesion_indices.max(dim=0).values
    centroid_zyx = lesion_indices.mean(dim=0)
    extents_zyx = (maxs_zyx - mins_zyx).clamp_min(1.0)
    raw_zyx = torch.stack(
        torch.meshgrid(
            torch.arange(depth, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        ),
        dim=0,
    )
    bbox_zyx = 2.0 * (raw_zyx - mins_zyx[:, None, None, None]) / extents_zyx[
        :, None, None, None
    ] - 1.0
    centroid_zyx_relative = (raw_zyx - centroid_zyx[:, None, None, None]) / extents_zyx[
        :, None, None, None
    ]
    bbox_xyz = bbox_zyx[(2, 1, 0), ...]
    centroid_xyz = centroid_zyx_relative[(2, 1, 0), ...]
    return [*coords, *bbox_xyz, *centroid_xyz]


def _local_statistics(image: torch.Tensor, mask: torch.Tensor, kernel: int) -> list[torch.Tensor]:
    image_5d = image[None, None]
    mask_5d = mask.to(torch.float32)[None, None]
    padding = (kernel // 2,) * 6
    padded_image = F.pad(image_5d, padding)
    padded_mask = F.pad(mask_5d, padding)
    mean = F.avg_pool3d(padded_image, kernel, stride=1)[0, 0]
    mean_sq = F.avg_pool3d(padded_image.square(), kernel, stride=1)[0, 0]
    variance = (mean_sq - mean.square()).clamp_min(0.0)
    occupancy = F.avg_pool3d(padded_mask, kernel, stride=1)[0, 0]
    return [mean, variance.sqrt(), occupancy]


def _neighborhood_channels(tensor: torch.Tensor, kernel: int) -> torch.Tensor:
    padded = F.pad(tensor[None, None], (kernel // 2,) * 6)
    windows = (
        padded.unfold(2, kernel, 1)
        .unfold(3, kernel, 1)
        .unfold(4, kernel, 1)
    )
    depth, height, width = tensor.shape
    return windows.contiguous().view(1, depth, height, width, kernel**3)[0].permute(3, 0, 1, 2)


def build_feature_volume(
    image: torch.Tensor,
    total_mask: torch.Tensor,
    kind: FeatureKind,
) -> torch.Tensor:
    """Return [C,D,H,W] features without consulting class identities."""

    image, total_mask = _validate(image, total_mask)
    if kind == "m0":
        return image[None]
    if kind == "m2":
        image_neighbors = _neighborhood_channels(image, 3)
        mask_neighbors = _neighborhood_channels(total_mask.to(torch.float32), 3)
        return torch.cat((image_neighbors, mask_neighbors), dim=0)
    if kind != "m1":
        raise ValueError(f"unsupported feature kind: {kind}")

    distance_np = distance_transform_edt(total_mask.numpy()).astype(np.float32, copy=False)
    distance = torch.from_numpy(distance_np)
    distance = distance / distance.max().clamp_min(1.0)
    coordinate_features = _coordinate_features(total_mask)
    local_3 = _local_statistics(image, total_mask, 3)
    local_5 = _local_statistics(image, total_mask, 5)
    # 1 intensity + 3 patch XYZ + 3 bbox XYZ + 3 centroid XYZ + 1 distance
    # + 2x (mean, std, mask occupancy) = 17 channels.
    result = torch.stack(
        [image, *coordinate_features, distance, *local_3, *local_5], dim=0
    )
    if result.shape[0] != FEATURE_CHANNELS[kind]:
        raise AssertionError(f"unexpected M1 feature count: {result.shape[0]}")
    return result


def masked_feature_rows(features: torch.Tensor, total_mask: torch.Tensor) -> torch.Tensor:
    if total_mask.ndim == 4:
        total_mask = total_mask[0]
    return features.permute(1, 2, 3, 0)[total_mask.to(torch.bool)]
