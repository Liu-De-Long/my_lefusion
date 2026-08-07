"""Training and evaluation engine for the p64 weak-supervision classifier."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from .data import (
    GLIClassifierPatchDataset,
    LABEL_VALUES,
    batch_records_by_anchor,
    class_weights_from_subset,
    load_labeled_subset,
    load_manifest_records,
    sha256_file,
)
from .features import FEATURE_CHANNELS, build_feature_volume, masked_feature_rows
from .metrics import PatientMetricAccumulator, reconstruct_prediction
from .models import build_classifier, count_parameters
from .tracking import classifier_wandb_run


FeatureRowCache = dict[str, tuple[torch.Tensor, torch.Tensor]]
SPATIAL_MODEL_KINDS = {"c0", "unet3d", "geometry_unet3d"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid classifier config: {config_path}")
    required = {"experiment_id", "data", "model", "training", "evaluation"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"config missing sections: {sorted(missing)}")
    payload["_config_path"] = str(config_path.resolve())
    payload["_config_sha256"] = sha256_file(config_path)
    return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    value = str(value)
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {value}")
    return device


def clean_git_provenance(config: Mapping[str, Any]) -> dict[str, str]:
    config_path = Path(str(config["_config_path"])).resolve()
    project_root = config_path.parents[2]

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError(
            "formal classifier training requires a clean Git worktree; "
            f"found:\n{status}"
        )
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "project_root": str(project_root),
    }


def _safe_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(record["relative_path"]),
            "case_id": str(record["case_id"]),
            "subject_id": str(record["subject_id"]),
            **(
                {
                    "anchor_label": int(record["anchor_label"]),
                    "sample_role": str(record["sample_role"]),
                }
                if "anchor_label" in record
                else {}
            ),
        }
        for record in records
    ]


def _model_from_config(config: Mapping[str, Any]) -> nn.Module:
    model_config = config["model"]
    return build_classifier(
        str(model_config["kind"]),
        hidden_dims=tuple(model_config.get("hidden_dims", [128, 64])),
        dropout=float(model_config.get("dropout", 0.1)),
        cnn_channels=int(model_config.get("cnn_channels", 24)),
        unet_base_channels=int(model_config.get("unet_base_channels", 24)),
    )


def _load_geometry_warmstart(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    subset_path: Path,
    device: torch.device,
) -> str:
    """Expand a two-channel U-Net checkpoint into the 18-channel geometry model."""

    payload = torch.load(checkpoint_path, map_location=device)
    if payload.get("model_kind") != "unet3d":
        raise ValueError("geometry warm-start requires an unet3d checkpoint")
    if payload.get("subset_sha256") != sha256_file(subset_path):
        raise ValueError("geometry warm-start labeled subset mismatch")
    source_state = payload.get("model")
    if not isinstance(source_state, Mapping):
        raise ValueError("geometry warm-start checkpoint has no model state")
    target_state = model.state_dict()
    expandable = {"encoder0.body.0.weight", "encoder0.skip.weight"}
    mismatched: list[str] = []
    for key, target_value in target_state.items():
        if key not in source_state:
            raise ValueError(f"geometry warm-start is missing parameter: {key}")
        source_value = source_state[key]
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            continue
        if key not in expandable or source_value.shape[1] != 2 or target_value.shape[1] != 18:
            mismatched.append(key)
            continue
        expanded = torch.zeros_like(target_value)
        expanded[:, 0] = source_value[:, 0]
        expanded[:, -1] = source_value[:, 1]
        target_state[key] = expanded
    unexpected = sorted(set(source_state).difference(target_state))
    if mismatched or unexpected:
        raise ValueError(
            "geometry warm-start architecture mismatch: "
            f"mismatched={sorted(mismatched)}, unexpected={unexpected}"
        )
    model.load_state_dict(target_state, strict=True)
    return sha256_file(checkpoint_path)


def update_selection_state(
    metric: float,
    *,
    best_metric: float,
    patience_metric: float,
    bad_epochs: int,
    min_delta: float,
) -> tuple[float, float, int, bool, bool]:
    """Track the absolute best checkpoint independently from early stopping."""

    metric = float(metric)
    is_absolute_best = metric > float(best_metric)
    if is_absolute_best:
        best_metric = metric
    is_significant_improvement = metric > float(patience_metric) + float(min_delta)
    if is_significant_improvement:
        patience_metric = metric
        bad_epochs = 0
    else:
        bad_epochs = int(bad_epochs) + 1
    return (
        float(best_metric),
        float(patience_metric),
        int(bad_epochs),
        bool(is_absolute_best),
        bool(is_significant_improvement),
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    config: Mapping[str, Any],
    subset_path: Path,
    epoch: int,
    best_metric: float,
    patience_metric: float,
    bad_epochs: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "model_kind": config["model"]["kind"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "patience_metric": float(patience_metric),
        "bad_epochs": int(bad_epochs),
        "config_sha256": config["_config_sha256"],
        "subset_sha256": sha256_file(subset_path),
        "git_head": config["_git_head"],
        "git_branch": config["_git_branch"],
        "wandb_run_id": str(config["wandb"]["run_id"]),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    config: Mapping[str, Any],
    subset_path: Path,
    device: torch.device,
) -> tuple[int, float, float, int]:
    payload = torch.load(path, map_location=device)
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported classifier checkpoint schema")
    if payload.get("experiment_id") != config["experiment_id"]:
        raise ValueError("checkpoint experiment ID mismatch")
    if payload.get("model_kind") != config["model"]["kind"]:
        raise ValueError("checkpoint model kind mismatch")
    if payload.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("checkpoint config hash mismatch")
    if payload.get("subset_sha256") != sha256_file(subset_path):
        raise ValueError("checkpoint subset hash mismatch")
    if payload.get("schema_version") == 2 and payload.get("wandb_run_id") != str(
        config["wandb"]["run_id"]
    ):
        raise ValueError("checkpoint W&B run ID mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["cuda_rng_state_all"]]
        )
    best_metric = float(payload["best_metric"])
    patience_metric = float(payload.get("patience_metric", best_metric))
    return (
        int(payload["epoch"]) + 1,
        best_metric,
        patience_metric,
        int(payload["bad_epochs"]),
    )


def validate_test_gate(
    checkpoint_path: str | Path,
    *,
    config_sha256: str,
    subset_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless the exact frozen best checkpoint passed every val gate."""

    checkpoint_path = Path(checkpoint_path)
    subset_path = Path(subset_path)
    if checkpoint_path.name != "best.pt":
        raise ValueError("test evaluation is restricted to the frozen best.pt checkpoint")
    gate_path = checkpoint_path.parent / "best_val_metrics.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"test requires frozen validation gate: {gate_path}")
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    required_gates = (
        "focus_miou_at_least_0_85",
        "et_iou_at_least_0_80",
        "rc_iou_at_least_0_80",
        "outside_nonzero_is_zero",
        "union_dice_is_one",
    )
    gates = gate_payload.get("gate", {})
    failed = [name for name in required_gates if gates.get(name) is not True]
    if failed:
        raise RuntimeError(f"test is sealed because validation gates failed: {failed}")
    if gate_payload.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("validation gate checkpoint hash does not match requested test checkpoint")
    if gate_payload.get("config_sha256") != config_sha256:
        raise ValueError("validation gate config hash mismatch")
    if gate_payload.get("subset_sha256") != sha256_file(subset_path):
        raise ValueError("validation gate subset hash mismatch")
    return gate_payload


def _sample_rows(
    rows: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if rows.shape[0] == 0:
        raise ValueError("cannot sample an empty class bucket")
    if rows.shape[0] >= count:
        indices = torch.randperm(rows.shape[0], generator=generator)[:count]
    else:
        indices = torch.randint(rows.shape[0], (count,), generator=generator)
    return rows[indices]


def _feature_rows_and_target(
    sample: Mapping[str, Any],
    *,
    feature_kind: str,
    cache: FeatureRowCache | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return immutable mask-inside rows, optionally reusing leak-safe features.

    The cache key is the frozen relative patch path.  Cached feature rows are
    derived only from T1c and the total lesion mask; target rows remain a
    separate tensor used exclusively for loss sampling.
    """

    key = str(sample["relative_path"])
    if cache is not None and key in cache:
        return cache[key]
    features = build_feature_volume(
        sample["image"], sample["total_mask"], feature_kind  # type: ignore[arg-type]
    )
    rows = masked_feature_rows(features, sample["total_mask"])  # type: ignore[arg-type]
    target = sample["target"][sample["total_mask"][0]] - 1  # type: ignore[index,operator]
    if cache is not None:
        cache[key] = (rows, target)
    return rows, target


def build_balanced_voxel_batch(
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    feature_kind: str,
    voxels_per_class: int,
    generator: torch.Generator,
    feature_cache: FeatureRowCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = GLIClassifierPatchDataset(dataset_root, _safe_records(records), load_targets=True)
    buckets: dict[int, list[torch.Tensor]] = {index: [] for index in range(4)}
    for sample in dataset:
        rows, target = _feature_rows_and_target(
            sample,
            feature_kind=feature_kind,
            cache=feature_cache,
        )
        for class_index in range(4):
            class_rows = rows[target == class_index]
            if class_rows.shape[0] > voxels_per_class:
                indices = torch.randperm(class_rows.shape[0], generator=generator)[:voxels_per_class]
                class_rows = class_rows[indices]
            if class_rows.numel():
                buckets[class_index].append(class_rows)
    missing = [index for index, values in buckets.items() if not values]
    if missing:
        raise RuntimeError(f"training patch group has no target voxels for classes: {missing}")
    features_by_class = [
        _sample_rows(
            torch.cat(buckets[class_index], dim=0),
            count=voxels_per_class,
            generator=generator,
        )
        for class_index in range(4)
    ]
    features = torch.cat(features_by_class, dim=0)
    target = torch.arange(4).repeat_interleave(voxels_per_class)
    permutation = torch.randperm(target.numel(), generator=generator)
    return features[permutation], target[permutation]


def _soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    class_multipliers: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    safe_target = target.clamp(0, 3)
    one_hot = F.one_hot(safe_target, 4).permute(0, 4, 1, 2, 3).to(probabilities.dtype)
    mask_float = mask[:, None].to(probabilities.dtype)
    probabilities = probabilities * mask_float
    one_hot = one_hot * mask_float
    reduction_dims = (0, 2, 3, 4)
    numerator = 2.0 * (probabilities * one_hot).sum(dim=reduction_dims)
    denominator = probabilities.sum(dim=reduction_dims) + one_hot.sum(dim=reduction_dims)
    valid = denominator > 0
    dice = (numerator[valid] + 1e-6) / (denominator[valid] + 1e-6)
    if class_multipliers is None:
        return 1.0 - dice.mean()
    weights = class_multipliers.to(dice.device, dice.dtype)[valid]
    return 1.0 - (dice * weights).sum() / weights.sum().clamp_min(1e-6)


def _interclass_boundary_mask(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return mask-inside voxels whose 3x3x3 neighborhood contains >1 class."""

    safe_target = target.clamp(0, 3)
    one_hot = F.one_hot(safe_target, 4).permute(0, 4, 1, 2, 3).to(torch.float32)
    one_hot = one_hot * mask[:, None].to(one_hot.dtype)
    neighborhood_presence = F.max_pool3d(one_hot, kernel_size=3, stride=1, padding=1)
    return (neighborhood_presence.sum(dim=1) > 1) & mask.to(torch.bool)


def _lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    count = sorted_foreground.numel()
    total = sorted_foreground.sum()
    intersection = total - sorted_foreground.cumsum(dim=0)
    union = total + (1.0 - sorted_foreground).cumsum(dim=0)
    gradient = 1.0 - intersection / union.clamp_min(1e-6)
    if count > 1:
        gradient[1:] = gradient[1:] - gradient[:-1]
    return gradient


def _lovasz_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_multipliers: torch.Tensor,
) -> torch.Tensor:
    """Multiclass Lovasz-Softmax over mask-inside voxels only."""

    probabilities = logits.softmax(dim=1).permute(0, 2, 3, 4, 1)[mask]
    labels = target[mask].clamp(0, 3)
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for class_index in range(4):
        foreground = (labels == class_index).to(probabilities.dtype)
        if not torch.any(foreground):
            continue
        errors = (foreground - probabilities[:, class_index]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, _lovasz_gradient(foreground_sorted)))
        weights.append(class_multipliers[class_index].to(probabilities.dtype))
    if not losses:
        return logits.sum() * 0.0
    stacked_losses = torch.stack(losses)
    stacked_weights = torch.stack(weights).to(stacked_losses.device)
    return (stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp_min(1e-6)


def _loss_settings(training_config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    loss_config = training_config.get("loss", {})
    focus_multiplier = float(loss_config.get("focus_class_multiplier", 1.0))
    return {
        "ce_weight": float(loss_config.get("ce_weight", 0.7)),
        "focal_weight": float(loss_config.get("focal_weight", 0.0)),
        "dice_weight": float(loss_config.get("dice_weight", 0.3)),
        "lovasz_weight": float(loss_config.get("lovasz_weight", 0.0)),
        "boundary_multiplier": float(loss_config.get("boundary_multiplier", 0.0)),
        "focal_gamma": float(loss_config.get("focal_gamma", 2.0)),
        "class_multipliers": torch.tensor(
            [1.0, 1.0, focus_multiplier, focus_multiplier], device=device
        ),
    }


def _masked_supervised_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    safe_target = target.clamp(0, 3)
    effective_class_weights = class_weights.to(logits.device) * settings["class_multipliers"]
    ce_map = F.cross_entropy(
        logits,
        safe_target,
        weight=effective_class_weights,
        reduction="none",
    )
    boundary_multiplier = float(settings["boundary_multiplier"])
    boundary = (
        _interclass_boundary_mask(safe_target, mask)
        if boundary_multiplier > 0.0
        else torch.zeros_like(mask, dtype=torch.bool)
    )
    voxel_weights = mask.to(logits.dtype) * (
        1.0 + boundary_multiplier * boundary.to(logits.dtype)
    )
    denominator = voxel_weights.sum().clamp_min(1.0)
    cross_entropy = (ce_map * voxel_weights).sum() / denominator
    target_probability = logits.softmax(dim=1).gather(1, safe_target[:, None]).squeeze(1)
    focal = (
        ((1.0 - target_probability) ** float(settings["focal_gamma"]))
        * ce_map
        * voxel_weights
    ).sum()
    focal = focal / denominator
    dice = _soft_dice_loss(
        logits,
        target,
        mask,
        class_multipliers=settings["class_multipliers"],
    )
    lovasz = (
        _lovasz_softmax_loss(
            logits,
            safe_target,
            mask,
            class_multipliers=settings["class_multipliers"],
        )
        if float(settings["lovasz_weight"]) > 0.0
        else logits.sum() * 0.0
    )
    loss = (
        float(settings["ce_weight"]) * cross_entropy
        + float(settings["focal_weight"]) * focal
        + float(settings["dice_weight"]) * dice
        + float(settings["lovasz_weight"]) * lovasz
    )
    return loss, {
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "focal": float(focal.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "lovasz_loss": float(lovasz.detach().cpu()),
        "boundary_fraction": float(
            (boundary.sum() / mask.sum().clamp_min(1)).detach().cpu()
        ),
    }


def _build_spatial_inputs(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    kind: str,
) -> torch.Tensor:
    """Build dense spatial inputs without consulting scalar class targets."""

    if image.ndim != 5 or image.shape[1] != 1 or mask.ndim != 4:
        raise ValueError(
            f"expected image [B,1,D,H,W] and mask [B,D,H,W], got {image.shape}, {mask.shape}"
        )
    if kind in {"c0", "unet3d"}:
        return torch.cat((image, mask[:, None].to(image.dtype)), dim=1)
    if kind != "geometry_unet3d":
        raise ValueError(f"unsupported spatial input kind: {kind}")
    feature_volumes = torch.stack(
        [
            build_feature_volume(image[index], mask[index], "m1")
            for index in range(image.shape[0])
        ]
    )
    return torch.cat((feature_volumes, mask[:, None].to(feature_volumes.dtype)), dim=1)


def _augment_spatial_batch(
    image: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor | None,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    for image_axis, spatial_axis in ((2, 1), (3, 2), (4, 3)):
        if float(torch.rand((), generator=generator)) < 0.5:
            image = torch.flip(image, dims=(image_axis,))
            mask = torch.flip(mask, dims=(spatial_axis,))
            if target is not None:
                target = torch.flip(target, dims=(spatial_axis,))
    return image, mask, target


def _augment_intensity(
    image: torch.Tensor,
    *,
    generator: torch.Generator,
    scale_range: float,
    shift_range: float,
    noise_std: float,
) -> torch.Tensor:
    batch = image.shape[0]
    scale = 1.0 + (torch.rand((batch, 1, 1, 1, 1), generator=generator) * 2.0 - 1.0) * scale_range
    shift = (torch.rand((batch, 1, 1, 1, 1), generator=generator) * 2.0 - 1.0) * shift_range
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype) * noise_std
    return image * scale.to(image.dtype) + shift.to(image.dtype) + noise


def _train_mlp_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    kind: str,
    epoch: int,
    seed: int,
    patches_per_class: int,
    voxels_per_class: int,
    device: torch.device,
    feature_cache: FeatureRowCache | None,
) -> float:
    model.train()
    losses: list[float] = []
    generator = torch.Generator().manual_seed(seed + 10_000_019 * epoch)
    for batch_records in batch_records_by_anchor(
        records,
        patches_per_class=patches_per_class,
        seed=seed,
        epoch=epoch,
    ):
        features, target = build_balanced_voxel_batch(
            dataset_root,
            batch_records,
            feature_kind=kind,
            voxels_per_class=voxels_per_class,
            generator=generator,
            feature_cache=feature_cache,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(features.to(device, non_blocking=True))
        loss = F.cross_entropy(logits, target.to(device, non_blocking=True))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _train_cnn_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    epoch: int,
    seed: int,
    batch_size: int,
    class_weights: torch.Tensor,
    device: torch.device,
    training_config: Mapping[str, Any],
    kind: str,
) -> float:
    model.train()
    generator = torch.Generator().manual_seed(seed + 10_000_019 * epoch)
    order = torch.randperm(len(records), generator=generator).tolist()
    losses: list[float] = []
    settings = _loss_settings(training_config, device)
    augmentation = training_config.get("augmentation", {})
    use_augmentation = bool(augmentation.get("enabled", False))
    for start in range(0, len(order), batch_size):
        batch_records = [_safe_records([records[index]])[0] for index in order[start : start + batch_size]]
        dataset = GLIClassifierPatchDataset(dataset_root, batch_records, load_targets=True)
        samples = [dataset[index] for index in range(len(dataset))]
        image = torch.stack([sample["image"] for sample in samples])  # type: ignore[list-item]
        mask = torch.stack([sample["total_mask"][0] for sample in samples])  # type: ignore[index]
        target = torch.stack([sample["target"] for sample in samples]) - 1  # type: ignore[list-item,operator]
        if use_augmentation:
            image, mask, target_augmented = _augment_spatial_batch(
                image, mask, target, generator=generator
            )
            image = _augment_intensity(
                image,
                generator=generator,
                scale_range=float(augmentation.get("scale_range", 0.1)),
                shift_range=float(augmentation.get("shift_range", 0.1)),
                noise_std=float(augmentation.get("noise_std", 0.03)),
            )
            if target_augmented is None:
                raise AssertionError("labeled augmentation lost its target")
            target = target_augmented
        inputs = _build_spatial_inputs(image, mask, kind=kind).to(device)
        mask = mask.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss, _ = _masked_supervised_loss(
            logits,
            target,
            mask,
            class_weights=class_weights,
            settings=settings,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _input_only_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "relative_path": str(record["relative_path"]),
            "case_id": str(record["case_id"]),
            "subject_id": str(record["subject_id"]),
        }
        for record in records
    ]


def _load_spatial_batch(
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    load_targets: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    dataset = GLIClassifierPatchDataset(
        dataset_root, _input_only_records(records), load_targets=load_targets
    )
    samples = [dataset[index] for index in range(len(dataset))]
    image = torch.stack([sample["image"] for sample in samples])  # type: ignore[list-item]
    mask = torch.stack([sample["total_mask"][0] for sample in samples])  # type: ignore[index]
    if not load_targets:
        if any("target" in sample for sample in samples):
            raise AssertionError("unlabeled classifier batch unexpectedly contains targets")
        return image, mask, None
    target = torch.stack([sample["target"] for sample in samples]) - 1  # type: ignore[list-item,operator]
    return image, mask, target


def _cyclic_epoch_records(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("cannot sample an empty record pool")
    count = min(int(count), len(records))
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(records), generator=generator).tolist()
    offset = (int(epoch) * count) % len(records)
    indices = [order[(offset + index) % len(order)] for index in range(count)]
    return [dict(records[index]) for index in indices]


def _select_balanced_pseudo_voxels(
    confidence: torch.Tensor,
    pseudo_target: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    max_per_class: int,
) -> torch.Tensor:
    selected = mask.to(torch.bool) & (confidence >= float(threshold))
    balanced = torch.zeros_like(selected)
    flat_confidence = confidence.flatten()
    flat_target = pseudo_target.flatten()
    flat_selected = selected.flatten()
    flat_balanced = balanced.flatten()
    for class_index in range(4):
        indices = torch.nonzero(
            flat_selected & (flat_target == class_index), as_tuple=False
        ).flatten()
        if indices.numel() > int(max_per_class):
            top = torch.topk(flat_confidence[indices], int(max_per_class)).indices
            indices = indices[top]
        flat_balanced[indices] = True
    return balanced


def _pseudo_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    max_per_class: int,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_probabilities = teacher_logits.softmax(dim=1)
    confidence, pseudo_target = teacher_probabilities.max(dim=1)
    selected = _select_balanced_pseudo_voxels(
        confidence,
        pseudo_target,
        mask,
        threshold=threshold,
        max_per_class=max_per_class,
    )
    selected_count = int(selected.sum())
    lesion_count = int(mask.sum())
    if selected_count == 0:
        return student_logits.sum() * 0.0, {
            "selected_voxels": 0.0,
            "selected_fraction": 0.0,
            "mean_confidence": 0.0,
        }
    ce_map = F.cross_entropy(
        student_logits,
        pseudo_target,
        weight=class_weights.to(student_logits.device),
        reduction="none",
    )
    pseudo_ce = (ce_map * selected).sum() / selected.sum()
    student_probabilities = student_logits.softmax(dim=1)
    consistency_map = (student_probabilities - teacher_probabilities).square().mean(dim=1)
    probability_consistency = (consistency_map * selected).sum() / selected.sum()
    loss = 0.8 * pseudo_ce + 0.2 * probability_consistency
    return loss, {
        "selected_voxels": float(selected_count),
        "selected_fraction": float(selected_count / max(1, lesion_count)),
        "mean_confidence": float(confidence[selected].mean().detach().cpu()),
    }


@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters(), strict=True
    ):
        teacher_parameter.mul_(float(decay)).add_(student_parameter, alpha=1.0 - float(decay))
    for teacher_buffer, student_buffer in zip(
        teacher.buffers(), student.buffers(), strict=True
    ):
        teacher_buffer.copy_(student_buffer)


def _train_mean_teacher_epoch(
    *,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset_root: str | Path,
    labeled_records: Sequence[Mapping[str, Any]],
    unlabeled_records: Sequence[Mapping[str, Any]],
    epoch: int,
    seed: int,
    training_config: Mapping[str, Any],
    class_weights: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    generator = torch.Generator().manual_seed(seed + 10_000_019 * epoch)
    batch_size = int(training_config.get("batch_size", 1))
    unlabeled_batch_size = int(training_config.get("unlabeled_batch_size", batch_size))
    labeled_order = torch.randperm(len(labeled_records), generator=generator).tolist()
    unlabeled_epoch_records = _cyclic_epoch_records(
        unlabeled_records,
        count=int(training_config.get("unlabeled_patches_per_epoch", len(labeled_records))),
        seed=seed + 91_003,
        epoch=epoch,
    )
    steps = max(
        math.ceil(len(labeled_order) / batch_size),
        math.ceil(len(unlabeled_epoch_records) / unlabeled_batch_size),
    )
    settings = _loss_settings(training_config, device)
    augmentation = training_config.get("augmentation", {})
    semi = training_config.get("semi_supervised", {})
    ramp_epochs = max(1, int(semi.get("ramp_epochs", 5)))
    unsupervised_weight = float(semi.get("weight", 1.0)) * min(1.0, (epoch + 1) / ramp_epochs)
    threshold = float(semi.get("confidence_threshold", 0.9))
    max_per_class = int(semi.get("max_pseudo_voxels_per_class", 8192))
    ema_decay = float(semi.get("ema_decay", 0.99))
    supervised_losses: list[float] = []
    unsupervised_losses: list[float] = []
    selected_fractions: list[float] = []
    mean_confidences: list[float] = []

    for step in range(steps):
        labeled_indices = [
            labeled_order[(step * batch_size + offset) % len(labeled_order)]
            for offset in range(batch_size)
        ]
        unlabeled_indices = [
            (step * unlabeled_batch_size + offset) % len(unlabeled_epoch_records)
            for offset in range(unlabeled_batch_size)
        ]
        labeled_batch = [labeled_records[index] for index in labeled_indices]
        unlabeled_batch = [unlabeled_epoch_records[index] for index in unlabeled_indices]

        image, mask, target = _load_spatial_batch(
            dataset_root, labeled_batch, load_targets=True
        )
        if target is None:
            raise AssertionError("labeled mean-teacher batch has no target")
        image, mask, target = _augment_spatial_batch(
            image, mask, target, generator=generator
        )
        image = _augment_intensity(
            image,
            generator=generator,
            scale_range=float(augmentation.get("scale_range", 0.1)),
            shift_range=float(augmentation.get("shift_range", 0.1)),
            noise_std=float(augmentation.get("noise_std", 0.03)),
        )
        image = image.to(device)
        mask = mask.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        supervised_logits = student(
            torch.cat((image, mask[:, None].to(image.dtype)), dim=1)
        )
        supervised_loss, _ = _masked_supervised_loss(
            supervised_logits,
            target,
            mask,
            class_weights=class_weights,
            settings=settings,
        )
        supervised_loss.backward()

        unlabeled_image, unlabeled_mask, unlabeled_target = _load_spatial_batch(
            dataset_root, unlabeled_batch, load_targets=False
        )
        if unlabeled_target is not None:
            raise AssertionError("unlabeled mean-teacher batch leaked a target")
        weak_image, unlabeled_mask, _ = _augment_spatial_batch(
            unlabeled_image, unlabeled_mask, None, generator=generator
        )
        strong_image = _augment_intensity(
            weak_image,
            generator=generator,
            scale_range=float(augmentation.get("strong_scale_range", 0.2)),
            shift_range=float(augmentation.get("strong_shift_range", 0.15)),
            noise_std=float(augmentation.get("strong_noise_std", 0.06)),
        )
        weak_inputs = torch.cat(
            (weak_image, unlabeled_mask[:, None].to(weak_image.dtype)), dim=1
        ).to(device)
        strong_inputs = torch.cat(
            (strong_image, unlabeled_mask[:, None].to(strong_image.dtype)), dim=1
        ).to(device)
        unlabeled_mask = unlabeled_mask.to(device)
        with torch.no_grad():
            teacher_logits = teacher(weak_inputs)
        student_logits = student(strong_inputs)
        unsupervised_loss, pseudo_stats = _pseudo_consistency_loss(
            student_logits,
            teacher_logits,
            unlabeled_mask,
            threshold=threshold,
            max_per_class=max_per_class,
            class_weights=class_weights,
        )
        (unsupervised_weight * unsupervised_loss).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()
        _ema_update(teacher, student, ema_decay)

        supervised_losses.append(float(supervised_loss.detach().cpu()))
        unsupervised_losses.append(float(unsupervised_loss.detach().cpu()))
        selected_fractions.append(pseudo_stats["selected_fraction"])
        mean_confidences.append(pseudo_stats["mean_confidence"])

    return {
        "train_loss": float(np.mean(supervised_losses))
        + unsupervised_weight * float(np.mean(unsupervised_losses)),
        "supervised_loss": float(np.mean(supervised_losses)),
        "unsupervised_loss": float(np.mean(unsupervised_losses)),
        "unsupervised_weight": unsupervised_weight,
        "pseudo_selected_fraction": float(np.mean(selected_fractions)),
        "pseudo_mean_confidence": float(np.mean(mean_confidences)),
        "unlabeled_patches": float(len(unlabeled_epoch_records)),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    *,
    kind: str,
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    inference_voxels: int,
    spatial_batch_size: int,
    bootstrap_samples: int,
    seed: int,
    feature_cache: FeatureRowCache | None = None,
) -> dict[str, Any]:
    model.eval()
    accumulator = PatientMetricAccumulator()
    dataset = GLIClassifierPatchDataset(dataset_root, _safe_records(records), load_targets=True)
    started = time.monotonic()
    if kind in SPATIAL_MODEL_KINDS:
        spatial_batch_size = max(1, int(spatial_batch_size))
        for start in range(0, len(dataset), spatial_batch_size):
            samples = [
                dataset[index]
                for index in range(start, min(start + spatial_batch_size, len(dataset)))
            ]
            images = torch.stack([sample["image"] for sample in samples])  # type: ignore[list-item]
            masks = torch.stack([sample["total_mask"] for sample in samples])  # type: ignore[list-item]
            inputs = _build_spatial_inputs(
                images, masks[:, 0], kind=kind
            ).to(device)
            batch_logits = model(inputs).cpu()
            for sample, logits in zip(samples, batch_logits, strict=True):
                mask = sample["total_mask"]  # type: ignore[assignment]
                target = sample["target"]  # type: ignore[assignment]
                inside_indices = logits[:, mask[0]].argmax(dim=0)
                prediction = reconstruct_prediction(inside_indices, mask)
                accumulator.update(
                    subject_id=str(sample["subject_id"]),
                    prediction=prediction,
                    target=target,
                    total_mask=mask,
                )
    else:
        for sample in dataset:
            mask = sample["total_mask"]  # type: ignore[assignment]
            target = sample["target"]  # type: ignore[assignment]
            rows, _ = _feature_rows_and_target(
                sample,
                feature_kind=kind,
                cache=feature_cache,
            )
            predictions: list[torch.Tensor] = []
            for start in range(0, rows.shape[0], inference_voxels):
                logits = model(rows[start : start + inference_voxels].to(device))
                predictions.append(logits.argmax(dim=1).cpu())
            inside_indices = torch.cat(predictions, dim=0)
            prediction = reconstruct_prediction(inside_indices, mask)
            accumulator.update(
                subject_id=str(sample["subject_id"]),
                prediction=prediction,
                target=target,
                total_mask=mask,
            )
    metrics = accumulator.compute(bootstrap_samples=bootstrap_samples, seed=seed)
    metrics["elapsed_seconds"] = time.monotonic() - started
    metrics["model_kind"] = kind
    return metrics


def _run_training_impl(
    config: dict[str, Any],
    *,
    resume: bool,
    git_provenance: Mapping[str, str],
    wandb_run: Any,
) -> dict[str, Any]:
    data_config = config["data"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    seed = int(training_config["seed"])
    seed_everything(seed)
    device = resolve_device(training_config.get("device", "auto"))
    dataset_root = Path(data_config["dataset_root"])
    split_file = Path(data_config["split_file"])
    subset_path = Path(data_config["labeled_subset"])
    train_records, subset_payload = load_labeled_subset(
        subset_path, dataset_root, split_file
    )
    val_records, _ = load_manifest_records(
        dataset_root, split_file, split="val"
    )
    kind = str(config["model"]["kind"]).lower()
    model = _model_from_config(config).to(device)
    initial_checkpoint_value = training_config.get("initial_checkpoint")
    initial_checkpoint = (
        Path(str(initial_checkpoint_value)) if initial_checkpoint_value else None
    )
    initial_checkpoint_sha256: str | None = None
    if initial_checkpoint is not None:
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(initial_checkpoint)
        if not resume:
            if kind != "geometry_unet3d":
                raise ValueError(
                    "supervised initial_checkpoint is only supported for geometry_unet3d"
                )
            initial_checkpoint_sha256 = _load_geometry_warmstart(
                model,
                initial_checkpoint,
                subset_path=subset_path,
                device=device,
            )
        else:
            initial_checkpoint_sha256 = sha256_file(initial_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(training_config.get("lr_patience", 3)),
        min_lr=float(training_config.get("min_learning_rate", 1e-6)),
    )
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"
    start_epoch, best_metric, patience_metric, bad_epochs = 0, -math.inf, -math.inf, 0
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {latest_path}")
        start_epoch, best_metric, patience_metric, bad_epochs = _restore_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            subset_path=subset_path,
            device=device,
        )
        latest_payload = torch.load(latest_path, map_location="cpu")
        if latest_payload.get("initial_checkpoint_sha256") != initial_checkpoint_sha256:
            raise ValueError("supervised initial checkpoint changed during resume")

    class_weights = class_weights_from_subset(subset_payload)
    train_feature_cache: FeatureRowCache | None = (
        {} if kind in FEATURE_CHANNELS else None
    )
    val_feature_cache: FeatureRowCache | None = (
        {} if kind in FEATURE_CHANNELS else None
    )
    metadata = {
        "experiment_id": config["experiment_id"],
        "config_sha256": config["_config_sha256"],
        "subset_sha256": sha256_file(subset_path),
        "model_kind": kind,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "train_patches": len(train_records),
        "val_patches": len(val_records),
        "val_subjects": len({record["subject_id"] for record in val_records}),
        "feature_cache": "masked_rows_memory" if kind in FEATURE_CHANNELS else "none",
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "wandb_run_id": str(wandb_run.id),
        "wandb_url": getattr(wandb_run, "url", None),
        **git_provenance,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    patience = int(training_config.get("early_stopping_patience", 8))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.005))
    max_epochs = int(training_config["epochs"])
    for epoch in range(start_epoch, max_epochs):
        if kind in FEATURE_CHANNELS:
            train_loss = _train_mlp_epoch(
                model=model,
                optimizer=optimizer,
                dataset_root=dataset_root,
                records=train_records,
                kind=kind,
                epoch=epoch,
                seed=seed,
                patches_per_class=int(training_config.get("patches_per_class", 2)),
                voxels_per_class=int(training_config.get("voxels_per_class", 2048)),
                device=device,
                feature_cache=train_feature_cache,
            )
        elif kind in SPATIAL_MODEL_KINDS:
            train_loss = _train_cnn_epoch(
                model=model,
                optimizer=optimizer,
                dataset_root=dataset_root,
                records=train_records,
                epoch=epoch,
                seed=seed,
                batch_size=int(training_config.get("batch_size", 4)),
                class_weights=class_weights,
                device=device,
                training_config=training_config,
                kind=kind,
            )
        else:
            raise ValueError(f"unsupported training kind: {kind}")

        metrics = evaluate_model(
            model,
            kind=kind,
            dataset_root=dataset_root,
            records=val_records,
            device=device,
            inference_voxels=int(evaluation_config.get("inference_voxels", 65536)),
            spatial_batch_size=int(evaluation_config.get("batch_size", 1)),
            bootstrap_samples=0,
            seed=seed,
            feature_cache=val_feature_cache,
        )
        focus_miou = float(metrics["focus_miou"])
        scheduler.step(focus_miou)
        (
            best_metric,
            patience_metric,
            bad_epochs,
            is_absolute_best,
            is_significant_improvement,
        ) = update_selection_state(
            focus_miou,
            best_metric=best_metric,
            patience_metric=patience_metric,
            bad_epochs=bad_epochs,
            min_delta=min_delta,
        )
        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            subset_path=subset_path,
            epoch=epoch,
            best_metric=best_metric,
            patience_metric=patience_metric,
            bad_epochs=bad_epochs,
        )
        checkpoint["initial_checkpoint_sha256"] = initial_checkpoint_sha256
        _atomic_torch_save(checkpoint, latest_path)
        if is_absolute_best:
            _atomic_torch_save(checkpoint, best_path)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_focus_miou": best_metric,
            "early_stopping_reference": patience_metric,
            "bad_epochs": bad_epochs,
            "is_absolute_best": is_absolute_best,
            "is_significant_improvement": is_significant_improvement,
            "val": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        wandb_run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "val/focus_miou": focus_miou,
                "val/macro_iou": float(metrics["macro_iou"]),
                "val/macro_dice": float(metrics["macro_dice"]),
                "val/balanced_accuracy": float(metrics["balanced_accuracy"]),
                "val/mask_inside_error_rate": float(metrics["mask_inside_error_rate"]),
                **{
                    f"val/{label.lower()}_iou": float(metrics["classes"][label]["iou"])
                    for label in ("NETC", "SNFH", "ET", "RC")
                },
                "selection/is_absolute_best": int(is_absolute_best),
                "selection/bad_epochs": bad_epochs,
            },
            step=epoch,
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if bad_epochs >= patience:
            break

    if not best_path.is_file():
        raise RuntimeError("training did not produce a best checkpoint")
    _restore_checkpoint(
        best_path,
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        subset_path=subset_path,
        device=device,
    )
    final_metrics = evaluate_model(
        model,
        kind=kind,
        dataset_root=dataset_root,
        records=val_records,
        device=device,
        inference_voxels=int(evaluation_config.get("inference_voxels", 65536)),
        spatial_batch_size=int(evaluation_config.get("batch_size", 1)),
        bootstrap_samples=int(evaluation_config.get("bootstrap_samples", 1000)),
        seed=seed,
        feature_cache=val_feature_cache,
    )
    final_metrics["checkpoint_sha256"] = sha256_file(best_path)
    final_metrics["config_sha256"] = config["_config_sha256"]
    final_metrics["subset_sha256"] = sha256_file(subset_path)
    final_metrics["initial_checkpoint_sha256"] = initial_checkpoint_sha256
    (output_dir / "best_val_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    wandb_run.summary["best/focus_miou"] = float(final_metrics["focus_miou"])
    wandb_run.summary["best/et_iou"] = float(final_metrics["classes"]["ET"]["iou"])
    wandb_run.summary["best/rc_iou"] = float(final_metrics["classes"]["RC"]["iou"])
    wandb_run.summary["best/checkpoint_sha256"] = final_metrics["checkpoint_sha256"]
    wandb_run.summary["gate/all_passed"] = bool(all(final_metrics["gate"].values()))
    return {**metadata, "best_val": final_metrics, "best_checkpoint": str(best_path)}


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        sorted(str(record["relative_path"]) for record in records),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_initial_model(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    config: Mapping[str, Any],
    subset_path: Path,
    device: torch.device,
) -> str:
    payload = torch.load(checkpoint_path, map_location=device)
    if payload.get("experiment_id") != config["experiment_id"]:
        raise ValueError("initial checkpoint experiment ID mismatch")
    if payload.get("model_kind") != config["model"]["kind"]:
        raise ValueError("initial checkpoint model kind mismatch")
    if payload.get("subset_sha256") != sha256_file(subset_path):
        raise ValueError("initial checkpoint labeled subset mismatch")
    model.load_state_dict(payload["model"])
    return sha256_file(checkpoint_path)


def _run_mean_teacher_training_impl(
    config: dict[str, Any],
    *,
    resume: bool,
    git_provenance: Mapping[str, str],
    wandb_run: Any,
) -> dict[str, Any]:
    data_config = config["data"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    if str(config["model"]["kind"]).lower() != "unet3d":
        raise ValueError("mean-teacher training currently requires model.kind=unet3d")
    seed = int(training_config["seed"])
    seed_everything(seed)
    device = resolve_device(training_config.get("device", "auto"))
    dataset_root = Path(data_config["dataset_root"])
    split_file = Path(data_config["split_file"])
    subset_path = Path(data_config["labeled_subset"])
    labeled_records, subset_payload = load_labeled_subset(
        subset_path, dataset_root, split_file
    )
    train_records, _ = load_manifest_records(dataset_root, split_file, split="train")
    labeled_paths = {str(record["relative_path"]) for record in labeled_records}
    unlabeled_records = [
        record
        for record in _input_only_records(train_records)
        if str(record["relative_path"]) not in labeled_paths
    ]
    if len(unlabeled_records) != len(train_records) - len(labeled_paths):
        raise RuntimeError("unexpected labeled/unlabeled train pool overlap")
    if not unlabeled_records:
        raise RuntimeError("mean-teacher training has no remaining unlabeled train patches")
    val_records, _ = load_manifest_records(dataset_root, split_file, split="val")

    student = _model_from_config(config).to(device)
    initial_checkpoint = Path(str(training_config["initial_checkpoint"]))
    teacher = deepcopy(student).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(training_config.get("lr_patience", 3)),
        min_lr=float(training_config.get("min_learning_rate", 1e-6)),
    )
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"
    start_epoch, best_metric, patience_metric, bad_epochs = 0, -math.inf, -math.inf, 0
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {latest_path}")
        start_epoch, best_metric, patience_metric, bad_epochs = _restore_checkpoint(
            latest_path,
            model=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            subset_path=subset_path,
            device=device,
        )
        resume_payload = torch.load(latest_path, map_location=device)
        if "student_model" not in resume_payload:
            raise ValueError("mean-teacher resume checkpoint has no student_model")
        student.load_state_dict(resume_payload["student_model"])
        initial_checkpoint_sha256 = str(resume_payload["initial_checkpoint_sha256"])
        if initial_checkpoint_sha256 != sha256_file(initial_checkpoint):
            raise ValueError("mean-teacher initial checkpoint changed during resume")
    else:
        initial_checkpoint_sha256 = _load_initial_model(
            student,
            initial_checkpoint,
            config=config,
            subset_path=subset_path,
            device=device,
        )
        teacher.load_state_dict(student.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    class_weights = class_weights_from_subset(subset_payload)
    unlabeled_pool_sha256 = _records_sha256(unlabeled_records)
    metadata = {
        "experiment_id": config["experiment_id"],
        "config_sha256": config["_config_sha256"],
        "subset_sha256": sha256_file(subset_path),
        "model_kind": "unet3d",
        "training_mode": "mean_teacher",
        "parameter_count": count_parameters(student),
        "device": str(device),
        "train_patches": len(labeled_records),
        "unlabeled_train_patches": len(unlabeled_records),
        "unlabeled_pool_sha256": unlabeled_pool_sha256,
        "val_patches": len(val_records),
        "val_subjects": len({record["subject_id"] for record in val_records}),
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "wandb_run_id": str(wandb_run.id),
        "wandb_url": getattr(wandb_run, "url", None),
        **git_provenance,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    patience = int(training_config.get("early_stopping_patience", 8))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.005))
    max_epochs = int(training_config["epochs"])
    for epoch in range(start_epoch, max_epochs):
        train_metrics = _train_mean_teacher_epoch(
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            dataset_root=dataset_root,
            labeled_records=labeled_records,
            unlabeled_records=unlabeled_records,
            epoch=epoch,
            seed=seed,
            training_config=training_config,
            class_weights=class_weights,
            device=device,
        )
        metrics = evaluate_model(
            teacher,
            kind="unet3d",
            dataset_root=dataset_root,
            records=val_records,
            device=device,
            inference_voxels=int(evaluation_config.get("inference_voxels", 65536)),
            spatial_batch_size=int(evaluation_config.get("batch_size", 1)),
            bootstrap_samples=0,
            seed=seed,
        )
        focus_miou = float(metrics["focus_miou"])
        scheduler.step(focus_miou)
        (
            best_metric,
            patience_metric,
            bad_epochs,
            is_absolute_best,
            is_significant_improvement,
        ) = update_selection_state(
            focus_miou,
            best_metric=best_metric,
            patience_metric=patience_metric,
            bad_epochs=bad_epochs,
            min_delta=min_delta,
        )
        checkpoint = _checkpoint_payload(
            model=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            subset_path=subset_path,
            epoch=epoch,
            best_metric=best_metric,
            patience_metric=patience_metric,
            bad_epochs=bad_epochs,
        )
        checkpoint["student_model"] = student.state_dict()
        checkpoint["initial_checkpoint_sha256"] = initial_checkpoint_sha256
        checkpoint["unlabeled_pool_sha256"] = unlabeled_pool_sha256
        _atomic_torch_save(checkpoint, latest_path)
        if is_absolute_best:
            _atomic_torch_save(checkpoint, best_path)
        record = {
            "epoch": epoch,
            **train_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_focus_miou": best_metric,
            "early_stopping_reference": patience_metric,
            "bad_epochs": bad_epochs,
            "is_absolute_best": is_absolute_best,
            "is_significant_improvement": is_significant_improvement,
            "val": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        wandb_run.log(
            {
                "epoch": epoch,
                "train/loss": train_metrics["train_loss"],
                "train/supervised_loss": train_metrics["supervised_loss"],
                "train/unsupervised_loss": train_metrics["unsupervised_loss"],
                "train/unsupervised_weight": train_metrics["unsupervised_weight"],
                "train/pseudo_selected_fraction": train_metrics["pseudo_selected_fraction"],
                "train/pseudo_mean_confidence": train_metrics["pseudo_mean_confidence"],
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "val/focus_miou": focus_miou,
                "val/macro_iou": float(metrics["macro_iou"]),
                "val/macro_dice": float(metrics["macro_dice"]),
                **{
                    f"val/{label.lower()}_iou": float(metrics["classes"][label]["iou"])
                    for label in ("NETC", "SNFH", "ET", "RC")
                },
                "selection/is_absolute_best": int(is_absolute_best),
                "selection/bad_epochs": bad_epochs,
            },
            step=epoch,
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if bad_epochs >= patience:
            break

    if not best_path.is_file():
        raise RuntimeError("mean-teacher training did not produce a best checkpoint")
    _restore_checkpoint(
        best_path,
        model=teacher,
        optimizer=None,
        scheduler=None,
        config=config,
        subset_path=subset_path,
        device=device,
    )
    final_metrics = evaluate_model(
        teacher,
        kind="unet3d",
        dataset_root=dataset_root,
        records=val_records,
        device=device,
        inference_voxels=int(evaluation_config.get("inference_voxels", 65536)),
        spatial_batch_size=int(evaluation_config.get("batch_size", 1)),
        bootstrap_samples=int(evaluation_config.get("bootstrap_samples", 1000)),
        seed=seed,
    )
    final_metrics["checkpoint_sha256"] = sha256_file(best_path)
    final_metrics["config_sha256"] = config["_config_sha256"]
    final_metrics["subset_sha256"] = sha256_file(subset_path)
    final_metrics["initial_checkpoint_sha256"] = initial_checkpoint_sha256
    final_metrics["unlabeled_pool_sha256"] = unlabeled_pool_sha256
    (output_dir / "best_val_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    wandb_run.summary["best/focus_miou"] = float(final_metrics["focus_miou"])
    wandb_run.summary["best/et_iou"] = float(final_metrics["classes"]["ET"]["iou"])
    wandb_run.summary["best/rc_iou"] = float(final_metrics["classes"]["RC"]["iou"])
    wandb_run.summary["best/checkpoint_sha256"] = final_metrics["checkpoint_sha256"]
    wandb_run.summary["gate/all_passed"] = bool(all(final_metrics["gate"].values()))
    return {**metadata, "best_val": final_metrics, "best_checkpoint": str(best_path)}


def run_training(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    git_provenance = clean_git_provenance(config)
    config["_git_head"] = git_provenance["git_head"]
    config["_git_branch"] = git_provenance["git_branch"]
    with classifier_wandb_run(config, resume=resume) as wandb_run:
        if str(config["training"].get("mode", "supervised")) == "mean_teacher":
            return _run_mean_teacher_training_impl(
                config,
                resume=resume,
                git_provenance=git_provenance,
                wandb_run=wandb_run,
            )
        return _run_training_impl(
            config,
            resume=resume,
            git_provenance=git_provenance,
            wandb_run=wandb_run,
        )


def _lesion_center_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    shape_dhw: Sequence[int] = (8, 16, 16),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_3d = mask[0].to(torch.bool)
    indices = torch.nonzero(mask_3d, as_tuple=False)
    if not indices.numel():
        raise ValueError("cannot crop an empty lesion mask")
    center = indices.to(torch.float32).mean(dim=0).round().to(torch.int64)
    slices: list[slice] = []
    for axis, requested in enumerate(shape_dhw):
        size = int(mask_3d.shape[axis])
        requested = min(int(requested), size)
        start = min(max(int(center[axis]) - requested // 2, 0), size - requested)
        slices.append(slice(start, start + requested))
    spatial = tuple(slices)
    cropped_image = image[(slice(None), *spatial)]
    cropped_mask = mask[(slice(None), *spatial)]
    cropped_target = target[spatial]
    if not torch.any(cropped_mask):
        raise AssertionError("lesion-centered crop unexpectedly lost the lesion")
    return cropped_image, cropped_mask, cropped_target


def run_cpu_preflight(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a zero-optimizer-step real-data forward/backward contract check."""

    config = load_config(config_path)
    data_config = config["data"]
    training_config = config["training"]
    seed = int(training_config["seed"])
    seed_everything(seed)
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    device = torch.device("cpu")
    dataset_root = Path(data_config["dataset_root"])
    split_file = Path(data_config["split_file"])
    subset_path = Path(data_config["labeled_subset"])
    records, subset_payload = load_labeled_subset(subset_path, dataset_root, split_file)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for record in records:
        stratum = (int(record["anchor_label"]), str(record["sample_role"]))
        if stratum not in seen:
            selected.append(dict(record))
            seen.add(stratum)
        if len(seen) == 8:
            break
    if len(selected) != 8:
        raise RuntimeError(f"preflight could not cover eight strata: {sorted(seen)}")

    kind = str(config["model"]["kind"]).lower()
    model = _model_from_config(config).to(device)
    preflight_initial_sha256: str | None = None
    if training_config.get("initial_checkpoint"):
        if kind != "geometry_unet3d":
            raise ValueError("preflight initial checkpoint requires geometry_unet3d")
        preflight_initial_sha256 = _load_geometry_warmstart(
            model,
            Path(str(training_config["initial_checkpoint"])),
            subset_path=subset_path,
            device=device,
        )
    model.train()
    started = time.monotonic()
    if kind in FEATURE_CHANNELS:
        generator = torch.Generator().manual_seed(seed)
        features, target = build_balanced_voxel_batch(
            dataset_root,
            selected,
            feature_kind=kind,
            voxels_per_class=64,
            generator=generator,
        )
        logits = model(features)
        loss = F.cross_entropy(logits, target)
        output_shape = list(logits.shape)
        input_shape = list(features.shape)
    elif kind in SPATIAL_MODEL_KINDS:
        dataset = GLIClassifierPatchDataset(
            dataset_root, _safe_records([selected[0]]), load_targets=True
        )
        sample = dataset[0]
        image, mask, target_original = _lesion_center_crop(
            sample["image"], sample["total_mask"], sample["target"]  # type: ignore[arg-type]
        )
        inputs = _build_spatial_inputs(
            image[None], mask[0][None], kind=kind
        )
        target = target_original[None] - 1
        logits = model(inputs)
        mask_3d = mask[0][None]
        loss, _ = _masked_supervised_loss(
            logits,
            target,
            mask_3d,
            class_weights=class_weights_from_subset(subset_payload),
            settings=_loss_settings(training_config, device),
        )
        output_shape = list(logits.shape)
        input_shape = list(inputs.shape)
    else:
        raise ValueError(f"unsupported preflight kind: {kind}")
    semi_preflight: dict[str, Any] = {}
    if str(training_config.get("mode", "supervised")) == "mean_teacher":
        train_pool, _ = load_manifest_records(dataset_root, split_file, split="train")
        labeled_paths = {str(record["relative_path"]) for record in records}
        unlabeled_pool = [
            record
            for record in _input_only_records(train_pool)
            if str(record["relative_path"]) not in labeled_paths
        ]
        unlabeled_dataset = GLIClassifierPatchDataset(
            dataset_root, [unlabeled_pool[0]], load_targets=False
        )
        unlabeled_sample = unlabeled_dataset[0]
        if "target" in unlabeled_sample:
            raise AssertionError("mean-teacher preflight detected target leakage")
        dummy_target = torch.zeros_like(
            unlabeled_sample["total_mask"][0], dtype=torch.int64  # type: ignore[index]
        )
        unlabeled_image, unlabeled_mask, _ = _lesion_center_crop(
            unlabeled_sample["image"],  # type: ignore[arg-type]
            unlabeled_sample["total_mask"],  # type: ignore[arg-type]
            dummy_target,
        )
        unlabeled_inputs = torch.cat(
            (unlabeled_image, unlabeled_mask.to(unlabeled_image.dtype)), dim=0
        )[None]
        teacher = deepcopy(model).eval()
        with torch.no_grad():
            teacher_logits = teacher(unlabeled_inputs)
        student_logits = model(unlabeled_inputs)
        consistency_loss, pseudo_stats = _pseudo_consistency_loss(
            student_logits,
            teacher_logits,
            unlabeled_mask[0][None],
            threshold=0.0,
            max_per_class=64,
            class_weights=class_weights_from_subset(subset_payload),
        )
        loss = loss + 0.1 * consistency_loss
        semi_preflight = {
            "unlabeled_pool_count": len(unlabeled_pool),
            "unlabeled_pool_sha256": _records_sha256(unlabeled_pool),
            "unlabeled_target_present": False,
            "pseudo_selected_voxels": int(pseudo_stats["selected_voxels"]),
        }
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite preflight loss: {loss}")
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
        raise RuntimeError(f"invalid preflight gradient norm: {gradient_norm}")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "model_kind": kind,
        "device": "cpu",
        "optimizer_steps": 0,
        "parameter_count": count_parameters(model),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "covered_strata": [f"{label}:{role}" for label, role in sorted(seen)],
        "record_count": len(selected),
        "subset_count": int(subset_payload["actual_count"]),
        "subset_sha256": sha256_file(subset_path),
        "config_sha256": config["_config_sha256"],
        "initial_checkpoint_sha256": preflight_initial_sha256,
        "elapsed_seconds": time.monotonic() - started,
        **semi_preflight,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def run_gpu_preflight(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Exercise the configured full-p64 training graph without an optimizer step."""

    config = load_config(config_path)
    data_config = config["data"]
    training_config = config["training"]
    seed = int(training_config["seed"])
    seed_everything(seed)
    device = resolve_device(training_config.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError("GPU preflight requires a CUDA device")
    dataset_root = Path(data_config["dataset_root"])
    split_file = Path(data_config["split_file"])
    subset_path = Path(data_config["labeled_subset"])
    labeled_records, subset_payload = load_labeled_subset(
        subset_path, dataset_root, split_file
    )
    kind = str(config["model"]["kind"]).lower()
    if kind not in SPATIAL_MODEL_KINDS:
        raise ValueError("GPU full-p64 preflight is only defined for spatial models")
    batch_size = int(training_config.get("batch_size", 1))
    model = _model_from_config(config).to(device).train()
    preflight_initial_sha256: str | None = None
    if training_config.get("initial_checkpoint"):
        if kind != "geometry_unet3d":
            raise ValueError("GPU preflight initial checkpoint requires geometry_unet3d")
        preflight_initial_sha256 = _load_geometry_warmstart(
            model,
            Path(str(training_config["initial_checkpoint"])),
            subset_path=subset_path,
            device=device,
        )
    class_weights = class_weights_from_subset(subset_payload)
    settings = _loss_settings(training_config, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    image, mask, target = _load_spatial_batch(
        dataset_root, labeled_records[:batch_size], load_targets=True
    )
    if target is None:
        raise AssertionError("GPU preflight labeled batch has no target")
    inputs = _build_spatial_inputs(image, mask, kind=kind).to(device)
    mask = mask.to(device)
    target = target.to(device)
    logits = model(inputs)
    supervised_loss, _ = _masked_supervised_loss(
        logits,
        target,
        mask,
        class_weights=class_weights,
        settings=settings,
    )
    loss = supervised_loss
    semi_result: dict[str, Any] = {}
    if str(training_config.get("mode", "supervised")) == "mean_teacher":
        train_pool, _ = load_manifest_records(dataset_root, split_file, split="train")
        labeled_paths = {str(record["relative_path"]) for record in labeled_records}
        unlabeled_pool = [
            record
            for record in _input_only_records(train_pool)
            if str(record["relative_path"]) not in labeled_paths
        ]
        unlabeled_batch_size = int(training_config.get("unlabeled_batch_size", 1))
        unlabeled_image, unlabeled_mask, unlabeled_target = _load_spatial_batch(
            dataset_root,
            unlabeled_pool[:unlabeled_batch_size],
            load_targets=False,
        )
        if unlabeled_target is not None:
            raise AssertionError("GPU preflight unlabeled batch leaked a target")
        unlabeled_inputs = torch.cat(
            (
                unlabeled_image,
                unlabeled_mask[:, None].to(unlabeled_image.dtype),
            ),
            dim=1,
        ).to(device)
        unlabeled_mask = unlabeled_mask.to(device)
        teacher = deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            teacher_logits = teacher(unlabeled_inputs)
        student_logits = model(unlabeled_inputs)
        consistency_loss, pseudo_stats = _pseudo_consistency_loss(
            student_logits,
            teacher_logits,
            unlabeled_mask,
            threshold=0.0,
            max_per_class=int(
                training_config.get("semi_supervised", {}).get(
                    "max_pseudo_voxels_per_class", 8192
                )
            ),
            class_weights=class_weights,
        )
        loss = loss + float(
            training_config.get("semi_supervised", {}).get("weight", 0.5)
        ) * consistency_loss
        semi_result = {
            "unlabeled_batch_size": unlabeled_batch_size,
            "unlabeled_target_present": False,
            "pseudo_selected_voxels": int(pseudo_stats["selected_voxels"]),
        }
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite GPU preflight loss: {loss}")
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
        raise RuntimeError(f"invalid GPU preflight gradient norm: {gradient_norm}")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "training_mode": str(training_config.get("mode", "supervised")),
        "model_kind": kind,
        "device": str(device),
        "optimizer_steps": 0,
        "parameter_count": count_parameters(model),
        "batch_size": batch_size,
        "input_shape": list(inputs.shape),
        "output_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "subset_sha256": sha256_file(subset_path),
        "config_sha256": config["_config_sha256"],
        "initial_checkpoint_sha256": preflight_initial_sha256,
        "elapsed_seconds": time.monotonic() - started,
        **semi_result,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def run_evaluation(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str,
    output_path: str | Path,
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    config = load_config(config_path)
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    training_config = config["training"]
    device = resolve_device(training_config.get("device", "auto"))
    dataset_root = Path(data_config["dataset_root"])
    split_file = Path(data_config["split_file"])
    subset_path = Path(data_config["labeled_subset"])
    checkpoint_path = Path(checkpoint_path)
    if split == "test":
        validate_test_gate(
            checkpoint_path,
            config_sha256=config["_config_sha256"],
            subset_path=subset_path,
        )
    records, _ = load_manifest_records(dataset_root, split_file, split=split)
    model = _model_from_config(config).to(device)
    _restore_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        subset_path=subset_path,
        device=device,
    )
    metrics = evaluate_model(
        model,
        kind=str(config["model"]["kind"]).lower(),
        dataset_root=dataset_root,
        records=records,
        device=device,
        inference_voxels=int(evaluation_config.get("inference_voxels", 65536)),
        spatial_batch_size=int(evaluation_config.get("batch_size", 1)),
        bootstrap_samples=int(evaluation_config.get("bootstrap_samples", 1000)),
        seed=int(training_config["seed"]),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
