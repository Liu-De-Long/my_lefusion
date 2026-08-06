"""Training and evaluation engine for the p64 weak-supervision classifier."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
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
    bad_epochs: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "model_kind": config["model"]["kind"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "bad_epochs": int(bad_epochs),
        "config_sha256": config["_config_sha256"],
        "subset_sha256": sha256_file(subset_path),
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
) -> tuple[int, float, int]:
    payload = torch.load(path, map_location=device)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported classifier checkpoint schema")
    if payload.get("experiment_id") != config["experiment_id"]:
        raise ValueError("checkpoint experiment ID mismatch")
    if payload.get("model_kind") != config["model"]["kind"]:
        raise ValueError("checkpoint model kind mismatch")
    if payload.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("checkpoint config hash mismatch")
    if payload.get("subset_sha256") != sha256_file(subset_path):
        raise ValueError("checkpoint subset hash mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return int(payload["epoch"]) + 1, float(payload["best_metric"]), int(payload["bad_epochs"])


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


def build_balanced_voxel_batch(
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    feature_kind: str,
    voxels_per_class: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = GLIClassifierPatchDataset(dataset_root, _safe_records(records), load_targets=True)
    buckets: dict[int, list[torch.Tensor]] = {index: [] for index in range(4)}
    for sample in dataset:
        features = build_feature_volume(sample["image"], sample["total_mask"], feature_kind)  # type: ignore[arg-type]
        rows = masked_feature_rows(features, sample["total_mask"])  # type: ignore[arg-type]
        target = sample["target"][sample["total_mask"][0]] - 1  # type: ignore[index,operator]
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


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
    return 1.0 - ((numerator[valid] + 1e-6) / (denominator[valid] + 1e-6)).mean()


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
) -> float:
    model.train()
    generator = torch.Generator().manual_seed(seed + 10_000_019 * epoch)
    order = torch.randperm(len(records), generator=generator).tolist()
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch_records = [_safe_records([records[index]])[0] for index in order[start : start + batch_size]]
        dataset = GLIClassifierPatchDataset(dataset_root, batch_records, load_targets=True)
        samples = [dataset[index] for index in range(len(dataset))]
        image = torch.stack([sample["image"] for sample in samples]).to(device)  # type: ignore[list-item]
        mask = torch.stack([sample["total_mask"][0] for sample in samples]).to(device)  # type: ignore[index]
        target = torch.stack([sample["target"] for sample in samples]).to(device) - 1  # type: ignore[list-item,operator]
        inputs = torch.cat((image, mask[:, None].to(image.dtype)), dim=1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        safe_target = target.clamp(0, 3)
        loss_map = F.cross_entropy(
            logits,
            safe_target,
            weight=class_weights.to(device),
            reduction="none",
        )
        cross_entropy = (loss_map * mask).sum() / mask.sum().clamp_min(1)
        dice = _soft_dice_loss(logits, target, mask)
        loss = 0.7 * cross_entropy + 0.3 * dice
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    *,
    kind: str,
    dataset_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    inference_voxels: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    accumulator = PatientMetricAccumulator()
    dataset = GLIClassifierPatchDataset(dataset_root, _safe_records(records), load_targets=True)
    started = time.monotonic()
    for sample in dataset:
        image = sample["image"]  # type: ignore[assignment]
        mask = sample["total_mask"]  # type: ignore[assignment]
        target = sample["target"]  # type: ignore[assignment]
        if kind == "c0":
            inputs = torch.cat((image, mask.to(image.dtype)), dim=0)[None].to(device)
            logits = model(inputs)[0]
            inside_indices = logits[:, mask[0].to(device)].argmax(dim=0).cpu()
        else:
            features = build_feature_volume(image, mask, kind)
            rows = masked_feature_rows(features, mask)
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


def run_training(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
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
    start_epoch, best_metric, bad_epochs = 0, -math.inf, 0
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {latest_path}")
        start_epoch, best_metric, bad_epochs = _restore_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            subset_path=subset_path,
            device=device,
        )

    class_weights = class_weights_from_subset(subset_payload)
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
            )
        elif kind == "c0":
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
            bootstrap_samples=0,
            seed=seed,
        )
        focus_miou = float(metrics["focus_miou"])
        scheduler.step(focus_miou)
        improved = focus_miou > best_metric + min_delta
        if improved:
            best_metric = focus_miou
            bad_epochs = 0
        else:
            bad_epochs += 1
        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            subset_path=subset_path,
            epoch=epoch,
            best_metric=best_metric,
            bad_epochs=bad_epochs,
        )
        _atomic_torch_save(checkpoint, latest_path)
        if improved:
            _atomic_torch_save(checkpoint, best_path)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_focus_miou": best_metric,
            "bad_epochs": bad_epochs,
            "val": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
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
        bootstrap_samples=int(evaluation_config.get("bootstrap_samples", 1000)),
        seed=seed,
    )
    (output_dir / "best_val_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**metadata, "best_val": final_metrics, "best_checkpoint": str(best_path)}


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
    records, _ = load_manifest_records(dataset_root, split_file, split=split)
    model = _model_from_config(config).to(device)
    _restore_checkpoint(
        Path(checkpoint_path),
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
        bootstrap_samples=int(evaluation_config.get("bootstrap_samples", 1000)),
        seed=int(training_config["seed"]),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
