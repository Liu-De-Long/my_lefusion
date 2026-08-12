"""Export, calibrate, filter, and audit exp016 GLI p64 pseudo masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LeFusion.classifier.data import (  # noqa: E402
    GLIClassifierPatchDataset,
    LABEL_NAMES,
    LABEL_VALUES,
    load_labeled_subset,
    load_manifest_records,
    sha256_file,
)
from LeFusion.classifier.engine import (  # noqa: E402
    _build_spatial_inputs,
    _modalities_from_config,
    _model_from_config,
    _restore_checkpoint,
    load_config,
    resolve_device,
)


PATCH_SIZE_XYZ = (64, 64, 32)
PATCH_DIR = "patch_64x64x32"
OVERLAY_KEYS = ("seg_xyz", "lesion_mask_xyz", "confidence_xyz", "hist")
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _one_hot_xyz(seg_xyz: np.ndarray) -> np.ndarray:
    return np.stack([(seg_xyz == value) for value in LABEL_VALUES], axis=0).astype(
        np.uint8, copy=False
    )


def lesion_histograms(t1c_xyz: np.ndarray, seg_xyz: np.ndarray) -> np.ndarray:
    hist = np.zeros((4, 16), dtype=np.float32)
    for index, label_value in enumerate(LABEL_VALUES):
        values = t1c_xyz[seg_xyz == label_value]
        if values.size:
            counts, _ = np.histogram(values, bins=16, range=(-1.0, 1.0))
            if int(counts.sum()):
                hist[index] = counts.astype(np.float32) / int(counts.sum())
    return hist


def validate_overlay(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(OVERLAY_KEYS):
            raise ValueError(f"invalid overlay keys in {path}: {sorted(arrays.files)}")
        payload = {key: np.asarray(arrays[key]) for key in OVERLAY_KEYS}
    seg = payload["seg_xyz"]
    mask = payload["lesion_mask_xyz"]
    confidence = payload["confidence_xyz"]
    hist = payload["hist"]
    if seg.shape != PATCH_SIZE_XYZ or seg.dtype != np.uint8:
        raise ValueError(f"invalid seg contract in {path}: {seg.shape}, {seg.dtype}")
    if mask.shape != (4, *PATCH_SIZE_XYZ) or mask.dtype != np.uint8:
        raise ValueError(f"invalid lesion mask contract in {path}: {mask.shape}, {mask.dtype}")
    if confidence.shape != PATCH_SIZE_XYZ or confidence.dtype != np.float16:
        raise ValueError(
            f"invalid confidence contract in {path}: {confidence.shape}, {confidence.dtype}"
        )
    if hist.shape != (4, 16) or hist.dtype != np.float32:
        raise ValueError(f"invalid histogram contract in {path}: {hist.shape}, {hist.dtype}")
    if not np.array_equal(mask, _one_hot_xyz(seg)):
        raise ValueError(f"scalar and four-channel overlay masks disagree: {path}")
    if not np.isfinite(confidence).all() or not np.isfinite(hist).all():
        raise ValueError(f"non-finite overlay values in {path}")
    for index, label_value in enumerate(LABEL_VALUES):
        expected = 1.0 if np.any(seg == label_value) else 0.0
        if not np.isclose(float(hist[index].sum()), expected, atol=1e-5):
            raise ValueError(f"invalid overlay histogram sum for label {label_value}: {path}")
    return payload


def component_records(
    seg_xyz: np.ndarray,
    confidence_xyz: np.ndarray,
    truth_xyz: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for class_index, label_value in enumerate(LABEL_VALUES):
        labeled, count = connected_components(seg_xyz == label_value, CONNECTIVITY_26)
        for component_id in range(1, int(count) + 1):
            component = labeled == component_id
            values = confidence_xyz[component].astype(np.float32, copy=False)
            row: dict[str, Any] = {
                "class_index": class_index,
                "label_value": label_value,
                "component_id": component_id,
                "voxel_count": int(component.sum()),
                "mean_confidence": float(values.mean()),
            }
            if truth_xyz is not None:
                row["correct_voxels"] = int(np.count_nonzero(truth_xyz[component] == label_value))
                row["incorrect_voxels"] = int(row["voxel_count"] - row["correct_voxels"])
            records.append(row)
    return records


def filter_components(
    seg_xyz: np.ndarray,
    confidence_xyz: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    filtered = np.zeros_like(seg_xyz, dtype=np.uint8)
    candidates: list[tuple[float, int, np.ndarray]] = []
    kept = 0
    removed = 0
    for label_value in LABEL_VALUES:
        labeled, count = connected_components(seg_xyz == label_value, CONNECTIVITY_26)
        for component_id in range(1, int(count) + 1):
            component = labeled == component_id
            mean_confidence = float(
                confidence_xyz[component].astype(np.float32, copy=False).mean()
            )
            candidates.append((mean_confidence, label_value, component))
            if mean_confidence >= float(threshold):
                filtered[component] = label_value
                kept += 1
            else:
                removed += 1
    fallback = False
    if not np.any(filtered):
        if not candidates:
            raise ValueError("direct pseudo mask has no components")
        _, label_value, component = max(candidates, key=lambda value: value[0])
        filtered[component] = label_value
        fallback = True
        kept += 1
        removed -= 1
    return filtered, {
        "kept_components": kept,
        "removed_components": removed,
        "fallback_kept_best_component": fallback,
        "retained_voxels": int(np.count_nonzero(filtered)),
        "direct_voxels": int(np.count_nonzero(seg_xyz)),
    }


def select_intersection(curve: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not curve:
        raise ValueError("threshold curve is empty")
    return dict(
        min(
            curve,
            key=lambda row: (
                float(row["absolute_gap"]),
                -float(row["balanced_mean"]),
                float(row["threshold"]),
            ),
        )
    )


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty overlay manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return sha256_file(path)


def _selection_paths(
    selection_manifest: Path | None,
    *,
    split: str,
) -> tuple[set[str] | None, str | None]:
    if selection_manifest is None:
        if split == "test":
            raise ValueError("test pseudo-mask export requires --selection-manifest")
        return None, None
    payload = json.loads(selection_manifest.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported selection manifest schema")
    if str(payload.get("split")) != split:
        raise ValueError("selection manifest split mismatch")
    paths = [str(value) for value in payload.get("selected_relative_paths", [])]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("selection manifest paths must be non-empty and unique")
    if int(payload.get("selected_count", -1)) != len(paths):
        raise ValueError("selection manifest count mismatch")
    if tuple(int(value) for value in payload.get("patch_size_xyz", [])) != PATCH_SIZE_XYZ:
        raise ValueError("selection manifest patch shape mismatch")
    return set(paths), sha256_file(selection_manifest)


def _restrict_records(
    records: Sequence[Mapping[str, Any]],
    selected_paths: set[str] | None,
) -> list[Mapping[str, Any]]:
    if selected_paths is None:
        return list(records)
    selected = [row for row in records if str(row["relative_path"]) in selected_paths]
    actual = {str(row["relative_path"]) for row in selected}
    if actual != selected_paths:
        missing = sorted(selected_paths.difference(actual))
        raise ValueError(f"selection contains paths outside source split: {missing[:3]}")
    return selected


def export_direct_masks(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    actual_checkpoint_sha = sha256_file(checkpoint)
    if actual_checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError(
            f"exp016 checkpoint hash mismatch: {actual_checkpoint_sha} != "
            f"{args.expected_checkpoint_sha256}"
        )
    device = resolve_device(args.device or config["training"].get("device", "auto"))
    dataset_root = Path(args.dataset_root or config["data"]["dataset_root"])
    split_file = Path(config["data"]["split_file"])
    subset_path = Path(config["data"]["labeled_subset"])
    modalities = _modalities_from_config(config)
    model = _model_from_config(config).to(device)
    _restore_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        subset_path=subset_path,
        device=device,
    )
    model.eval()
    output_root = Path(args.output_root)
    output_size_root = output_root / PATCH_DIR
    manifest_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    selection_sha256: str | None = None
    kind = str(config["model"]["kind"]).lower()
    with torch.no_grad():
        for split in args.split:
            selected_paths, current_selection_sha = _selection_paths(
                args.selection_manifest, split=split
            )
            if selection_sha256 is not None and current_selection_sha != selection_sha256:
                raise ValueError("one export may use only one selection manifest")
            selection_sha256 = current_selection_sha or selection_sha256
            records, source_manifest = load_manifest_records(
                dataset_root, split_file, split=split
            )
            records = _restrict_records(records, selected_paths)
            dataset = GLIClassifierPatchDataset(
                dataset_root, records, load_targets=True, modalities=modalities
            )
            split_counts[split] = len(dataset)
            loader = DataLoader(
                dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                drop_last=False,
                num_workers=int(args.num_workers),
                pin_memory=device.type == "cuda",
                persistent_workers=int(args.num_workers) > 0,
            )
            for batch in loader:
                images = batch["image"]
                masks = batch["total_mask"][:, 0]
                inputs = _build_spatial_inputs(images, masks, kind=kind).to(
                    device, non_blocking=True
                )
                probabilities = model(inputs).softmax(dim=1).cpu()
                for batch_index, probability in enumerate(probabilities):
                    relative_path = str(batch["relative_path"][batch_index])
                    output_path = output_size_root / relative_path
                    if output_path.is_file() and args.resume:
                        validate_overlay(output_path)
                    else:
                        total_mask = masks[batch_index].numpy().astype(bool, copy=False)
                        max_confidence, class_index = probability.max(dim=0)
                        seg_dhw = np.zeros(total_mask.shape, dtype=np.uint8)
                        seg_dhw[total_mask] = (
                            class_index.numpy()[total_mask].astype(np.uint8, copy=False) + 1
                        )
                        confidence_dhw = np.zeros(total_mask.shape, dtype=np.float32)
                        confidence_dhw[total_mask] = max_confidence.numpy()[total_mask]
                        seg_xyz = np.transpose(seg_dhw, (1, 2, 0)).copy()
                        confidence_xyz = np.transpose(confidence_dhw, (1, 2, 0)).astype(
                            np.float16, copy=False
                        )
                        t1c_xyz = np.transpose(images[batch_index, 0].numpy(), (1, 2, 0)).copy()
                        if not np.array_equal(seg_xyz > 0, np.transpose(total_mask, (1, 2, 0))):
                            raise RuntimeError(f"direct mask union mismatch: {relative_path}")
                        _atomic_npz(
                            output_path,
                            seg_xyz=seg_xyz,
                            lesion_mask_xyz=_one_hot_xyz(seg_xyz),
                            confidence_xyz=confidence_xyz,
                            hist=lesion_histograms(t1c_xyz, seg_xyz),
                        )
                        validate_overlay(output_path)
                    manifest_rows.append(
                        {
                            "relative_path": relative_path,
                            "split": split,
                            "case_id": str(batch["case_id"][batch_index]),
                            "subject_id": str(batch["subject_id"][batch_index]),
                            "sha256": sha256_file(output_path),
                        }
                    )
    manifest_rows.sort(key=lambda row: (str(row["split"]), str(row["relative_path"])))
    manifest_sha = _write_manifest(output_root / "files.csv", manifest_rows)
    contract = {
        "schema_version": 1,
        "variant": "direct_exp016_argmax_inside_true_total_mask",
        "checkpoint_sha256": actual_checkpoint_sha,
        "classifier_config_sha256": config["_config_sha256"],
        "subset_sha256": sha256_file(subset_path),
        "split_file_sha256": sha256_file(split_file),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_dataset_root": str(dataset_root.resolve()),
        "splits": split_counts,
        "file_count": len(manifest_rows),
        "files_manifest_sha256": manifest_sha,
        "selection_manifest_sha256": selection_sha256,
        "test_accessed": "test" in split_counts,
        "unselected_test_accessed": False,
    }
    _json(output_root / "contract.json", contract)
    return contract


def _load_source_arrays(source_root: Path, relative_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = source_root / PATCH_DIR / relative_path
    with np.load(path, allow_pickle=False) as arrays:
        return np.asarray(arrays["t1c"]), np.asarray(arrays["seg"])


def calibrate_threshold(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    direct_root = Path(args.direct_root)
    records, _ = load_manifest_records(source_root, args.split_file, split="val")
    pairs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    component_rows: list[dict[str, Any]] = []
    for record in records:
        relative_path = str(record["relative_path"])
        overlay = validate_overlay(direct_root / PATCH_DIR / relative_path)
        _, truth = _load_source_arrays(source_root, relative_path)
        for row in component_records(
            overlay["seg_xyz"], overlay["confidence_xyz"], truth
        ):
            row = dict(row)
            row.update(
                {
                    "relative_path": relative_path,
                    "subject_id": str(record["subject_id"]),
                }
            )
            component_rows.append(row)
            pairs[(str(record["subject_id"]), int(row["label_value"]))].append(row)

    curve: list[dict[str, float]] = []
    for threshold in np.arange(0.25, 1.0, 0.001, dtype=np.float64):
        crr_values: list[float] = []
        err_values: list[float] = []
        for values in pairs.values():
            correct_total = sum(int(row["correct_voxels"]) for row in values)
            incorrect_total = sum(int(row["incorrect_voxels"]) for row in values)
            correct_kept = sum(
                int(row["correct_voxels"])
                for row in values
                if float(row["mean_confidence"]) >= float(threshold)
            )
            incorrect_removed = sum(
                int(row["incorrect_voxels"])
                for row in values
                if float(row["mean_confidence"]) < float(threshold)
            )
            if correct_total:
                crr_values.append(correct_kept / correct_total)
            if incorrect_total:
                err_values.append(incorrect_removed / incorrect_total)
        crr = float(np.mean(crr_values))
        err = float(np.mean(err_values))
        curve.append(
            {
                "threshold": round(float(threshold), 3),
                "crr": crr,
                "err": err,
                "absolute_gap": abs(crr - err),
                "balanced_mean": (crr + err) / 2.0,
            }
        )
    selected = select_intersection(curve)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "threshold_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    with (output_dir / "val_components.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot([row["threshold"] for row in curve], [row["crr"] for row in curve], label="CRR")
        axis.plot([row["threshold"] for row in curve], [row["err"] for row in curve], label="ERR")
        axis.axvline(selected["threshold"], color="black", linestyle="--", label="selected")
        axis.set(xlabel="component mean-confidence threshold", ylabel="patient-class equal rate", ylim=(0, 1))
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "threshold_curve.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    payload = {
        "schema_version": 1,
        "method": "patient_class_equal_correct_retention_error_rejection_intersection",
        "component_connectivity": 26,
        "component_confidence": "mean_max_softmax",
        "candidate_range": [0.25, 0.999],
        "candidate_step": 0.001,
        "selected": selected,
        "val_patch_count": len(records),
        "val_component_count": len(component_rows),
        "direct_contract_sha256": sha256_file(direct_root / "contract.json"),
        "split_file_sha256": sha256_file(args.split_file),
        "test_accessed": False,
    }
    _json(output_dir / "threshold.json", payload)
    return payload


def derive_filtered_masks(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    direct_root = Path(args.direct_root)
    output_root = Path(args.output_root)
    threshold_payload = json.loads(Path(args.threshold_json).read_text(encoding="utf-8"))
    threshold = float(threshold_payload["selected"]["threshold"])
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    fallback_count = 0
    kept_components = 0
    removed_components = 0
    selection_sha256: str | None = None
    for split in args.split:
        selected_paths, current_selection_sha = _selection_paths(
            args.selection_manifest, split=split
        )
        if selection_sha256 is not None and current_selection_sha != selection_sha256:
            raise ValueError("one export may use only one selection manifest")
        selection_sha256 = current_selection_sha or selection_sha256
        records, source_manifest = load_manifest_records(source_root, args.split_file, split=split)
        records = _restrict_records(records, selected_paths)
        split_counts[split] = len(records)
        for record in records:
            relative_path = str(record["relative_path"])
            direct = validate_overlay(direct_root / PATCH_DIR / relative_path)
            output_path = output_root / PATCH_DIR / relative_path
            if output_path.is_file() and args.resume:
                validate_overlay(output_path)
                filtered = validate_overlay(output_path)["seg_xyz"]
                stats = {"fallback_kept_best_component": False, "kept_components": 0, "removed_components": 0}
            else:
                filtered, stats = filter_components(
                    direct["seg_xyz"], direct["confidence_xyz"], threshold
                )
                t1c, _ = _load_source_arrays(source_root, relative_path)
                _atomic_npz(
                    output_path,
                    seg_xyz=filtered,
                    lesion_mask_xyz=_one_hot_xyz(filtered),
                    confidence_xyz=direct["confidence_xyz"],
                    hist=lesion_histograms(t1c, filtered),
                )
                validate_overlay(output_path)
            fallback_count += int(bool(stats["fallback_kept_best_component"]))
            kept_components += int(stats["kept_components"])
            removed_components += int(stats["removed_components"])
            rows.append(
                {
                    "relative_path": relative_path,
                    "split": split,
                    "case_id": str(record["case_id"]),
                    "subject_id": str(record["subject_id"]),
                    "retained_voxels": int(np.count_nonzero(filtered)),
                    "sha256": sha256_file(output_path),
                }
            )
    rows.sort(key=lambda row: (str(row["split"]), str(row["relative_path"])))
    manifest_sha = _write_manifest(output_root / "files.csv", rows)
    contract = {
        "schema_version": 1,
        "variant": "filtered_exp016_component_mean_confidence",
        "threshold": threshold,
        "threshold_contract_sha256": sha256_file(args.threshold_json),
        "direct_contract_sha256": sha256_file(direct_root / "contract.json"),
        "split_file_sha256": sha256_file(args.split_file),
        "source_manifest_sha256": sha256_file(source_manifest),
        "splits": split_counts,
        "file_count": len(rows),
        "fallback_patch_count": fallback_count,
        "kept_component_count": kept_components,
        "removed_component_count": removed_components,
        "files_manifest_sha256": manifest_sha,
        "selection_manifest_sha256": selection_sha256,
        "test_accessed": "test" in split_counts,
        "unselected_test_accessed": False,
    }
    _json(output_root / "contract.json", contract)
    return contract


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _class_metrics(confusion: np.ndarray, class_index: int) -> dict[str, float]:
    tp = float(confusion[class_index, class_index])
    fn = float(confusion[class_index].sum() - tp)
    fp = float(confusion[:, class_index].sum() - tp)
    return {
        "iou": _safe_ratio(tp, tp + fp + fn),
        "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
    }


def _metrics_from_patient_confusions(
    patient_confusions: Mapping[str, np.ndarray], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    subjects = sorted(patient_confusions)
    confusions = [patient_confusions[subject] for subject in subjects]
    classes: dict[str, Any] = {}
    for index, name in enumerate(LABEL_NAMES):
        values = [_class_metrics(confusion, index) for confusion in confusions]
        classes[name] = {
            metric: float(np.nanmean([value[metric] for value in values]))
            for metric in ("iou", "dice", "precision", "recall")
        }
    focus = float(np.mean([classes["ET"]["iou"], classes["RC"]["iou"]]))
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(int(bootstrap_samples)):
        indices = rng.integers(0, len(confusions), len(confusions))
        sampled = [confusions[index] for index in indices]
        et = float(np.nanmean([_class_metrics(value, 2)["iou"] for value in sampled]))
        rc = float(np.nanmean([_class_metrics(value, 3)["iou"] for value in sampled]))
        boot.append((et + rc) / 2.0)
    pooled = np.sum(np.stack(confusions), axis=0)
    normalized_confusions: list[np.ndarray] = []
    for confusion in confusions:
        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized_confusions.append(
            np.divide(
                confusion,
                row_sums,
                out=np.full_like(confusion, np.nan, dtype=np.float64),
                where=row_sums > 0,
            )
        )
    return {
        "patient_count": len(subjects),
        "classes": classes,
        "macro_iou": float(np.mean([value["iou"] for value in classes.values()])),
        "macro_dice": float(np.mean([value["dice"] for value in classes.values()])),
        "balanced_accuracy": float(np.mean([value["recall"] for value in classes.values()])),
        "focus_miou": focus,
        "focus_miou_bootstrap_95ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "pooled_confusion_true4_by_pred4plus_abstain": pooled.tolist(),
        "patient_normalized_confusion_true4_by_pred4plus_abstain": np.nanmean(
            np.stack(normalized_confusions), axis=0
        ).tolist(),
        "retained_coverage": _safe_ratio(float(pooled[:, :4].sum()), float(pooled.sum())),
        "abstention_rate": _safe_ratio(float(pooled[:, 4].sum()), float(pooled.sum())),
    }


def audit_masks(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    overlay_root = Path(args.overlay_root)
    records, _ = load_manifest_records(source_root, args.split_file, split=args.split)
    seen_records, _ = load_labeled_subset(args.labeled_subset, source_root, args.split_file)
    seen = {str(record["relative_path"]) for record in seen_records}
    groups: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.zeros((4, 5), dtype=np.int64))
    )
    patch_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    small_region: dict[str, dict[str, list[float]]] = {
        name: {"1-100": [], "101-1000": [], ">1000": []} for name in LABEL_NAMES
    }
    for record in records:
        relative_path = str(record["relative_path"])
        overlay = validate_overlay(overlay_root / PATCH_DIR / relative_path)
        _, truth = _load_source_arrays(source_root, relative_path)
        prediction = overlay["seg_xyz"]
        total_mask = truth > 0
        if np.any(prediction[~total_mask] != 0):
            raise ValueError(f"overlay predicts outside true total mask: {relative_path}")
        truth_inside = truth[total_mask].astype(np.int64) - 1
        prediction_inside = prediction[total_mask].astype(np.int64)
        prediction_encoded = np.where(prediction_inside > 0, prediction_inside - 1, 4)
        confusion = np.bincount(
            truth_inside * 5 + prediction_encoded, minlength=20
        ).reshape(4, 5)
        subject = str(record["subject_id"])
        group_names = ["all", "seen_1000" if relative_path in seen else "unseen_6772"]
        group_names.append(str(record["sample_role"]))
        is_padded = any(int(record[name]) for name in (
            "pad_before_x", "pad_before_y", "pad_before_z",
            "pad_after_x", "pad_after_y", "pad_after_z",
        ))
        group_names.append("padded" if is_padded else "not_padded")
        for group_name in group_names:
            groups[group_name][subject] += confusion
        correct = int(np.count_nonzero((prediction == truth) & total_mask))
        retained = int(np.count_nonzero(prediction))
        patch_row: dict[str, Any] = {
                "relative_path": relative_path,
                "subject_id": subject,
                "sample_role": str(record["sample_role"]),
                "is_padded": int(is_padded),
                "teacher_seen": int(relative_path in seen),
                "true_lesion_voxels": int(total_mask.sum()),
                "retained_voxels": retained,
                "coverage": _safe_ratio(retained, int(total_mask.sum())),
                "correct_voxels": correct,
                "accuracy_with_abstention_as_error": _safe_ratio(correct, int(total_mask.sum())),
                "mean_confidence_inside": float(overlay["confidence_xyz"][total_mask].astype(np.float32).mean()),
            }
        for class_index, class_name in enumerate(LABEL_NAMES, start=1):
            truth_class = truth == class_index
            predicted_class = prediction == class_index
            intersection = int(np.count_nonzero(truth_class & predicted_class))
            union = int(np.count_nonzero(truth_class | predicted_class))
            denominator = int(np.count_nonzero(truth_class) + np.count_nonzero(predicted_class))
            patch_row[f"{class_name.lower()}_iou"] = _safe_ratio(intersection, union)
            patch_row[f"{class_name.lower()}_dice"] = _safe_ratio(2 * intersection, denominator)
        patch_rows.append(patch_row)
        for row in component_records(prediction, overlay["confidence_xyz"], truth):
            row = dict(row)
            row.update({"relative_path": relative_path, "subject_id": subject})
            component_rows.append(row)
        for class_index, class_name in enumerate(LABEL_NAMES, start=1):
            class_voxels = int(np.count_nonzero(truth == class_index))
            if not class_voxels:
                continue
            band = "1-100" if class_voxels <= 100 else "101-1000" if class_voxels <= 1000 else ">1000"
            intersection = int(np.count_nonzero((truth == class_index) & (prediction == class_index)))
            denominator = int(np.count_nonzero(truth == class_index) + np.count_nonzero(prediction == class_index))
            small_region[class_name][band].append(_safe_ratio(2 * intersection, denominator))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "patch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0]))
        writer.writeheader()
        writer.writerows(patch_rows)
    with (output_dir / "component_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)
    patient_rows: list[dict[str, Any]] = []
    for subject, confusion in sorted(groups["all"].items()):
        patient_rows.append({"subject_id": subject, **{f"true_{i+1}_pred_{j+1 if j < 4 else 0}": int(confusion[i, j]) for i in range(4) for j in range(5)}})
    with (output_dir / "patient_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_rows[0]))
        writer.writeheader()
        writer.writerows(patient_rows)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        worst = sorted(
            patch_rows,
            key=lambda row: (float(row["accuracy_with_abstention_as_error"]), str(row["relative_path"])),
        )[:4]
        worst_paths = {str(row["relative_path"]) for row in worst}
        remaining = sorted(
            (row for row in patch_rows if str(row["relative_path"]) not in worst_paths),
            key=lambda row: hashlib.sha256(
                f"{args.seed}|{row['relative_path']}".encode("utf-8")
            ).hexdigest(),
        )[:4]
        selected = worst + remaining
        figure, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
        label_cmap = ListedColormap(["black", "#3b82f6", "#22c55e", "#ef4444", "#eab308"])
        for row_index, row in enumerate(selected):
            relative_path = str(row["relative_path"])
            t1c, truth = _load_source_arrays(source_root, relative_path)
            prediction = validate_overlay(overlay_root / PATCH_DIR / relative_path)["seg_xyz"]
            slice_index = int(np.argmax(np.count_nonzero(truth > 0, axis=(0, 1))))
            axes[row_index, 0].imshow(t1c[:, :, slice_index].T, cmap="gray", origin="lower", vmin=-1, vmax=1)
            axes[row_index, 1].imshow(truth[:, :, slice_index].T, cmap=label_cmap, origin="lower", vmin=0, vmax=4)
            axes[row_index, 2].imshow(prediction[:, :, slice_index].T, cmap=label_cmap, origin="lower", vmin=0, vmax=4)
            axes[row_index, 0].set_ylabel(Path(relative_path).stem, fontsize=7)
            for axis in axes[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])
        for column, title in enumerate(("T1c", "true mask", "pseudo mask")):
            axes[0, column].set_title(title)
        figure.tight_layout()
        figure.savefig(output_dir / "qa_contact_sheet.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
    summary = {
        "schema_version": 1,
        "split": args.split,
        "patch_count": len(records),
        "overlay_contract_sha256": sha256_file(overlay_root / "contract.json"),
        "groups": {
            name: _metrics_from_patient_confusions(
                values, bootstrap_samples=args.bootstrap_samples, seed=args.seed
            )
            for name, values in groups.items()
            if values
        },
        "small_region_patch_dice": {
            name: {
                band: {"patch_count": len(values), "mean_dice": float(np.nanmean(values)) if values else None}
                for band, values in bands.items()
            }
            for name, bands in small_region.items()
        },
        "test_accessed": False,
    }
    _json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export-direct")
    export.add_argument("--config", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--expected-checkpoint-sha256", required=True)
    export.add_argument("--split", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    export.add_argument("--selection-manifest", type=Path)
    export.add_argument(
        "--dataset-root",
        type=Path,
        help="Override only the classifier input dataset root; model config remains frozen.",
    )
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--device", default=None)
    export.add_argument("--batch-size", type=int, default=8)
    export.add_argument("--num-workers", type=int, default=8)
    export.add_argument("--resume", action="store_true")

    calibrate = subparsers.add_parser("calibrate-threshold")
    calibrate.add_argument("--source-root", type=Path, required=True)
    calibrate.add_argument("--direct-root", type=Path, required=True)
    calibrate.add_argument("--split-file", type=Path, required=True)
    calibrate.add_argument("--output-dir", type=Path, required=True)

    filtered = subparsers.add_parser("derive-filtered")
    filtered.add_argument("--source-root", type=Path, required=True)
    filtered.add_argument("--direct-root", type=Path, required=True)
    filtered.add_argument("--output-root", type=Path, required=True)
    filtered.add_argument("--split-file", type=Path, required=True)
    filtered.add_argument("--threshold-json", type=Path, required=True)
    filtered.add_argument("--split", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    filtered.add_argument("--selection-manifest", type=Path)
    filtered.add_argument("--resume", action="store_true")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--source-root", type=Path, required=True)
    audit.add_argument("--overlay-root", type=Path, required=True)
    audit.add_argument("--split-file", type=Path, required=True)
    audit.add_argument("--labeled-subset", type=Path, required=True)
    audit.add_argument("--split", choices=("train", "val"), required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--bootstrap-samples", type=int, default=1000)
    audit.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export-direct":
        result = export_direct_masks(args)
    elif args.command == "calibrate-threshold":
        result = calibrate_threshold(args)
    elif args.command == "derive-filtered":
        result = derive_filtered_masks(args)
    else:
        result = audit_masks(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
