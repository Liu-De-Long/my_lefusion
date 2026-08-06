"""Patient-equal segmentation metrics for reconstructed p64 predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .data import LABEL_NAMES, LABEL_VALUES


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _class_metrics(confusion: np.ndarray, index: int) -> dict[str, float]:
    true_positive = float(confusion[index, index])
    false_negative = float(confusion[index, :].sum() - true_positive)
    false_positive = float(confusion[:, index].sum() - true_positive)
    return {
        "iou": _safe_ratio(true_positive, true_positive + false_positive + false_negative),
        "dice": _safe_ratio(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative),
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "recall": _safe_ratio(true_positive, true_positive + false_negative),
    }


def _focus_from_confusions(confusions: list[np.ndarray]) -> float:
    et = _nanmean([_class_metrics(confusion, 2)["iou"] for confusion in confusions])
    rc = _nanmean([_class_metrics(confusion, 3)["iou"] for confusion in confusions])
    return _nanmean([et, rc])


class PatientMetricAccumulator:
    """Accumulate complete p64 predictions, then aggregate subjects equally."""

    def __init__(self) -> None:
        self.patient_confusions: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((4, 4), dtype=np.int64)
        )
        self.patient_errors: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.patch_count = 0
        self.outside_nonzero = 0
        self.outside_voxels = 0
        self.union_intersection = 0
        self.union_sum = 0
        self.small_region: dict[str, dict[str, list[float]]] = {
            name: {"1-100": [], "101-1000": [], ">1000": []}
            for name in LABEL_NAMES
        }

    def update(
        self,
        *,
        subject_id: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        total_mask: torch.Tensor,
    ) -> None:
        prediction = prediction.detach().to("cpu", torch.int64).squeeze()
        target = target.detach().to("cpu", torch.int64).squeeze()
        total_mask = total_mask.detach().to("cpu", torch.bool).squeeze()
        if prediction.shape != target.shape or target.shape != total_mask.shape or target.ndim != 3:
            raise ValueError(
                f"metrics require full matching [D,H,W], got {prediction.shape}, {target.shape}, {total_mask.shape}"
            )
        if not torch.equal(total_mask, target > 0):
            raise ValueError("input total mask does not equal target class union")
        if torch.any((prediction < 0) | (prediction > 4)):
            raise ValueError("prediction contains labels outside 0..4")

        outside = ~total_mask
        self.outside_nonzero += int(torch.count_nonzero(prediction[outside]))
        self.outside_voxels += int(outside.sum())
        predicted_union = prediction > 0
        self.union_intersection += int((predicted_union & total_mask).sum())
        self.union_sum += int(predicted_union.sum() + total_mask.sum())

        truth_inside = target[total_mask] - 1
        prediction_inside = prediction[total_mask] - 1
        if torch.any((prediction_inside < 0) | (prediction_inside > 3)):
            raise ValueError("every total-mask voxel must receive one of four classes")
        encoded = truth_inside * 4 + prediction_inside
        confusion = torch.bincount(encoded, minlength=16).reshape(4, 4).numpy()
        self.patient_confusions[str(subject_id)] += confusion
        errors = int(torch.count_nonzero(truth_inside != prediction_inside))
        self.patient_errors[str(subject_id)][0] += errors
        self.patient_errors[str(subject_id)][1] += int(total_mask.sum())
        self.patch_count += 1

        for class_index, class_name in enumerate(LABEL_NAMES, start=1):
            class_voxels = int(torch.count_nonzero(target == class_index))
            if class_voxels == 0:
                continue
            band = "1-100" if class_voxels <= 100 else "101-1000" if class_voxels <= 1000 else ">1000"
            truth_class = target == class_index
            predicted_class = prediction == class_index
            intersection = int((truth_class & predicted_class).sum())
            denominator = int(truth_class.sum() + predicted_class.sum())
            dice = _safe_ratio(2.0 * intersection, denominator)
            self.small_region[class_name][band].append(dice)

    def compute(self, *, bootstrap_samples: int = 1000, seed: int = 20260806) -> dict[str, Any]:
        if not self.patient_confusions:
            raise RuntimeError("no predictions were accumulated")
        subject_ids = sorted(self.patient_confusions)
        confusions = [self.patient_confusions[subject] for subject in subject_ids]
        class_payload: dict[str, dict[str, float | int]] = {}
        for class_index, class_name in enumerate(LABEL_NAMES):
            per_patient = [_class_metrics(confusion, class_index) for confusion in confusions]
            class_payload[class_name] = {
                metric: _nanmean([values[metric] for values in per_patient])
                for metric in ("iou", "dice", "precision", "recall")
            }
            class_payload[class_name]["evaluable_patients"] = sum(
                np.isfinite(values["iou"]) for values in per_patient
            )

        macro_iou = _nanmean([float(values["iou"]) for values in class_payload.values()])
        macro_dice = _nanmean([float(values["dice"]) for values in class_payload.values()])
        balanced_accuracy = _nanmean(
            [float(values["recall"]) for values in class_payload.values()]
        )
        focus_miou = _nanmean(
            [float(class_payload["ET"]["iou"]), float(class_payload["RC"]["iou"])]
        )
        patient_error_rates = [
            _safe_ratio(*self.patient_errors[subject]) for subject in subject_ids
        ]

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
        patient_normalized_confusion = np.nanmean(
            np.stack(normalized_confusions, axis=0), axis=0
        )

        rng = np.random.default_rng(seed)
        bootstrap_values: list[float] = []
        if bootstrap_samples > 0:
            for _ in range(int(bootstrap_samples)):
                indices = rng.integers(0, len(confusions), size=len(confusions))
                value = _focus_from_confusions([confusions[index] for index in indices])
                if np.isfinite(value):
                    bootstrap_values.append(value)
        focus_ci = (
            [float(np.percentile(bootstrap_values, 2.5)), float(np.percentile(bootstrap_values, 97.5))]
            if bootstrap_values
            else [float("nan"), float("nan")]
        )

        small_region_payload: dict[str, dict[str, dict[str, float | int]]] = {}
        for class_name, bands in self.small_region.items():
            small_region_payload[class_name] = {
                band: {
                    "patch_count": len(values),
                    "mean_dice": _nanmean(values),
                }
                for band, values in bands.items()
            }

        pooled = np.sum(np.stack(confusions, axis=0), axis=0)
        return {
            "aggregation": "full_p64_then_patient_equal",
            "patient_count": len(subject_ids),
            "patch_count": self.patch_count,
            "classes": class_payload,
            "macro_iou": macro_iou,
            "macro_dice": macro_dice,
            "balanced_accuracy": balanced_accuracy,
            "focus_classes": ["ET", "RC"],
            "focus_miou": focus_miou,
            "focus_miou_bootstrap_95ci": focus_ci,
            "mask_inside_error_rate": _nanmean(patient_error_rates),
            "outside_nonzero_rate": _safe_ratio(self.outside_nonzero, self.outside_voxels),
            "union_dice": _safe_ratio(2.0 * self.union_intersection, self.union_sum),
            "patient_normalized_confusion_matrix": patient_normalized_confusion.tolist(),
            "pooled_confusion_matrix_reference_only": pooled.tolist(),
            "small_region_patch_dice": small_region_payload,
            "gate": {
                "focus_miou_at_least_0_85": bool(focus_miou >= 0.85),
                "et_iou_at_least_0_80": bool(float(class_payload["ET"]["iou"]) >= 0.80),
                "rc_iou_at_least_0_80": bool(float(class_payload["RC"]["iou"]) >= 0.80),
                "outside_nonzero_is_zero": self.outside_nonzero == 0,
                "union_dice_is_one": self.union_intersection * 2 == self.union_sum,
            },
        }


def reconstruct_prediction(
    class_indices_inside: torch.Tensor,
    total_mask: torch.Tensor,
) -> torch.Tensor:
    """Restore scalar [D,H,W] labels and force mask exterior to background 0."""

    if total_mask.ndim == 4 and total_mask.shape[0] == 1:
        total_mask = total_mask[0]
    total_mask = total_mask.to(torch.bool)
    class_indices_inside = class_indices_inside.to(torch.int64).flatten()
    if class_indices_inside.numel() != int(total_mask.sum()):
        raise ValueError("inside predictions do not match total-mask voxel count")
    if torch.any((class_indices_inside < 0) | (class_indices_inside > 3)):
        raise ValueError("inside class indices must be 0..3")
    result = torch.zeros_like(total_mask, dtype=torch.int64)
    result[total_mask] = class_indices_inside + 1
    return result
