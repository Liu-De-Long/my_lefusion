#!/usr/bin/env python3
"""Paired exp020 audit with GT, retained, rejected, and outside-union regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity

from gli_exp019_test200_metrics import (
    EXPECTED,
    LABEL_NAMES,
    MODEL_NAMES,
    _hist_w1,
    _inception_features,
    _load_manifest,
    _swav_features,
    frechet_distance,
    kid_repeated,
    orthogonal_lesion_crops,
    sha256_file,
)


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_runs(
    name: str,
    roots: list[Path],
    manifest: dict,
    *,
    expected_mask_source: str,
    expected_overlay_sha256: str | None = None,
) -> tuple[dict[str, Path], list[dict]]:
    expected_step, expected_sha = EXPECTED[name]
    paths: dict[str, Path] = {}
    summaries = []
    for shard_index, root in enumerate(roots):
        summary = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        expected_paths = set(manifest["shards"][shard_index]["relative_paths"])
        actual_paths = {str(row["source_relative_path"]) for row in summary["samples"]}
        if actual_paths != expected_paths:
            raise ValueError(f"{name} shard {shard_index} does not match manifest")
        if int(summary["checkpoint_step"]) != expected_step or summary["checkpoint_sha256"] != expected_sha:
            raise ValueError(f"{name} checkpoint mismatch")
        if summary["checkpoint_weights_key"] != "ema" or summary["mask_source"] != expected_mask_source:
            raise ValueError(f"{name} mask/weights contract mismatch")
        if int(summary["model_calls_per_sample"]) != 300 or not bool(summary["memory_stable"]):
            raise ValueError(f"{name} sampling or memory contract mismatch")
        if expected_mask_source == "overlay":
            if summary.get("mask_overlay_contract_sha256") != expected_overlay_sha256:
                raise ValueError(f"{name} overlay contract hash mismatch")
        for relative in expected_paths:
            paths[relative] = root / "generated_npz" / f"{Path(relative).stem}.npz"
        summaries.append(summary)
    if len(paths) != 200 or any(not path.is_file() for path in paths.values()):
        raise ValueError(f"{name} does not contain exactly 200 output NPZ files")
    return paths, summaries


def _psnr_payload(mse: float) -> dict:
    if mse == 0.0:
        return {"mse": 0.0, "psnr_db": None, "psnr_infinite": True}
    return {
        "mse": float(mse),
        "psnr_db": float(10.0 * math.log10(4.0 / mse)),
        "psnr_infinite": False,
    }


def _patient_bootstrap(values: dict[str, float], *, repeats: int = 2000) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray([value for value in values.values() if np.isfinite(value)], dtype=np.float64)
    if not array.size:
        return [None, None]
    rng = np.random.default_rng(20260806)
    estimates = np.asarray(
        [np.mean(rng.choice(array, size=len(array), replace=True)) for _ in range(repeats)]
    )
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _summarize_region(rows: list[dict]) -> dict:
    voxels = sum(int(row["voxel_count"]) for row in rows)
    sse = sum(float(row["sse"]) for row in rows)
    pooled_mse = float(sse / voxels) if voxels else float("nan")
    finite_psnr = [float(row["psnr_db"]) for row in rows if np.isfinite(float(row["psnr_db"]))]
    patient_sse: dict[str, float] = defaultdict(float)
    patient_voxels: dict[str, int] = defaultdict(int)
    patient_ssim: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        patient = str(row["subject_id"])
        patient_sse[patient] += float(row["sse"])
        patient_voxels[patient] += int(row["voxel_count"])
        patient_ssim[patient].append(float(row["ssim_region_mean"]))
    patient_psnr = {
        patient: (float("inf") if patient_sse[patient] == 0 else 10.0 * math.log10(4.0 / (patient_sse[patient] / count)))
        for patient, count in patient_voxels.items()
    }
    patient_ssim_mean = {key: float(np.mean(value)) for key, value in patient_ssim.items()}
    patient_psnr_finite = [value for value in patient_psnr.values() if np.isfinite(value)]
    changed_voxels = sum(
        float(row["changed_voxel_fraction"]) * int(row["voxel_count"])
        for row in rows
    )
    return {
        "patch_count": len(rows),
        "patient_count": len(patient_voxels),
        "voxel_count": voxels,
        "pooled": _psnr_payload(pooled_mse),
        "patch_psnr_finite_mean": float(np.mean(finite_psnr)) if finite_psnr else None,
        "patch_psnr_finite_median": float(np.median(finite_psnr)) if finite_psnr else None,
        "patch_psnr_infinite_count": len(rows) - len(finite_psnr),
        "patient_equal_psnr_finite_mean": (
            float(np.mean(patient_psnr_finite)) if patient_psnr_finite else None
        ),
        "patient_equal_psnr_finite_median": (
            float(np.median(patient_psnr_finite)) if patient_psnr_finite else None
        ),
        "patient_psnr_infinite_count": sum(np.isinf(value) for value in patient_psnr.values()),
        "patient_equal_psnr_finite_bootstrap_95ci": _patient_bootstrap(patient_psnr),
        "pooled_ssim_region_mean": float(
            sum(float(row["ssim_region_mean"]) * int(row["voxel_count"]) for row in rows) / voxels
        ) if voxels else None,
        "patient_equal_ssim_mean": float(np.mean(list(patient_ssim_mean.values()))) if patient_ssim_mean else None,
        "patient_equal_ssim_median": float(np.median(list(patient_ssim_mean.values()))) if patient_ssim_mean else None,
        "patient_equal_ssim_bootstrap_95ci": _patient_bootstrap(patient_ssim_mean),
        "max_abs_change": max(float(row["max_abs_change"]) for row in rows),
        "changed_voxel_fraction": float(changed_voxels / voxels) if voxels else None,
        "exact_copy_voxel_fraction": float(1.0 - changed_voxels / voxels) if voxels else None,
    }


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.bincount((truth.astype(np.int64) * 5 + prediction).ravel(), minlength=25).reshape(5, 5)


def _mask_metrics(confusion: np.ndarray) -> dict:
    result = {"confusion_truth_rows_prediction_columns": confusion.tolist(), "per_class": {}}
    for label, name in LABEL_NAMES.items():
        tp = float(confusion[label, label])
        fn = float(confusion[label].sum() - tp)
        fp = float(confusion[:, label].sum() - tp)
        result["per_class"][name] = {
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    return result


def _region_rows(
    *,
    model: str,
    relative: str,
    subject: str,
    reference: np.ndarray,
    generated: np.ndarray,
    truth_seg: np.ndarray,
    retained: np.ndarray,
    support: np.ndarray,
) -> list[dict]:
    difference = generated.astype(np.float64) - reference.astype(np.float64)
    _, ssim_map = structural_similarity(reference, generated, data_range=2.0, win_size=7, full=True)
    union = truth_seg > 0
    regions: dict[str, np.ndarray] = {
        **{f"gt_{name.lower()}": truth_seg == label for label, name in LABEL_NAMES.items()},
        "gt_union": union,
        "brain_outside_gt_union": support & ~union,
        "brain_outside_gt_union_safe": support & ~binary_dilation(union, iterations=3),
        "filtered_retained_union": retained,
        "filtered_rejected_gt_union": union & ~retained,
        **{
            f"gt_{name.lower()}_within_filtered_retained": (truth_seg == label) & retained
            for label, name in LABEL_NAMES.items()
        },
    }
    rows = []
    for region, mask in regions.items():
        count = int(mask.sum())
        if not count:
            continue
        sse = float(np.square(difference[mask]).sum())
        mse = sse / count
        rows.append(
            {
                "model": model,
                "relative_path": relative,
                "subject_id": subject,
                "region": region,
                "voxel_count": count,
                "sse": sse,
                "mse": mse,
                "psnr_db": float("inf") if mse == 0 else 10.0 * math.log10(4.0 / mse),
                "ssim_region_mean": float(ssim_map[mask].mean()),
                "max_abs_change": float(np.max(np.abs(difference[mask]))),
                "changed_voxel_fraction": float(np.count_nonzero(difference[mask]) / count),
                "exact_invariance": bool(sse == 0.0),
            }
        )
    return rows


def _masked_crops(image: np.ndarray, seg: np.ndarray) -> list[np.ndarray]:
    masked = np.zeros_like(image, dtype=np.float32)
    masked[seg > 0] = image[seg > 0]
    return orthogonal_lesion_crops(masked, seg)


def _qa(manifest: dict, paths: dict[str, dict[str, Path]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates: dict[tuple[int, str], list[str]] = defaultdict(list)
    for row in manifest["selected_records"]:
        candidates[(int(row["anchor_label"]), str(row["sample_role"]))].append(str(row["relative_path"]))
    selected = [
        min(candidates[(label, role)], key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        for label in (1, 2, 3, 4) for role in ("interior", "boundary")
    ]
    titles = ["original", "GT mask", "direct mask", "filtered mask", "exp010", "direct", "filtered", "|exp010 err|", "|direct err|", "|filtered err|", "retained", "rejected"]
    fig, axes = plt.subplots(8, len(titles), figsize=(30, 20))
    for row_index, relative in enumerate(selected):
        arrays = {name: np.load(paths[name][relative], allow_pickle=False) for name in MODEL_NAMES}
        reference = np.asarray(arrays["exp010"]["original_input_t1c_xyz"])
        truth = np.asarray(arrays["exp010"]["conditioning_seg_xyz"])
        direct = np.asarray(arrays["direct"]["conditioning_seg_xyz"])
        filtered = np.asarray(arrays["filtered"]["conditioning_seg_xyz"])
        retained = filtered > 0
        rejected = (truth > 0) & ~retained
        z = int(np.argmax((truth > 0).sum(axis=(0, 1))))
        generated = {name: np.asarray(arrays[name]["generated_t1c_xyz"]) for name in MODEL_NAMES}
        panels = [reference[:, :, z], truth[:, :, z], direct[:, :, z], filtered[:, :, z]]
        panels += [generated[name][:, :, z] for name in MODEL_NAMES]
        panels += [np.abs(generated[name][:, :, z] - reference[:, :, z]) for name in MODEL_NAMES]
        panels += [retained[:, :, z], rejected[:, :, z]]
        for index, (axis, panel, title) in enumerate(zip(axes[row_index], panels, titles)):
            if 1 <= index <= 3:
                axis.imshow(panel.T, cmap="viridis", origin="lower", vmin=0, vmax=4)
            elif 7 <= index <= 9:
                axis.imshow(panel.T, cmap="magma", origin="lower", vmin=0, vmax=2)
            else:
                axis.imshow(panel.T, cmap="gray", origin="lower", vmin=-1 if index in (0, 4, 5, 6) else 0, vmax=1)
            axis.set_title(title if row_index == 0 else "")
            axis.axis("off")
        for value in arrays.values():
            value.close()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    for name in MODEL_NAMES:
        parser.add_argument(f"--{name}-shard", type=Path, action="append", required=True)
    parser.add_argument("--direct-overlay-contract", type=Path, required=True)
    parser.add_argument("--filtered-overlay-contract", type=Path, required=True)
    parser.add_argument("--swav-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, metadata = _load_manifest(args.manifest)
    contracts = {
        "direct": json.loads(args.direct_overlay_contract.read_text(encoding="utf-8")),
        "filtered": json.loads(args.filtered_overlay_contract.read_text(encoding="utf-8")),
    }
    manifest_sha = sha256_file(args.manifest)
    for name, contract in contracts.items():
        if contract.get("selection_manifest_sha256") != manifest_sha:
            raise ValueError(f"{name} overlay selection hash mismatch")
        if int(contract.get("file_count", -1)) != 200 or bool(contract.get("unselected_test_accessed", True)):
            raise ValueError(f"{name} overlay access contract mismatch")
    roots = {name: list(getattr(args, f"{name}_shard")) for name in MODEL_NAMES}
    overlay_shas = {
        "direct": sha256_file(args.direct_overlay_contract),
        "filtered": sha256_file(args.filtered_overlay_contract),
    }
    paths = {}
    run_summaries = {}
    for name in MODEL_NAMES:
        paths[name], run_summaries[name] = _load_runs(
            name, roots[name], manifest,
            expected_mask_source="ground_truth" if name == "exp010" else "overlay",
            expected_overlay_sha256=overlay_shas.get(name),
        )

    region_rows: list[dict] = []
    hist_rows: list[dict] = []
    conditioning_rows: list[dict] = []
    confusion = {"direct": np.zeros((5, 5), dtype=np.int64), "filtered": np.zeros((5, 5), dtype=np.int64)}
    real_composite: list[np.ndarray] = []
    real_retained: list[np.ndarray] = []
    fake_composite = {name: [] for name in MODEL_NAMES}
    fake_retained = {name: [] for name in MODEL_NAMES}
    for relative in manifest["selected_relative_paths"]:
        loaded = {name: np.load(paths[name][relative], allow_pickle=False) for name in MODEL_NAMES}
        reference = np.asarray(loaded["exp010"]["original_input_t1c_xyz"], dtype=np.float32)
        truth = np.asarray(loaded["exp010"]["conditioning_seg_xyz"], dtype=np.uint8)
        support = np.asarray(loaded["exp010"]["explicit_brain_support_mask_xyz"], dtype=bool)
        direct_seg = np.asarray(loaded["direct"]["conditioning_seg_xyz"], dtype=np.uint8)
        filtered_seg = np.asarray(loaded["filtered"]["conditioning_seg_xyz"], dtype=np.uint8)
        retained = filtered_seg > 0
        if not np.array_equal(direct_seg > 0, truth > 0):
            raise ValueError(f"direct union mismatch: {relative}")
        if np.any(retained & ~(direct_seg > 0)):
            raise ValueError(f"filtered union is not a direct subset: {relative}")
        sample_seed = int(loaded["exp010"]["sample_seed"])
        for name in MODEL_NAMES:
            np.testing.assert_allclose(loaded[name]["original_input_t1c_xyz"], reference, atol=0, rtol=0)
            if int(loaded[name]["sample_seed"]) != sample_seed:
                raise ValueError(f"sample seed mismatch: {relative}")
        confusion["direct"] += _confusion(truth, direct_seg)
        confusion["filtered"] += _confusion(truth, filtered_seg)
        real_composite.extend(orthogonal_lesion_crops(reference, truth))
        real_retained.extend(_masked_crops(reference, filtered_seg))
        for name in MODEL_NAMES:
            generated = np.asarray(loaded[name]["generated_t1c_xyz"], dtype=np.float32)
            if not np.isfinite(generated).all():
                raise ValueError(f"non-finite generated output: {name} {relative}")
            region_rows.extend(
                _region_rows(
                    model=name, relative=relative, subject=str(metadata[relative]["subject_id"]),
                    reference=reference, generated=generated, truth_seg=truth,
                    retained=retained, support=support,
                )
            )
            fake_composite[name].extend(orthogonal_lesion_crops(generated, truth))
            fake_retained[name].extend(_masked_crops(generated, filtered_seg))
            conditioning = np.asarray(loaded[name]["conditioning_seg_xyz"], dtype=np.uint8)
            for label, label_name in LABEL_NAMES.items():
                for source, seg in (("gt", truth), ("conditioning", conditioning)):
                    mask = seg == label
                    if mask.any():
                        hist_rows.append({
                            "model": name, "relative_path": relative,
                            "subject_id": str(metadata[relative]["subject_id"]),
                            "mask_source": source, "label": label, "label_name": label_name,
                            "voxel_count": int(mask.sum()),
                            "hist_w1": _hist_w1(reference, generated, mask),
                        })
            conditioning_rows.append({
                "model": name, "relative_path": relative,
                "subject_id": str(metadata[relative]["subject_id"]),
                "gt_union_voxels": int(np.count_nonzero(truth)),
                "conditioning_union_voxels": int(np.count_nonzero(conditioning)),
                "filtered_retained_voxels": int(retained.sum()),
                "filtered_rejected_gt_voxels": int(np.count_nonzero((truth > 0) & ~retained)),
            })
        for value in loaded.values():
            value.close()

    if len(real_composite) != 600 or len(real_retained) != 600:
        raise RuntimeError("distribution evaluation requires 600 crops per mode")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "region_metrics_per_patch.csv", region_rows)
    _write_csv(output / "hist_w1_per_class.csv", hist_rows)
    _write_csv(output / "conditioning_coverage.csv", conditioning_rows)
    region_summary = {
        name: {
            region: _summarize_region([row for row in region_rows if row["model"] == name and row["region"] == region])
            for region in sorted({row["region"] for row in region_rows if row["model"] == name})
        }
        for name in MODEL_NAMES
    }
    region_index = {
        (str(row["model"]), str(row["relative_path"]), str(row["region"])): row
        for row in region_rows
    }
    paired_rows: list[dict] = []
    for left, right in (("direct", "exp010"), ("filtered", "exp010"), ("filtered", "direct")):
        for relative in manifest["selected_relative_paths"]:
            common_regions = {
                key[2] for key in region_index if key[0] == left and key[1] == relative
            }.intersection(
                key[2] for key in region_index if key[0] == right and key[1] == relative
            )
            for region in sorted(common_regions):
                left_row = region_index[(left, relative, region)]
                right_row = region_index[(right, relative, region)]
                for metric in ("psnr_db", "ssim_region_mean"):
                    left_value = float(left_row[metric])
                    right_value = float(right_row[metric])
                    if not np.isfinite(left_value) or not np.isfinite(right_value):
                        continue
                    paired_rows.append({
                        "left": left, "right": right, "region": region, "metric": metric,
                        "relative_path": relative, "subject_id": left_row["subject_id"],
                        "delta_left_minus_right": left_value - right_value,
                    })
    _write_csv(output / "paired_region_differences.csv", paired_rows)
    paired_summary = {}
    for left, right in (("direct", "exp010"), ("filtered", "exp010"), ("filtered", "direct")):
        pair = f"{left}_minus_{right}"
        paired_summary[pair] = {}
        for region in sorted({row["region"] for row in paired_rows if row["left"] == left and row["right"] == right}):
            paired_summary[pair][region] = {}
            for metric in ("psnr_db", "ssim_region_mean"):
                chosen = [row for row in paired_rows if row["left"] == left and row["right"] == right and row["region"] == region and row["metric"] == metric]
                grouped: dict[str, list[float]] = defaultdict(list)
                for row in chosen:
                    grouped[str(row["subject_id"])].append(float(row["delta_left_minus_right"]))
                patient_values = {key: float(np.mean(value)) for key, value in grouped.items()}
                paired_summary[pair][region][metric] = {
                    "patch_count": len(chosen),
                    "patient_equal_mean": float(np.mean(list(patient_values.values()))) if patient_values else None,
                    "patient_bootstrap_95ci": _patient_bootstrap(patient_values),
                }
    mask_audit = {name: _mask_metrics(matrix) for name, matrix in confusion.items()}
    for name, matrix in confusion.items():
        union_tp = float(matrix[1:, 1:].sum())
        union_fn = float(matrix[1:, 0].sum())
        union_fp = float(matrix[0, 1:].sum())
        mask_audit[name]["union"] = {
            "dice": 2 * union_tp / (2 * union_tp + union_fp + union_fn),
            "iou": union_tp / (union_tp + union_fp + union_fn),
            "precision": union_tp / (union_tp + union_fp),
            "recall": union_tp / (union_tp + union_fn),
            "coverage": union_tp / (union_tp + union_fn),
            "rejection_rate": union_fn / (union_tp + union_fn),
        }

    feature_cache = output / "feature_cache"
    feature_cache.mkdir(exist_ok=True)
    distribution = {}
    for mode, real, fake in (
        ("gt_union_composite", real_composite, fake_composite),
        ("common_filtered_retained", real_retained, fake_retained),
    ):
        distribution[mode] = {"models": {}, "real_split_baseline": {}}
        real_inc_path = feature_cache / f"{mode}_real_inception.npz"
        real_swav_path = feature_cache / f"{mode}_real_swav.npz"
        real_inc = np.load(real_inc_path)["features"] if real_inc_path.is_file() else _inception_features(real, args.device, args.batch_size)
        real_swv = np.load(real_swav_path)["features"] if real_swav_path.is_file() else _swav_features(real, args.swav_weights, args.device, args.batch_size)
        if not real_inc_path.is_file(): np.savez_compressed(real_inc_path, features=real_inc)
        if not real_swav_path.is_file(): np.savez_compressed(real_swav_path, features=real_swv)
        for name in MODEL_NAMES:
            inc_path = feature_cache / f"{mode}_{name}_inception.npz"
            swv_path = feature_cache / f"{mode}_{name}_swav.npz"
            inc = np.load(inc_path)["features"] if inc_path.is_file() else _inception_features(fake[name], args.device, args.batch_size)
            swv = np.load(swv_path)["features"] if swv_path.is_file() else _swav_features(fake[name], args.swav_weights, args.device, args.batch_size)
            if not inc_path.is_file(): np.savez_compressed(inc_path, features=inc)
            if not swv_path.is_file(): np.savez_compressed(swv_path, features=swv)
            distribution[mode]["models"][name] = {
                "fid": frechet_distance(real_inc, inc),
                "fsd_swav": frechet_distance(real_swv, swv),
                "kid": kid_repeated(real_inc, inc),
            }
        split_a = np.concatenate(
            [np.arange(index * 3, index * 3 + 3) for index in range(0, 200, 2)]
        )
        split_b = np.concatenate(
            [np.arange(index * 3, index * 3 + 3) for index in range(1, 200, 2)]
        )
        distribution[mode]["real_split_baseline"] = {
            "fid": frechet_distance(real_inc[split_a], real_inc[split_b]),
            "fsd_swav": frechet_distance(real_swv[split_a], real_swv[split_b]),
            "kid": kid_repeated(real_inc[split_a], real_inc[split_b]),
        }
    _json(output / "distribution_metrics.json", distribution)
    hist_summary = {}
    for name in MODEL_NAMES:
        hist_summary[name] = {}
        for source in ("gt", "conditioning"):
            chosen = [row for row in hist_rows if row["model"] == name and row["mask_source"] == source]
            by_patient: dict[str, list[float]] = defaultdict(list)
            for row in chosen:
                by_patient[str(row["subject_id"])].append(float(row["hist_w1"]))
            patient_hist = {key: float(np.mean(value)) for key, value in by_patient.items()}
            hist_summary[name][source] = {
                "valid_class_units": len(chosen),
                "macro_mean": float(np.mean([row["hist_w1"] for row in chosen])),
                "patient_equal_macro_mean": float(np.mean(list(patient_hist.values()))),
                "patient_bootstrap_95ci": _patient_bootstrap(patient_hist),
                "per_class": {
                    label_name: float(np.mean([row["hist_w1"] for row in chosen if row["label"] == label]))
                    if any(row["label"] == label for row in chosen) else None
                    for label, label_name in LABEL_NAMES.items()
                },
            }
    summary = {
        "schema_version": 1,
        "experiment_id": "20260813_exp020_gli_exp010_pseudomask_conditioned_test200",
        "manifest_sha256": manifest_sha,
        "selected_count": 200,
        "test_accessed_count": 200,
        "unselected_test_accessed": False,
        "mask_sources": {"exp010": "ground_truth", "direct": "overlay", "filtered": "overlay"},
        "overlay_contract_sha256": {
            "direct": overlay_shas["direct"],
            "filtered": overlay_shas["filtered"],
        },
        "regions": region_summary,
        "paired_region_differences": paired_summary,
        "mask_audit": mask_audit,
        "hist_w1": hist_summary,
        "distribution": distribution,
        "filtered_composite_warning": "GT-union metrics include exact copied rejected voxels; use filtered-retained regions for generated-only comparison.",
    }
    _qa(manifest, paths, output / "qa_montage.png")
    _json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
