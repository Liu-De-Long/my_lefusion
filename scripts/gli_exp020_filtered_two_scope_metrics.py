#!/usr/bin/env python3
"""Filtered exp020 metrics for retained-mask and full-patch evaluation scopes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FEATURE_CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
SCOPES = ("filtered_retained_union", "full_patch_64x64x32")


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _project_helpers():
    """Import torch-backed project helpers only when the full CLI is executed."""
    from classifier.data import load_manifest_records
    from gli_exp019_test200_metrics import (
        _hist_w1,
        _inception_features,
        _swav_features,
        frechet_distance,
        kid_repeated,
    )
    from gli_exp020_test200_metrics import sha256_file

    return {
        "load_manifest_records": load_manifest_records,
        "hist_w1": _hist_w1,
        "inception_features": _inception_features,
        "swav_features": _swav_features,
        "frechet_distance": frechet_distance,
        "kid_repeated": kid_repeated,
        "sha256_file": sha256_file,
    }


def _read_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = [str(value) for value in payload["selected_relative_paths"]]
    if len(paths) != 200 or len(paths) != len(set(paths)):
        raise ValueError("manifest must contain exactly 200 unique paths")
    return payload


def _outputs(roots: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in roots:
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("mask_source") != "overlay" or int(metrics.get("checkpoint_step", -1)) != 46000:
            raise ValueError(f"invalid filtered run contract: {root}")
        for row in metrics["samples"]:
            relative = str(row["source_relative_path"])
            result[relative] = root / "generated_npz" / f"{Path(relative).stem}.npz"
    if len(result) != 200 or any(not value.is_file() for value in result.values()):
        raise ValueError("filtered outputs must contain exactly 200 files")
    return result


def _full_fov_slices(image: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    slices = []
    for axis in range(3):
        other = tuple(index for index in range(3) if index != axis)
        position = int(np.argmax(mask.sum(axis=other)))
        slices.append(np.take(image, position, axis=axis).astype(np.float32, copy=False))
    return slices


def _masked_crops(image: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    from gli_exp019_test200_metrics import orthogonal_lesion_crops

    neutral = np.zeros_like(image, dtype=np.float32)
    neutral[mask] = image[mask]
    return orthogonal_lesion_crops(neutral, mask.astype(np.uint8))


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _radiomics_job(job: dict[str, str]) -> dict[str, object]:
    import SimpleITK as sitk
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=0.05, force2D=False)
    extractor.disableAllFeatures()
    for name in FEATURE_CLASSES:
        extractor.enableFeatureClassByName(name)
    if job["kind"] == "train":
        with np.load(job["source_path"], allow_pickle=False) as data:
            image = np.asarray(data["t1c"], dtype=np.float32)
            lesion = np.asarray(data["seg"]) > 0
    else:
        with np.load(job["output_path"], allow_pickle=False) as data:
            key = "original_input_t1c_xyz" if job["kind"] == "real" else "generated_t1c_xyz"
            image = np.asarray(data[key], dtype=np.float32)
            lesion = np.asarray(data["conditioning_seg_xyz"]) > 0
    mask = lesion if job["scope"] == "filtered_retained_union" else np.ones(image.shape, dtype=bool)
    record: dict[str, object] = {
        "kind": job["kind"], "scope": job["scope"], "key": job["key"], "status": "ok"
    }
    if int(mask.sum()) < 8:
        record.update(status="failed", error="mask_too_small")
        return record
    try:
        normalized = np.clip((image + 1.0) / 2.0, 0.0, 1.0)
        if job["scope"] == "full_patch_64x64x32":
            # PyRadiomics requires at least one background voxel. Padding the ROI
            # retains every one of the 64*64*32 patch voxels while satisfying that
            # contract; no evaluated patch voxel is discarded.
            normalized = np.pad(normalized, 1, mode="constant", constant_values=0.0)
            mask = np.pad(mask, 1, mode="constant", constant_values=False)
        result = extractor.execute(
            sitk.GetImageFromArray(normalized), sitk.GetImageFromArray(mask.astype(np.uint8))
        )
        for name, value in result.items():
            if name.startswith("original_") and (number := _finite(value)) is not None:
                record[name] = number
    except Exception as exc:  # noqa: BLE001
        record.update(status="failed", error=repr(exc))
    return record


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["kind", "scope", "key", "status", "error"] + sorted(
        {key for row in rows for key in row if key.startswith("original_")}
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _raw(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    return np.asarray(
        [[_finite(row.get(column)) if _finite(row.get(column)) is not None else np.nan for column in columns] for row in rows],
        dtype=np.float64,
    )


def _fill_z(rows: list[dict[str, str]], columns: list[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = _raw(rows, columns)
    missing = np.where(~np.isfinite(values))
    values[missing] = np.take(mean, missing[1])
    return (values - mean) / std


def _distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )


def rad_mmd2(real: np.ndarray, generated: np.ndarray, gamma: float) -> float:
    return float(
        np.exp(-gamma * _distances(real, real)).mean()
        + np.exp(-gamma * _distances(generated, generated)).mean()
        - 2.0 * np.exp(-gamma * _distances(real, generated)).mean()
    )


def _radiomics(
    *, train_records: list[dict], source_root: Path, selected: list[str], outputs: dict[str, Path],
    cache: Path, workers: int,
) -> dict[str, dict[str, float | int]]:
    if not cache.is_file():
        jobs = []
        for record in train_records:
            for scope in SCOPES:
                jobs.append({
                    "kind": "train", "scope": scope, "key": str(record["relative_path"]),
                    "source_path": str(source_root / "patch_64x64x32" / str(record["relative_path"])),
                })
        for relative in selected:
            for scope in SCOPES:
                for kind in ("real", "generated"):
                    jobs.append({
                        "kind": kind, "scope": scope, "key": relative,
                        "output_path": str(outputs[relative]),
                    })
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_radiomics_job, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if index % 500 == 0:
                    print(f"radiomics {index}/{len(jobs)}", flush=True)
        _write_rows(cache, rows)
    rows = _read_rows(cache)
    result = {}
    for scope in SCOPES:
        scoped = [row for row in rows if row["scope"] == scope]
        train = sorted((row for row in scoped if row["kind"] == "train" and row["status"] == "ok"), key=lambda row: row["key"])
        real = sorted((row for row in scoped if row["kind"] == "real" and row["status"] == "ok"), key=lambda row: row["key"])
        generated = sorted((row for row in scoped if row["kind"] == "generated" and row["status"] == "ok"), key=lambda row: row["key"])
        failed = [row for row in scoped if row["status"] != "ok"]
        if not train or not real or [r["key"] for r in real] != [r["key"] for r in generated]:
            raise ValueError(f"radiomics pairing mismatch: {scope}")
        columns = sorted(
            name for name in train[0] if name.startswith("original_")
            and sum(_finite(row.get(name)) is not None for row in train) >= 8
        )
        train_raw = _raw(train, columns)
        mean = np.nanmean(train_raw, axis=0)
        std = np.nanstd(train_raw, axis=0)
        std[~np.isfinite(std) | (std < 1e-8)] = 1.0
        train_z = _fill_z(train, columns, mean, std)
        rng = np.random.default_rng(20260729)
        sample = train_z[rng.choice(len(train_z), size=min(512, len(train_z)), replace=False)]
        distances = _distances(sample, sample)
        bandwidth = float(np.median(distances[distances > 0]))
        gamma = 1.0 / max(2.0 * bandwidth, 1e-8)
        result[scope] = {
            "rad_mmd": rad_mmd2(_fill_z(real, columns, mean, std), _fill_z(generated, columns, mean, std), gamma),
            "feature_count": len(columns), "train_patches": len(train),
            "real_patches": len(real), "generated_patches": len(generated),
            "failed_records": len(failed),
            "failed_keys": sorted({row["key"] for row in failed}),
            "rbf_bandwidth_train_median_sqdist": bandwidth, "gamma": gamma,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--region-csv", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--filtered-shard", type=Path, action="append", required=True)
    parser.add_argument("--swav-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helpers = _project_helpers()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(args.manifest)
    selected = [str(value) for value in manifest["selected_relative_paths"]]
    outputs = _outputs(args.filtered_shard)
    if set(selected) != set(outputs):
        raise ValueError("output paths do not match selection")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    region_rows = _read_rows(args.region_csv)
    real_mask, generated_mask, real_patch, generated_patch = [], [], [], []
    hist_values = {scope: [] for scope in SCOPES}
    for relative in selected:
        with np.load(outputs[relative], allow_pickle=False) as data:
            real = np.asarray(data["original_input_t1c_xyz"], dtype=np.float32)
            generated = np.asarray(data["generated_t1c_xyz"], dtype=np.float32)
            mask = np.asarray(data["conditioning_seg_xyz"]) > 0
        real_mask.extend(_masked_crops(real, mask))
        generated_mask.extend(_masked_crops(generated, mask))
        real_patch.extend(_full_fov_slices(real, mask))
        generated_patch.extend(_full_fov_slices(generated, mask))
        hist_values["filtered_retained_union"].append(helpers["hist_w1"](real, generated, mask))
        hist_values["full_patch_64x64x32"].append(helpers["hist_w1"](real, generated, np.ones(real.shape, dtype=bool)))

    cache = args.output / "feature_cache"
    cache.mkdir(exist_ok=True)
    distribution = {}
    for scope, real_images, generated_images in (
        ("filtered_retained_union", real_mask, generated_mask),
        ("full_patch_64x64x32", real_patch, generated_patch),
    ):
        inc_real_path = cache / f"{scope}_real_inception.npz"
        inc_gen_path = cache / f"{scope}_filtered_inception.npz"
        swav_real_path = cache / f"{scope}_real_swav.npz"
        swav_gen_path = cache / f"{scope}_filtered_swav.npz"
        inc_real = np.load(inc_real_path)["features"] if inc_real_path.is_file() else helpers["inception_features"](real_images, args.device, args.batch_size)
        inc_gen = np.load(inc_gen_path)["features"] if inc_gen_path.is_file() else helpers["inception_features"](generated_images, args.device, args.batch_size)
        swav_real = np.load(swav_real_path)["features"] if swav_real_path.is_file() else helpers["swav_features"](real_images, args.swav_weights, args.device, args.batch_size)
        swav_gen = np.load(swav_gen_path)["features"] if swav_gen_path.is_file() else helpers["swav_features"](generated_images, args.swav_weights, args.device, args.batch_size)
        for path, values in ((inc_real_path, inc_real), (inc_gen_path, inc_gen), (swav_real_path, swav_real), (swav_gen_path, swav_gen)):
            if not path.is_file():
                np.savez_compressed(path, features=values)
        distribution[scope] = {
            "fsd_swav": helpers["frechet_distance"](swav_real, swav_gen),
            "kid": helpers["kid_repeated"](inc_real, inc_gen),
            "image_count": len(real_images),
        }

    train_records, train_manifest = helpers["load_manifest_records"](args.source_root, args.split_file, split="train")
    radiomics = _radiomics(
        train_records=list(train_records), source_root=args.source_root, selected=selected,
        outputs=outputs, cache=args.output / "radiomics_features.csv", workers=args.workers,
    )
    rows = []
    for scope in SCOPES:
        region = summary["regions"]["filtered"][scope]
        patch_scope_rows = [
            row for row in region_rows
            if row["model"] == "filtered" and row["region"] == scope
        ]
        if len(patch_scope_rows) != 200:
            raise ValueError(f"expected 200 filtered region rows for {scope}")
        row = {
            "scope": scope,
            "fsd_swav": distribution[scope]["fsd_swav"],
            "kid_mean": distribution[scope]["kid"]["mean"],
            "kid_std": distribution[scope]["kid"]["std"],
            "psnr_patch_mean": region["patch_psnr_finite_mean"],
            "psnr_pooled": region["pooled"]["psnr_db"],
            "ssim_patch_mean": float(np.mean([float(value["ssim_region_mean"]) for value in patch_scope_rows])),
            "ssim_patient_equal_mean": region["patient_equal_ssim_mean"],
            "ssim_pooled": region["pooled_ssim_region_mean"],
            "hist_w1_mean": float(np.mean(hist_values[scope])),
            "rad_mmd": radiomics[scope]["rad_mmd"],
        }
        rows.append(row)
    with (args.output / "filtered_two_scope_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "schema_version": 1, "experiment": "exp020_filtered_two_scope_metrics",
        "manifest_sha256": helpers["sha256_file"](args.manifest), "summary_sha256": helpers["sha256_file"](args.summary),
        "region_csv_sha256": helpers["sha256_file"](args.region_csv),
        "source_train_manifest_sha256": helpers["sha256_file"](train_manifest), "selected_count": 200,
        "fsd_definition": "SwAV-ResNet50 avgpool; 3 orthogonal maximal-mask slices",
        "kid_definition": "Inception-v3 features; unbiased polynomial MMD; 100 deterministic subsets",
        "mask_scope_image_policy": "outside retained union set to neutral 0 then bbox+8 crop",
        "patch_scope_image_policy": "same orthogonal positions, full slice FOV without bbox crop",
        "hist_w1_definition": "16 bins in [-1,1], union distribution",
        "rad_mmd_definition": {
            "reference": "/workspace/LeFusion-main/LeFusion-main/tools/t1c_main_20k/compute_rad_mmd.py",
            "feature_classes": FEATURE_CLASSES, "shape_features": False, "bin_width": 0.05,
            "intensity": "clip((x+1)/2,0,1)", "zscore": "real train patches",
            "full_patch_roi": "all 64x64x32 voxels; one-voxel exterior padding only to satisfy PyRadiomics background-label contract",
            "kernel": "Gaussian RBF", "bandwidth": "train median squared distance, sample<=512 seed20260729",
            "estimator": "biased MMD^2 matching reference implementation",
        },
        "radiomics": radiomics, "rows": rows,
        "filtered_copy_warning": "full-patch metrics include exact copied background and rejected lesion regions",
    }
    _json(args.output / "filtered_two_scope_table.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
