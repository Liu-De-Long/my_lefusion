#!/usr/bin/env python3
"""Audit and compare the three paired exp019 test-200 inference runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity


MODEL_NAMES = ("exp010", "direct", "filtered")
LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
EXPECTED = {
    "exp010": (14000, "58e939798fc8d3427b70aafdd61877a14984d2478cb7a39796c6019407f07358"),
    "direct": (44000, "3e3c30585ef27c3540a8f3e3bf0bc3f4b48e1c1fb84ac7b85c06adfac3f70e0a"),
    "filtered": (46000, "5ddd8f112e0d63fc671f641351010cd2fe0fe11a5009f25f3df4e31cf617ff6c"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def psnr(reference: np.ndarray, generated: np.ndarray, mask: np.ndarray | None = None) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if mask is not None:
        chosen = np.asarray(mask, dtype=bool)
        if not chosen.any():
            raise ValueError("PSNR mask is empty")
        difference = generated[chosen] - reference[chosen]
    else:
        difference = generated - reference
    mse = float(np.mean(np.square(difference)))
    return float("inf") if mse == 0 else float(10.0 * math.log10(4.0 / mse))


def _crop_slice(image: np.ndarray, mask: np.ndarray, margin: int = 8) -> np.ndarray:
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("slice lesion mask is empty")
    lo = np.maximum(points.min(axis=0) - int(margin), 0)
    hi = np.minimum(points.max(axis=0) + int(margin) + 1, mask.shape)
    return np.asarray(image[lo[0] : hi[0], lo[1] : hi[1]], dtype=np.float32)


def orthogonal_lesion_crops(
    image_xyz: np.ndarray, seg_xyz: np.ndarray, margin: int = 8
) -> list[np.ndarray]:
    lesion = np.asarray(seg_xyz) > 0
    if not lesion.any():
        raise ValueError("patch lesion union is empty")
    crops = []
    for axis in range(3):
        reduce_axes = tuple(index for index in range(3) if index != axis)
        slice_index = int(np.argmax(lesion.sum(axis=reduce_axes)))
        image_slice = np.take(image_xyz, slice_index, axis=axis)
        mask_slice = np.take(lesion, slice_index, axis=axis)
        crops.append(_crop_slice(image_slice, mask_slice, margin=margin))
    return crops


def lesion_crop_ssim(reference: np.ndarray, generated: np.ndarray, seg: np.ndarray) -> float:
    real_crops = orthogonal_lesion_crops(reference, seg)
    fake_crops = orthogonal_lesion_crops(generated, seg)
    values = []
    for real, fake in zip(real_crops, fake_crops):
        win_size = min(7, real.shape[0], real.shape[1])
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            raise ValueError(f"lesion crop too small for SSIM: {real.shape}")
        values.append(
            float(structural_similarity(real, fake, data_range=2.0, win_size=win_size))
        )
    return float(np.mean(values))


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Exact sample FD using a low-rank cross-covariance identity."""
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("feature matrices must be Nxd with matching d")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Fréchet distance requires at least two samples per set")
    mean_delta = a.mean(axis=0) - b.mean(axis=0)
    centered_a = (a - a.mean(axis=0)) / math.sqrt(len(a) - 1)
    centered_b = (b - b.mean(axis=0)) / math.sqrt(len(b) - 1)
    trace_a = float(np.square(centered_a).sum())
    trace_b = float(np.square(centered_b).sum())
    cross_trace = float(np.linalg.svd(centered_a @ centered_b.T, compute_uv=False).sum())
    result = float(mean_delta @ mean_delta + trace_a + trace_b - 2.0 * cross_trace)
    return max(result, 0.0)


def polynomial_mmd_unbiased(features_a: np.ndarray, features_b: np.ndarray) -> float:
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2 or a.shape[1] != b.shape[1]:
        raise ValueError("KID inputs require matching features and at least two samples")
    dimension = a.shape[1]
    kernel_aa = (a @ a.T / dimension + 1.0) ** 3
    kernel_bb = (b @ b.T / dimension + 1.0) ** 3
    kernel_ab = (a @ b.T / dimension + 1.0) ** 3
    return float(
        (kernel_aa.sum() - np.trace(kernel_aa)) / (len(a) * (len(a) - 1))
        + (kernel_bb.sum() - np.trace(kernel_bb)) / (len(b) * (len(b) - 1))
        - 2.0 * kernel_ab.mean()
    )


def kid_repeated(
    features_a: np.ndarray,
    features_b: np.ndarray,
    *,
    repeats: int = 100,
    subset_size: int = 100,
    seed: int = 20260806,
) -> dict:
    rng = np.random.default_rng(seed)
    size = min(int(subset_size), len(features_a), len(features_b))
    values = []
    for _ in range(int(repeats)):
        left = rng.choice(len(features_a), size=size, replace=False)
        right = rng.choice(len(features_b), size=size, replace=False)
        values.append(polynomial_mmd_unbiased(features_a[left], features_b[right]))
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "subsets": repeats, "subset_size": size}


def _hist_w1(reference: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> float:
    edges = np.linspace(-1.0, 1.0, 17)
    centers = (edges[:-1] + edges[1:]) / 2
    real_hist, _ = np.histogram(reference[mask], bins=edges)
    fake_hist, _ = np.histogram(generated[mask], bins=edges)
    real_hist = real_hist.astype(np.float64) / max(float(real_hist.sum()), 1.0)
    fake_hist = fake_hist.astype(np.float64) / max(float(fake_hist.sum()), 1.0)
    return float(wasserstein_distance(centers, centers, real_hist, fake_hist))


def _summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def patient_bootstrap_ci(rows: list[dict], key: str, repeats: int = 2000) -> list[float | None]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            grouped[str(row["subject_id"])].append(value)
    subjects = sorted(grouped)
    if not subjects:
        return [None, None]
    rng = np.random.default_rng(20260806)
    values = []
    for _ in range(int(repeats)):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        flat = [value for subject in sampled for value in grouped[str(subject)]]
        values.append(float(np.mean(flat)))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def patient_equal_summary(rows: list[dict], key: str) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            grouped[str(row["subject_id"])].append(value)
    patient_means = [float(np.mean(values)) for values in grouped.values()]
    return {
        **_summary(patient_means),
        "patient_bootstrap_95ci": patient_bootstrap_ci(rows, key),
    }


def _load_manifest(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "test" or int(payload.get("selected_count", -1)) != 200:
        raise ValueError("exp019 requires an exact 200-patch test manifest")
    if int(payload.get("shard_count", -1)) != 2:
        raise ValueError("exp019 manifest must contain two shards")
    if [len(shard["relative_paths"]) for shard in payload["shards"]] != [100, 100]:
        raise ValueError("exp019 shard counts must be 100/100")
    metadata = {str(row["relative_path"]): dict(row) for row in payload["selected_records"]}
    return payload, metadata


def _load_model_run(name: str, roots: list[Path], manifest: dict) -> tuple[dict[str, Path], dict[str, dict]]:
    expected_step, expected_sha = EXPECTED[name]
    npz_by_path: dict[str, Path] = {}
    metrics_by_path: dict[str, dict] = {}
    for shard_index, root in enumerate(roots):
        summary = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        expected_paths = set(manifest["shards"][shard_index]["relative_paths"])
        actual_paths = {str(row["source_relative_path"]) for row in summary["samples"]}
        if actual_paths != expected_paths:
            raise ValueError(f"{name} shard {shard_index} does not match the frozen manifest")
        if int(summary["checkpoint_step"]) != expected_step or summary["checkpoint_sha256"] != expected_sha:
            raise ValueError(f"{name} checkpoint provenance mismatch")
        if summary["checkpoint_weights_key"] != "ema" or summary["mask_source"] != "ground_truth":
            raise ValueError(f"{name} weights or mask source mismatch")
        if int(summary["model_calls_per_sample"]) != 300:
            raise ValueError(f"{name} did not use the 300-step schedule")
        if not bool(summary["memory_stable"]):
            raise ValueError(f"{name} reports unstable GPU memory")
        for row in summary["samples"]:
            relative = str(row["source_relative_path"])
            stem = Path(relative).stem
            npz_by_path[relative] = root / "generated_npz" / f"{stem}.npz"
            metrics_by_path[relative] = row
    if len(npz_by_path) != 200:
        raise ValueError(f"{name} expected 200 unique NPZ files, got {len(npz_by_path)}")
    return npz_by_path, metrics_by_path


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resize_uint8(crops: list[np.ndarray], size: int) -> "object":
    import torch
    import torch.nn.functional as functional

    tensors = []
    for crop in crops:
        scaled = np.clip((np.asarray(crop, dtype=np.float32) + 1.0) * 127.5, 0, 255)
        tensor = torch.from_numpy(scaled).unsqueeze(0).unsqueeze(0)
        tensor = functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        tensors.append(tensor[0].repeat(3, 1, 1).round().to(torch.uint8))
    return torch.stack(tensors)


def _inception_features(crops: list[np.ndarray], device: str, batch_size: int) -> np.ndarray:
    import torch
    try:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    except ImportError as error:
        raise RuntimeError("canonical FID/KID requires torch-fidelity==0.3.0") from error
    model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
    images = _resize_uint8(crops, 299)
    output = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            output.append(model(images[start : start + batch_size].to(device))[0].cpu().numpy())
    return np.concatenate(output, axis=0)


def _swav_features(crops: list[np.ndarray], weights: Path, device: str, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    state = {key.removeprefix("module."): value for key, value in state.items() if not key.removeprefix("module.").startswith("projection_head") and not key.removeprefix("module.").startswith("prototypes")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"fc.weight", "fc.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(f"unexpected SwAV checkpoint keys: missing={missing}, unexpected={unexpected}")
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    images = _resize_uint8(crops, 224).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = (images - mean) / std
    output = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            output.append(model(images[start : start + batch_size].to(device)).cpu().numpy())
    return np.concatenate(output, axis=0)


def _distribution_metrics(real: np.ndarray, fake: np.ndarray) -> dict:
    return {
        "frechet": frechet_distance(real, fake),
        "kid": kid_repeated(real, fake),
    }


def _build_montage(rows: list[dict], paths: dict[str, dict[str, Path]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = {}
    for row in rows:
        key = (int(row["anchor_label"]), str(row["sample_role"]))
        chosen.setdefault(key, str(row["relative_path"]))
    selected = [chosen[(label, role)] for label in (1, 2, 3, 4) for role in ("interior", "boundary")]
    fig, axes = plt.subplots(8, 6, figsize=(18, 24))
    for row_index, relative in enumerate(selected):
        arrays = {name: np.load(paths[name][relative], allow_pickle=False) for name in MODEL_NAMES}
        reference = np.asarray(arrays["exp010"]["original_input_t1c_xyz"], dtype=np.float32)
        masked = np.asarray(arrays["exp010"]["masked_input_t1c_xyz"], dtype=np.float32)
        seg = np.asarray(arrays["exp010"]["conditioning_seg_xyz"], dtype=np.uint8)
        z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
        panels = [reference[:, :, z], masked[:, :, z]] + [np.asarray(arrays[name]["generated_t1c_xyz"])[:, :, z] for name in MODEL_NAMES] + [(seg[:, :, z] > 0).astype(float)]
        titles = ["original", "masked", "exp010", "direct", "filtered", "true mask"]
        for axis, image, title in zip(axes[row_index], panels, titles):
            axis.imshow(image.T, cmap="gray", origin="lower", vmin=-1, vmax=1)
            axis.set_title(title if row_index == 0 else "")
            axis.axis("off")
        axes[row_index, 0].set_ylabel(Path(relative).stem, fontsize=7)
        for arrays_for_model in arrays.values():
            arrays_for_model.close()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    for name in MODEL_NAMES:
        parser.add_argument(f"--{name}-shard", type=Path, action="append", required=True)
    parser.add_argument("--swav-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, metadata = _load_manifest(args.manifest)
    roots = {name: list(getattr(args, f"{name}_shard")) for name in MODEL_NAMES}
    if any(len(value) != 2 for value in roots.values()):
        raise ValueError("each model requires exactly two shard roots")
    npz_paths = {}
    inference_metrics = {}
    for name in MODEL_NAMES:
        npz_paths[name], inference_metrics[name] = _load_model_run(name, roots[name], manifest)

    per_patch = []
    per_class = []
    real_crops: list[np.ndarray] = []
    fake_crops: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_NAMES}
    ordered_paths = list(manifest["selected_relative_paths"])
    for relative in ordered_paths:
        loaded = {name: np.load(npz_paths[name][relative], allow_pickle=False) for name in MODEL_NAMES}
        reference = np.asarray(loaded["exp010"]["original_input_t1c_xyz"], dtype=np.float32)
        seg = np.asarray(loaded["exp010"]["conditioning_seg_xyz"], dtype=np.uint8)
        union = seg > 0
        expected_seed = int(loaded["exp010"]["sample_seed"])
        for name in MODEL_NAMES:
            if int(loaded[name]["sample_seed"]) != expected_seed:
                raise ValueError(f"sample seed mismatch for {relative}")
            np.testing.assert_array_equal(loaded[name]["conditioning_seg_xyz"], seg)
            np.testing.assert_allclose(loaded[name]["condition_hist_64"], loaded["exp010"]["condition_hist_64"], atol=0, rtol=0)
            np.testing.assert_allclose(loaded[name]["original_input_t1c_xyz"], reference, atol=0, rtol=0)
        real_crops.extend(orthogonal_lesion_crops(reference, seg))
        meta = metadata[relative]
        for name in MODEL_NAMES:
            generated = np.asarray(loaded[name]["generated_t1c_xyz"], dtype=np.float32)
            if not np.isfinite(generated).all():
                raise ValueError(f"non-finite output: {name} {relative}")
            if float(np.max(np.abs(generated[~union] - reference[~union]))) != 0.0:
                raise ValueError(f"background invariance failed: {name} {relative}")
            row = {
                "model": name,
                "relative_path": relative,
                "subject_id": meta["subject_id"],
                "anchor_label": int(meta["anchor_label"]),
                "sample_role": meta["sample_role"],
                "sample_seed": expected_seed,
                "lesion_psnr": psnr(reference, generated, union),
                "lesion_ssim": lesion_crop_ssim(reference, generated, seg),
                "full_patch_psnr": psnr(reference, generated),
                "full_patch_ssim": float(structural_similarity(reference, generated, data_range=2.0, win_size=7)),
                "lesion_mae": float(np.abs(generated[union] - reference[union]).mean()),
            }
            per_patch.append(row)
            fake_crops[name].extend(orthogonal_lesion_crops(generated, seg))
            for label, label_name in LABEL_NAMES.items():
                mask = seg == label
                if mask.any():
                    per_class.append({
                        "model": name,
                        "relative_path": relative,
                        "subject_id": meta["subject_id"],
                        "anchor_label": int(meta["anchor_label"]),
                        "sample_role": meta["sample_role"],
                        "label": label,
                        "label_name": label_name,
                        "voxel_count": int(mask.sum()),
                        "hist_w1": _hist_w1(reference, generated, mask),
                    })
        for arrays in loaded.values():
            arrays.close()

    if len(real_crops) != 600 or any(len(value) != 600 for value in fake_crops.values()):
        raise RuntimeError("distribution metrics require exactly 600 real and generated crops")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_patch.csv", per_patch)
    _write_csv(output / "per_class.csv", per_class)

    cache = output / "feature_cache"
    cache.mkdir(exist_ok=True)
    inception = {}
    swav = {}
    real_inception_path = cache / "real_inception.npz"
    real_swav_path = cache / "real_swav.npz"
    if real_inception_path.is_file():
        inception["real"] = np.load(real_inception_path)["features"]
    else:
        inception["real"] = _inception_features(real_crops, args.device, args.batch_size)
        np.savez_compressed(real_inception_path, features=inception["real"])
    if real_swav_path.is_file():
        swav["real"] = np.load(real_swav_path)["features"]
    else:
        swav["real"] = _swav_features(real_crops, args.swav_weights, args.device, args.batch_size)
        np.savez_compressed(real_swav_path, features=swav["real"])
    distribution = {"models": {}, "real_split_baseline": {}}
    for name in MODEL_NAMES:
        inception_path = cache / f"{name}_inception.npz"
        swav_path = cache / f"{name}_swav.npz"
        inception[name] = np.load(inception_path)["features"] if inception_path.is_file() else _inception_features(fake_crops[name], args.device, args.batch_size)
        swav[name] = np.load(swav_path)["features"] if swav_path.is_file() else _swav_features(fake_crops[name], args.swav_weights, args.device, args.batch_size)
        if not inception_path.is_file():
            np.savez_compressed(inception_path, features=inception[name])
        if not swav_path.is_file():
            np.savez_compressed(swav_path, features=swav[name])
        distribution["models"][name] = {
            "fid": frechet_distance(inception["real"], inception[name]),
            "fsd_swav": frechet_distance(swav["real"], swav[name]),
            "kid": kid_repeated(inception["real"], inception[name]),
        }
    split_a = np.concatenate([np.arange(index * 3, index * 3 + 3) for index in range(0, 200, 2)])
    split_b = np.concatenate([np.arange(index * 3, index * 3 + 3) for index in range(1, 200, 2)])
    distribution["real_split_baseline"] = {
        "fid": frechet_distance(inception["real"][split_a], inception["real"][split_b]),
        "fsd_swav": frechet_distance(swav["real"][split_a], swav["real"][split_b]),
        "kid": kid_repeated(inception["real"][split_a], inception["real"][split_b]),
    }
    (output / "distribution_metrics.json").write_text(json.dumps(distribution, indent=2) + "\n", encoding="utf-8")

    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "selected_count": 200,
        "distribution_crop_count": 600,
        "test_accessed_count": 200,
        "unselected_test_accessed": False,
        "models": {},
        "distribution": distribution,
    }
    for name in MODEL_NAMES:
        model_rows = [row for row in per_patch if row["model"] == name]
        class_rows = [row for row in per_class if row["model"] == name]
        summary["models"][name] = {
            "checkpoint_step": EXPECTED[name][0],
            "checkpoint_sha256": EXPECTED[name][1],
            "patch_count": len(model_rows),
            "subject_count": len({row["subject_id"] for row in model_rows}),
            "paired_metrics": {
                key: {**_summary(row[key] for row in model_rows), "patient_bootstrap_95ci": patient_bootstrap_ci(model_rows, key)}
                for key in ("lesion_psnr", "lesion_ssim", "full_patch_psnr", "full_patch_ssim", "lesion_mae")
            },
            "hist_w1": {
                "macro": _summary(row["hist_w1"] for row in class_rows),
                "patient_equal_macro": patient_equal_summary(class_rows, "hist_w1"),
                "per_class": {
                    LABEL_NAMES[label]: patient_equal_summary(
                        [row for row in class_rows if row["label"] == label], "hist_w1"
                    )
                    for label in LABEL_NAMES
                },
            },
        }
    comparisons = []
    indexed = {(row["model"], row["relative_path"]): row for row in per_patch}
    for left, right in (("direct", "exp010"), ("filtered", "exp010"), ("filtered", "direct")):
        for relative in ordered_paths:
            for metric in ("lesion_psnr", "lesion_ssim", "full_patch_psnr", "full_patch_ssim", "lesion_mae"):
                row = indexed[(left, relative)]
                comparisons.append({
                    "left": left, "right": right, "metric": metric,
                    "relative_path": relative, "subject_id": row["subject_id"],
                    "delta_left_minus_right": float(row[metric]) - float(indexed[(right, relative)][metric]),
                })
    _write_csv(output / "paired_comparisons.csv", comparisons)
    comparison_summary = {}
    for left, right in (("direct", "exp010"), ("filtered", "exp010"), ("filtered", "direct")):
        pair_name = f"{left}_minus_{right}"
        comparison_summary[pair_name] = {}
        for metric in ("lesion_psnr", "lesion_ssim", "full_patch_psnr", "full_patch_ssim", "lesion_mae"):
            chosen = [
                {
                    "subject_id": row["subject_id"],
                    "delta": row["delta_left_minus_right"],
                }
                for row in comparisons
                if row["left"] == left and row["right"] == right and row["metric"] == metric
            ]
            comparison_summary[pair_name][metric] = patient_equal_summary(chosen, "delta")
    summary["paired_comparisons"] = comparison_summary
    _build_montage(manifest["selected_records"], npz_paths, output / "qa_montage.png")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
