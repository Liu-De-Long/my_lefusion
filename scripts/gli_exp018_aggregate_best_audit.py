"""Aggregate exp018 full-val loss and paired fixed-QA generation audits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


CHANNELS = ("NETC", "SNFH", "ET", "RC")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def _qa_summary(directory: Path) -> dict:
    metrics_path = directory / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    if len(samples) != 8:
        raise ValueError(f"fixed QA must contain 8 patches: {metrics_path}")
    lesion_mae = [float(row["lesion_change_mae"]) for row in samples]
    means = [float(row["generated_lesion_mean_abs"]) for row in samples]
    stds = [float(row["generated_lesion_std"]) for row in samples]
    background = [float(row["background_max_abs_change"]) for row in samples]
    per_class = {}
    for label, name in enumerate(CHANNELS, start=1):
        class_mae: list[float] = []
        histogram_l1: list[float] = []
        for row in samples:
            relative_path = str(row["source_relative_path"])
            npz_path = directory / "generated_npz" / f"{Path(relative_path).stem}.npz"
            with np.load(npz_path, allow_pickle=False) as arrays:
                generated = np.asarray(arrays["generated_t1c_xyz"], dtype=np.float32)
                original = np.asarray(arrays["input_t1c_xyz"], dtype=np.float32)
                segmentation = np.asarray(arrays["conditioning_seg_xyz"], dtype=np.uint8)
            mask = segmentation == label
            if mask.any():
                class_mae.append(float(np.abs(generated[mask] - original[mask]).mean()))
            histogram = row.get("per_label_histogram", {}).get(str(label))
            if histogram is not None:
                histogram_l1.append(float(histogram["l1"]))
        per_class[str(label)] = {
            "name": name,
            "lesion_mae": _summary(class_mae),
            "histogram_l1": _summary(histogram_l1),
        }
    return {
        "sample_count": len(samples),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "checkpoint_step": int(payload["checkpoint_step"]),
        "mask_source": payload["mask_source"],
        "mask_overlay_contract_sha256": payload.get("mask_overlay_contract_sha256"),
        "lesion_mae": _summary(lesion_mae),
        "histogram_l1": _summary(
            [
                float(value["l1"])
                for row in samples
                for value in row.get("per_label_histogram", {}).values()
            ]
        ),
        "nonzero_patch_rate": float(np.mean(np.asarray(means) > 1e-6)),
        "nonflat_patch_rate": float(np.mean(np.asarray(stds) > 1e-6)),
        "background_invariance": {
            "exact_patch_rate": float(np.mean(np.asarray(background) == 0.0)),
            "max_abs_change": float(max(background)),
        },
        "per_class": per_class,
    }


def _montage(paths: list[Path], output: Path) -> None:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + 1) // 2
    canvas = Image.new("RGB", (width * 2, (height + 28) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, image) in enumerate(zip(paths, images)):
        x = (index % 2) * width
        y = (index // 2) * (height + 28)
        canvas.paste(image, (x, y + 28))
        draw.text((x + 4, y + 6), path.parent.parent.name + "/" + path.stem, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    result = {"schema_version": 1, "routes": {}}
    montage_paths: list[Path] = []
    for route in ("direct", "filtered"):
        route_root = root / "outputs" / f"{route}_mask_fp32_50k"
        loss = json.loads((route_root / "best_validation/metrics.json").read_text(encoding="utf-8"))
        true_dir = route_root / "best_validation/true_mask_qa"
        overlay_dir = route_root / "best_validation/overlay_mask_qa"
        true_summary = _qa_summary(true_dir)
        overlay_summary = _qa_summary(overlay_dir)
        if true_summary["checkpoint_sha256"] != loss["checkpoint_sha256"]:
            raise ValueError(f"{route} true-QA checkpoint does not match loss audit")
        if overlay_summary["checkpoint_sha256"] != loss["checkpoint_sha256"]:
            raise ValueError(f"{route} overlay-QA checkpoint does not match loss audit")
        result["routes"][route] = {
            "full_val_loss": loss,
            "fixed_qa_true_mask": true_summary,
            "fixed_qa_overlay_mask": overlay_summary,
        }
        for directory in (true_dir, overlay_dir):
            montage_paths.extend(sorted((directory / "qa").glob("*.png")))
    if len(montage_paths) != 32:
        raise ValueError(f"expected 32 QA images, found {len(montage_paths)}")
    output = root / "outputs/best_validation"
    _montage(montage_paths, output / "qa_montage.png")
    _atomic_json(output / "summary.json", result)
    print(json.dumps({"summary": str(output / "summary.json"), "qa_images": 32}, indent=2))


if __name__ == "__main__":
    main()
