"""Audit same-case four-class exp012 outputs without mask or histogram label leakage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
BIN_EDGES = np.linspace(-1.0, 1.0, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def _load_records(root: Path) -> dict[int, dict[str, dict]]:
    by_label: dict[int, dict[str, dict]] = {}
    for label, name in LABELS.items():
        variant = root / f"target_{label}_{name.lower()}"
        progress_path = variant / "progress.jsonl"
        records = [
            json.loads(line)
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != 8:
            raise ValueError(f"{variant} must contain exactly 8 records, got {len(records)}")
        by_label[label] = {
            str(item["manifest"]["source_relative_path"]): item for item in records
        }
    paths = [set(items) for items in by_label.values()]
    if any(current != paths[0] for current in paths[1:]):
        raise ValueError("the four target classes do not contain the same frozen cases")
    return by_label


def _normalize_hist(values: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=BIN_EDGES)
    hist = hist.astype(np.float64)
    return hist / max(float(hist.sum()), 1.0)


def _glcm_features(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lesion_values = values[mask].astype(np.float64)
    mean = float(lesion_values.mean())
    std = float(lesion_values.std())
    normalized = (values.astype(np.float64) - mean) / max(std, 1e-6)
    quantized = np.clip(((normalized + 3.0) / 6.0 * 8).astype(np.int64), 0, 7)
    result: list[float] = []
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        pair_mask = mask[tuple(left)] & mask[tuple(right)]
        a = quantized[tuple(left)][pair_mask]
        b = quantized[tuple(right)][pair_mask]
        matrix = np.zeros((8, 8), dtype=np.float64)
        if a.size:
            np.add.at(matrix, (a, b), 1.0)
            np.add.at(matrix, (b, a), 1.0)
        matrix /= max(float(matrix.sum()), 1.0)
        i, j = np.indices(matrix.shape)
        contrast = float((matrix * (i - j) ** 2).sum())
        homogeneity = float((matrix / (1.0 + np.abs(i - j))).sum())
        energy = float((matrix**2).sum())
        nonzero = matrix[matrix > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum()) if nonzero.size else 0.0
        result.extend((contrast, homogeneity, energy, entropy))
    return np.asarray(result, dtype=np.float64)


def _leave_one_case_out(features: np.ndarray, labels: np.ndarray, cases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros_like(labels)
    margins = np.zeros(labels.shape[0], dtype=np.float64)
    for case in np.unique(cases):
        test = cases == case
        train = ~test
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0)
        scaled_train = (features[train] - mean) / np.maximum(std, 1e-6)
        scaled_test = (features[test] - mean) / np.maximum(std, 1e-6)
        centroids = {
            label: scaled_train[labels[train] == label].mean(axis=0)
            for label in LABELS
        }
        test_indices = np.flatnonzero(test)
        for row, index in zip(scaled_test, test_indices):
            distances = {label: float(np.linalg.norm(row - center)) for label, center in centroids.items()}
            ordered = sorted(distances, key=distances.get)
            predictions[index] = ordered[0]
            target = int(labels[index])
            margins[index] = min(value for label, value in distances.items() if label != target) - distances[target]
    return predictions, margins


def _save_case_sheet(path: Path, arrays: dict[int, dict[str, np.ndarray]], title: str) -> None:
    original = arrays[1]["original"]
    masked = arrays[1]["masked"]
    mask = arrays[1]["mask"]
    z = int(np.argmax(mask.transpose(2, 0, 1).sum(axis=(1, 2))))
    original_d = original.transpose(2, 0, 1)
    masked_d = masked.transpose(2, 0, 1)
    mask_d = mask.transpose(2, 0, 1)
    fig, axes = plt.subplots(1, 11, figsize=(27.5, 2.8))
    panels: list[tuple[np.ndarray, str, str]] = [
        (original_d[z], "original", "gray"),
        (masked_d[z], "masked", "gray"),
    ]
    for label, name in LABELS.items():
        generated = arrays[label]["generated"].transpose(2, 0, 1)
        panels.append((generated[z], name, "gray"))
        panels.append((np.abs(generated[z] - original_d[z]), f"|{name}-original|", "magma"))
    panels.append((mask_d[z].astype(np.uint8), "union mask", "gray"))
    for axis, (image, panel_title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, origin="lower", vmin=(-1 if cmap == "gray" else None), vmax=(1 if cmap == "gray" else None))
        axis.set_title(panel_title, fontsize=8)
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.input_root
    by_label = _load_records(root)
    cases = sorted(by_label[1])
    rows: list[dict] = []
    texture_rows: list[dict] = []
    features: list[np.ndarray] = []
    feature_labels: list[int] = []
    feature_cases: list[str] = []
    case_sheet_paths: list[Path] = []
    references: dict[int, np.ndarray] = {}
    for label, name in LABELS.items():
        first_record = by_label[label][cases[0]]
        npz_path = root / f"target_{label}_{name.lower()}" / first_record["manifest"]["generated_npz"]
        with np.load(npz_path) as data:
            block = data["condition_hist_64"][(label - 1) * 16 : label * 16].astype(np.float64)
        references[label] = block / max(float(block.sum()), 1e-12)
    for case_index, relative_path in enumerate(cases):
        arrays: dict[int, dict[str, np.ndarray]] = {}
        common_seed = None
        common_original = None
        common_masked = None
        common_source_seg = None
        for label, name in LABELS.items():
            record = by_label[label][relative_path]
            variant = root / f"target_{label}_{name.lower()}"
            npz_path = variant / record["manifest"]["generated_npz"]
            with np.load(npz_path) as data:
                generated = data["generated_t1c_xyz"].astype(np.float32)
                original = data["original_input_t1c_xyz"].astype(np.float32)
                masked = data["masked_input_t1c_xyz"].astype(np.float32)
                source_seg = data["source_conditioning_seg_xyz"].astype(np.uint8)
                target_seg = data["conditioning_seg_xyz"].astype(np.uint8)
                lesion_mask = data["lesion_mask_cdhw"].astype(np.uint8)
                condition = data["condition_hist_64"].astype(np.float64)
                sample_seed = int(data["sample_seed"])
                saved_target = int(data["target_anchor_label"])
            union = source_seg > 0
            if saved_target != label or not np.array_equal(target_seg[union], np.full(union.sum(), label, dtype=np.uint8)):
                raise ValueError(f"target label contract failed for {relative_path}, label {label}")
            if np.any(target_seg[~union]) or lesion_mask[label - 1].sum() != union.sum() or lesion_mask.sum() != union.sum():
                raise ValueError(f"mask-channel contract failed for {relative_path}, label {label}")
            active_blocks = [bool(np.any(condition[i * 16 : (i + 1) * 16])) for i in range(4)]
            if active_blocks != [i == label - 1 for i in range(4)]:
                raise ValueError(f"hist-block contract failed for {relative_path}, label {label}")
            if common_seed is None:
                common_seed, common_original, common_masked, common_source_seg = sample_seed, original, masked, source_seg
            elif sample_seed != common_seed or not np.array_equal(original, common_original) or not np.array_equal(masked, common_masked) or not np.array_equal(source_seg, common_source_seg):
                raise ValueError(f"same-case frozen-input contract failed for {relative_path}")
            generated_hist = _normalize_hist(generated[union])
            distances = {candidate: float(np.abs(generated_hist - references[candidate]).sum()) for candidate in LABELS}
            ordered = sorted(distances, key=distances.get)
            target_distance = distances[label]
            non_target_min = min(value for candidate, value in distances.items() if candidate != label)
            background_max = float(np.abs(generated[~union] - original[~union]).max())
            feature = _glcm_features(generated, union)
            rows.append(
                {
                    "case_index": case_index,
                    "source_relative_path": relative_path,
                    "target_label": label,
                    "target_name": name,
                    "sample_seed": sample_seed,
                    "hist_target_rank": ordered.index(label) + 1,
                    "hist_target_top1": int(ordered[0] == label),
                    "hist_target_l1": target_distance,
                    "hist_nearest_other_l1": non_target_min,
                    "hist_margin": non_target_min - target_distance,
                    "background_max_abs_change": background_max,
                    **{f"hist_l1_to_{LABELS[candidate].lower()}": distances[candidate] for candidate in LABELS},
                }
            )
            texture_rows.append(
                {
                    "case_index": case_index,
                    "source_relative_path": relative_path,
                    "target_label": label,
                    "target_name": name,
                    **{f"texture_{i:02d}": float(value) for i, value in enumerate(feature)},
                }
            )
            features.append(feature)
            feature_labels.append(label)
            feature_cases.append(relative_path)
            arrays[label] = {"generated": generated, "original": original, "masked": masked, "mask": union}
        sheet = root / "case_contact_sheets" / f"case_{case_index:02d}.png"
        _save_case_sheet(sheet, arrays, f"case {case_index:02d}: same input / mask / noise, four target classes")
        case_sheet_paths.append(sheet)
    feature_array = np.stack(features)
    label_array = np.asarray(feature_labels, dtype=np.int64)
    case_array = np.asarray(feature_cases)
    predictions, texture_margins = _leave_one_case_out(feature_array, label_array, case_array)
    for row, prediction, margin in zip(rows, predictions, texture_margins):
        row["texture_loco_prediction"] = int(prediction)
        row["texture_loco_correct"] = int(prediction == int(row["target_label"]))
        row["texture_loco_margin"] = float(margin)
    with (root / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "texture_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(texture_rows[0]))
        writer.writeheader()
        writer.writerows(texture_rows)
    per_class = {
        name: {
            "hist_top1": int(sum(row["hist_target_top1"] for row in rows if row["target_label"] == label)),
            "hist_mean_rank": float(np.mean([row["hist_target_rank"] for row in rows if row["target_label"] == label])),
            "hist_mean_margin": float(np.mean([row["hist_margin"] for row in rows if row["target_label"] == label])),
            "texture_loco_correct": int(sum(row["texture_loco_correct"] for row in rows if row["target_label"] == label)),
        }
        for label, name in LABELS.items()
    }
    summary = {
        "experiment_id": "20260808_exp014_gli_four_class_counterfactual_qa",
        "case_count": 8,
        "generation_count": 32,
        "same_case_same_input_mask_noise_contract": True,
        "histogram_target_top1_count": int(sum(row["hist_target_top1"] for row in rows)),
        "histogram_target_top1_fraction": float(np.mean([row["hist_target_top1"] for row in rows])),
        "histogram_mean_target_rank": float(np.mean([row["hist_target_rank"] for row in rows])),
        "histogram_mean_margin": float(np.mean([row["hist_margin"] for row in rows])),
        "texture_loco_accuracy": float(np.mean(predictions == label_array)),
        "texture_loco_positive_margin_count": int(np.sum(texture_margins > 0)),
        "background_max_abs_change": float(max(row["background_max_abs_change"] for row in rows)),
        "per_class": per_class,
        "limitations": [
            "Histogram ranking measures intensity-distribution control, not medical subtype semantics.",
            "Texture LOCO uses intensity-normalized GLCM features and no mask/hist metadata, but it is descriptive on 32 generated samples.",
            "A leakage-safe classifier ceiling on real held-out T1c lesions is still required before claiming semantic subtype control.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    case_images = []
    for sheet_path in case_sheet_paths:
        with Image.open(sheet_path) as source:
            case_images.append(source.convert("RGB"))
    width = max(image.width for image in case_images)
    height = sum(image.height for image in case_images)
    overview = Image.new("RGB", (width, height), "white")
    y = 0
    for image in case_images:
        overview.paste(image, (0, y))
        y += image.height
    overview.save(root / "four_class_contact_sheet.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
