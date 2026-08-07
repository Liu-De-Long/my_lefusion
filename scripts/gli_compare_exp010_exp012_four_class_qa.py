"""Pair frozen exp010/exp012 same-case four-class QA outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
MODELS = ("exp010", "exp012")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp010-root", type=Path, required=True)
    parser.add_argument("--exp012-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _record_map(root: Path) -> dict[tuple[str, int], tuple[Path, dict]]:
    records: dict[tuple[str, int], tuple[Path, dict]] = {}
    for label, name in LABELS.items():
        variant = root / f"target_{label}_{name.lower()}"
        items = [
            json.loads(line)
            for line in (variant / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(items) != 8:
            raise ValueError(f"{variant} must contain 8 records, got {len(items)}")
        for item in items:
            key = (str(item["manifest"]["source_relative_path"]), label)
            if key in records:
                raise ValueError(f"duplicate record: {key}")
            records[key] = (variant / item["manifest"]["generated_npz"], item)
    return records


def _metric_map(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = _read_csv(root / "case_metrics.csv")
    result = {(row["source_relative_path"], int(row["target_label"])): row for row in rows}
    if len(rows) != 32 or len(result) != 32:
        raise ValueError(f"{root}/case_metrics.csv must contain 32 unique rows")
    return result


def _winner(left: float, right: float, higher_is_better: bool = True, tolerance: float = 1e-12) -> str:
    delta = left - right
    if abs(delta) <= tolerance:
        return "tie"
    if higher_is_better:
        return "exp010" if delta > 0 else "exp012"
    return "exp010" if delta < 0 else "exp012"


def _confusion(rows: list[dict[str, str]], prediction_field: str) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for row in rows:
        target = int(row["target_label"])
        if prediction_field == "hist":
            prediction = min(
                LABELS,
                key=lambda label: float(row[f"hist_l1_to_{LABELS[label].lower()}"]),
            )
        else:
            prediction = int(row[prediction_field])
        matrix[target - 1, prediction - 1] += 1
    return matrix


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target\\predicted", *LABELS.values()])
        for label, name in LABELS.items():
            writer.writerow([name, *matrix[label - 1].tolist()])


def _load_arrays(path: Path) -> dict[str, np.ndarray | int]:
    with np.load(path) as data:
        return {
            "generated": data["generated_t1c_xyz"].astype(np.float32),
            "original": data["original_input_t1c_xyz"].astype(np.float32),
            "masked": data["masked_input_t1c_xyz"].astype(np.float32),
            "source_seg": data["source_conditioning_seg_xyz"].astype(np.uint8),
            "target_seg": data["conditioning_seg_xyz"].astype(np.uint8),
            "lesion_mask": data["lesion_mask_cdhw"].astype(np.uint8),
            "condition_hist": data["condition_hist_64"].astype(np.float64),
            "sample_seed": int(data["sample_seed"]),
            "target_label": int(data["target_anchor_label"]),
        }


def _validate_pair(left: dict, right: dict, key: tuple[str, int]) -> None:
    for field in ("original", "masked", "source_seg", "target_seg", "lesion_mask", "condition_hist"):
        if not np.array_equal(left[field], right[field]):
            raise ValueError(f"paired contract differs at {key}: {field}")
    for field in ("sample_seed", "target_label"):
        if left[field] != right[field]:
            raise ValueError(f"paired contract differs at {key}: {field}")


def _save_case_sheet(path: Path, case: str, model_arrays: dict[str, dict[int, dict]]) -> None:
    reference = model_arrays["exp010"][1]
    original = reference["original"]
    masked = reference["masked"]
    union = reference["source_seg"] > 0
    z = int(np.argmax(union.transpose(2, 0, 1).sum(axis=(1, 2))))
    original_d = original.transpose(2, 0, 1)
    masked_d = masked.transpose(2, 0, 1)
    union_d = union.transpose(2, 0, 1)
    fig, axes = plt.subplots(2, 11, figsize=(27.5, 5.6))
    for row_index, model in enumerate(MODELS):
        panels: list[tuple[np.ndarray, str, str]] = [
            (original_d[z], "original", "gray"),
            (masked_d[z], "masked", "gray"),
        ]
        for label, name in LABELS.items():
            generated = model_arrays[model][label]["generated"].transpose(2, 0, 1)
            panels.append((generated[z], name, "gray"))
            panels.append((np.abs(generated[z] - original_d[z]), f"|{name}-original|", "magma"))
        panels.append((union_d[z].astype(np.uint8), "union mask", "gray"))
        for axis, (image, title, cmap) in zip(axes[row_index], panels):
            axis.imshow(
                image,
                cmap=cmap,
                origin="lower",
                vmin=(-1 if cmap == "gray" else None),
                vmax=(1 if cmap == "gray" else None),
            )
            axis.set_title(title, fontsize=8)
            axis.axis("off")
        axes[row_index, 0].set_ylabel(model, fontsize=10)
    fig.suptitle(f"{case}: same input / union / hist / sampling seed", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _append_overview(paths: list[Path], output: Path) -> None:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    canvas = Image.new("RGB", (max(image.width for image in images), sum(image.height for image in images)), "white")
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roots = {"exp010": args.exp010_root, "exp012": args.exp012_root}
    metrics = {model: _metric_map(root) for model, root in roots.items()}
    records = {model: _record_map(root) for model, root in roots.items()}
    keys = sorted(metrics["exp010"])
    if set(keys) != set(metrics["exp012"]) or set(keys) != set(records["exp010"]) or set(keys) != set(records["exp012"]):
        raise ValueError("exp010 and exp012 do not contain the same 32 case/class pairs")

    paired_rows: list[dict] = []
    arrays_by_case: dict[str, dict[str, dict[int, dict]]] = {}
    for case, label in keys:
        arrays = {model: _load_arrays(records[model][(case, label)][0]) for model in MODELS}
        _validate_pair(arrays["exp010"], arrays["exp012"], (case, label))
        arrays_by_case.setdefault(case, {model: {} for model in MODELS})
        for model in MODELS:
            arrays_by_case[case][model][label] = arrays[model]
        left, right = metrics["exp010"][(case, label)], metrics["exp012"][(case, label)]
        paired_rows.append(
            {
                "source_relative_path": case,
                "target_label": label,
                "target_name": LABELS[label],
                "sample_seed": int(left["sample_seed"]),
                "exp010_hist_rank": int(left["hist_target_rank"]),
                "exp012_hist_rank": int(right["hist_target_rank"]),
                "hist_rank_winner": _winner(float(left["hist_target_rank"]), float(right["hist_target_rank"]), False),
                "exp010_hist_top1": int(left["hist_target_top1"]),
                "exp012_hist_top1": int(right["hist_target_top1"]),
                "exp010_hist_margin": float(left["hist_margin"]),
                "exp012_hist_margin": float(right["hist_margin"]),
                "hist_margin_winner": _winner(float(left["hist_margin"]), float(right["hist_margin"])),
                "exp010_texture_correct": int(left["texture_loco_correct"]),
                "exp012_texture_correct": int(right["texture_loco_correct"]),
                "exp010_texture_margin": float(left["texture_loco_margin"]),
                "exp012_texture_margin": float(right["texture_loco_margin"]),
                "texture_margin_winner": _winner(float(left["texture_loco_margin"]), float(right["texture_loco_margin"])),
                "exp010_background_max_abs_change": float(left["background_max_abs_change"]),
                "exp012_background_max_abs_change": float(right["background_max_abs_change"]),
            }
        )

    with (args.output_root / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    summaries: dict[str, dict] = {}
    for model in MODELS:
        rows = list(metrics[model].values())
        hist_matrix = _confusion(rows, "hist")
        texture_matrix = _confusion(rows, "texture_loco_prediction")
        _write_matrix(args.output_root / f"{model}_hist_confusion.csv", hist_matrix)
        _write_matrix(args.output_root / f"{model}_texture_confusion.csv", texture_matrix)
        summaries[model] = {
            "hist_top1": int(np.trace(hist_matrix)),
            "hist_mean_target_rank": float(np.mean([float(row["hist_target_rank"]) for row in rows])),
            "hist_mean_margin": float(np.mean([float(row["hist_margin"]) for row in rows])),
            "texture_top1": int(np.trace(texture_matrix)),
            "texture_mean_margin": float(np.mean([float(row["texture_loco_margin"]) for row in rows])),
            "background_max_abs_change": float(max(float(row["background_max_abs_change"]) for row in rows)),
            "hist_confusion": hist_matrix.tolist(),
            "texture_confusion": texture_matrix.tolist(),
            "per_class": {
                name: {
                    "hist_top1": int(hist_matrix[label - 1, label - 1]),
                    "texture_top1": int(texture_matrix[label - 1, label - 1]),
                    "hist_mean_margin": float(np.mean([float(row["hist_margin"]) for row in rows if int(row["target_label"]) == label])),
                    "texture_mean_margin": float(np.mean([float(row["texture_loco_margin"]) for row in rows if int(row["target_label"]) == label])),
                }
                for label, name in LABELS.items()
            },
        }

    per_case_rows: list[dict] = []
    for case in sorted(arrays_by_case):
        case_rows = [row for row in paired_rows if row["source_relative_path"] == case]
        left_hist = float(np.mean([row["exp010_hist_margin"] for row in case_rows]))
        right_hist = float(np.mean([row["exp012_hist_margin"] for row in case_rows]))
        left_texture = float(np.mean([row["exp010_texture_margin"] for row in case_rows]))
        right_texture = float(np.mean([row["exp012_texture_margin"] for row in case_rows]))
        per_case_rows.append(
            {
                "source_relative_path": case,
                "exp010_hist_top1": sum(row["exp010_hist_top1"] for row in case_rows),
                "exp012_hist_top1": sum(row["exp012_hist_top1"] for row in case_rows),
                "exp010_hist_mean_margin": left_hist,
                "exp012_hist_mean_margin": right_hist,
                "hist_margin_winner": _winner(left_hist, right_hist),
                "exp010_texture_correct": sum(row["exp010_texture_correct"] for row in case_rows),
                "exp012_texture_correct": sum(row["exp012_texture_correct"] for row in case_rows),
                "exp010_texture_mean_margin": left_texture,
                "exp012_texture_mean_margin": right_texture,
                "texture_margin_winner": _winner(left_texture, right_texture),
            }
        )
    with (args.output_root / "per_case_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case_rows[0]))
        writer.writeheader()
        writer.writerows(per_case_rows)

    sheet_paths: list[Path] = []
    for case_index, case in enumerate(sorted(arrays_by_case)):
        sheet = args.output_root / "case_contact_sheets" / f"case_{case_index:02d}.png"
        _save_case_sheet(sheet, case, arrays_by_case[case])
        sheet_paths.append(sheet)
    _append_overview(sheet_paths, args.output_root / "exp010_exp012_four_class_contact_sheet.png")

    paired_wins = {
        field: dict(Counter(row[field] for row in paired_rows))
        for field in ("hist_rank_winner", "hist_margin_winner", "texture_margin_winner")
    }
    summary = {
        "experiment_id": "20260808_exp016_gli_exp010_exp012_four_class_comparison",
        "case_count": 8,
        "paired_generation_count": 32,
        "new_exp010_generation_count": 32,
        "reused_exp012_generation_count": 32,
        "same_input_union_hist_seed_contract": True,
        "models": summaries,
        "paired_wins": paired_wins,
        "per_case_wins": {
            field: dict(Counter(row[field] for row in per_case_rows))
            for field in ("hist_margin_winner", "texture_margin_winner")
        },
        "limitations": [
            "Histogram Top-1 measures intensity-distribution control, not medical subtype semantics.",
            "GLCM LOCO is fit and evaluated only on each model's 32 generated samples.",
            "The comparison is paired and leakage-audited, but it does not establish real-lesion semantic validity.",
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
