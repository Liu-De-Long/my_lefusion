"""Compare the fixed eight-case QA outputs of short GLI generation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


VARIANTS = (
    "qa_original_real",
    "qa_original_cluster_first",
    "qa_original_cluster_last",
    "qa_union_cluster_first",
    "qa_union_cluster_last",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=seed-output-root; pass once for each experiment",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_progress(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 8:
        raise ValueError(f"expected exactly 8 QA records in {path}, got {len(records)}")
    return records


def histogram(values: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.linspace(-1.0, 1.0, 17))
    counts = counts.astype(np.float64)
    return counts / max(float(counts.sum()), 1.0)


def counterfactual_case_pass(first_path: Path, last_path: Path) -> tuple[bool, float, float]:
    with np.load(first_path, allow_pickle=False) as first, np.load(last_path, allow_pickle=False) as last:
        first_image = np.transpose(first["generated_t1c_xyz"], (2, 0, 1))
        last_image = np.transpose(last["generated_t1c_xyz"], (2, 0, 1))
        first_seg = np.transpose(first["conditioning_seg_xyz"], (2, 0, 1))
        last_seg = np.transpose(last["conditioning_seg_xyz"], (2, 0, 1))
        if not np.array_equal(first_seg, last_seg):
            raise ValueError(f"counterfactual masks differ: {first_path} / {last_path}")
        first_condition = first["condition_hist_64"].astype(np.float64)
        last_condition = last["condition_hist_64"].astype(np.float64)
        own = []
        swapped = []
        for label in (1, 2, 3, 4):
            mask = first_seg == label
            if not mask.any():
                continue
            start = (label - 1) * 16
            first_target = first_condition[start:start + 16]
            last_target = last_condition[start:start + 16]
            first_hist = histogram(first_image[mask])
            last_hist = histogram(last_image[mask])
            own.extend((np.abs(first_hist - first_target).sum(), np.abs(last_hist - last_target).sum()))
            swapped.extend((np.abs(first_hist - last_target).sum(), np.abs(last_hist - first_target).sum()))
    own_mean = float(np.mean(own))
    swapped_mean = float(np.mean(swapped))
    return own_mean < swapped_mean, own_mean, swapped_mean


def main() -> None:
    args = parse_args()
    runs = {}
    for item in args.run:
        name, separator, root = item.partition("=")
        if not separator or not name or name in runs:
            raise ValueError(f"invalid or duplicate --run: {item}")
        runs[name] = Path(root)
    if len(runs) != 3:
        raise ValueError(f"comparison requires exactly three runs, got {len(runs)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    indexed = {}
    for name, root in runs.items():
        indexed[name] = {}
        for variant in VARIANTS:
            variant_root = root / variant
            records = load_progress(variant_root / "progress.jsonl")
            indexed[name][variant] = {
                record["manifest"]["source_relative_path"]: record for record in records
            }
            for record in records:
                metrics = record["metrics"]
                rows.append(
                    {
                        "experiment": name,
                        "variant": variant,
                        "source_relative_path": record["manifest"]["source_relative_path"],
                        "lesion_mae": metrics["lesion_change_mae"],
                        "zero_fill_baseline_mae": metrics["zero_fill_baseline_lesion_mae"],
                        "lesion_mae_improvement_fraction": metrics["lesion_mae_improvement_fraction"],
                        "generated_lesion_mean_abs": metrics["generated_lesion_mean_abs"],
                        "generated_lesion_std": metrics["generated_lesion_std"],
                        "background_max_abs_change": metrics["background_max_abs_change"],
                    }
                )

    summary = {}
    for name, root in runs.items():
        real_records = indexed[name]["qa_original_real"]
        lesion_pass = sum(
            float(record["metrics"]["lesion_mae_improvement_fraction"]) >= 0.2
            and float(record["metrics"]["generated_lesion_mean_abs"]) >= 0.02
            and float(record["metrics"]["generated_lesion_std"]) >= 0.02
            for record in real_records.values()
        )
        hist_pass = 0
        union_pass = 0
        hist_pairs = []
        union_pairs = []
        for prefix, counter, storage in (
            ("qa_original_cluster", "hist", hist_pairs),
            ("qa_union_cluster", "union", union_pairs),
        ):
            first_records = indexed[name][f"{prefix}_first"]
            last_records = indexed[name][f"{prefix}_last"]
            for source_path in sorted(first_records):
                first_npz = root / f"{prefix}_first" / first_records[source_path]["manifest"]["generated_npz"]
                last_npz = root / f"{prefix}_last" / last_records[source_path]["manifest"]["generated_npz"]
                passed, own, swapped = counterfactual_case_pass(first_npz, last_npz)
                storage.append({"source_relative_path": source_path, "passed": passed, "own_l1": own, "swapped_l1": swapped})
                if counter == "hist":
                    hist_pass += int(passed)
                else:
                    union_pass += int(passed)
        background_max = max(float(row["background_max_abs_change"]) for row in rows if row["experiment"] == name)
        summary[name] = {
            "lesion_pass_count": lesion_pass,
            "hist_counterfactual_pass_count": hist_pass,
            "union_counterfactual_pass_count": union_pass,
            "background_max_abs_change": background_max,
            "passes_all_thresholds": bool(
                lesion_pass >= 6 and hist_pass >= 6 and union_pass >= 6 and background_max <= 1e-5
            ),
            "hist_pairs": hist_pairs,
            "union_pairs": union_pairs,
        }

    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    case_paths = sorted(next(iter(indexed.values()))["qa_original_real"])
    cell_width = 900
    title_height = 56
    rendered = []
    for source_path in case_paths:
        row_images = []
        for name, root in runs.items():
            record = indexed[name]["qa_original_real"][source_path]
            stem = Path(record["manifest"]["generated_npz"]).stem
            with Image.open(root / "qa_original_real" / "qa" / f"{stem}.png") as source:
                image = source.convert("RGB")
            height = round(image.height * cell_width / image.width)
            row_images.append(image.resize((cell_width, height), Image.Resampling.LANCZOS))
        rendered.append(row_images)
    cell_height = max(image.height for row in rendered for image in row)
    canvas = Image.new(
        "RGB", (cell_width * len(runs), title_height + cell_height * len(rendered)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(runs):
        draw.text((column * cell_width + 12, 18), name, fill="black")
    for row_index, row_images in enumerate(rendered):
        for column, image in enumerate(row_images):
            canvas.paste(image, (column * cell_width, title_height + row_index * cell_height))
    canvas.save(args.output_dir / "three_method_original_real_contact_sheet.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
