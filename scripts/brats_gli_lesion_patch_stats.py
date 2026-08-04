#!/usr/bin/env python
"""Compute BraTS2024 GLI lesion statistics for patch-size design."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


MODALITIES = ("t1c", "t1n", "t2f", "t2w")
REGIONS = {
    "label_1_netc": (1,),
    "label_2_snfh": (2,),
    "label_3_et": (3,),
    "label_4_rc": (4,),
    "wt_without_rc": (1, 2, 3),
    "tc_without_rc": (1, 3),
    "abnormality_including_rc": (1, 2, 3, 4),
}
PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-cases", default=0, type=int)
    parser.add_argument("--skip-intensity", action="store_true")
    parser.add_argument(
        "--intensity-max-cases",
        default=0,
        type=int,
        help="Limit modality intensity statistics to the first N cases. 0 means all selected cases.",
    )
    parser.add_argument(
        "--intensity-sample-limit",
        default=200_000,
        type=int,
        help="Maximum lesion voxels sampled per case/region/modality for intensity percentiles.",
    )
    parser.add_argument("--seed", default=20260804, type=int)
    return parser.parse_args()


def case_dirs(split_dir: Path, max_cases: int) -> list[Path]:
    cases = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if max_cases > 0:
        cases = cases[:max_cases]
    return cases


def load_array(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def bbox(mask: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    coords = np.where(mask)
    if coords[0].size == 0:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0]
    starts = [int(c.min()) for c in coords]
    ends = [int(c.max()) + 1 for c in coords]
    sizes = [e - s for s, e in zip(starts, ends)]
    return starts, ends, sizes


def percentiles(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {f"p{p}": 0.0 for p in PERCENTILES}
    vals = np.percentile(arr, PERCENTILES)
    return {f"p{p}": round(float(v), 6) for p, v in zip(PERCENTILES, vals)}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_numeric(rows: list[dict], keys: list[str], group_key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    summary = []
    for group, group_rows in sorted(grouped.items()):
        out = {group_key: group, "n": len(group_rows)}
        for key in keys:
            values = [float(row[key]) for row in group_rows]
            for pname, pval in percentiles(values).items():
                out[f"{key}_{pname}"] = pval
            out[f"{key}_mean"] = round(float(np.mean(values)), 6) if values else 0.0
        summary.append(out)
    return summary


def recommend_patch(region_rows: list[dict], target_region: str) -> dict:
    rows = [row for row in region_rows if row["region"] == target_region and row["voxel_count"] > 0]
    if not rows:
        return {"region": target_region, "error": "no non-empty rows"}
    dims = {
        "bbox_x": [row["bbox_x"] for row in rows],
        "bbox_y": [row["bbox_y"] for row in rows],
        "bbox_z": [row["bbox_z"] for row in rows],
    }
    rec = {"region": target_region, "non_empty_cases": len(rows)}
    for key, values in dims.items():
        p = percentiles(values)
        rec[f"{key}_p90"] = p["p90"]
        rec[f"{key}_p95"] = p["p95"]
        rec[f"{key}_p99"] = p["p99"]
    rec["suggested_patch_cover_p90"] = [
        int(np.ceil(rec["bbox_x_p90"] / 16) * 16),
        int(np.ceil(rec["bbox_y_p90"] / 16) * 16),
        int(np.ceil(rec["bbox_z_p90"] / 8) * 8),
    ]
    rec["suggested_patch_cover_p95"] = [
        int(np.ceil(rec["bbox_x_p95"] / 16) * 16),
        int(np.ceil(rec["bbox_y_p95"] / 16) * 16),
        int(np.ceil(rec["bbox_z_p95"] / 8) * 8),
    ]
    return rec


def markdown_report(summary: dict) -> str:
    lines = [
        "# BraTS2024 GLI 病灶统计与 patch 建议",
        "",
        "## 数据范围",
        "",
        f"- 数据路径：`{summary['data_root']}`",
        f"- split：`{summary['split']}`",
        f"- 病例数：`{summary['case_count']}`",
        "",
        "## patch 初步建议",
        "",
    ]
    for rec in summary["patch_recommendations"]:
        if "error" in rec:
            lines.append(f"- `{rec['region']}`：无有效病灶。")
            continue
        lines.append(
            "- `{region}`：p90 bbox 约 `{p90x} x {p90y} x {p90z}`，"
            "p95 bbox 约 `{p95x} x {p95y} x {p95z}`，"
            "建议覆盖 p90 的 patch 为 `{r90}`，覆盖 p95 的 patch 为 `{r95}`。".format(
                region=rec["region"],
                p90x=rec["bbox_x_p90"],
                p90y=rec["bbox_y_p90"],
                p90z=rec["bbox_z_p90"],
                p95x=rec["bbox_x_p95"],
                p95y=rec["bbox_y_p95"],
                p95z=rec["bbox_z_p95"],
                r90=" x ".join(map(str, rec["suggested_patch_cover_p90"])),
                r95=" x ".join(map(str, rec["suggested_patch_cover_p95"])),
            )
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `case_region_stats.csv`：每个病例、每个病灶区域的体素量和 bbox。",
            "- `region_summary.csv`：按病灶区域聚合后的体素量和 bbox 分位数。",
            "- `modality_intensity_stats.csv`：每个病例、区域、模态的病灶强度统计。",
            "- `modality_intensity_summary.csv`：按区域和模态聚合后的强度统计。",
            "- `summary.json`：机器可读汇总和 patch 建议。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    split_dir = args.data_root / args.split
    args.out_dir.mkdir(parents=True, exist_ok=True)

    region_rows: list[dict] = []
    intensity_rows: list[dict] = []
    shape_rows: list[dict] = []

    cases = case_dirs(split_dir, args.max_cases)
    for case_index, case_dir in enumerate(cases, start=1):
        case_id = case_dir.name
        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        if not seg_path.exists():
            continue

        seg_img = nib.load(str(seg_path))
        seg = np.asanyarray(seg_img.dataobj)
        spacing = tuple(float(v) for v in seg_img.header.get_zooms()[:3])
        voxel_volume = float(np.prod(spacing))

        shape_rows.append(
            {
                "case_id": case_id,
                "shape_x": int(seg.shape[0]),
                "shape_y": int(seg.shape[1]),
                "shape_z": int(seg.shape[2]),
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
            }
        )

        region_masks = {}
        for region_name, labels in REGIONS.items():
            mask = np.isin(seg, labels)
            starts, ends, sizes = bbox(mask)
            voxel_count = int(mask.sum())
            region_masks[region_name] = mask
            region_rows.append(
                {
                    "case_id": case_id,
                    "region": region_name,
                    "labels": "+".join(map(str, labels)),
                    "voxel_count": voxel_count,
                    "volume_mm3": round(voxel_count * voxel_volume, 6),
                    "bbox_x": sizes[0],
                    "bbox_y": sizes[1],
                    "bbox_z": sizes[2],
                    "bbox_start_x": starts[0],
                    "bbox_start_y": starts[1],
                    "bbox_start_z": starts[2],
                    "bbox_end_x": ends[0],
                    "bbox_end_y": ends[1],
                    "bbox_end_z": ends[2],
                }
            )

        should_compute_intensity = not args.skip_intensity and (
            args.intensity_max_cases <= 0 or case_index <= args.intensity_max_cases
        )
        if not should_compute_intensity:
            continue

        for modality in MODALITIES:
            image_path = case_dir / f"{case_id}-{modality}.nii.gz"
            if not image_path.exists():
                continue
            image = load_array(image_path)
            for region_name, mask in region_masks.items():
                if not mask.any():
                    values = np.asarray([], dtype=np.float32)
                else:
                    values = image[mask]
                    if values.size > args.intensity_sample_limit > 0:
                        indices = rng.choice(values.size, size=args.intensity_sample_limit, replace=False)
                        values = values[indices]
                stats = percentiles(values)
                intensity_rows.append(
                    {
                        "case_id": case_id,
                        "region": region_name,
                        "modality": modality,
                        "sampled_voxels": int(values.size),
                        "mean": round(float(np.mean(values)), 6) if values.size else 0.0,
                        "std": round(float(np.std(values)), 6) if values.size else 0.0,
                        **stats,
                    }
                )

        if case_index % 50 == 0:
            print(f"processed {case_index}/{len(cases)} cases", flush=True)

    write_csv(args.out_dir / "case_shape_stats.csv", shape_rows)
    write_csv(args.out_dir / "case_region_stats.csv", region_rows)
    write_csv(
        args.out_dir / "region_summary.csv",
        summarize_numeric(region_rows, ["voxel_count", "volume_mm3", "bbox_x", "bbox_y", "bbox_z"], "region"),
    )
    write_csv(args.out_dir / "modality_intensity_stats.csv", intensity_rows)

    modality_grouped = []
    for region in REGIONS:
        for modality in MODALITIES:
            rows = [r for r in intensity_rows if r["region"] == region and r["modality"] == modality]
            if not rows:
                continue
            out = {"region": region, "modality": modality, "n": len(rows)}
            for key in ("mean", "std", "p1", "p5", "p50", "p95", "p99"):
                vals = [float(r[key]) for r in rows]
                out[f"{key}_mean"] = round(float(np.mean(vals)), 6)
                out[f"{key}_p50"] = round(float(np.percentile(vals, 50)), 6)
                out[f"{key}_p95"] = round(float(np.percentile(vals, 95)), 6)
            modality_grouped.append(out)
    write_csv(args.out_dir / "modality_intensity_summary.csv", modality_grouped)

    summary = {
        "data_root": str(args.data_root),
        "split": args.split,
        "case_count": len(cases),
        "skip_intensity": bool(args.skip_intensity),
        "intensity_max_cases": int(args.intensity_max_cases),
        "regions": REGIONS,
        "modalities": MODALITIES,
        "patch_recommendations": [
            recommend_patch(region_rows, "abnormality_including_rc"),
            recommend_patch(region_rows, "wt_without_rc"),
            recommend_patch(region_rows, "tc_without_rc"),
            recommend_patch(region_rows, "label_1_netc"),
            recommend_patch(region_rows, "label_2_snfh"),
            recommend_patch(region_rows, "label_3_et"),
            recommend_patch(region_rows, "label_4_rc"),
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "patch_recommendations.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary["patch_recommendations"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
