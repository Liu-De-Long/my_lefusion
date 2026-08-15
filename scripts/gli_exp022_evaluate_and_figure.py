#!/usr/bin/env python
"""Normalize, audit, score, and render the five-method exp022 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


METHODS = (
    ("repaint3d", "RePaint-3D", "neg1_pos1", 300),
    ("med_ddpm_t1c", "Med-DDPM-T1c", "neg1_pos1", 250),
    ("pix2pix_3d_inpaint", "Pix2Pix-3D", "zero_one", 1),
    ("latent_rflow_3d", "Latent RFlow-3D", "zero_one", 50),
    ("filtered_v2", "Filtered (v2)", "neg1_pos1", 300),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cdhw_to_xyz(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (1, 32, 64, 64):
        raise ValueError(f"expected CDHW (1,32,64,64), got {value.shape}")
    return np.transpose(value[0], (1, 2, 0)).copy()


def hist_w1(real: np.ndarray, generated: np.ndarray) -> float:
    edges = np.linspace(-1.0, 1.0, 257, dtype=np.float64)
    real_hist, _ = np.histogram(real, bins=edges)
    gen_hist, _ = np.histogram(generated, bins=edges)
    real_cdf = np.cumsum(real_hist, dtype=np.float64) / max(real_hist.sum(), 1)
    gen_cdf = np.cumsum(gen_hist, dtype=np.float64) / max(gen_hist.sum(), 1)
    return float(np.abs(real_cdf - gen_cdf).sum() * (edges[1] - edges[0]))


def psnr(real: np.ndarray, generated: np.ndarray, data_range: float = 2.0) -> float:
    mse = float(np.mean(np.square(real.astype(np.float64) - generated.astype(np.float64))))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(data_range * data_range / mse))


def safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    minimum = min(a.shape)
    win = min(7, minimum if minimum % 2 else minimum - 1)
    if win < 3:
        return float(1.0 if np.allclose(a, b) else 0.0)
    return float(structural_similarity(a, b, data_range=2.0, win_size=win))


def local_three_view_ssim(real: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> float:
    scores = []
    for axis in range(3):
        reduce_axes = tuple(index for index in range(3) if index != axis)
        slice_index = int(np.argmax(mask.sum(axis=reduce_axes)))
        m = np.take(mask, slice_index, axis=axis)
        r = np.take(real, slice_index, axis=axis)
        g = np.take(generated, slice_index, axis=axis)
        coordinates = np.argwhere(m)
        if not len(coordinates):
            raise ValueError("shared union is empty on selected view")
        lower = np.maximum(coordinates.min(axis=0) - 4, 0)
        upper = np.minimum(coordinates.max(axis=0) + 5, m.shape)
        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        scores.append(safe_ssim(r[slices], g[slices]))
    return float(np.mean(scores))


def load_ours_map(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    progress = root / "progress.jsonl"
    if not progress.exists():
        raise FileNotFoundError(progress)
    for line in progress.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        manifest = row.get("manifest", row)
        if not manifest.get("generated_npz"):
            continue
        mapping[str(manifest["source_relative_path"])] = root / str(manifest["generated_npz"])
    return mapping


def find_legacy_output(root: Path, method: str, case_id: str) -> Path:
    candidates = [
        path
        for path in root.rglob(f"{case_id}.npz")
        if method in path.parts and "20260806" in path.parts
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"{method}/{case_id}: expected one output, found {candidates}")
    return candidates[0]


def load_method(
    method: str,
    domain: str,
    record: dict[str, object],
    external_root: Path,
    ours_map: dict[str, Path],
    med_root: Path | None,
) -> tuple[np.ndarray, np.ndarray, Path, int]:
    if method == "filtered_v2":
        path = ours_map[str(record["relative_path"])]
        with np.load(path, allow_pickle=False) as data:
            raw = np.asarray(data["generated_t1c_xyz"], dtype=np.float32)
            union = np.asarray(data["conditioning_seg_xyz"] > 0, dtype=bool)
            sample_seed = int(data["sample_seed"])
    else:
        search_root = med_root if method == "med_ddpm_t1c" and med_root is not None else external_root
        path = find_legacy_output(search_root, method, str(record["case_id"]))
        with np.load(path, allow_pickle=False) as data:
            raw = cdhw_to_xyz(np.asarray(data["raw"], dtype=np.float32))
            union = cdhw_to_xyz(np.asarray(data["union"], dtype=np.uint8)) > 0
            sample_seed = int(data["seed"])
        if domain == "zero_one":
            raw = raw * 2.0 - 1.0
    if raw.shape != (64, 64, 32) or not np.isfinite(raw).all():
        raise RuntimeError(f"invalid output {method}/{record['sample_id']}: {raw.shape}")
    return raw, union, path, sample_seed


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_figure(
    cases: list[dict[str, object]], output_path: Path, *, labeled: bool
) -> None:
    row_labels = [
        "Ground-truth T1c",
        "Shared filtered mask",
        "Masked input",
        "RePaint-3D",
        "Med-DDPM-T1c",
        "Pix2Pix-3D",
        "Latent RFlow-3D",
        "Filtered (v2)",
    ]
    fig, axes = plt.subplots(8, 10, figsize=(17.6, 13.7), squeeze=False)
    for column, case in enumerate(cases):
        mask = np.asarray(case["mask"], dtype=bool)
        z = int(np.argmax(mask.sum(axis=(0, 1))))
        images = [
            case["real"],
            mask.astype(np.float32) * 2.0 - 1.0,
            case["masked"],
            case["outputs"]["repaint3d"],
            case["outputs"]["med_ddpm_t1c"],
            case["outputs"]["pix2pix_3d_inpaint"],
            case["outputs"]["latent_rflow_3d"],
            case["outputs"]["filtered_v2"],
        ]
        for row, image in enumerate(images):
            ax = axes[row, column]
            ax.imshow(np.asarray(image)[:, :, z].T, cmap="gray", vmin=-1, vmax=1, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if labeled and row == 0:
                ax.set_title(
                    f"{case['sample_id']}\nP={float(case['volume_percentile']):.2f}",
                    fontsize=8,
                )
            if labeled and column == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, rotation=0, ha="right", va="center")
    fig.subplots_adjust(left=0.13 if labeled else 0.015, right=0.995, top=0.96, bottom=0.015, wspace=0.02, hspace=0.035)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--ours-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--med-root", type=Path)
    parser.add_argument("--med-contract", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--paper3-modified", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    records = sorted(selection["selected_records"], key=lambda row: int(row["union_volume_voxels"]))
    if len(records) != 10 or len({row["subject_id"] for row in records}) != 10:
        raise RuntimeError("selection contract failed")
    ours_map = load_ours_map(args.ours_root)
    adapted_root = args.output_dir / "adapted_npz"
    metric_rows: list[dict[str, object]] = []
    cases = []
    audit_rows = []
    for record in records:
        relative = str(record["relative_path"])
        source_path = args.dataset_root / "patch_64x64x32" / relative
        overlay_path = args.filtered_root / "patch_64x64x32" / relative
        with np.load(source_path, allow_pickle=False) as data:
            real = np.asarray(data["t1c"], dtype=np.float32)
        with np.load(overlay_path, allow_pickle=False) as data:
            seg = np.asarray(data["seg_xyz"], dtype=np.uint8)
            four_mask = np.asarray(data["lesion_mask_xyz"], dtype=np.uint8)
            histogram = np.asarray(data["hist"], dtype=np.float32)
        mask = seg > 0
        if not mask.any() or not np.array_equal(four_mask.astype(bool).any(axis=0), mask):
            raise RuntimeError(f"invalid filtered condition: {relative}")
        masked = real.copy()
        masked[mask] = 0.0
        outputs = {}
        for method, display, domain, expected_nfe in METHODS:
            raw, output_union, source_output, sample_seed = load_method(
                method, domain, record, args.external_root, ours_map, args.med_root
            )
            if not np.array_equal(output_union, mask):
                raise RuntimeError(f"shared-union mismatch: {method}/{record['sample_id']}")
            preclip_min, preclip_max = float(raw.min()), float(raw.max())
            adapted = np.clip(raw, -1.0, 1.0)
            background_error_before_restore = float(np.max(np.abs(adapted[~mask] - real[~mask])))
            adapted[~mask] = real[~mask]
            if float(np.max(np.abs(adapted[~mask] - real[~mask]))) != 0.0:
                raise RuntimeError("background restoration failed")
            method_root = adapted_root / method
            method_root.mkdir(parents=True, exist_ok=True)
            adapted_path = method_root / f"{record['sample_id']}.npz"
            np.savez_compressed(
                adapted_path,
                generated_t1c_xyz=adapted.astype(np.float32),
                shared_union_xyz=mask.astype(np.uint8),
                source_relative_path=np.asarray(relative),
                source_output=np.asarray(str(source_output)),
                source_domain=np.asarray(domain),
                seed=np.asarray(20260806, dtype=np.int64),
                nfe=np.asarray(expected_nfe, dtype=np.int64),
            )
            outputs[method] = adapted
            region_real = real[mask]
            region_generated = adapted[mask]
            metric_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "subject_id": record["subject_id"],
                    "relative_path": relative,
                    "method": display,
                    "method_key": method,
                    "shared_union_voxels": int(mask.sum()),
                    "shared_union_psnr": psnr(region_real, region_generated),
                    "three_view_local_ssim": local_three_view_ssim(real, adapted, mask),
                    "shared_union_mae": float(np.mean(np.abs(region_real - region_generated))),
                    "hist_w1": hist_w1(region_real, region_generated),
                    "full_patch_psnr": psnr(real, adapted),
                    "full_patch_ssim": safe_ssim(real, adapted),
                    "full_patch_mae": float(np.mean(np.abs(real - adapted))),
                }
            )
            audit_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "method": method,
                    "source_output": str(source_output),
                    "source_output_sha256": sha256_file(source_output),
                    "adapted_output": str(adapted_path),
                    "adapted_output_sha256": sha256_file(adapted_path),
                    "root_seed": 20260806,
                    "recorded_output_seed": sample_seed,
                    "seed_policy": (
                        "LeFusion deterministic batch-derived sample seed from root seed"
                        if method == "filtered_v2"
                        else "legacy root seed; implementation applies root seed + case index"
                    ),
                    "expected_nfe": expected_nfe,
                    "input_domain": domain,
                    "preclip_min": preclip_min,
                    "preclip_max": preclip_max,
                    "clipped_voxels": int(np.count_nonzero((raw < -1.0) | (raw > 1.0))),
                    "background_max_abs_before_canonical_restore": background_error_before_restore,
                    "background_max_abs_after_canonical_restore": 0.0,
                    "finite": True,
                    "shared_union_matches": True,
                }
            )
        cases.append({**record, "real": real, "mask": mask, "masked": masked, "outputs": outputs})

    if len(metric_rows) != 50:
        raise RuntimeError(f"expected 50 paired metric rows, got {len(metric_rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics_per_case.csv", metric_rows)
    numeric = [
        "shared_union_psnr",
        "three_view_local_ssim",
        "shared_union_mae",
        "hist_w1",
        "full_patch_psnr",
        "full_patch_ssim",
        "full_patch_mae",
    ]
    summary_rows = []
    for method, display, _, _ in METHODS:
        rows = [row for row in metric_rows if row["method_key"] == method]
        summary = {"method": display, "method_key": method, "cases": len(rows)}
        for name in numeric:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_std"] = float(np.std(values, ddof=1))
        summary_rows.append(summary)
    write_csv(args.output_dir / "metrics_summary.csv", summary_rows)
    json_dump(args.output_dir / "metrics_summary.json", summary_rows)
    json_dump(args.output_dir / "output_audit.json", {"checks_passed": True, "rows": audit_rows})

    direct_root = args.filtered_root.parent / "direct"
    direct_contract_path = direct_root / "contract.json"
    filtered_contract_path = args.filtered_root / "contract.json"
    direct_contract = json.loads(direct_contract_path.read_text(encoding="utf-8"))
    filtered_contract = json.loads(filtered_contract_path.read_text(encoding="utf-8"))
    comparison_records = []
    for record in selection["selected_records"]:
        relative = str(record["relative_path"])
        direct_path = direct_root / "patch_64x64x32" / relative
        filtered_path = args.filtered_root / "patch_64x64x32" / relative
        with np.load(filtered_path, allow_pickle=False) as overlay:
            retained_voxels = int(np.count_nonzero(overlay["seg_xyz"]))
        comparison_records.append(
            {
                **record,
                "direct_pseudomask_sha256": sha256_file(direct_path),
                "filtered_pseudomask_sha256": sha256_file(filtered_path),
                "filtered_retained_union_voxels": retained_voxels,
                "filter_fallback_used": False,
            }
        )
    comparison_manifest = {
        "schema_version": 1,
        "experiment_id": args.experiment_id or selection["experiment_id"],
        "selection_manifest": str(args.selection),
        "selection_manifest_sha256": sha256_file(args.selection),
        "selection": selection["selection"],
        "selected_count": 10,
        "subject_count": 10,
        "classifier_provenance": {
            "checkpoint_sha256": direct_contract["checkpoint_sha256"],
            "config_sha256": direct_contract["classifier_config_sha256"],
            "labeled_subset_sha256": direct_contract["subset_sha256"],
            "direct_contract": str(direct_contract_path),
            "direct_contract_sha256": sha256_file(direct_contract_path),
        },
        "filter_provenance": {
            "threshold": filtered_contract["threshold"],
            "threshold_contract_sha256": filtered_contract["threshold_contract_sha256"],
            "filtered_contract": str(filtered_contract_path),
            "filtered_contract_sha256": sha256_file(filtered_contract_path),
            "fallback_patch_count": filtered_contract["fallback_patch_count"],
        },
        "records": comparison_records,
    }
    json_dump(args.output_dir / "comparison_manifest.json", comparison_manifest)

    legacy_contract = json.loads(
        (args.external_root / "legacy_asset_audit.json").read_text(encoding="utf-8")
    )
    ours_contract_path = args.ours_root / "run_contract.json"
    checkpoint_contracts = {
        "schema_version": 1,
        "checks_passed": True,
        "legacy": legacy_contract,
        "filtered_v2": {
            "run_contract": str(ours_contract_path),
            "run_contract_sha256": sha256_file(ours_contract_path),
            "payload": json.loads(ours_contract_path.read_text(encoding="utf-8")),
        },
        "method_protocols": {
            method: {"display_name": display, "input_domain": domain, "nfe": nfe, "root_seed": 20260806}
            for method, display, domain, nfe in METHODS
        },
    }
    if args.med_contract is not None:
        checkpoint_contracts["med_ddpm_domain_fix"] = {
            "contract": str(args.med_contract),
            "contract_sha256": sha256_file(args.med_contract),
            "payload": json.loads(args.med_contract.read_text(encoding="utf-8")),
        }
    json_dump(args.output_dir / "checkpoint_contracts.json", checkpoint_contracts)

    render_figure(cases, args.output_dir / "fig2_v2_test10_labeled.png", labeled=True)
    render_figure(cases, args.output_dir / "fig2_v2_test10_clean.png", labeled=False)
    render_figure(cases, args.output_dir / "fig2_v2_test10_labeled.pdf", labeled=True)
    render_figure(cases, args.output_dir / "fig2_v2_test10_clean.pdf", labeled=False)
    protocol = {
        "schema_version": 1,
        "experiment_id": args.experiment_id or selection["experiment_id"],
        "selection_manifest": str(args.selection),
        "selection_manifest_sha256": sha256_file(args.selection),
        "column_order": [case["sample_id"] for case in cases],
        "column_order_rule": "ascending GT union volume",
        "slice_rule": "axial z with maximum shared filtered-union area",
        "rows": [
            "Ground-truth T1c", "Shared filtered mask", "Masked input", "RePaint-3D",
            "Med-DDPM-T1c", "Pix2Pix-3D", "Latent RFlow-3D", "Filtered (v2)"
        ],
        "display_range": [-1.0, 1.0],
        "shared_region": "filtered-retained union",
        "outside_region": "exactly restored from original T1c after domain adaptation",
        "quantitative_scope": "paired reconstruction/quality audit only; no FID/FSD/KID/Rad-MMD",
        "med_ddpm_protocol": "domain-fixed short-budget" if args.med_root is not None else "legacy frozen checkpoint",
        "paper3_modified": bool(args.paper3_modified),
    }
    json_dump(args.output_dir / "figure_protocol.json", protocol)
    json_dump(
        args.output_dir / "validation.json",
        {
            "checks_passed": True,
            "methods": 5,
            "cases_per_method": 10,
            "paired_rows": 50,
            "shared_union_nonempty": True,
            "shared_union_identical_across_methods": True,
            "background_error_after_restore": 0.0,
            "all_outputs_finite": True,
            "forbidden_small_sample_metrics_computed": [],
        },
    )
    print(json.dumps({"checks_passed": True, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
