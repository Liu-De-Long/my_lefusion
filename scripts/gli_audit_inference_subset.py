#!/usr/bin/env python3
"""Audit completed GLI inference shards and optionally compare a legacy run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CHANNEL_NAMES = ("NETC", "SNFH", "ET", "RC")
REGION_METRICS = (
    "healthy_brain_mae",
    "healthy_brain_p95_abs_change",
    "healthy_brain_max_abs",
    "healthy_brain_changed_fraction_gt_0p1",
    "outside_change_mae",
    "outside_change_p95_abs",
    "outside_change_max_abs",
    "outside_changed_fraction_gt_0p1",
    "boundary_outer_shell_mae",
    "boundary_outer_shell_p95_abs_change",
    "boundary_outer_shell_max_abs",
    "boundary_outer_shell_changed_fraction_gt_0p1",
    "boundary_jump_mean_increase",
    "boundary_jump_p95_increase",
    "lesion_change_mae",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _aggregate(records: list[dict]) -> dict:
    result = {}
    for key in REGION_METRICS:
        values = [float(item[key]) for item in records if item.get(key) is not None]
        result[key] = _summary(values)
    return result


def _load_run(name: str, shard_dirs: list[Path], selection: dict) -> dict:
    selected_records = selection["selected_records"]
    selected_paths = [str(item["relative_path"]) for item in selected_records]
    metadata = {str(item["relative_path"]): item for item in selected_records}
    expected_shards = [set(selected_paths[index::len(shard_dirs)]) for index in range(len(shard_dirs))]
    summaries = []
    samples = []
    actual_shards = []
    sample_dirs = {}
    for index, shard_dir in enumerate(shard_dirs):
        metrics_path = shard_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"{name} shard {index} has no completed metrics.json: {metrics_path}")
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        shard_samples = summary["samples"]
        paths = [str(item["source_relative_path"]) for item in shard_samples]
        if len(paths) != len(set(paths)):
            raise ValueError(f"{name} shard {index} contains duplicate paths")
        actual = set(paths)
        if actual != expected_shards[index]:
            raise ValueError(
                f"{name} shard {index} mismatch: missing={len(expected_shards[index] - actual)}, "
                f"unexpected={len(actual - expected_shards[index])}"
            )
        for item in shard_samples:
            path = str(item["source_relative_path"])
            item = dict(item)
            item.update(
                anchor_label=int(metadata[path]["anchor_label"]),
                sample_role=str(metadata[path]["sample_role"]),
                subject_id=str(metadata[path]["subject_id"]),
            )
            samples.append(item)
            sample_dirs[path] = shard_dir
        summaries.append(summary)
        actual_shards.append(actual)

    if any(actual_shards[i] & actual_shards[j] for i in range(len(actual_shards)) for j in range(i + 1, len(actual_shards))):
        raise ValueError(f"{name} shards overlap")
    if set().union(*actual_shards) != set(selected_paths):
        raise ValueError(f"{name} shard union does not equal the frozen subset")

    invariant_keys = (
        "checkpoint_sha256",
        "checkpoint_weights_key",
        "checkpoint_step",
        "cluster_sha256",
        "selection_manifest_sha256",
        "selection_shard_count",
        "lesion_channel_label_values",
        "lesion_channel_names",
        "model_calls_per_sample",
    )
    provenance = {key: summaries[0].get(key) for key in invariant_keys}
    for summary in summaries[1:]:
        for key, expected in provenance.items():
            if summary.get(key) != expected:
                raise ValueError(f"{name} shard provenance mismatch for {key}")
    if provenance["lesion_channel_label_values"] != [1, 2, 3, 4] or provenance["lesion_channel_names"] != list(CHANNEL_NAMES):
        raise ValueError(f"{name} lesion channel contract mismatch")
    if int(provenance["model_calls_per_sample"]) != 300:
        raise ValueError(f"{name} did not use the full 300-step schedule")

    for item in samples:
        if item["output_shape_dhw"] != [32, 64, 64] or item["output_shape_xyz"] != [64, 64, 32]:
            raise ValueError(f"{name} output shape mismatch: {item['source_relative_path']}")
        if item["internal_channels_shape_cdhw"] != [4, 32, 64, 64]:
            raise ValueError(f"{name} channel shape mismatch: {item['source_relative_path']}")
        if int(item["lesion_outside_support_voxels"]) != 0:
            raise ValueError(f"{name} lesion outside support: {item['source_relative_path']}")

    grouped = {"all": _aggregate(samples), "anchor_label": {}, "sample_role": {}}
    for label in (1, 2, 3, 4):
        grouped["anchor_label"][str(label)] = _aggregate([x for x in samples if x["anchor_label"] == label])
    for role in ("interior", "boundary"):
        grouped["sample_role"][role] = _aggregate([x for x in samples if x["sample_role"] == role])

    per_label = {}
    for label, channel_name in enumerate(CHANNEL_NAMES, start=1):
        patch_changes = []
        histogram_l1 = []
        histogram_wasserstein = []
        voxel_count = 0
        present_count = 0
        for item in samples:
            path = str(item["source_relative_path"])
            if int(item["lesion_channel_voxels"][label - 1]) == 0:
                continue
            present_count += 1
            stem = Path(path).stem
            npz_path = sample_dirs[path] / "generated_npz" / f"{stem}.npz"
            with np.load(npz_path, allow_pickle=False) as payload:
                generated = np.asarray(payload["generated_t1c_xyz"], dtype=np.float32)
                original = np.asarray(payload["input_t1c_xyz"], dtype=np.float32)
                segmentation = np.asarray(payload["conditioning_seg_xyz"], dtype=np.uint8)
            mask = segmentation == label
            voxel_count += int(mask.sum())
            patch_changes.append(float(np.abs(generated[mask] - original[mask]).mean()))
            hist = item.get("per_label_histogram", {}).get(str(label))
            if hist:
                histogram_l1.append(float(hist["l1"]))
                histogram_wasserstein.append(float(hist["wasserstein"]))
        per_label[str(label)] = {
            "name": channel_name,
            "present_patch_count": present_count,
            "voxel_count": voxel_count,
            "lesion_change_mae_per_patch": _summary(patch_changes),
            "histogram_l1": _summary(histogram_l1),
            "histogram_wasserstein": _summary(histogram_wasserstein),
        }

    anomaly_rows = []
    anomaly_keys = (
        "healthy_brain_mae",
        "outside_change_mae",
        "boundary_outer_shell_mae",
        "boundary_jump_p95_increase",
    )
    for key in anomaly_keys:
        eligible = [item for item in samples if item.get(key) is not None]
        for rank, item in enumerate(sorted(eligible, key=lambda x: float(x[key]), reverse=True)[:10], start=1):
            anomaly_rows.append(
                {
                    "metric": key,
                    "rank": rank,
                    "value": float(item[key]),
                    "source_relative_path": item["source_relative_path"],
                    "anchor_label": item["anchor_label"],
                    "sample_role": item["sample_role"],
                    "subject_id": item["subject_id"],
                }
            )

    multilabel = Counter(sum(int(value > 0) for value in item["lesion_channel_voxels"]) for item in samples)
    return {
        "name": name,
        "sample_count": len(samples),
        "shard_counts": [len(paths) for paths in actual_shards],
        "shard_overlap": 0,
        "shard_union_matches_frozen_subset": True,
        "provenance": provenance,
        "elapsed_seconds_by_shard": [float(item["elapsed_seconds"]) for item in summaries],
        "peak_memory_mib_by_shard": [float(item["peak_memory_mib"]) for item in summaries],
        "memory_stable_by_shard": [bool(item["memory_stable"]) for item in summaries],
        "anchor_label_counts": dict(sorted(Counter(item["anchor_label"] for item in samples).items())),
        "sample_role_counts": dict(sorted(Counter(item["sample_role"] for item in samples).items())),
        "subject_count": len({item["subject_id"] for item in samples}),
        "present_lesion_label_count_per_patch": dict(sorted(multilabel.items())),
        "lesion_outside_support_voxels_total": sum(int(item["lesion_outside_support_voxels"]) for item in samples),
        "metrics": grouped,
        "per_lesion_channel": per_label,
        "anomalies": anomaly_rows,
        "samples_by_path": {str(item["source_relative_path"]): item for item in samples},
    }


def _comparison(current: dict, legacy: dict) -> dict:
    shared = sorted(set(current["samples_by_path"]) & set(legacy["samples_by_path"]))
    result = {"paired_sample_count": len(shared), "metric_delta_current_minus_legacy": {}}
    for key in REGION_METRICS:
        deltas = []
        for path in shared:
            new_value = current["samples_by_path"][path].get(key)
            old_value = legacy["samples_by_path"][path].get(key)
            if new_value is not None and old_value is not None:
                deltas.append(float(new_value) - float(old_value))
        result["metric_delta_current_minus_legacy"][key] = _summary(deltas)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--legacy-shard-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    current = _load_run("exp006_official_repaint", args.shard_dir, selection)
    payload = {
        "schema_version": 1,
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": _sha256(args.selection_manifest),
        "current": {key: value for key, value in current.items() if key != "samples_by_path"},
    }
    if args.legacy_shard_dir:
        legacy = _load_run("exp005_hard_clamp", args.legacy_shard_dir, selection)
        payload["legacy"] = {key: value for key, value in legacy.items() if key != "samples_by_path"}
        payload["paired_comparison"] = _comparison(current, legacy)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "anomaly_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["metric", "rank", "value", "source_relative_path", "anchor_label", "sample_role", "subject_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(current["anomalies"])
    print(json.dumps({"sample_count": current["sample_count"], "audit": str(args.output / "audit.json")}, indent=2))


if __name__ == "__main__":
    main()
