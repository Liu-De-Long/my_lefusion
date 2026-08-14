#!/usr/bin/env python
"""Freeze ten patient-distinct, stratified GLI test patches for exp022."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


TARGET_QUANTILES = tuple((index + 0.5) / 10.0 for index in range(10))
STRATA = tuple(
    (label, role)
    for label in (1, 2, 3, 4)
    for role in ("boundary", "interior")
)
STRATUM_QUOTAS = {
    (1, "boundary"): 1,
    (1, "interior"): 1,
    (2, "boundary"): 2,
    (2, "interior"): 2,
    (3, "boundary"): 1,
    (3, "interior"): 1,
    (4, "boundary"): 1,
    (4, "interior"): 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(path: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}\0{path}".encode("utf-8")).hexdigest()


def read_test_records(
    dataset_root: Path, split_file: Path, *, expected_count: int, expected_subjects: int
) -> tuple[list[dict[str, object]], Path]:
    size_root = dataset_root / "patch_64x64x32"
    manifest_path = size_root / "manifest.csv"
    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    subject_split = {str(k): str(v) for k, v in split_payload["subject_split"].items()}
    records: list[dict[str, object]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            if subject_split[str(source["subject_id"])] != "test":
                continue
            row: dict[str, object] = dict(source)
            row["anchor_label"] = int(source["anchor_label"])
            row["union_volume_voxels"] = sum(
                int(source[f"label_{label}_voxels"]) for label in (1, 2, 3, 4)
            )
            row["source_npz"] = str(size_root / str(source["relative_path"]))
            records.append(row)
    subjects = {str(row["subject_id"]) for row in records}
    if len(records) != expected_count or len(subjects) != expected_subjects:
        raise RuntimeError(
            f"unexpected test population: patches={len(records)}, subjects={len(subjects)}"
        )
    ranked = sorted(
        records,
        key=lambda row: (
            int(row["union_volume_voxels"]),
            str(row["relative_path"]),
        ),
    )
    denominator = max(len(ranked) - 1, 1)
    for rank, row in enumerate(ranked):
        row["volume_rank_zero_based"] = rank
        row["volume_percentile"] = rank / denominator
    return records, manifest_path


def select_test10(
    records: list[dict[str, object]], *, seed: int, beam_width: int = 8000
) -> list[dict[str, object]]:
    """Use deterministic beam search to satisfy quotas, subjects and volume deciles."""
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {
        stratum: [] for stratum in STRATA
    }
    for row in records:
        grouped[(int(row["anchor_label"]), str(row["sample_role"]))].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                stable_digest(str(row["relative_path"]), seed),
                str(row["relative_path"]),
            )
        )

    initial_remaining = tuple(STRATUM_QUOTAS[stratum] for stratum in STRATA)
    # state = (cost, tie, remaining quotas, used subjects, chosen rows)
    states: list[
        tuple[float, tuple[str, ...], tuple[int, ...], frozenset[str], tuple[dict[str, object], ...]]
    ] = [(0.0, (), initial_remaining, frozenset(), ())]
    for quantile in TARGET_QUANTILES:
        expanded = []
        for cost, tie, remaining, used_subjects, chosen in states:
            for stratum_index, stratum in enumerate(STRATA):
                if remaining[stratum_index] <= 0:
                    continue
                candidates = sorted(
                    (
                        row
                        for row in grouped[stratum]
                        if str(row["subject_id"]) not in used_subjects
                    ),
                    key=lambda row: (
                        abs(float(row["volume_percentile"]) - quantile),
                        stable_digest(str(row["relative_path"]), seed),
                    ),
                )[:10]
                for row in candidates:
                    updated_remaining = list(remaining)
                    updated_remaining[stratum_index] -= 1
                    path = str(row["relative_path"])
                    selected = dict(row)
                    selected["selection_target_quantile"] = quantile
                    expanded.append(
                        (
                            cost + abs(float(row["volume_percentile"]) - quantile),
                            tie + (stable_digest(path, seed),),
                            tuple(updated_remaining),
                            used_subjects | {str(row["subject_id"])},
                            chosen + (selected,),
                        )
                    )
        if not expanded:
            raise RuntimeError(f"selection became infeasible at quantile={quantile}")
        best_by_key = {}
        for state in sorted(expanded, key=lambda item: (item[0], item[1])):
            key = (state[2], state[3])
            best_by_key.setdefault(key, state)
        states = sorted(best_by_key.values(), key=lambda item: (item[0], item[1]))[
            :beam_width
        ]
    feasible = [state for state in states if not any(state[2])]
    if not feasible:
        raise RuntimeError("selection did not satisfy all stratum quotas")
    selected = list(min(feasible, key=lambda item: (item[0], item[1]))[4])
    if len(selected) != 10 or len({str(row["subject_id"]) for row in selected}) != 10:
        raise RuntimeError("selection is not ten patient-distinct patches")
    return selected


def build_payload(
    records: list[dict[str, object]],
    selected: list[dict[str, object]],
    *,
    dataset_root: Path,
    manifest_path: Path,
    split_file: Path,
    seed: int,
) -> dict[str, object]:
    selected_records = []
    for sample_index, row in enumerate(selected, start=1):
        source_npz = Path(str(row["source_npz"]))
        selected_records.append(
            {
                "sample_id": f"S{sample_index:02d}",
                "relative_path": str(row["relative_path"]),
                "case_id": str(row["case_id"]),
                "subject_id": str(row["subject_id"]),
                "anchor_label": int(row["anchor_label"]),
                "anchor_name": str(row["anchor_name"]),
                "sample_role": str(row["sample_role"]),
                "union_volume_voxels": int(row["union_volume_voxels"]),
                "volume_rank_zero_based": int(row["volume_rank_zero_based"]),
                "volume_percentile": float(row["volume_percentile"]),
                "selection_target_quantile": float(row["selection_target_quantile"]),
                "source_npz_sha256": sha256_file(source_npz),
            }
        )
    strata = Counter(
        f"{row['anchor_label']}:{row['sample_role']}" for row in selected_records
    )
    labels = Counter(str(row["anchor_label"]) for row in selected_records)
    roles = Counter(str(row["sample_role"]) for row in selected_records)
    paths = [str(row["relative_path"]) for row in selected_records]
    return {
        # Keep the established v1 envelope so the existing multimodal rebuild
        # and inference readers accept it; exp022 adds richer optional fields.
        "schema_version": 1,
        "experiment_id": "20260815_exp022_gli_fig2_v2_test10_comparison",
        "split": "test",
        "patch_size_xyz": [64, 64, 32],
        "selection": {
            "mode": "patient_distinct_stratified_volume_deciles",
            "seed": int(seed),
            "target_count": 10,
            "target_quantiles": list(TARGET_QUANTILES),
            "label_quotas": {"1": 2, "2": 4, "3": 2, "4": 2},
            "role_quotas": {"boundary": 5, "interior": 5},
            "stratum_quotas": {
                f"{label}:{role}": count
                for (label, role), count in STRATUM_QUOTAS.items()
            },
            "objective": "minimize absolute GT-union volume-percentile distance",
            "tie_break": "sha256(seed + NUL + relative_path)",
            "visual_cherry_picking": False,
            "full_distribution": {
                "patches": len(records),
                "subjects": len({str(row["subject_id"]) for row in records}),
            },
            "selected_distribution": {
                "patches": len(selected_records),
                "subjects": len({str(row["subject_id"]) for row in selected_records}),
                "anchor_labels": dict(sorted(labels.items())),
                "sample_roles": dict(sorted(roles.items())),
                "strata": dict(sorted(strata.items())),
            },
        },
        "selected_count": 10,
        "subject_count": 10,
        "selected_records": selected_records,
        "selected_relative_paths": paths,
        "shard_count": 1,
        "shards": [{"index": 0, "count": 10, "relative_paths": paths}],
        "provenance": {
            "dataset_root": str(dataset_root),
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "split_file": str(split_file),
            "split_sha256": sha256_file(split_file),
            "selection_uses_generated_outputs": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--expected-count", type=int, default=1038)
    parser.add_argument("--expected-subjects", type=int, default=74)
    args = parser.parse_args()
    records, manifest_path = read_test_records(
        args.dataset_root,
        args.split_file,
        expected_count=args.expected_count,
        expected_subjects=args.expected_subjects,
    )
    selected = select_test10(records, seed=args.seed)
    payload = build_payload(
        records,
        selected,
        dataset_root=args.dataset_root,
        manifest_path=manifest_path,
        split_file=args.split_file,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
