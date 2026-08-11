"""Deterministic selection manifests for GLI validation QA and test subsets."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence


LABEL_NAMES = {1: "netc", 2: "snfh", 3: "et", 4: "rc"}
QA_ROLES = ("interior", "boundary")


def _stable_digest(relative_path: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}\0{relative_path}".encode("utf-8")).hexdigest()


def _record_key(record: Mapping[str, object]) -> tuple[int, str]:
    return int(record["anchor_label"]), str(record["sample_role"])


def _record_summary(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "relative_path": str(record["relative_path"]),
        "case_id": str(record["case_id"]),
        "subject_id": str(record["subject_id"]),
        "anchor_label": int(record["anchor_label"]),
        "sample_role": str(record["sample_role"]),
    }


def _distribution(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    strata = Counter(f"{int(row['anchor_label'])}:{row['sample_role']}" for row in records)
    labels = Counter(str(int(row["anchor_label"])) for row in records)
    roles = Counter(str(row["sample_role"]) for row in records)
    return {
        "patches": len(records),
        "subjects": len({str(row["subject_id"]) for row in records}),
        "anchor_labels": dict(sorted(labels.items())),
        "sample_roles": dict(sorted(roles.items())),
        "strata": dict(sorted(strata.items())),
    }


def select_stratified_fraction(
    records: Sequence[Mapping[str, object]], *, fraction: float, seed: int
) -> tuple[list[int], dict[str, object]]:
    """Select exactly floor(N*fraction) with largest-remainder stratum quotas."""
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if not records:
        raise ValueError("cannot select from empty records")
    target = math.floor(len(records) * float(fraction))
    if target <= 0:
        raise ValueError("fraction selects zero records")
    selected, payload = select_stratified_count(records, count=target, seed=seed)
    payload.update(mode="stratified_fraction", fraction=float(fraction))
    return selected, payload


def select_stratified_count(
    records: Sequence[Mapping[str, object]], *, count: int, seed: int
) -> tuple[list[int], dict[str, object]]:
    """Select an exact count with largest-remainder label/role quotas."""
    if not records:
        raise ValueError("cannot select from empty records")
    target = int(count)
    if not 0 < target <= len(records):
        raise ValueError("count must be in [1, len(records)]")
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[_record_key(record)].append(index)
    exact = {key: len(indices) * target / len(records) for key, indices in grouped.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = target - sum(quotas.values())
    order = sorted(grouped, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    selected: list[int] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda index: (
                _stable_digest(str(records[index]["relative_path"]), seed),
                str(records[index]["relative_path"]),
            ),
        )
        selected.extend(ranked[: quotas[key]])
    selected.sort(
        key=lambda index: (
            _stable_digest(str(records[index]["relative_path"]), seed),
            str(records[index]["relative_path"]),
        )
    )
    selected_records = [records[index] for index in selected]
    payload = {
        "mode": "stratified_count",
        "count": target,
        "seed": int(seed),
        "target_count": target,
        "full_distribution": _distribution(records),
        "selected_distribution": _distribution(selected_records),
        "stratum_quotas": {
            f"{key[0]}:{key[1]}": quotas[key] for key in sorted(quotas)
        },
    }
    return selected, payload


def select_val_qa(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int,
    lesion_label_counts: Mapping[str, int] | None = None,
) -> tuple[list[int], dict[str, object]]:
    """Select one deterministic, preferably multi-label patch per label/role."""
    selected: list[int] = []
    lesion_label_counts = lesion_label_counts or {}
    for label in sorted(LABEL_NAMES):
        for role in QA_ROLES:
            candidates = [
                index
                for index, record in enumerate(records)
                if int(record["anchor_label"]) == label
                and str(record["sample_role"]) == role
            ]
            if not candidates:
                raise ValueError(f"val QA has no candidate for label={label}, role={role}")
            candidates.sort(
                key=lambda index: (
                    -int(lesion_label_counts.get(str(records[index]["relative_path"]), 0)),
                    _stable_digest(str(records[index]["relative_path"]), seed),
                    str(records[index]["relative_path"]),
                )
            )
            selected.append(candidates[0])
    chosen = [records[index] for index in selected]
    return selected, {
        "mode": "val_label_role_qa",
        "seed": int(seed),
        "target_count": len(selected),
        "full_distribution": _distribution(records),
        "selected_distribution": _distribution(chosen),
        "selected_lesion_label_counts": {
            str(record["relative_path"]): int(
                lesion_label_counts.get(str(record["relative_path"]), 0)
            )
            for record in chosen
        },
    }


def build_selection_manifest(
    records: Sequence[Mapping[str, object]],
    selected_indices: Sequence[int],
    *,
    selection: Mapping[str, object],
    split: str,
    patch_size_xyz: Sequence[int],
    shard_count: int,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    if int(shard_count) <= 0:
        raise ValueError("shard_count must be positive")
    chosen = [_record_summary(records[int(index)]) for index in selected_indices]
    paths = [str(row["relative_path"]) for row in chosen]
    if len(paths) != len(set(paths)):
        raise ValueError("selection contains duplicate relative paths")
    shards = [paths[index :: int(shard_count)] for index in range(int(shard_count))]
    if sorted(path for shard in shards for path in shard) != sorted(paths):
        raise RuntimeError("selection shards do not partition selected paths")
    return {
        "schema_version": 1,
        "split": str(split),
        "patch_size_xyz": [int(value) for value in patch_size_xyz],
        "selection": dict(selection),
        "selected_count": len(paths),
        "selected_records": chosen,
        "selected_relative_paths": paths,
        "subject_count": len({str(row["subject_id"]) for row in chosen}),
        "shard_count": int(shard_count),
        "shards": [
            {"index": index, "count": len(shard), "relative_paths": shard}
            for index, shard in enumerate(shards)
        ],
        "provenance": dict(provenance),
    }


def manifest_shard_paths(
    payload: Mapping[str, object], *, shard_index: int, shard_count: int
) -> list[str]:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported GLI selection manifest schema")
    if int(payload.get("shard_count", -1)) != int(shard_count):
        raise ValueError("selection manifest shard_count mismatch")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not 0 <= int(shard_index) < len(shards):
        raise ValueError("invalid selection manifest shard index")
    shard = shards[int(shard_index)]
    if not isinstance(shard, Mapping) or int(shard.get("index", -1)) != int(shard_index):
        raise ValueError("selection manifest shard metadata mismatch")
    paths = shard.get("relative_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("selection manifest shard is empty")
    return [str(path) for path in paths]
