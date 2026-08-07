"""Leak-safe data contract and labeled-subset selection for the p64 classifier."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_VALUES = (1, 2, 3, 4)
LABEL_NAMES = ("NETC", "SNFH", "ET", "RC")
PATCH_SIZE_XYZ = (64, 64, 32)
PATCH_SIZE_DHW = (32, 64, 64)
MODALITY_KEYS = ("t1c", "t1n", "t2f", "t2w")
NON_INPUT_NPZ_KEYS = frozenset({"seg", "hist", "affine"})
SAFE_SAMPLE_KEYS = frozenset(
    {
        "image",
        "total_mask",
        "target",
        "case_id",
        "subject_id",
        "relative_path",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_subject_split(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("subject_split")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"split file has no subject_split mapping: {path}")
    result = {str(key): str(value) for key, value in mapping.items()}
    invalid = sorted(set(result.values()).difference({"train", "val", "test"}))
    if invalid:
        raise ValueError(f"invalid split names: {invalid}")
    return result


def load_manifest_records(
    dataset_root: str | Path,
    split_file: str | Path,
    *,
    split: str | None,
    patch_size_xyz: Sequence[int] = PATCH_SIZE_XYZ,
) -> tuple[list[dict[str, str]], Path]:
    patch_size_xyz = tuple(int(value) for value in patch_size_xyz)
    if patch_size_xyz != PATCH_SIZE_XYZ:
        raise ValueError(f"exp009 only supports p64 {PATCH_SIZE_XYZ}, got {patch_size_xyz}")
    size_root = Path(dataset_root) / "patch_64x64x32"
    manifest_path = size_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    subject_split = load_subject_split(split_file)
    records: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "relative_path",
            "case_id",
            "subject_id",
            "patch_size_xyz",
            "anchor_label",
            "sample_role",
            "label_1_voxels",
            "label_2_voxels",
            "label_3_voxels",
            "label_4_voxels",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest missing fields: {sorted(missing)}")
        for raw in reader:
            row = dict(raw)
            relative_path = row["relative_path"]
            if relative_path in seen_paths:
                raise ValueError(f"duplicate manifest path: {relative_path}")
            seen_paths.add(relative_path)
            subject_id = row["subject_id"]
            if subject_id not in subject_split:
                raise ValueError(f"subject missing from frozen split: {subject_id}")
            effective_split = subject_split[subject_id]
            if split is not None and effective_split != split:
                continue
            if row["patch_size_xyz"] != "64x64x32":
                raise ValueError(f"unexpected patch size: {row['patch_size_xyz']}")
            if int(row["anchor_label"]) not in LABEL_VALUES:
                raise ValueError(f"invalid anchor label: {row['anchor_label']}")
            if row["sample_role"] not in {"interior", "boundary"}:
                raise ValueError(f"invalid sample role: {row['sample_role']}")
            row["effective_split"] = effective_split
            records.append(row)
    if not records:
        raise RuntimeError(f"no records matched split={split!r}")
    return records, manifest_path


def _round_robin_subject_select(
    records: Sequence[Mapping[str, str]],
    *,
    count: int,
    seed: int,
    stratum: tuple[int, str],
) -> list[dict[str, str]]:
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_subject[str(record["subject_id"])].append(dict(record))
    if sum(len(values) for values in by_subject.values()) < count:
        raise ValueError(f"stratum {stratum} has fewer than {count} records")
    subjects = sorted(
        by_subject,
        key=lambda subject: _stable_key(seed, stratum[0], stratum[1], subject),
    )
    for subject, values in by_subject.items():
        values.sort(
            key=lambda row: _stable_key(
                seed, stratum[0], stratum[1], subject, row["relative_path"]
            )
        )
    positions = {subject: 0 for subject in subjects}
    selected: list[dict[str, str]] = []
    while len(selected) < count:
        progressed = False
        for subject in subjects:
            position = positions[subject]
            values = by_subject[subject]
            if position >= len(values):
                continue
            selected.append(values[position])
            positions[subject] = position + 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError(f"unable to fill stratum {stratum}")
    return selected


def build_labeled_subset(
    dataset_root: str | Path,
    split_file: str | Path,
    output_path: str | Path,
    *,
    count: int = 1000,
    seed: int = 20260806,
) -> dict[str, Any]:
    if count <= 0 or count % 8:
        raise ValueError("subset count must be positive and divisible by 8")
    records, manifest_path = load_manifest_records(
        dataset_root, split_file, split="train"
    )
    per_stratum = count // 8
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["anchor_label"]), record["sample_role"])].append(record)
    expected = {
        (label, role)
        for label in LABEL_VALUES
        for role in ("interior", "boundary")
    }
    missing = expected.difference(grouped)
    if missing:
        raise ValueError(f"missing subset strata: {sorted(missing)}")

    selected: list[dict[str, str]] = []
    for stratum in sorted(expected):
        selected.extend(
            _round_robin_subject_select(
                grouped[stratum], count=per_stratum, seed=seed, stratum=stratum
            )
        )
    selected.sort(key=lambda row: row["relative_path"])
    relative_paths = [row["relative_path"] for row in selected]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("labeled subset contains duplicate patch paths")
    if any(row["effective_split"] != "train" for row in selected):
        raise RuntimeError("labeled subset contains non-train records")

    anchor_counts = Counter(int(row["anchor_label"]) for row in selected)
    role_counts = Counter(row["sample_role"] for row in selected)
    stratum_counts = Counter(
        f"{row['anchor_label']}:{row['sample_role']}" for row in selected
    )
    subject_counts = Counter(row["subject_id"] for row in selected)
    voxel_counts = {
        str(label): sum(int(row[f"label_{label}_voxels"]) for row in selected)
        for label in LABEL_VALUES
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "strategy": "train_only_anchor_role_subject_round_robin",
        "seed": seed,
        "requested_count": count,
        "actual_count": len(selected),
        "patch_size_xyz": list(PATCH_SIZE_XYZ),
        "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": sha256_file(split_file),
        "anchor_counts": {str(key): value for key, value in sorted(anchor_counts.items())},
        "role_counts": dict(sorted(role_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "subject_count": len(subject_counts),
        "max_patches_per_subject": max(subject_counts.values()),
        "class_voxel_counts_for_loss_only": voxel_counts,
        "records": [
            {
                "relative_path": row["relative_path"],
                "case_id": row["case_id"],
                "subject_id": row["subject_id"],
                "anchor_label": int(row["anchor_label"]),
                "sample_role": row["sample_role"],
            }
            for row in selected
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def load_labeled_subset(
    path: str | Path,
    dataset_root: str | Path,
    split_file: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported subset schema: {payload.get('schema_version')}")
    if payload.get("patch_size_xyz") != list(PATCH_SIZE_XYZ):
        raise ValueError("subset is not p64")
    _, manifest_path = load_manifest_records(dataset_root, split_file, split="train")
    if payload.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("dataset manifest hash does not match frozen subset")
    if payload.get("split_sha256") != sha256_file(split_file):
        raise ValueError("split hash does not match frozen subset")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != int(payload["actual_count"]):
        raise ValueError("invalid subset record list")
    return [dict(record) for record in records], payload


class GLIClassifierPatchDataset(Dataset):
    """Expose configured MRI modalities, total lesion mask and optional target.

    The NPZ histogram and manifest-derived per-class fields are deliberately not
    returned.  Model code receives a strict whitelist so label-derived metadata
    cannot silently become a feature.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        records: Sequence[Mapping[str, Any]],
        *,
        load_targets: bool = True,
        modalities: Sequence[str] = ("t1c",),
    ) -> None:
        self.size_root = Path(dataset_root) / "patch_64x64x32"
        self.records = [dict(record) for record in records]
        self.load_targets = bool(load_targets)
        self.modalities = tuple(str(value).lower() for value in modalities)
        if not self.records:
            raise ValueError("classifier dataset requires at least one record")
        if not self.modalities or len(self.modalities) != len(set(self.modalities)):
            raise ValueError(f"modalities must be non-empty and unique: {self.modalities}")
        invalid_modalities = sorted(set(self.modalities).difference(MODALITY_KEYS))
        if invalid_modalities:
            raise ValueError(f"unsupported MRI modalities: {invalid_modalities}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        relative_path = str(record["relative_path"])
        path = (self.size_root / relative_path).resolve()
        if self.size_root.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as arrays:
            file_keys = set(arrays.files)
            required_keys = {*self.modalities, "seg", "affine"}
            allowed_keys = {*self.modalities} | NON_INPUT_NPZ_KEYS
            if not required_keys.issubset(file_keys) or not file_keys.issubset(allowed_keys):
                raise ValueError(f"unexpected NPZ keys in {path}: {sorted(arrays.files)}")
            images_xyz = [np.asarray(arrays[modality]) for modality in self.modalities]
            seg_xyz = np.asarray(arrays["seg"])
        invalid_shapes = [
            (modality, image.shape)
            for modality, image in zip(self.modalities, images_xyz, strict=True)
            if image.shape != PATCH_SIZE_XYZ
        ]
        if invalid_shapes or seg_xyz.shape != PATCH_SIZE_XYZ:
            raise ValueError(
                f"invalid p64 shape in {path}: modalities={invalid_shapes}, seg={seg_xyz.shape}"
            )
        invalid_dtypes = [
            (modality, str(image.dtype))
            for modality, image in zip(self.modalities, images_xyz, strict=True)
            if image.dtype != np.float32
        ]
        if invalid_dtypes or seg_xyz.dtype != np.uint8:
            raise ValueError(
                f"invalid p64 dtypes in {path}: modalities={invalid_dtypes}, seg={seg_xyz.dtype}"
            )
        non_finite = [
            modality
            for modality, image in zip(self.modalities, images_xyz, strict=True)
            if not np.isfinite(image).all()
        ]
        if non_finite:
            raise ValueError(f"non-finite MRI values in {path}: {non_finite}")
        out_of_range = [
            modality
            for modality, image in zip(self.modalities, images_xyz, strict=True)
            if image.min() < -1.00001 or image.max() > 1.00001
        ]
        if out_of_range:
            raise ValueError(f"MRI values outside [-1,1] in {path}: {out_of_range}")
        labels = set(np.unique(seg_xyz).tolist())
        if not labels.issubset({0, *LABEL_VALUES}):
            raise ValueError(f"invalid labels in {path}: {sorted(labels)}")

        image_dhw = np.stack(
            [np.transpose(image, (2, 0, 1)) for image in images_xyz], axis=0
        ).copy()
        seg_dhw = np.transpose(seg_xyz, (2, 0, 1)).copy()
        total_mask = seg_dhw > 0
        sample: dict[str, Any] = {
            "image": torch.from_numpy(image_dhw),
            "total_mask": torch.from_numpy(total_mask[None]),
            "case_id": str(record["case_id"]),
            "subject_id": str(record["subject_id"]),
            "relative_path": relative_path,
        }
        if self.load_targets:
            sample["target"] = torch.from_numpy(seg_dhw.astype(np.int64, copy=False))
        if not set(sample).issubset(SAFE_SAMPLE_KEYS):
            raise AssertionError(f"unsafe classifier sample keys: {sorted(set(sample) - SAFE_SAMPLE_KEYS)}")
        return sample


def class_weights_from_subset(payload: Mapping[str, Any]) -> torch.Tensor:
    counts = payload.get("class_voxel_counts_for_loss_only")
    if not isinstance(counts, Mapping):
        raise ValueError("subset has no class voxel counts")
    values = torch.tensor([float(counts[str(label)]) for label in LABEL_VALUES])
    if torch.any(values <= 0):
        raise ValueError(f"invalid class voxel counts: {values.tolist()}")
    weights = torch.rsqrt(values)
    return weights / weights.mean()


def batch_records_by_anchor(
    records: Sequence[Mapping[str, Any]],
    *,
    patches_per_class: int,
    seed: int,
    epoch: int,
) -> Iterable[list[dict[str, Any]]]:
    if patches_per_class <= 0:
        raise ValueError("patches_per_class must be positive")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["anchor_label"])].append(dict(record))
    if set(grouped) != set(LABEL_VALUES):
        raise ValueError("training subset must cover all anchor labels")
    generator = torch.Generator().manual_seed(int(seed) + 1_000_003 * int(epoch))
    for label in LABEL_VALUES:
        order = torch.randperm(len(grouped[label]), generator=generator).tolist()
        grouped[label] = [grouped[label][index] for index in order]
    steps = max(math.ceil(len(grouped[label]) / patches_per_class) for label in LABEL_VALUES)
    for step in range(steps):
        batch: list[dict[str, Any]] = []
        for label in LABEL_VALUES:
            values = grouped[label]
            for offset in range(patches_per_class):
                index = (step * patches_per_class + offset) % len(values)
                batch.append(values[index])
        yield batch
