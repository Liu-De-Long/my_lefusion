#!/usr/bin/env python
"""Replay frozen p64 windows and materialize leak-safe four-modality patches."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from LeFusion.classifier.data import load_subject_split, sha256_file
from brats_gli_crop_local_patches import (
    atomic_save_npz,
    extract_patch,
    robust_normalize_t1c,
    translated_affine,
)


MODALITIES = ("t1c", "t1n", "t2f", "t2w")
PATCH_SIZE_XYZ = (64, 64, 32)
OUTPUT_KEYS = frozenset({*MODALITIES, "seg", "affine"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--raw-split", default="train")
    parser.add_argument("--source-dataset-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--include-split",
        action="append",
        choices=("train", "val", "test"),
        dest="include_splits",
        help="Repeat to materialize selected patient splits; defaults to train and val.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.include_splits = tuple(args.include_splits or ("train", "val"))
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_cases < 0:
        parser.error("--max-cases cannot be negative")
    return args


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required = {
            "relative_path",
            "case_id",
            "subject_id",
            "patch_size_xyz",
            "origin_x",
            "origin_y",
            "origin_z",
        }
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"source manifest missing fields: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"source manifest is empty: {path}")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("source manifest has duplicate relative paths")
    if any(row["patch_size_xyz"] != "64x64x32" for row in rows):
        raise ValueError("source manifest contains a non-p64 record")
    return rows, fields


def _validate_output_patch(path: Path) -> int:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != OUTPUT_KEYS:
            raise ValueError(f"invalid multimodal NPZ keys in {path}: {sorted(arrays.files)}")
        for modality in MODALITIES:
            value = np.asarray(arrays[modality])
            if value.shape != PATCH_SIZE_XYZ or value.dtype != np.float32:
                raise ValueError(f"invalid {modality} contract in {path}: {value.shape}/{value.dtype}")
            if not np.isfinite(value).all() or value.min() < -1.00001 or value.max() > 1.00001:
                raise ValueError(f"invalid normalized {modality} values in {path}")
        seg = np.asarray(arrays["seg"])
        affine = np.asarray(arrays["affine"])
        if seg.shape != PATCH_SIZE_XYZ or seg.dtype != np.uint8:
            raise ValueError(f"invalid segmentation contract in {path}: {seg.shape}/{seg.dtype}")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(f"invalid affine in {path}")
        if not set(np.unique(seg).tolist()).issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"invalid segmentation labels in {path}")
    return path.stat().st_size


def _load_case_volumes(
    raw_root: Path,
    raw_split: str,
    case_id: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, dict[str, float | bool]]]:
    case_root = raw_root / raw_split / case_id
    seg_image = nib.load(str(case_root / f"{case_id}-seg.nii.gz"))
    seg = seg_image.get_fdata(dtype=np.float32).astype(np.uint8)
    normalized: dict[str, np.ndarray] = {}
    normalization: dict[str, dict[str, float | bool]] = {}
    for modality in MODALITIES:
        image = nib.load(str(case_root / f"{case_id}-{modality}.nii.gz"))
        raw = image.get_fdata(dtype=np.float32)
        if raw.shape != seg.shape:
            raise ValueError(
                f"shape mismatch for {case_id}/{modality}: {raw.shape} vs {seg.shape}"
            )
        if not np.allclose(image.affine, seg_image.affine, atol=1e-4):
            raise ValueError(f"affine mismatch for {case_id}/{modality}")
        value, low, high, degenerate = robust_normalize_t1c(raw)
        normalized[modality] = value
        normalization[modality] = {
            "p005": low,
            "p995": high,
            "degenerate": degenerate,
        }
    return normalized, seg, np.asarray(seg_image.affine), normalization


def _process_case(
    case_id: str,
    rows: Sequence[Mapping[str, str]],
    *,
    raw_root: str,
    raw_split: str,
    source_size_root: str,
    output_size_root: str,
    resume: bool,
) -> dict[str, Any]:
    normalized, raw_seg, raw_affine, normalization = _load_case_volumes(
        Path(raw_root), raw_split, case_id
    )
    source_root = Path(source_size_root)
    output_root = Path(output_size_root)
    written = 0
    reused = 0
    compressed_bytes = 0
    for row in rows:
        relative_path = str(row["relative_path"])
        source_path = source_root / relative_path
        output_path = output_root / relative_path
        if output_path.is_file() and resume:
            compressed_bytes += _validate_output_patch(output_path)
            reused += 1
            continue
        with np.load(source_path, allow_pickle=False) as source:
            source_t1c = np.asarray(source["t1c"])
            source_seg = np.asarray(source["seg"])
            source_affine = np.asarray(source["affine"])
        origin = tuple(int(row[f"origin_{axis}"]) for axis in "xyz")
        seg_patch, pad_before, pad_after = extract_patch(
            raw_seg, origin, PATCH_SIZE_XYZ, 0
        )
        expected_before = tuple(int(row[f"pad_before_{axis}"]) for axis in "xyz")
        expected_after = tuple(int(row[f"pad_after_{axis}"]) for axis in "xyz")
        if pad_before != expected_before or pad_after != expected_after:
            raise ValueError(f"padding replay mismatch for {relative_path}")
        if not np.array_equal(seg_patch, source_seg):
            raise ValueError(f"segmentation replay mismatch for {relative_path}")
        patch_arrays: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            patch, _, _ = extract_patch(normalized[modality], origin, PATCH_SIZE_XYZ, 0.0)
            patch_arrays[modality] = patch.astype(np.float32, copy=False)
        if not np.allclose(patch_arrays["t1c"], source_t1c, atol=1e-6):
            raise ValueError(f"T1c normalization replay mismatch for {relative_path}")
        replay_affine = translated_affine(raw_affine, origin)
        if not np.allclose(replay_affine, source_affine, atol=1e-5):
            raise ValueError(f"affine replay mismatch for {relative_path}")
        atomic_save_npz(
            output_path,
            **patch_arrays,
            seg=seg_patch.astype(np.uint8, copy=False),
            affine=replay_affine,
        )
        compressed_bytes += _validate_output_patch(output_path)
        written += 1
    return {
        "case_id": case_id,
        "patch_count": len(rows),
        "written": written,
        "reused": reused,
        "compressed_bytes": compressed_bytes,
        "normalization": normalization,
    }


def build_multimodal_p64(
    *,
    raw_root: Path,
    raw_split: str,
    source_dataset_root: Path,
    split_file: Path,
    output_root: Path,
    include_splits: Sequence[str] = ("train", "val"),
    workers: int = 4,
    max_cases: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    source_size_root = source_dataset_root / "patch_64x64x32"
    source_manifest = source_size_root / "manifest.csv"
    rows, _ = _read_manifest(source_manifest)
    subject_split = load_subject_split(split_file)
    include = set(include_splits)
    selected = [
        row for row in rows if subject_split.get(row["subject_id"]) in include
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[row["case_id"]].append(row)
    case_ids = sorted(grouped)
    if max_cases:
        case_ids = case_ids[:max_cases]
        selected = [row for case_id in case_ids for row in grouped[case_id]]
    if not selected:
        raise ValueError(f"no patches matched include_splits={sorted(include)}")

    output_size_root = output_root / "patch_64x64x32"
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(
            f"output root is not empty; use --resume after auditing it: {output_root}"
        )
    output_size_root.mkdir(parents=True, exist_ok=True)
    worker_arguments = [
        (
            case_id,
            grouped[case_id],
        )
        for case_id in case_ids
    ]
    if workers == 1:
        results = [
            _process_case(
                case_id,
                rows_for_case,
                raw_root=str(raw_root),
                raw_split=raw_split,
                source_size_root=str(source_size_root),
                output_size_root=str(output_size_root),
                resume=resume,
            )
            for case_id, rows_for_case in worker_arguments
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_case,
                    case_id,
                    rows_for_case,
                    raw_root=str(raw_root),
                    raw_split=raw_split,
                    source_size_root=str(source_size_root),
                    output_size_root=str(output_size_root),
                    resume=resume,
                ): case_id
                for case_id, rows_for_case in worker_arguments
            }
            for future in as_completed(futures):
                results.append(future.result())

    # Preserve the source bytes exactly so the frozen 1000-patch manifest hash
    # remains valid. Missing test NPZ files intentionally keep test physically sealed.
    shutil.copy2(source_manifest, output_size_root / "manifest.csv")
    normalization_degenerate = Counter()
    for result in results:
        for modality, values in result["normalization"].items():
            normalization_degenerate[modality] += int(bool(values["degenerate"]))
    split_counts = Counter(subject_split[row["subject_id"]] for row in selected)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "frozen_origin_multimodal_p64_no_hist_input",
        "modalities": list(MODALITIES),
        "patch_size_xyz": list(PATCH_SIZE_XYZ),
        "include_splits": sorted(include),
        "test_materialized": "test" in include,
        "case_count": len(case_ids),
        "patch_count": len(selected),
        "split_patch_counts": dict(sorted(split_counts.items())),
        "written_patch_count": sum(int(result["written"]) for result in results),
        "reused_patch_count": sum(int(result["reused"]) for result in results),
        "compressed_bytes": sum(int(result["compressed_bytes"]) for result in results),
        "normalization_degenerate_case_counts": dict(
            sorted(normalization_degenerate.items())
        ),
        "source_manifest_sha256": sha256_file(source_manifest),
        "output_manifest_sha256": sha256_file(output_size_root / "manifest.csv"),
        "split_sha256": sha256_file(split_file),
        "label_derived_input_keys": [],
    }
    if payload["source_manifest_sha256"] != payload["output_manifest_sha256"]:
        raise AssertionError("copied manifest hash changed")
    audit_path = output_root / "multimodal_build_audit.json"
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    args = parse_args()
    result = build_multimodal_p64(
        raw_root=args.raw_root,
        raw_split=args.raw_split,
        source_dataset_root=args.source_dataset_root,
        split_file=args.split_file,
        output_root=args.output_root,
        include_splits=args.include_splits,
        workers=args.workers,
        max_cases=args.max_cases,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
