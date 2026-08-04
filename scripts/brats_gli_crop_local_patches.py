#!/usr/bin/env python
"""Build deterministic local T1c lesion patches from BraTS 2024 GLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage


LABEL_NAMES = {1: "netc", 2: "snfh", 3: "et", 4: "rc"}
MANIFEST_FIELDS = (
    "relative_path",
    "case_id",
    "subject_id",
    "split",
    "patch_size_xyz",
    "anchor_label",
    "anchor_name",
    "sample_role",
    "center_x",
    "center_y",
    "center_z",
    "origin_x",
    "origin_y",
    "origin_z",
    "pad_before_x",
    "pad_before_y",
    "pad_before_z",
    "pad_after_x",
    "pad_after_y",
    "pad_after_z",
    "label_1_voxels",
    "label_2_voxels",
    "label_3_voxels",
    "label_4_voxels",
    "anchor_voxels",
    "anchor_fraction",
    "normalization_p005",
    "normalization_p995",
    "normalization_degenerate",
    "distinct_origin",
)


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    subject_id: str
    case_dir: str
    labels_present: tuple[int, ...]
    split: str


def parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [part for part in re.split(r"[xX,]", value) if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be XxYxZ, for example 64x64x32")
    size = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in size):
        raise argparse.ArgumentTypeError("patch dimensions must be positive")
    return size  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    default_root = os.environ.get("REMOTE_DATA_ROOT_BRATS_GLI", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(default_root) if default_root else None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--patch-size",
        action="append",
        type=parse_patch_size,
        dest="patch_sizes",
        help="Repeat for multiple XYZ patch sizes; defaults to 64x64x32 and 80x96x80.",
    )
    parser.add_argument("--seed", default=20260804, type=int)
    parser.add_argument("--val-fraction", default=0.2, type=float)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--max-cases", default=0, type=int)
    parser.add_argument("--boundary-candidates", default=512, type=int)
    parser.add_argument("--qa-per-label", default=4, type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Build and validate a limited staging dataset, then remove only that staging directory.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root is required when REMOTE_DATA_ROOT_BRATS_GLI is unset")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.smoke and args.max_cases <= 0:
        parser.error("--smoke requires --max-cases")
    args.patch_sizes = args.patch_sizes or [(64, 64, 32), (80, 96, 80)]
    if len(set(args.patch_sizes)) != len(args.patch_sizes):
        parser.error("duplicate patch sizes are not allowed")
    return args


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def fallback_subject_id(case_id: str) -> str:
    match = re.match(r"^(BraTS-GLI-\d+)-\d+$", case_id)
    return match.group(1) if match else case_id


def load_metadata_subject_map(data_root: Path) -> tuple[dict[str, str], str]:
    metadata_files = sorted(data_root.glob("*.xlsx"))
    if not metadata_files:
        return {}, "metadata xlsx not found; using case-id fallback"
    try:
        import pandas as pd

        frame = pd.read_excel(metadata_files[0])
    except Exception as exc:  # pragma: no cover - depends on optional Excel engine
        return {}, f"metadata unreadable ({exc}); using case-id fallback"

    columns = {str(column).strip(): column for column in frame.columns}
    case_candidates = [
        name for name in columns if "brats" in name.lower() or "case" in name.lower() or "scan" in name.lower()
    ]
    subject_candidates = [
        name
        for name in columns
        if "subject" in name.lower() or "patient" in name.lower() or "participant" in name.lower()
    ]
    if not case_candidates or not subject_candidates:
        return {}, "metadata columns are ambiguous; using case-id fallback"

    best_mapping: dict[str, str] = {}
    best_columns: tuple[str, str] | None = None
    for case_name in case_candidates:
        for subject_name in subject_candidates:
            mapping: dict[str, str] = {}
            for case_value, subject_value in zip(frame[columns[case_name]], frame[columns[subject_name]]):
                if pd.isna(case_value) or pd.isna(subject_value):
                    continue
                case_text = str(case_value).strip()
                subject_text = str(subject_value).strip()
                match = re.search(r"BraTS-GLI-\d+(?:-\d+)?", case_text)
                if match:
                    mapping[match.group(0)] = subject_text
            if len(mapping) > len(best_mapping):
                best_mapping = mapping
                best_columns = (case_name, subject_name)
    if not best_mapping or best_columns is None:
        return {}, "metadata has no usable BraTS case mapping; using case-id fallback"
    return best_mapping, f"metadata columns: case={best_columns[0]!r}, subject={best_columns[1]!r}"


def labels_in_seg(seg_path: Path) -> tuple[int, ...]:
    seg = np.asanyarray(nib.load(str(seg_path)).dataobj)
    values = set(int(value) for value in np.unique(seg))
    invalid = values.difference({0, 1, 2, 3, 4})
    if invalid:
        raise ValueError(f"unexpected labels in {seg_path}: {sorted(invalid)}")
    return tuple(label for label in LABEL_NAMES if label in values)


def discover_cases(
    data_root: Path,
    split: str,
    max_cases: int,
    subject_map: dict[str, str],
    workers: int,
) -> list[CaseInfo]:
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    case_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if max_cases > 0:
        case_dirs = case_dirs[:max_cases]
    def inspect_case(case_dir: Path) -> CaseInfo:
        case_id = case_dir.name
        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        t1c_path = case_dir / f"{case_id}-t1c.nii.gz"
        if not seg_path.is_file() or not t1c_path.is_file():
            raise FileNotFoundError(f"missing t1c or seg for {case_id}")
        subject_id = subject_map.get(case_id, fallback_subject_id(case_id))
        return CaseInfo(
            case_id=case_id,
            subject_id=subject_id,
            case_dir=str(case_dir),
            labels_present=labels_in_seg(seg_path),
            split="",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        cases = list(executor.map(inspect_case, case_dirs))
    if not cases:
        raise RuntimeError(f"no cases found in {split_dir}")
    return cases


def stratified_subject_split(
    cases: Sequence[CaseInfo], val_fraction: float, seed: int
) -> tuple[list[CaseInfo], dict[str, str]]:
    by_subject: dict[str, list[CaseInfo]] = {}
    for case in cases:
        by_subject.setdefault(case.subject_id, []).append(case)

    signatures: dict[tuple[int, ...], list[str]] = {}
    for subject_id, subject_cases in by_subject.items():
        signature = tuple(
            int(any(label in case.labels_present for case in subject_cases)) for label in LABEL_NAMES
        )
        signatures.setdefault(signature, []).append(subject_id)

    split_by_subject: dict[str, str] = {}
    for signature, subject_ids in sorted(signatures.items()):
        ordered = sorted(subject_ids)
        rng = np.random.default_rng(stable_seed(seed, "split", signature))
        rng.shuffle(ordered)
        val_count = int(round(len(ordered) * val_fraction))
        if len(ordered) > 1:
            val_count = min(max(val_count, 1), len(ordered) - 1)
        else:
            val_count = int(rng.random() < val_fraction)
        val_subjects = set(ordered[:val_count])
        for subject_id in ordered:
            split_by_subject[subject_id] = "val" if subject_id in val_subjects else "train"

    assigned = [
        CaseInfo(
            case_id=case.case_id,
            subject_id=case.subject_id,
            case_dir=case.case_dir,
            labels_present=case.labels_present,
            split=split_by_subject[case.subject_id],
        )
        for case in cases
    ]
    return assigned, split_by_subject


def robust_normalize_t1c(image: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    image = np.asarray(image, dtype=np.float32)
    foreground = image != 0
    normalized = np.zeros(image.shape, dtype=np.float32)
    if not foreground.any():
        return normalized, 0.0, 0.0, True
    low, high = np.percentile(image[foreground], [0.5, 99.5])
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return normalized, low, high, True
    values = np.clip(image[foreground], low, high)
    scaled = ((values - low) / (high - low) * 2.0 - 1.0).astype(np.float32)
    normalized[foreground] = np.clip(scaled, -1.0, 1.0)
    return normalized, low, high, False


def crop_origin(center: Sequence[int], patch_size: Sequence[int]) -> tuple[int, int, int]:
    return tuple(int(center[axis]) - int(patch_size[axis]) // 2 for axis in range(3))


def clipped_box(
    origin: Sequence[int], patch_size: Sequence[int], shape: Sequence[int]
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    starts = tuple(max(0, int(origin[axis])) for axis in range(3))
    ends = tuple(min(int(shape[axis]), int(origin[axis]) + int(patch_size[axis])) for axis in range(3))
    return starts, ends


def box_sum(
    integral: np.ndarray, origin: Sequence[int], patch_size: Sequence[int], shape: Sequence[int]
) -> int:
    starts, ends = clipped_box(origin, patch_size, shape)
    x0, y0, z0 = starts
    x1, y1, z1 = ends
    return int(
        integral[x1, y1, z1]
        - integral[x0, y1, z1]
        - integral[x1, y0, z1]
        - integral[x1, y1, z0]
        + integral[x0, y0, z1]
        + integral[x0, y1, z0]
        + integral[x1, y0, z0]
        - integral[x0, y0, z0]
    )


def mask_integral(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int64), ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)


def interior_center(mask: np.ndarray) -> tuple[int, int, int]:
    distance = ndimage.distance_transform_edt(mask)
    return tuple(int(value) for value in np.unravel_index(int(np.argmax(distance)), mask.shape))


def boundary_center(
    mask: np.ndarray,
    patch_size: Sequence[int],
    seed: int,
    forbidden_origin: Sequence[int],
    max_candidates: int,
) -> tuple[tuple[int, int, int], bool]:
    boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    coords = np.argwhere(boundary)
    if coords.size == 0:
        coords = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    if len(coords) > max_candidates:
        coords = coords[rng.choice(len(coords), size=max_candidates, replace=False)]
    else:
        coords = coords[rng.permutation(len(coords))]

    integral = mask_integral(mask)
    patch_volume = int(np.prod(patch_size))
    candidates = []
    forbidden = tuple(int(value) for value in forbidden_origin)
    for coord in coords:
        center = tuple(int(value) for value in coord)
        origin = crop_origin(center, patch_size)
        fraction = box_sum(integral, origin, patch_size, mask.shape) / patch_volume
        in_target_range = 0.10 <= fraction <= 0.70
        candidates.append((origin == forbidden, not in_target_range, abs(fraction - 0.25), center))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    chosen = candidates[0]
    center = chosen[3]
    return center, crop_origin(center, patch_size) != forbidden


def extract_patch(
    array: np.ndarray, origin: Sequence[int], patch_size: Sequence[int], fill_value: int | float
) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    patch = np.full(tuple(int(value) for value in patch_size), fill_value, dtype=array.dtype)
    source_starts, source_ends = clipped_box(origin, patch_size, array.shape)
    target_starts = tuple(max(0, -int(origin[axis])) for axis in range(3))
    target_ends = tuple(
        target_starts[axis] + source_ends[axis] - source_starts[axis] for axis in range(3)
    )
    source_slices = tuple(slice(source_starts[axis], source_ends[axis]) for axis in range(3))
    target_slices = tuple(slice(target_starts[axis], target_ends[axis]) for axis in range(3))
    patch[target_slices] = array[source_slices]
    pad_before = target_starts
    pad_after = tuple(int(patch_size[axis]) - target_ends[axis] for axis in range(3))
    return patch, pad_before, pad_after


def lesion_histograms(t1c: np.ndarray, seg: np.ndarray) -> np.ndarray:
    hist = np.zeros((4, 16), dtype=np.float32)
    for index, label in enumerate(LABEL_NAMES):
        values = t1c[seg == label]
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=16, range=(-1.0, 1.0))
        total = int(counts.sum())
        if total:
            hist[index] = counts.astype(np.float32) / total
    return hist


def translated_affine(affine: np.ndarray, origin: Sequence[int]) -> np.ndarray:
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = np.asarray(origin, dtype=np.float64)
    return np.asarray(affine, dtype=np.float64) @ translation


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def patch_dir_name(patch_size: Sequence[int]) -> str:
    return "patch_" + "x".join(str(int(value)) for value in patch_size)


def process_case(
    case: CaseInfo,
    patch_sizes: Sequence[tuple[int, int, int]],
    staging_root: str,
    seed: int,
    boundary_candidates: int,
) -> dict:
    case_dir = Path(case.case_dir)
    t1c_path = case_dir / f"{case.case_id}-t1c.nii.gz"
    seg_path = case_dir / f"{case.case_id}-seg.nii.gz"
    t1c_image = nib.load(str(t1c_path))
    seg_image = nib.load(str(seg_path))
    t1c_raw = t1c_image.get_fdata(dtype=np.float32)
    seg = np.asanyarray(seg_image.dataobj).astype(np.uint8, copy=False)
    if t1c_raw.shape != seg.shape:
        raise ValueError(f"shape mismatch for {case.case_id}: {t1c_raw.shape} vs {seg.shape}")
    if not np.allclose(t1c_image.affine, seg_image.affine, atol=1e-4):
        raise ValueError(f"affine mismatch for {case.case_id}")
    invalid = set(int(value) for value in np.unique(seg)).difference({0, 1, 2, 3, 4})
    if invalid:
        raise ValueError(f"unexpected labels for {case.case_id}: {sorted(invalid)}")

    t1c, norm_low, norm_high, norm_degenerate = robust_normalize_t1c(t1c_raw)
    rows_by_size: dict[str, list[dict]] = {patch_dir_name(size): [] for size in patch_sizes}
    staging = Path(staging_root)

    for label in case.labels_present:
        anchor_mask = seg == label
        inner_center = interior_center(anchor_mask)
        for patch_size in patch_sizes:
            size_name = patch_dir_name(patch_size)
            inner_origin = crop_origin(inner_center, patch_size)
            edge_center, distinct = boundary_center(
                anchor_mask,
                patch_size,
                stable_seed(seed, case.case_id, label, patch_size, "boundary"),
                inner_origin,
                boundary_candidates,
            )
            for role, center, origin_is_distinct in (
                ("interior", inner_center, True),
                ("boundary", edge_center, distinct),
            ):
                origin = crop_origin(center, patch_size)
                t1c_patch, pad_before, pad_after = extract_patch(t1c, origin, patch_size, 0.0)
                seg_patch, _, _ = extract_patch(seg, origin, patch_size, 0)
                label_counts = {value: int((seg_patch == value).sum()) for value in LABEL_NAMES}
                anchor_voxels = label_counts[label]
                if anchor_voxels <= 0:
                    raise RuntimeError(
                        f"anchor label {label} missing after crop for {case.case_id} {patch_size} {role}"
                    )
                hist = lesion_histograms(t1c_patch, seg_patch)
                name = (
                    f"{case.case_id}__label{label}_{LABEL_NAMES[label]}__{role}"
                    f"__x{origin[0]}_y{origin[1]}_z{origin[2]}.npz"
                )
                relative_path = Path("patches") / name
                output_path = staging / size_name / relative_path
                if not output_path.exists():
                    atomic_save_npz(
                        output_path,
                        t1c=t1c_patch.astype(np.float32, copy=False),
                        seg=seg_patch.astype(np.uint8, copy=False),
                        hist=hist,
                        affine=translated_affine(t1c_image.affine, origin),
                    )
                row = {
                    "relative_path": relative_path.as_posix(),
                    "case_id": case.case_id,
                    "subject_id": case.subject_id,
                    "split": case.split,
                    "patch_size_xyz": "x".join(map(str, patch_size)),
                    "anchor_label": label,
                    "anchor_name": LABEL_NAMES[label],
                    "sample_role": role,
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "origin_x": origin[0],
                    "origin_y": origin[1],
                    "origin_z": origin[2],
                    "pad_before_x": pad_before[0],
                    "pad_before_y": pad_before[1],
                    "pad_before_z": pad_before[2],
                    "pad_after_x": pad_after[0],
                    "pad_after_y": pad_after[1],
                    "pad_after_z": pad_after[2],
                    "label_1_voxels": label_counts[1],
                    "label_2_voxels": label_counts[2],
                    "label_3_voxels": label_counts[3],
                    "label_4_voxels": label_counts[4],
                    "anchor_voxels": anchor_voxels,
                    "anchor_fraction": round(anchor_voxels / int(np.prod(patch_size)), 8),
                    "normalization_p005": round(norm_low, 6),
                    "normalization_p995": round(norm_high, 6),
                    "normalization_degenerate": int(norm_degenerate),
                    "distinct_origin": int(origin_is_distinct),
                }
                rows_by_size[size_name].append(row)
    return {
        "case_id": case.case_id,
        "rows_by_size": rows_by_size,
        "normalization_degenerate": norm_degenerate,
    }


def write_manifest(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_patch_file(path: Path, row: dict[str, str], patch_size: tuple[int, int, int]) -> int:
    with np.load(path) as arrays:
        required = {"t1c", "seg", "hist", "affine"}
        if set(arrays.files) != required:
            raise ValueError(f"invalid NPZ keys in {path}: {arrays.files}")
        t1c = arrays["t1c"]
        seg = arrays["seg"]
        hist = arrays["hist"]
        affine = arrays["affine"]
        if t1c.shape != patch_size or seg.shape != patch_size:
            raise ValueError(f"invalid patch shape in {path}: {t1c.shape}, {seg.shape}")
        if t1c.dtype != np.float32 or seg.dtype != np.uint8:
            raise ValueError(f"invalid dtypes in {path}: {t1c.dtype}, {seg.dtype}")
        if hist.shape != (4, 16) or hist.dtype != np.float32 or affine.shape != (4, 4):
            raise ValueError(f"invalid metadata arrays in {path}")
        if not np.isfinite(t1c).all() or not np.isfinite(hist).all() or not np.isfinite(affine).all():
            raise ValueError(f"non-finite values in {path}")
        if t1c.min() < -1.00001 or t1c.max() > 1.00001:
            raise ValueError(f"T1c outside [-1,1] in {path}")
        if not set(int(value) for value in np.unique(seg)).issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"invalid segmentation labels in {path}")
        anchor_label = int(row["anchor_label"])
        if not np.any(seg == anchor_label):
            raise ValueError(f"anchor label missing in {path}")
        for index, label in enumerate(LABEL_NAMES):
            expected = 1.0 if np.any(seg == label) else 0.0
            if not np.isclose(float(hist[index].sum()), expected, atol=1e-5):
                raise ValueError(f"invalid histogram sum for label {label} in {path}")
    return path.stat().st_size


def validate_dataset(staging_root: Path, patch_sizes: Sequence[tuple[int, int, int]]) -> dict:
    result: dict[str, dict] = {}
    split_subjects: dict[str, set[str]] = {"train": set(), "val": set()}
    for patch_size in patch_sizes:
        size_name = patch_dir_name(patch_size)
        size_root = staging_root / size_name
        rows = load_manifest(size_root / "manifest.csv")
        seen_paths: set[str] = set()
        total_bytes = 0
        counts = {str(label): 0 for label in LABEL_NAMES}
        roles = {"interior": 0, "boundary": 0}
        padded = 0
        for row in rows:
            relative_path = row["relative_path"]
            if relative_path in seen_paths:
                raise ValueError(f"duplicate manifest path: {size_name}/{relative_path}")
            seen_paths.add(relative_path)
            path = size_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            total_bytes += validate_patch_file(path, row, patch_size)
            counts[row["anchor_label"]] += 1
            roles[row["sample_role"]] += 1
            split_subjects[row["split"]].add(row["subject_id"])
            pad_total = sum(int(row[f"pad_{side}_{axis}"]) for side in ("before", "after") for axis in "xyz")
            padded += int(pad_total > 0)
        result[size_name] = {
            "patch_count": len(rows),
            "anchor_counts": counts,
            "role_counts": roles,
            "padded_patch_count": padded,
            "compressed_bytes": total_bytes,
        }
    leakage = split_subjects["train"].intersection(split_subjects["val"])
    if leakage:
        raise ValueError(f"subject leakage between train and val: {sorted(leakage)[:10]}")
    return result


def create_qa_images(staging_root: Path, patch_sizes: Sequence[tuple[int, int, int]], per_label: int) -> None:
    if per_label <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "tab:red", 2: "tab:green", 3: "tab:orange", 4: "tab:blue"}
    for patch_size in patch_sizes:
        size_root = staging_root / patch_dir_name(patch_size)
        rows = load_manifest(size_root / "manifest.csv")
        qa_dir = size_root / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        for label in LABEL_NAMES:
            selected = [row for row in rows if int(row["anchor_label"]) == label][:per_label]
            for index, row in enumerate(selected):
                with np.load(size_root / row["relative_path"]) as arrays:
                    t1c = arrays["t1c"]
                    seg = arrays["seg"]
                z_index = t1c.shape[2] // 2
                figure, axis = plt.subplots(figsize=(5, 5), dpi=120)
                axis.imshow(t1c[:, :, z_index].T, cmap="gray", vmin=-1, vmax=1, origin="lower")
                for contour_label, color in colors.items():
                    mask_slice = (seg[:, :, z_index] == contour_label).T
                    if mask_slice.any():
                        axis.contour(mask_slice.astype(float), levels=[0.5], colors=[color], linewidths=0.8)
                axis.set_title(
                    f"{row['case_id']} label {label} {row['sample_role']} z={z_index}", fontsize=8
                )
                axis.axis("off")
                figure.tight_layout()
                figure.savefig(qa_dir / f"label{label}_{index:02d}_{row['sample_role']}.png")
                plt.close(figure)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def expected_patch_count(cases: Iterable[CaseInfo]) -> int:
    return sum(len(case.labels_present) * 2 for case in cases)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    subject_map, metadata_note = load_metadata_subject_map(data_root)
    cases = discover_cases(data_root, args.split, args.max_cases, subject_map, args.workers)
    cases, split_by_subject = stratified_subject_split(cases, args.val_fraction, args.seed)
    expected_per_size = expected_patch_count(cases)
    audit = {
        "data_root": str(data_root),
        "source_split": args.split,
        "case_count": len(cases),
        "subject_count": len(split_by_subject),
        "metadata": metadata_note,
        "patch_sizes_xyz": [list(size) for size in args.patch_sizes],
        "expected_patches_per_size": expected_per_size,
        "split_subject_counts": {
            split: sum(value == split for value in split_by_subject.values()) for split in ("train", "val")
        },
    }
    if args.validate_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    final_root = args.out_root.resolve()
    staging_root = final_root.parent / f".{final_root.name}.staging"
    if final_root.exists():
        raise FileExistsError(f"final output already exists: {final_root}")
    if staging_root.exists() and not args.resume:
        raise FileExistsError(f"staging output already exists; use --resume: {staging_root}")
    staging_root.mkdir(parents=True, exist_ok=True)
    for patch_size in args.patch_sizes:
        (staging_root / patch_dir_name(patch_size) / "patches").mkdir(parents=True, exist_ok=True)

    write_json(
        staging_root / "splits.json",
        {
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "subject_split": dict(sorted(split_by_subject.items())),
        },
    )

    rows_by_size: dict[str, list[dict]] = {patch_dir_name(size): [] for size in args.patch_sizes}
    failures: list[dict[str, str]] = []
    degenerate_cases: list[str] = []
    processed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_case,
                case,
                args.patch_sizes,
                str(staging_root),
                args.seed,
                args.boundary_candidates,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
                for size_name, rows in result["rows_by_size"].items():
                    rows_by_size[size_name].extend(rows)
                if result["normalization_degenerate"]:
                    degenerate_cases.append(case.case_id)
            except Exception as exc:
                failures.append({"case_id": case.case_id, "error": repr(exc)})
            processed += 1
            if processed % 25 == 0 or processed == len(cases):
                print(f"processed {processed}/{len(cases)} cases; failures={len(failures)}", flush=True)

    for patch_size in args.patch_sizes:
        size_name = patch_dir_name(patch_size)
        rows = sorted(
            rows_by_size[size_name],
            key=lambda row: (row["case_id"], int(row["anchor_label"]), row["sample_role"]),
        )
        write_manifest(staging_root / size_name / "manifest.csv", rows)

    prevalidation_summary = {
        **audit,
        "processed_cases": processed,
        "failed_cases": failures,
        "normalization_degenerate_cases": sorted(degenerate_cases),
    }
    write_json(staging_root / "dataset_summary.json", prevalidation_summary)
    if failures:
        raise RuntimeError(f"{len(failures)} cases failed; staging retained at {staging_root}")
    for size_name, rows in rows_by_size.items():
        if len(rows) != expected_per_size:
            raise RuntimeError(
                f"unexpected patch count for {size_name}: {len(rows)} != {expected_per_size}"
            )

    create_qa_images(staging_root, args.patch_sizes, args.qa_per_label)
    validation = validate_dataset(staging_root, args.patch_sizes)
    final_summary = {**prevalidation_summary, "validation": validation, "published": True}
    if args.smoke:
        final_summary["published"] = False
        final_summary["smoke_validation"] = True
        write_json(staging_root / "dataset_summary.json", final_summary)
        shutil.rmtree(staging_root)
        print(json.dumps(final_summary, ensure_ascii=False, indent=2))
        print(f"smoke validation passed; removed staging dataset: {staging_root}")
        return
    write_json(staging_root / "dataset_summary.json", final_summary)
    os.replace(staging_root, final_root)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"published dataset: {final_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
