"""Audit the existing T1c nonzero normalization support against a multimodal support mask."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LeFusion"))
from dataset.gli_hist_in import MODALITIES, build_explicit_brain_support  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(seed: int, value: object) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def support_metrics(raw_nonzero: np.ndarray, support: np.ndarray) -> dict[str, float | int]:
    raw_nonzero = np.asarray(raw_nonzero, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if raw_nonzero.shape != support.shape:
        raise ValueError("support audit shape mismatch")
    intersection = int(np.count_nonzero(raw_nonzero & support))
    raw_count = int(np.count_nonzero(raw_nonzero))
    support_count = int(np.count_nonzero(support))
    extra = int(np.count_nonzero(raw_nonzero & ~support))
    missing = int(np.count_nonzero(support & ~raw_nonzero))
    denominator = raw_count + support_count
    return {
        "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "raw_nonzero_voxels": raw_count,
        "support_voxels": support_count,
        "extra_voxels": extra,
        "missing_voxels": missing,
        "extra_fraction": extra / max(raw_count, 1),
        "missing_fraction": missing / max(support_count, 1),
    }


def _load_case(case_dir: Path, case_id: str) -> tuple[list[np.ndarray], np.ndarray]:
    modalities = []
    affine = None
    shape = None
    for modality in MODALITIES:
        path = case_dir / f"{case_id}-{modality}.nii.gz"
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
        affine = image.affine if affine is None else affine
        shape = array.shape if shape is None else shape
        if array.shape != shape:
            raise ValueError(f"multimodal shape mismatch for {case_id}")
        modalities.append(array)
    segmentation = np.asanyarray(
        nib.load(str(case_dir / f"{case_id}-seg.nii.gz")).dataobj
    )
    if segmentation.shape != shape:
        raise ValueError(f"segmentation shape mismatch for {case_id}")
    if not set(np.unique(segmentation).tolist()).issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"invalid segmentation values for {case_id}")
    return modalities, segmentation


def _subject_index(manifest_path: Path, split_file: Path) -> dict[str, dict]:
    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    split_map = split_payload["subject_split"]
    subjects: dict[str, dict] = defaultdict(lambda: {"cases": set(), "labels": set()})
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subject = row["subject_id"]
            if split_map.get(subject) != "train":
                continue
            subjects[subject]["cases"].add(row["case_id"])
            subjects[subject]["labels"].add(int(row["anchor_label"]))
    if len(subjects) != 584:
        raise ValueError(f"expected 584 train subjects, got {len(subjects)}")
    return subjects


def _select_subjects(
    subjects: dict[str, dict], raw_train_root: Path, *, seed: int, count: int
) -> list[dict]:
    candidates = []
    for subject, metadata in sorted(subjects.items()):
        case_id = sorted(metadata["cases"])[0]
        t1c_path = raw_train_root / case_id / f"{case_id}-t1c.nii.gz"
        foreground_volume = int(np.count_nonzero(np.asanyarray(nib.load(str(t1c_path)).dataobj)))
        candidates.append(
            {
                "subject_id": subject,
                "case_id": case_id,
                "label_signature": "".join(
                    str(int(value in metadata["labels"])) for value in (1, 2, 3, 4)
                ),
                "foreground_volume": foreground_volume,
            }
        )
    if not 0 < count <= len(candidates):
        raise ValueError(f"invalid audit subject count: {count}")
    volumes = np.asarray([item["foreground_volume"] for item in candidates])
    boundaries = np.quantile(volumes, [0.25, 0.5, 0.75])
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for candidate in candidates:
        volume_bin = int(np.searchsorted(boundaries, candidate["foreground_volume"], side="right"))
        candidate["foreground_volume_quartile"] = volume_bin
        groups[(candidate["label_signature"], volume_bin)].append(candidate)

    allocations = {}
    fractional = []
    for key, members in groups.items():
        ideal = len(members) * count / len(candidates)
        allocations[key] = int(np.floor(ideal))
        fractional.append((ideal - allocations[key], key))
    remainder = count - sum(allocations.values())
    fractional.sort(key=lambda item: (-item[0], item[1]))
    for _, key in fractional:
        if remainder <= 0:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError("could not allocate the requested audit sample")

    selected = []
    for key, members in sorted(groups.items()):
        rng = np.random.default_rng(_stable_seed(seed, key))
        order = np.arange(len(members))
        rng.shuffle(order)
        selected.extend(members[index] for index in order[: allocations[key]])
    return sorted(selected, key=lambda item: item["subject_id"])


def _plot_cases(records: list[dict], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for axis, record in zip(axes.flat, records):
        case_dir = Path(record["_case_dir"])
        modalities, segmentation = _load_case(case_dir, record["case_id"])
        t1c = modalities[0]
        support, _ = build_explicit_brain_support(modalities, segmentation)
        raw = t1c != 0
        difference = raw ^ support
        z = int(np.argmax(difference.sum(axis=(0, 1)))) if difference.any() else t1c.shape[2] // 2
        values = t1c[raw]
        low, high = np.percentile(values, [0.5, 99.5]) if values.size else (0, 1)
        axis.imshow(t1c[:, :, z].T, cmap="gray", vmin=low, vmax=max(high, low + 1e-6), origin="lower")
        extra = raw[:, :, z] & ~support[:, :, z]
        missing = support[:, :, z] & ~raw[:, :, z]
        if extra.any():
            axis.contour(extra.T, levels=[0.5], colors="red", linewidths=0.7, origin="lower")
        if missing.any():
            axis.contour(missing.T, levels=[0.5], colors="cyan", linewidths=0.7, origin="lower")
        axis.set_title(f"{record['case_id']}\nDice={record['dice']:.4f}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(records) :]:
        axis.axis("off")
    fig.suptitle(title + " (red=raw-only, cyan=support-only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_audit(
    raw_root: Path,
    manifest_path: Path,
    split_file: Path,
    output_dir: Path,
    *,
    seed: int,
    subject_count: int = 64,
) -> dict:
    subjects = _subject_index(manifest_path, split_file)
    raw_train_root = raw_root / "train"
    selected = _select_subjects(
        subjects, raw_train_root, seed=seed, count=subject_count
    )
    records = []
    visuals = []
    for selected_subject in selected:
        case_id = selected_subject["case_id"]
        modalities, segmentation = _load_case(raw_train_root / case_id, case_id)
        support, lesion_outside_consensus = build_explicit_brain_support(
            modalities, segmentation
        )
        metrics = support_metrics(modalities[0] != 0, support)
        record = {
            **selected_subject,
            **metrics,
            "lesion_outside_consensus_voxels": lesion_outside_consensus,
        }
        records.append(record)
        visuals.append(
            {**record, "_case_dir": str(raw_train_root / case_id)}
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "subject_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "selected_subjects.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    by_dice = sorted(visuals, key=lambda item: item["dice"])
    worst = [{**item} for item in by_dice[:16]]
    rng = np.random.default_rng(seed)
    random_indices = np.sort(rng.choice(len(visuals), size=min(16, len(visuals)), replace=False))
    random_records = [{**visuals[index]} for index in random_indices]
    _plot_cases(worst, output_dir / "worst_16_cases.png", "Worst explicit-support agreement")
    _plot_cases(random_records, output_dir / "random_16_cases.png", "Random explicit-support audit")

    dice = np.asarray([record["dice"] for record in records])
    extra = np.asarray([record["extra_fraction"] for record in records])
    gates = {
        "median_dice_ge_0_99": bool(np.median(dice) >= 0.99),
        "p05_dice_ge_0_97": bool(np.percentile(dice, 5) >= 0.97),
        "median_extra_le_0_005": bool(np.median(extra) <= 0.005),
        "p95_extra_le_0_02": bool(np.percentile(extra, 95) <= 0.02),
    }
    summary = {
        "schema_version": 1,
        "method": "representative_case_per_train_subject_stratified_by_label_presence_and_foreground_volume",
        "seed": seed,
        "subject_count": len(records),
        "source_split": "train",
        "split_sha256": _sha256(split_file),
        "manifest_sha256": _sha256(manifest_path),
        "metrics": {
            "median_dice": float(np.median(dice)),
            "p05_dice": float(np.percentile(dice, 5)),
            "median_extra_fraction": float(np.median(extra)),
            "p95_extra_fraction": float(np.percentile(extra, 95)),
        },
        "automated_gates": gates,
        "automated_gates_pass": bool(all(gates.values())),
        "manual_qa_status": "pending",
        "normalization_decision": "pending_manual_qa",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--subject-count", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        args.raw_root,
        args.manifest,
        args.split_file,
        args.output_dir,
        seed=args.seed,
        subject_count=args.subject_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
