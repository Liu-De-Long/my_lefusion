#!/usr/bin/env python
"""Package the frozen exp022 v2 inputs for legacy baseline inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xyz_to_cdhw(array: np.ndarray) -> np.ndarray:
    return np.transpose(array, (2, 0, 1))[None].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--legacy-stage-root", required=True)
    args = parser.parse_args()

    payload = json.loads(args.selection.read_text(encoding="utf-8"))
    if payload.get("split") != "test" or int(payload.get("selected_count", -1)) != 10:
        raise ValueError("selection must be the frozen ten-patch test manifest")
    size_root = args.dataset_root / "patch_64x64x32"
    overlay_root = args.filtered_root / "patch_64x64x32"
    inputs_root = args.output_root / "inputs"
    manifests_root = args.output_root / "manifests"
    inputs_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    rows = []
    audit_rows = []
    for record in payload["selected_records"]:
        sample_id = str(record["sample_id"])
        relative_path = str(record["relative_path"])
        source_path = size_root / relative_path
        overlay_path = overlay_root / relative_path
        with np.load(source_path, allow_pickle=False) as source:
            t1c_xyz = np.asarray(source["t1c"], dtype=np.float32)
        with np.load(overlay_path, allow_pickle=False) as overlay:
            seg_xyz = np.asarray(overlay["seg_xyz"], dtype=np.uint8)
            lesion_mask_xyz = np.asarray(overlay["lesion_mask_xyz"], dtype=np.uint8)
            hist = np.asarray(overlay["hist"], dtype=np.float32)
        if t1c_xyz.shape != (64, 64, 32) or seg_xyz.shape != (64, 64, 32):
            raise ValueError(f"shape mismatch for {relative_path}")
        if lesion_mask_xyz.shape != (4, 64, 64, 32) or hist.shape != (4, 16):
            raise ValueError(f"filtered condition mismatch for {relative_path}")
        union_xyz = seg_xyz > 0
        if not bool(union_xyz.any()):
            raise ValueError(f"empty filtered union for {relative_path}")
        if not np.array_equal(lesion_mask_xyz.astype(bool).any(axis=0), union_xyz):
            raise ValueError(f"filtered scalar/four-channel mismatch for {relative_path}")

        real_neg = xyz_to_cdhw(t1c_xyz).astype(np.float32)
        union = xyz_to_cdhw(union_xyz.astype(np.uint8))
        real_zero_one = np.clip((real_neg + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        masked_neg = real_neg.copy()
        masked_neg[union > 0] = -1.0
        masked_zero_one = real_zero_one.copy()
        masked_zero_one[union > 0] = 0.0
        sample_root = inputs_root / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        local_paths = {
            "real_neg": sample_root / "real_t1c_neg1_pos1.npy",
            "real_zero_one": sample_root / "real_t1c_zero_one.npy",
            "masked_neg": sample_root / "masked_t1c_neg1_pos1.npy",
            "masked_zero_one": sample_root / "masked_t1c_zero_one.npy",
            "union": sample_root / "shared_filtered_union.npy",
            "four_mask": sample_root / "filtered_four_mask_xyz.npy",
            "hist": sample_root / "filtered_hist.npy",
        }
        np.save(local_paths["real_neg"], real_neg)
        np.save(local_paths["real_zero_one"], real_zero_one)
        np.save(local_paths["masked_neg"], masked_neg)
        np.save(local_paths["masked_zero_one"], masked_zero_one)
        np.save(local_paths["union"], union.astype(np.uint8))
        np.save(local_paths["four_mask"], lesion_mask_xyz)
        np.save(local_paths["hist"], hist)

        legacy_sample_root = Path(args.legacy_stage_root) / "inputs" / sample_id
        rows.append(
            {
                "sample_id": sample_id,
                "case_id": str(record["case_id"]),
                "subject_id": str(record["subject_id"]),
                "real_t1c_path": str(legacy_sample_root / "real_t1c_neg1_pos1.npy"),
                "real_t1c_zero_one_path": str(legacy_sample_root / "real_t1c_zero_one.npy"),
                "masked_background_path": str(legacy_sample_root / "masked_t1c_neg1_pos1.npy"),
                "masked_zero_one_path": str(legacy_sample_root / "masked_t1c_zero_one.npy"),
                "target_union_path": str(legacy_sample_root / "shared_filtered_union.npy"),
            }
        )
        audit_rows.append(
            {
                "sample_id": sample_id,
                "relative_path": relative_path,
                "case_id": str(record["case_id"]),
                "subject_id": str(record["subject_id"]),
                "filtered_union_voxels": int(union.sum()),
                "gt_union_voxels": int(record["union_volume_voxels"]),
                "retained_coverage": float(union.sum() / int(record["union_volume_voxels"])),
                "source_npz_sha256": sha256_file(source_path),
                "filtered_overlay_sha256": sha256_file(overlay_path),
                "inside_masked_neg_max": float(np.max(np.abs(masked_neg[union > 0] + 1.0))),
                "inside_masked_zero_one_max": float(np.max(np.abs(masked_zero_one[union > 0]))),
            }
        )

    manifest_path = manifests_root / "test_generation.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema_version": 1,
        "selection": str(args.selection),
        "selection_sha256": sha256_file(args.selection),
        "filtered_contract": str(args.filtered_root / "contract.json"),
        "filtered_contract_sha256": sha256_file(args.filtered_root / "contract.json"),
        "sample_count": len(rows),
        "all_unique_subjects": len({row["subject_id"] for row in rows}) == 10,
        "all_filtered_unions_nonempty": all(row["filtered_union_voxels"] > 0 for row in audit_rows),
        "records": audit_rows,
    }
    (args.output_root / "input_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
