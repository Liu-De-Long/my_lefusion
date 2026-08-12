#!/usr/bin/env python3
"""Fail-closed audit for the frozen exp020 test-200 pseudo-mask sidecars."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gli_pseudomask_pipeline import PATCH_DIR, lesion_histograms, sha256_file, validate_overlay


def _file_paths(root: Path) -> set[str]:
    with (root / "files.csv").open(newline="", encoding="utf-8") as handle:
        return {str(row["relative_path"]) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = [str(value) for value in selection["selected_relative_paths"]]
    expected = set(paths)
    if len(paths) != 200 or len(expected) != 200:
        raise ValueError("selection must contain exactly 200 unique paths")
    manifest_sha = sha256_file(args.manifest)
    contracts = {}
    for name, root in (("direct", args.direct_root), ("filtered", args.filtered_root)):
        contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
        if contract.get("selection_manifest_sha256") != manifest_sha:
            raise ValueError(f"{name} selection hash mismatch")
        if int(contract.get("file_count", -1)) != 200:
            raise ValueError(f"{name} file count mismatch")
        if bool(contract.get("unselected_test_accessed", True)):
            raise ValueError(f"{name} accessed unselected test data")
        if _file_paths(root) != expected:
            raise ValueError(f"{name} files.csv does not equal the selection")
        contracts[name] = contract

    direct_voxels = filtered_voxels = fallback_nonempty = 0
    for relative in paths:
        with np.load(args.source_root / PATCH_DIR / relative, allow_pickle=False) as source:
            t1c = np.asarray(source["t1c"], dtype=np.float32)
            truth_union = np.asarray(source["seg"]) > 0
        direct = validate_overlay(args.direct_root / PATCH_DIR / relative)
        filtered = validate_overlay(args.filtered_root / PATCH_DIR / relative)
        direct_union = direct["seg_xyz"] > 0
        filtered_union = filtered["seg_xyz"] > 0
        if not np.array_equal(direct_union, truth_union):
            raise ValueError(f"direct union mismatch: {relative}")
        if np.any(filtered_union & ~direct_union):
            raise ValueError(f"filtered is not a direct subset: {relative}")
        if direct_union.any() and not filtered_union.any():
            raise ValueError(f"filtered fallback failed: {relative}")
        for name, overlay in (("direct", direct), ("filtered", filtered)):
            expected_hist = lesion_histograms(t1c, overlay["seg_xyz"])
            if not np.allclose(overlay["hist"], expected_hist, atol=1e-6, rtol=0):
                raise ValueError(f"{name} histogram mismatch: {relative}")
        direct_voxels += int(direct_union.sum())
        filtered_voxels += int(filtered_union.sum())
        fallback_nonempty += int(direct_union.any() and filtered_union.any())

    payload = {
        "schema_version": 1,
        "selection_manifest_sha256": manifest_sha,
        "selected_count": len(paths),
        "direct_contract_sha256": sha256_file(args.direct_root / "contract.json"),
        "filtered_contract_sha256": sha256_file(args.filtered_root / "contract.json"),
        "direct_union_exact_gt_total": True,
        "filtered_union_subset_direct": True,
        "overlay_histograms_recomputed": True,
        "direct_union_voxels": direct_voxels,
        "filtered_union_voxels": filtered_voxels,
        "filtered_union_coverage": filtered_voxels / direct_voxels,
        "nonempty_filtered_patch_count": fallback_nonempty,
        "unselected_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
