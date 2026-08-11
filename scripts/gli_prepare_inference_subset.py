"""Freeze deterministic GLI validation-QA or fractional-test selection manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
import sys

sys.path.insert(0, str(ROOT / "LeFusion"))

from checkpointing import sha256_file  # noqa: E402
from dataset.gli_hist import GLIDataset  # noqa: E402
from inference.gli_selection import (  # noqa: E402
    build_selection_manifest,
    select_stratified_count,
    select_stratified_fraction,
    select_val_qa,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _lesion_label_counts(dataset: GLIDataset) -> dict[str, int]:
    result = {}
    for record in dataset.records:
        path = dataset.size_root / str(record["relative_path"])
        with np.load(path, allow_pickle=False) as arrays:
            labels = np.unique(np.asarray(arrays["seg"], dtype=np.uint8))
        result[str(record["relative_path"])] = int(np.count_nonzero(labels > 0))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--patch-size", default="64x64x32")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--mode", choices=("val-qa", "fraction"), required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = GLIDataset(
        args.dataset_root,
        args.patch_size,
        split=args.split,
        split_file=args.split_file,
    )
    if args.mode == "val-qa":
        indices, selection = select_val_qa(
            dataset.records,
            seed=args.seed,
            lesion_label_counts=_lesion_label_counts(dataset),
        )
    else:
        if args.split != "test":
            raise ValueError("fraction mode is reserved for the held-out test split")
        if args.count is not None:
            indices, selection = select_stratified_count(
                dataset.records, count=args.count, seed=args.seed
            )
        else:
            indices, selection = select_stratified_fraction(
                dataset.records, fraction=args.fraction, seed=args.seed
            )
    split_file = Path(args.split_file).expanduser()
    payload = build_selection_manifest(
        dataset.records,
        indices,
        selection=selection,
        split=args.split,
        patch_size_xyz=dataset.patch_size_xyz,
        shard_count=args.shard_count,
        provenance={
            "dataset_manifest": str(dataset.manifest_path),
            "dataset_manifest_sha256": sha256_file(dataset.manifest_path),
            "split_file": str(split_file),
            "split_sha256": sha256_file(split_file),
        },
    )
    output = Path(args.output).expanduser()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"frozen selection manifest already exists with different content: {output}"
            )
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return
    _atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
