"""CLI for the BraTS2024 GLI p64 weak-supervision classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LeFusion.classifier.data import build_labeled_subset
from LeFusion.classifier.engine import (
    run_cpu_preflight,
    run_evaluation,
    run_gpu_preflight,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-subset")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--split-file", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--count", type=int, default=1000)
    prepare.add_argument("--seed", type=int, default=20260806)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume", action="store_true")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path)

    gpu_preflight = subparsers.add_parser("gpu-preflight")
    gpu_preflight.add_argument("--config", type=Path, required=True)
    gpu_preflight.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-subset":
        result = build_labeled_subset(
            args.dataset_root,
            args.split_file,
            args.output,
            count=args.count,
            seed=args.seed,
        )
    elif args.command == "train":
        result = run_training(args.config, resume=args.resume)
    elif args.command == "preflight":
        result = run_cpu_preflight(args.config, output_path=args.output)
    elif args.command == "gpu-preflight":
        result = run_gpu_preflight(args.config, output_path=args.output)
    else:
        result = run_evaluation(
            args.config,
            args.checkpoint,
            split=args.split,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
