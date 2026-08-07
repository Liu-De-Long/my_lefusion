"""Run exp012 on the same eight cases with four fixed union target classes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LABELS = {1: "netc", 2: "snfh", 3: "et", 4: "rc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name", default="gli_exp012_p64_four_class_qa"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=list(LABELS),
        choices=list(LABELS),
        help="Disjoint label subset, useful for two mutually exclusive workers.",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    inference = repo / "LeFusion" / "inference" / "inference.py"
    analysis = repo / "scripts" / "gli_analyze_four_class_counterfactual_qa.py"
    for label in args.labels:
        output = args.output_root / f"target_{label}_{LABELS[label]}"
        command = [
            sys.executable,
            str(inference),
            "--config-name",
            args.config_name,
            f"output.root={output}",
            f"input_policy.target_label={label}",
            "output.max_batches=null",
            "output.overwrite=false",
            "output.resume=true",
        ]
        subprocess.run(command, cwd=repo, check=True)
    if not args.skip_analysis and set(args.labels) == set(LABELS):
        subprocess.run(
            [sys.executable, str(analysis), "--input-root", str(args.output_root)],
            cwd=repo,
            check=True,
        )


if __name__ == "__main__":
    main()
