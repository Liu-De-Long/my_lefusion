"""Run the five frozen eight-case QA variants for one short GLI checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "qa_original_real": {
        "conditioning.source": "real",
        "conditioning.selection": "nearest",
        "input_policy.lesion_mode": "original_multilabel",
    },
    "qa_original_cluster_first": {
        "conditioning.source": "cluster",
        "conditioning.selection": "first",
        "input_policy.lesion_mode": "original_multilabel",
    },
    "qa_original_cluster_last": {
        "conditioning.source": "cluster",
        "conditioning.selection": "last",
        "input_policy.lesion_mode": "original_multilabel",
    },
    "qa_union_cluster_first": {
        "conditioning.source": "cluster",
        "conditioning.selection": "first",
        "input_policy.lesion_mode": "union_single_label_cycle",
    },
    "qa_union_cluster_last": {
        "conditioning.source": "cluster",
        "conditioning.selection": "last",
        "input_policy.lesion_mode": "union_single_label_cycle",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    inference = repo / "LeFusion" / "inference" / "inference.py"
    contact_sheet = repo / "scripts" / "gli_build_qa_contact_sheet.py"
    for variant, overrides in VARIANTS.items():
        output = args.output_root / variant
        command = [
            sys.executable,
            str(inference),
            "--config-name",
            args.config_name,
            f"output.root={output}",
            "output.max_batches=null",
            "output.overwrite=false",
            "output.resume=true",
        ]
        command.extend(f"{key}={value}" for key, value in overrides.items())
        subprocess.run(command, cwd=repo, check=True)
        subprocess.run(
            [
                sys.executable,
                str(contact_sheet),
                "--input-dir",
                str(output / "qa"),
                "--output",
                str(output / "qa_contact_sheet.png"),
                "--title",
                f"{args.config_name} {variant}",
            ],
            cwd=repo,
            check=True,
        )


if __name__ == "__main__":
    main()
