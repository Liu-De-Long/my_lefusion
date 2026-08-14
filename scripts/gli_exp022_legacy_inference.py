#!/usr/bin/env python
"""Audit and run the four frozen legacy Fig. 2 baselines for exp022.

This file is copied into the bounded LeFusion-main workspace.  It never trains
or rewrites a checkpoint; all generated files stay below ``--stage-root``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import numpy as np


EXPECTED_ASSETS = {
    "repaint3d": (
        "outputs/repaint3d_t1c_norm_neg1pos1_repaired_20260731/full/checkpoints/model-step004000.pt",
        "4d4c0c50c04cc33e86baf872dd61b9fb81520db49fa8979ebb4afbfa9309ff3d",
    ),
    "med_ddpm_t1c": (
        "outputs/t1c_repaint_med_ours2_main_20k_20260728/checkpoints/med_ddpm_t1c/model-20.pt",
        "3e789a8daf21c1c40ce221e75fc5f8bdce92a7f88920126d9a63db81d883ca0c",
    ),
    "pix2pix_3d_inpaint": (
        "outputs/t1c_two_day_external_addon_20260729/checkpoints/pix2pix_3d_inpaint/model-20000.pt",
        "9aacf05293d96858fe8800ef37e9914c49ff70beb7615e6cbebf7b80a20405e8",
    ),
    "latent_rflow_3d": (
        "outputs/t1c_two_day_external_addon_20260729/checkpoints/latent_rflow_3d/model-20000.pt",
        "16572746329772be44d61e0daf2853b9918d7439004042a30246d105def319f3",
    ),
    "latent_rflow_vae": (
        "outputs/t1c_two_day_external_addon_20260729/checkpoints/latent_rflow_vae/model-20000.pt",
        "e6d091760c96d415b2a01b095ef73daa035e0328581125d7df99c32e163f38b5",
    ),
}

EXPECTED_SOURCE_ASSETS = {
    "repaint_infer": ("project", ".tmp_t1c_adapters/repaint/infer_t1c_repaint3d.py", "d7a0a861806a244c8cdd0f9160d47dc5e44cfa309edf02ed91a42c4d6c5b476d"),
    "repaint_model": ("project", ".tmp_t1c_adapters/repaint/train_t1c_3d_prior_20k.py", "c62536147473700e20edbe904ff586d6d280d314247ce2702f7015e033586ece"),
    "repaint_config": ("project", ".tmp_t1c_adapters/repaint/configs/t1c_main_20k_repaint3d_20260727.yaml", "38c4ff9e769fb1f66257699fe6954661011a7a4ad5cb0adf51c44332f27e8b8a"),
    "med_infer": ("project", ".tmp_t1c_adapters/med/infer_t1c_seed20260721.py", "5ab478b4241626e4a86422950a08f1c6326f77979b9e372995671db8df71a368"),
    "med_adapter": ("project", ".tmp_t1c_adapters/med/train_t1c_main_20k.py", "2b32f07799a7d7dba39ab9d729ff627a8c5aaafeb97a6aca4bee872ec56024d9"),
    "med_config": ("project", ".tmp_t1c_adapters/med/configs/t1c_main_20k_med_ddpm_t1c.yaml", "dee5ab9effdd6266c0e5c2f9577df3f66c1d44a81c8b1c1f209e0b7f8af2828a"),
    "med_diffusion": ("workspace", "method_reference_repos/med-ddpm-main/diffusion_model/trainer_brats.py", "d48d3fd192cb870b862cef3955472f077f1fad30c074f1aefa2ad6fb69455387"),
    "med_unet": ("workspace", "method_reference_repos/med-ddpm-main/diffusion_model/unet_brats.py", "45f7cb9a1ad0e7cb802b15eb8d86e8929aaf81d2514ae1e45e1dd9e5848ea6cc"),
    "pix_rflow_experiment": ("project", "tools/t1c_two_day_addon/experiment.py", "956d86b3d4d397d8f6dda7e0e4859309e2f91a2dea6c2d99466c7eccb3b78553"),
}
EXPECTED_NFE = {
    "repaint3d": 300,
    "med_ddpm_t1c": 250,
    "pix2pix_3d_inpaint": 1,
    "latent_rflow_3d": 50,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_assets(project: Path, stage_root: Path) -> dict[str, object]:
    rows = []
    for method, (relative, expected_sha) in EXPECTED_ASSETS.items():
        path = project / relative
        actual_sha = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("ema") or checkpoint.get("model")
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"{method}: checkpoint has no usable state dict")
        row = {
            "method": method,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "sha256_matches": actual_sha == expected_sha,
            "checkpoint_step": int(checkpoint.get("step", -1)),
            "state_key": "ema" if checkpoint.get("ema") is not None else "model",
            "state_tensors": len(state),
            "all_state_tensors_finite": all(
                bool(torch.isfinite(value).all())
                for value in state.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            ),
        }
        if not row["sha256_matches"] or not row["all_state_tensors_finite"]:
            raise RuntimeError(json.dumps(row, indent=2))
        rows.append(row)
    source_rows = []
    for name, (root_kind, relative, expected_sha) in EXPECTED_SOURCE_ASSETS.items():
        root = project if root_kind == "project" else project.parent
        path = root / relative
        actual_sha = sha256_file(path)
        row = {
            "name": name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "sha256_matches": actual_sha == expected_sha,
        }
        if not row["sha256_matches"]:
            raise RuntimeError(json.dumps(row, indent=2))
        source_rows.append(row)
    payload = {
        "schema_version": 1,
        "checks_passed": True,
        "checkpoint_assets": rows,
        "inference_source_assets": source_rows,
    }
    stage_root.mkdir(parents=True, exist_ok=True)
    (stage_root / "legacy_asset_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def write_config(source: Path, target: Path, output_root: Path) -> None:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["paths"] = dict(payload["paths"])
    payload["paths"]["output_root"] = str(output_root)
    payload["runtime"] = dict(payload["runtime"])
    payload["runtime"]["gpus"] = [0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_checked(command: list[str], cwd: Path, project: Path) -> None:
    print(json.dumps({"cwd": str(cwd), "command": command}), flush=True)
    env = os.environ.copy()
    med_reference = project.parent / "method_reference_repos/med-ddpm-main"
    env["PYTHONPATH"] = (
        str(project) + os.pathsep + str(med_reference) + os.pathsep + env.get("PYTHONPATH", "")
    )
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_repaint(project: Path, stage_root: Path, seed: int, limit: int) -> None:
    adapter = project / ".tmp_t1c_adapters/repaint"
    config = stage_root / "configs/repaint.yaml"
    write_config(
        adapter / "configs/t1c_main_20k_repaint3d_20260727.yaml",
        config,
        stage_root / "outputs/repaint",
    )
    checkpoint = project / EXPECTED_ASSETS["repaint3d"][0]
    command = [
        sys.executable,
        "infer_t1c_repaint3d.py",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
        "--manifest",
        str(stage_root / "manifests/test_generation.csv"),
        "--jump-length",
        "5",
        "--jump-n-sample",
        "1",
        "--overwrite",
    ]
    if limit:
        command += ["--limit", str(limit)]
    run_checked(command, adapter, project)


def run_med(project: Path, stage_root: Path, seed: int, limit: int) -> None:
    adapter = project / ".tmp_t1c_adapters/med"
    config = stage_root / "configs/med.yaml"
    write_config(
        adapter / "configs/t1c_main_20k_med_ddpm_t1c.yaml",
        config,
        stage_root / "outputs/med",
    )
    checkpoint = project / EXPECTED_ASSETS["med_ddpm_t1c"][0]
    command = [
        sys.executable,
        "infer_t1c_seed20260721.py",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
        "--manifest",
        str(stage_root / "manifests/test_generation.csv"),
        "--overwrite",
    ]
    if limit:
        command += ["--limit", str(limit)]
    run_checked(command, adapter, project)


def run_external(
    project: Path, stage_root: Path, seed: int, limit: int, stage: str
) -> None:
    source = project / "tools/t1c_two_day_addon/experiment.py"
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    module = load_module(source, "exp022_legacy_external")
    args = SimpleNamespace(
        stage=stage,
        manifest_root=str(stage_root),
        run_root=str(stage_root / "outputs/external"),
        latent_channels=4,
        limit_cases=int(limit),
        cache_root="",
        checkpoint=str(project / EXPECTED_ASSETS[stage][0]),
        vae_checkpoint=(
            str(project / EXPECTED_ASSETS["latent_rflow_vae"][0])
            if stage == "latent_rflow_3d"
            else ""
        ),
        seed=int(seed),
        device="cuda:0",
        inference_steps=50,
        overwrite=True,
    )
    status = int(module.infer(args))
    if status != 0:
        raise RuntimeError(f"{stage} inference failed with status {status}")


def validate_outputs(stage_root: Path, methods: tuple[str, ...], seed: int, limit: int) -> None:
    def external_native_normalize(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        values = image[np.isfinite(image) & (image != 0)]
        if values.size < 16:
            values = image[np.isfinite(image)]
        if values.size == 0:
            return np.zeros_like(image, dtype=np.float32)
        lo, hi = np.percentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.clip(image, 0.0, 1.0).astype(np.float32)
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    with (stage_root / "manifests/test_generation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]
    locations = {
        "repaint3d": stage_root / f"outputs/repaint/generations/repaint3d/{seed}",
        "med_ddpm_t1c": stage_root / f"outputs/med/generations/med_ddpm_t1c/{seed}",
        "pix2pix_3d_inpaint": stage_root / f"outputs/external/generations/pix2pix_3d_inpaint/{seed}",
        "latent_rflow_3d": stage_root / f"outputs/external/generations/latent_rflow_3d/{seed}",
    }
    zero_one_methods = {"pix2pix_3d_inpaint", "latent_rflow_3d"}
    checks = []
    for method in methods:
        for row in rows:
            case_id = row["case_id"]
            path = locations[method] / f"{case_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                raw = np.asarray(data["raw"], dtype=np.float32)
                composited = np.asarray(data["composited"], dtype=np.float32)
                output_union = np.asarray(data["union"], dtype=np.uint8) > 0
                output_nfe = int(data["nfe"])
            union = np.load(row["target_union_path"]).astype(bool)
            if method in zero_one_methods:
                # The frozen Pix2Pix/RFlow definition applies its own 0.5/99.5
                # percentile normalization to the manifest's native T1c path.
                real = external_native_normalize(np.load(row["real_t1c_path"]))
                masked = real * (~union)
                fill_value = 0.0
                valid_range = (-1.0e-6, 1.0 + 1.0e-6)
            else:
                real = np.load(row["real_t1c_path"]).astype(np.float32)
                masked = np.load(row["masked_background_path"]).astype(np.float32)
                fill_value = -1.0
                valid_range = (-1.000001, 1.000001)
            if raw.shape != (1, 32, 64, 64) or not np.isfinite(raw).all():
                raise RuntimeError(f"{method}/{case_id}: invalid raw output")
            if output_nfe != EXPECTED_NFE[method]:
                raise RuntimeError(f"{method}/{case_id}: NFE mismatch {output_nfe}")
            if not np.array_equal(output_union, union):
                raise RuntimeError(f"{method}/{case_id}: shared union mismatch")
            if not np.all(masked[union] == fill_value):
                raise RuntimeError(f"{method}/{case_id}: masked input leaks hole intensities")
            background_error = float(np.max(np.abs(composited[~union] - real[~union])))
            if background_error != 0.0:
                raise RuntimeError(f"{method}/{case_id}: nonzero background error")
            if float(raw.min()) < valid_range[0] or float(raw.max()) > valid_range[1]:
                raise RuntimeError(f"{method}/{case_id}: output outside declared domain")
            checks.append(
                {
                    "method": method,
                    "case_id": case_id,
                    "output": str(path),
                    "output_sha256": sha256_file(path),
                    "shape": list(raw.shape),
                    "nfe": output_nfe,
                    "raw_min": float(raw.min()),
                    "raw_max": float(raw.max()),
                    "finite": True,
                    "shared_union_matches": True,
                    "masked_hole_fill_value": fill_value,
                    "masked_hole_has_no_real_intensity": True,
                    "background_max_abs_error": background_error,
                }
            )
    payload = {
        "schema_version": 1,
        "checks_passed": True,
        "seed": seed,
        "limit": limit,
        "methods": list(methods),
        "checks": checks,
    }
    suffix = "preflight" if limit == 1 else "formal"
    (stage_root / f"legacy_output_validation_{suffix}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("audit", "validate", "repaint3d", "med_ddpm_t1c", "pix2pix_3d_inpaint", "latent_rflow_3d", "all"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    audit = audit_assets(args.project, args.stage_root)
    print(json.dumps(audit, indent=2), flush=True)
    all_methods = ("repaint3d", "med_ddpm_t1c", "pix2pix_3d_inpaint", "latent_rflow_3d")
    methods = (
        all_methods
        if args.method in {"all", "validate"}
        else (() if args.method == "audit" else (args.method,))
    )
    for method in (() if args.method == "validate" else methods):
        if method == "repaint3d":
            run_repaint(args.project, args.stage_root, args.seed, args.limit)
        elif method == "med_ddpm_t1c":
            run_med(args.project, args.stage_root, args.seed, args.limit)
        else:
            run_external(args.project, args.stage_root, args.seed, args.limit, method)
    if methods:
        validate_outputs(args.stage_root, tuple(methods), args.seed, args.limit)


if __name__ == "__main__":
    main()
