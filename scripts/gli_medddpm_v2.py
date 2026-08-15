#!/usr/bin/env python
"""Train and evaluate the mask-only Med-DDPM baseline on GLI v2 patches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


EXPECTED_REFERENCE_SHA = {
    "diffusion_model/trainer_brats.py": "d48d3fd192cb870b862cef3955472f077f1fad30c074f1aefa2ad6fb69455387",
    "diffusion_model/unet_brats.py": "45f7cb9a1ad0e7cb802b15eb8d86e8929aaf81d2514ae1e45e1dd9e5848ea6cc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def xyz_to_cdhw(array: np.ndarray) -> np.ndarray:
    if array.shape != (64, 64, 32):
        raise ValueError(f"expected XYZ (64,64,32), got {array.shape}")
    return np.transpose(array, (2, 0, 1))[None].copy()


def audit_reference(cfg: dict) -> list[dict[str, object]]:
    root = Path(cfg["paths"]["reference_root"])
    rows = []
    for relative, expected in EXPECTED_REFERENCE_SHA.items():
        path = root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"reference hash mismatch: {path}: {actual} != {expected}")
        rows.append({"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size})
    return rows


def read_split_rows(cfg: dict, split: str) -> list[dict[str, str]]:
    dataset_root = Path(cfg["paths"]["dataset_root"])
    manifest = dataset_root / "patch_64x64x32" / "manifest.csv"
    split_path = Path(cfg["paths"]["split_file"])
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    subject_split = split_payload["subject_split"]
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if subject_split[row["subject_id"]] == split]
    for row in selected:
        row["npz_path"] = str(dataset_root / "patch_64x64x32" / row["relative_path"])
    return selected


class V2MaskDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]):
        if not rows:
            raise RuntimeError("empty dataset")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        with np.load(row["npz_path"], allow_pickle=False) as data:
            target_xyz = np.asarray(data["t1c"], dtype=np.float32)
            seg_xyz = np.asarray(data["seg"], dtype=np.uint8)
        if not np.isfinite(target_xyz).all():
            raise ValueError(f"non-finite target: {row['relative_path']}")
        if float(target_xyz.min()) < -1.000001 or float(target_xyz.max()) > 1.000001:
            raise ValueError(f"target outside [-1,1]: {row['relative_path']}")
        target = xyz_to_cdhw(target_xyz)
        condition = xyz_to_cdhw((seg_xyz > 0).astype(np.float32))
        if not condition.any():
            raise ValueError(f"empty union: {row['relative_path']}")
        return {
            "target": torch.from_numpy(target),
            "condition": torch.from_numpy(condition),
            "relative_path": row["relative_path"],
            "subject_id": row["subject_id"],
        }


def audit_data(cfg: dict) -> dict[str, object]:
    counts = {}
    subjects = {}
    samples = []
    subject_sets = {}
    for split in ("train", "val", "test"):
        rows = read_split_rows(cfg, split)
        counts[split] = len(rows)
        subject_sets[split] = {row["subject_id"] for row in rows}
        subjects[split] = len(subject_sets[split])
        for row in rows[:8]:
            item = V2MaskDataset([row])[0]
            samples.append(
                {
                    "split": split,
                    "relative_path": row["relative_path"],
                    "target_shape": list(item["target"].shape),
                    "condition_shape": list(item["condition"].shape),
                    "target_min": float(item["target"].min()),
                    "target_max": float(item["target"].max()),
                    "union_voxels": int(item["condition"].sum()),
                }
            )
    if counts != {"train": 7772, "val": 1032, "test": 1038}:
        raise RuntimeError(f"patch count mismatch: {counts}")
    if subjects != {"train": 584, "val": 73, "test": 74}:
        raise RuntimeError(f"subject count mismatch: {subjects}")
    if any(subject_sets[a] & subject_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("patient split leakage")
    payload = {
        "checks_passed": True,
        "patch_counts": counts,
        "subject_counts": subjects,
        "manifest_sha256": sha256_file(Path(cfg["paths"]["dataset_root"]) / "patch_64x64x32" / "manifest.csv"),
        "split_sha256": sha256_file(Path(cfg["paths"]["split_file"])),
        "reference_assets": audit_reference(cfg),
        "sample_checks": samples,
    }
    json_dump(Path(cfg["paths"]["output_root"]) / "audit" / "data_contract.json", payload)
    return payload


def import_med_modules(cfg: dict):
    root = str(Path(cfg["paths"]["reference_root"]))
    if root not in sys.path:
        sys.path.insert(0, root)
    from diffusion_model.trainer_brats import EMA, GaussianDiffusion
    from diffusion_model.unet_brats import create_model

    return EMA, GaussianDiffusion, create_model


def build_diffusion(cfg: dict, device: torch.device, multi_gpu: bool):
    _, GaussianDiffusion, create_model = import_med_modules(cfg)
    model_cfg = cfg["model"]
    model = create_model(
        image_size=64,
        num_channels=int(model_cfg["num_channels"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        in_channels=2,
        out_channels=1,
        attention_resolutions=str(model_cfg["attention_resolutions"]),
    )
    if multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return GaussianDiffusion(
        model,
        image_size=64,
        depth_size=32,
        timesteps=int(model_cfg["timesteps"]),
        loss_type="l1",
        with_condition=True,
        channels=1,
    ).to(device)


def normalize_state_dict(state: dict[str, torch.Tensor], multi_gpu: bool) -> dict[str, torch.Tensor]:
    has_module = any(key.startswith("denoise_fn.module.") for key in state)
    if has_module and not multi_gpu:
        return {
            ("denoise_fn." + key[len("denoise_fn.module.") :]) if key.startswith("denoise_fn.module.") else key: value
            for key, value in state.items()
        }
    if not has_module and multi_gpu:
        return {
            ("denoise_fn.module." + key[len("denoise_fn.") :]) if key.startswith("denoise_fn.") else key: value
            for key, value in state.items()
        }
    return state


def checkpoint_path(cfg: dict, step: int) -> Path:
    return Path(cfg["paths"]["output_root"]) / "checkpoints" / f"model-step{step:06d}.pt"


def save_checkpoint(path: Path, step: int, model, ema_model, optimizer, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
        },
        path,
    )


def evaluate_loss(model, loader: DataLoader, device: torch.device, seed: int) -> float:
    seed_everything(seed)
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            target = batch["target"].to(device, non_blocking=True).float()
            condition = batch["condition"].to(device, non_blocking=True).float()
            values.append(float(model(target, condition_tensors=condition).detach().cpu()))
    return float(np.mean(values))


def run_train(cfg: dict, target_step: int) -> dict[str, object]:
    if target_step not in {500, 3000, 5000}:
        raise ValueError("target step must be one of 500, 3000, 5000")
    audit_data(cfg)
    seed = int(cfg["runtime"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda")
    train_rows = read_split_rows(cfg, "train")
    val_rows = read_split_rows(cfg, "val")
    train_loader = DataLoader(
        V2MaskDataset(train_rows),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        V2MaskDataset(val_rows),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
    )
    model = build_diffusion(cfg, device, multi_gpu=True)
    ema_model = deepcopy(model).eval()
    optimizer = Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    EMA, _, _ = import_med_modules(cfg)
    ema = EMA(float(cfg["training"]["ema_decay"]))
    start_step = 0
    candidates = [step for step in (500, 3000, 5000) if step < target_step and checkpoint_path(cfg, step).exists()]
    if candidates:
        resume = checkpoint_path(cfg, max(candidates))
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(normalize_state_dict(payload["model"], multi_gpu=True))
        ema_model.load_state_dict(normalize_state_dict(payload["ema"], multi_gpu=True))
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
    baseline_path = Path(cfg["paths"]["output_root"]) / "validation" / "step000000.json"
    if start_step == 0 and not baseline_path.exists():
        baseline_loss = evaluate_loss(ema_model, val_loader, device, seed + 17)
        json_dump(baseline_path, {"step": 0, "validation_loss": baseline_loss})
    iterator = iter(train_loader)
    log_path = Path(cfg["paths"]["output_root"]) / "logs" / "train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for step in range(start_step + 1, target_step + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        target = batch["target"].to(device, non_blocking=True).float()
        condition = batch["condition"].to(device, non_blocking=True).float()
        loss = model(target, condition_tensors=condition)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["max_grad_norm"])))
        if not np.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step % int(cfg["training"]["update_ema_every"]) == 0:
            ema.update_model_average(ema_model, model)
        if step == 1 or step % int(cfg["training"]["log_every"]) == 0:
            rec = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
                "elapsed_sec": time.time() - started,
                "sec_per_step": (time.time() - started) / max(step - start_step, 1),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    path = checkpoint_path(cfg, target_step)
    save_checkpoint(path, target_step, model, ema_model, optimizer, cfg)
    val_loss = evaluate_loss(ema_model, val_loader, device, seed + 17)
    result = {
        "step": target_step,
        "validation_loss": val_loss,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "runtime_sec": time.time() - started,
    }
    json_dump(Path(cfg["paths"]["output_root"]) / "validation" / f"step{target_step:06d}.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def load_checkpoint_model(cfg: dict, path: Path, device: torch.device):
    model = build_diffusion(cfg, device, multi_gpu=False)
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["ema"] if payload.get("ema") is not None else payload["model"]
    model.load_state_dict(normalize_state_dict(state, multi_gpu=False))
    return model.eval(), payload


def run_validate(cfg: dict, checkpoint: Path) -> dict[str, object]:
    device = torch.device("cuda")
    rows = read_split_rows(cfg, "val")
    loader = DataLoader(V2MaskDataset(rows), batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=4)
    model, payload = load_checkpoint_model(cfg, checkpoint, device)
    loss = evaluate_loss(model, loader, device, int(cfg["runtime"]["seed"]) + 17)
    item = V2MaskDataset(rows)[0]
    condition = item["condition"].unsqueeze(0).to(device)
    seed_everything(int(cfg["runtime"]["seed"]) + 101)
    with torch.inference_mode():
        sample = model.sample(batch_size=1, condition_tensors=condition).detach().cpu().numpy()
    result = {
        "step": int(payload["step"]),
        "validation_loss": loss,
        "sample_shape": list(sample.shape),
        "sample_min": float(sample.min()),
        "sample_max": float(sample.max()),
        "sample_std": float(sample.std()),
        "sample_finite": bool(np.isfinite(sample).all()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    if not result["sample_finite"] or result["sample_std"] <= 1e-4:
        raise RuntimeError(json.dumps(result, indent=2))
    json_dump(Path(cfg["paths"]["output_root"]) / "validation" / f"gate-step{int(payload['step']):06d}.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_infer(cfg: dict, checkpoint: Path, selection: Path, filtered_root: Path) -> dict[str, object]:
    device = torch.device("cuda")
    model, payload = load_checkpoint_model(cfg, checkpoint, device)
    selected = json.loads(selection.read_text(encoding="utf-8"))["selected_records"]
    output_root = Path(cfg["paths"]["output_root"]) / "test10" / "generations" / "med_ddpm_t1c" / "20260806"
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    seed = 20260806
    dataset_root = Path(cfg["paths"]["dataset_root"]) / "patch_64x64x32"
    for index, record in enumerate(selected):
        relative = str(record["relative_path"])
        with np.load(dataset_root / relative, allow_pickle=False) as source:
            real_xyz = np.asarray(source["t1c"], dtype=np.float32)
        with np.load(filtered_root / "patch_64x64x32" / relative, allow_pickle=False) as overlay:
            union_xyz = np.asarray(overlay["seg_xyz"], dtype=np.uint8) > 0
        real = xyz_to_cdhw(real_xyz)
        union = xyz_to_cdhw(union_xyz.astype(np.uint8))
        condition = torch.from_numpy(union.astype(np.float32)).unsqueeze(0).to(device)
        seed_everything(seed + index)
        started = time.time()
        with torch.inference_mode():
            raw = model.sample(batch_size=1, condition_tensors=condition)[0].detach().cpu().numpy().astype(np.float32)
        composited = real.copy()
        composited[union > 0] = raw[union > 0]
        if float(np.max(np.abs(composited[union == 0] - real[union == 0]))) != 0.0:
            raise RuntimeError("background restoration failed")
        path = output_root / f"{record['case_id']}.npz"
        np.savez_compressed(
            path,
            raw=raw,
            composited=composited,
            union=union.astype(np.uint8),
            case_id=np.asarray(record["case_id"]),
            seed=np.asarray(seed + index, dtype=np.int64),
            root_seed=np.asarray(seed, dtype=np.int64),
            nfe=np.asarray(250, dtype=np.int64),
            checkpoint_step=np.asarray(int(payload["step"]), dtype=np.int64),
        )
        records.append(
            {
                "sample_id": record["sample_id"],
                "case_id": record["case_id"],
                "output": str(path),
                "sha256": sha256_file(path),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "raw_std": float(raw.std()),
                "finite": bool(np.isfinite(raw).all()),
                "background_max_abs_error": 0.0,
                "runtime_sec": time.time() - started,
            }
        )
    result = {
        "checks_passed": len(records) == 10 and all(row["finite"] for row in records),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(payload["step"]),
        "seed": seed,
        "nfe": 250,
        "records": records,
    }
    json_dump(Path(cfg["paths"]["output_root"]) / "test10" / "inference_contract.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-data")
    train = sub.add_parser("train")
    train.add_argument("--target-step", type=int, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--checkpoint", type=Path, required=True)
    infer = sub.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--selection", type=Path, required=True)
    infer.add_argument("--filtered-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "audit-data":
        print(json.dumps(audit_data(cfg), indent=2))
    elif args.command == "train":
        run_train(cfg, args.target_step)
    elif args.command == "validate":
        run_validate(cfg, args.checkpoint)
    else:
        run_infer(cfg, args.checkpoint, args.selection, args.filtered_root)


if __name__ == "__main__":
    main()
