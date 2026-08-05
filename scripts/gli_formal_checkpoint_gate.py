"""Validate formal GLI best-EMA inference and zero-update latest resume."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).parents[1]
import sys

sys.path.insert(0, str(ROOT / "LeFusion"))

from checkpointing import (  # noqa: E402
    canonical_config_hash,
    load_diffusion_checkpoint,
    load_training_checkpoint,
    sha256_file,
)
from ddpm import Trainer  # noqa: E402
from get_dataset.get_dataset import get_train_dataset, get_validation_dataset  # noqa: E402
from train.train import build_model_and_diffusion, set_global_seed, validate_training_config  # noqa: E402
from train.validation import run_gli_validation  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _expected_model_metadata(cfg: DictConfig) -> dict:
    return {
        "data_type": "gli",
        "diffusion_num_channels": int(cfg.model.diffusion_num_channels),
        "cond_dim": int(cfg.model.cond_dim),
        "base_dim": int(cfg.model.get("base_dim") or cfg.model.diffusion_img_size),
        "spatial_shape_dhw": [int(value) for value in cfg.model.spatial_shape_dhw],
        "timesteps": int(cfg.model.timesteps),
        "temporal_max_distance": int(cfg.model.temporal_max_distance),
    }


def _validate_provenance(cfg: DictConfig, dataset, checkpoint: dict) -> None:
    metadata = checkpoint["metadata"]
    resolved = checkpoint["resolved_config"]
    if canonical_config_hash(resolved) != str(metadata["config_hash"]):
        raise ValueError("checkpoint resolved_config hash does not match metadata")
    current = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if canonical_config_hash(current) != str(metadata["config_hash"]):
        raise ValueError("current formal config does not match checkpoint config hash")
    if sha256_file(dataset.manifest_path) != str(metadata["manifest_hash"]):
        raise ValueError("current dataset manifest does not match checkpoint")
    if sha256_file(cfg.dataset.split_file) != str(metadata["split_hash"]):
        raise ValueError("current split file does not match checkpoint")


def _create_resume_trainer(
    cfg: DictConfig,
    device: torch.device,
    checkpoint: dict,
):
    train_dataset, train_sampler = get_train_dataset(cfg)
    validation_dataset = get_validation_dataset(cfg)
    diffusion = build_model_and_diffusion(cfg, device, use_data_parallel=True)
    return Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=int(cfg.model.batch_size),
        save_and_sample_every=int(cfg.model.save_and_sample_every),
        train_lr=float(cfg.model.train_lr),
        train_num_steps=int(cfg.model.train_num_steps),
        gradient_accumulate_every=int(cfg.model.gradient_accumulate_every),
        ema_decay=float(cfg.model.ema_decay),
        amp=bool(cfg.model.amp),
        num_sample_rows=int(cfg.model.num_sample_rows),
        results_folder=cfg.model.results_folder,
        num_workers=int(cfg.model.num_workers),
        device=device,
        max_grad_norm=float(cfg.model.max_grad_norm),
        train_sampler=train_sampler,
        validation_dataset=validation_dataset,
        validation_config=cfg.validation,
        checkpoint_config=cfg.checkpoint,
        checkpoint_metadata=checkpoint["metadata"],
        resolved_config=checkpoint["resolved_config"],
    )


def _next_batch_fingerprint(cfg: DictConfig, device: torch.device, path: Path) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    trainer = _create_resume_trainer(cfg, device, raw)
    try:
        trainer.load(path)
        restored = {
            "optimizer_step": trainer.step,
            "train_epoch": trainer.train_epoch,
            "batch_in_epoch": trainer.batch_in_epoch,
        }
        batch = trainer._next_train_batch()
        paths = [str(value) for value in batch["relative_path"]]
        encoded = json.dumps(paths, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        restored["next_batch_relative_paths"] = paths
        restored["next_batch_sha256"] = hashlib.sha256(encoded).hexdigest()
        return restored
    finally:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


@hydra.main(
    config_path="../LeFusion/train/config",
    config_name="base_cfg",
    version_base=None,
)
def run(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal checkpoint gate requires CUDA")
    validate_training_config(cfg)
    if str(cfg.dataset.data_type) != "gli" or tuple(cfg.dataset.patch_size_xyz) != (64, 64, 32):
        raise ValueError("formal checkpoint gate only accepts exp005 p64")
    results = Path(str(cfg.model.results_folder)).expanduser()
    best_path = results / "best.pt"
    latest_path = results / "latest.pt"
    if not best_path.is_file() or not latest_path.is_file():
        raise FileNotFoundError("best.pt/latest.pt is missing")
    output_root = results.parent
    set_global_seed(int(cfg.seed))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    best_raw = torch.load(best_path, map_location="cpu", weights_only=False)
    latest_raw = torch.load(latest_path, map_location="cpu", weights_only=False)
    load_training_checkpoint(best_path, expected_metadata=best_raw["metadata"])
    load_training_checkpoint(latest_path, expected_metadata=latest_raw["metadata"])
    train_dataset, _ = get_train_dataset(cfg)
    _validate_provenance(cfg, train_dataset, best_raw)
    _validate_provenance(cfg, train_dataset, latest_raw)
    if best_raw["metadata"]["wandb_run_id"] != latest_raw["metadata"]["wandb_run_id"]:
        raise ValueError("best/latest W&B run IDs differ")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    diffusion = build_model_and_diffusion(cfg, device, use_data_parallel=False)
    load_diffusion_checkpoint(
        diffusion,
        best_path,
        weights_key="ema",
        expected_metadata=_expected_model_metadata(cfg),
    )
    validation_dataset = get_validation_dataset(cfg)
    validation = run_gli_validation(
        diffusion,
        validation_dataset,
        device=device,
        batch_size=int(cfg.validation.batch_size),
        num_workers=int(cfg.validation.num_workers),
        seed=int(cfg.validation.seed),
    )
    expected = float(best_raw["early_stopping"]["best"])
    actual = float(validation["val/ema/total_loss"])
    tolerance = 1e-6
    if abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"best EMA validation mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
        )
    validation_payload = {
        "passed": True,
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint_step": int(best_raw["step"]),
        "weights_key": "ema",
        "expected_total_loss": expected,
        "absolute_error": abs(actual - expected),
        "tolerance": tolerance,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        **validation,
    }
    _atomic_json(output_root / "validation_best_ema" / "metrics.json", validation_payload)
    del diffusion
    del best_raw
    gc.collect()
    torch.cuda.empty_cache()

    first = _next_batch_fingerprint(cfg, device, latest_path)
    second = _next_batch_fingerprint(cfg, device, latest_path)
    if first != second:
        raise RuntimeError("actual latest.pt resume is not deterministic")
    resume_payload = {
        "passed": True,
        "optimizer_updates": 0,
        "checkpoint_path": str(latest_path),
        "checkpoint_sha256": sha256_file(latest_path),
        "checkpoint_step": int(latest_raw["step"]),
        "restored_twice": True,
        "next_batch_reproducible": True,
        "state": first,
    }
    _atomic_json(output_root / "resume_preflight" / "metrics.json", resume_payload)
    gate = {
        "passed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "validation": validation_payload,
        "resume": resume_payload,
    }
    _atomic_json(output_root / "checkpoint_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
