"""Audit an exp018 best EMA on complete real-mask and route-overlay validation."""

from __future__ import annotations

import gc
import json
import os
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
        "spatial_condition_channels": int(cfg.model.spatial_condition_channels),
        "objective": str(cfg.lesion_generation.objective),
        "gli_state_mode": str(cfg.lesion_generation.state_mode),
    }


def _validate_checkpoint_provenance(cfg: DictConfig, checkpoint: dict) -> None:
    metadata = checkpoint["metadata"]
    resolved = checkpoint["resolved_config"]
    if canonical_config_hash(resolved) != str(metadata["config_hash"]):
        raise ValueError("checkpoint embedded config hash mismatch")
    current = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if canonical_config_hash(current) != str(metadata["config_hash"]):
        raise ValueError("current formal config does not match checkpoint")
    train_dataset, _ = get_train_dataset(cfg)
    if sha256_file(train_dataset.manifest_path) != str(metadata["manifest_hash"]):
        raise ValueError("dataset manifest hash mismatch")
    if sha256_file(cfg.dataset.split_file) != str(metadata["split_hash"]):
        raise ValueError("split hash mismatch")
    overlay_contract = train_dataset.mask_overlay_contract_path
    if overlay_contract is None:
        raise ValueError("exp018 training dataset has no mask overlay")
    if sha256_file(overlay_contract) != str(metadata["mask_overlay_contract_hash"]):
        raise ValueError("training overlay contract hash mismatch")


@hydra.main(config_path="../LeFusion/train/config", config_name="base_cfg", version_base=None)
def run(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("exp018 best validation requires CUDA")
    validate_training_config(cfg)
    set_global_seed(int(cfg.seed))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    results = Path(str(cfg.model.results_folder)).expanduser()
    best_path = results / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    load_training_checkpoint(best_path, expected_metadata=checkpoint["metadata"])
    _validate_checkpoint_provenance(cfg, checkpoint)

    diffusion = build_model_and_diffusion(cfg, device, use_data_parallel=False)
    load_diffusion_checkpoint(
        diffusion,
        best_path,
        weights_key="ema",
        expected_metadata=_expected_model_metadata(cfg),
    )
    real_dataset = get_validation_dataset(cfg)
    real_metrics = run_gli_validation(
        diffusion,
        real_dataset,
        device=device,
        batch_size=int(cfg.validation.batch_size),
        num_workers=int(cfg.validation.num_workers),
        seed=int(cfg.validation.seed),
        prefix="val/real/ema",
    )
    expected = float(checkpoint["early_stopping"]["best"])
    actual = float(real_metrics["val/real/ema/total_loss"])
    if abs(actual - expected) > 1e-6:
        raise RuntimeError(f"best real-val loss mismatch: {actual} != {expected}")

    overlay_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    overlay_cfg.dataset.mask_overlay.enabled = True
    overlay_cfg.dataset.mask_overlay.apply_splits = ["val"]
    overlay_dataset = get_validation_dataset(overlay_cfg)
    if len(real_dataset) != 1032 or len(overlay_dataset) != 1032:
        raise RuntimeError(
            f"expected 1032 val patches, got real={len(real_dataset)}, overlay={len(overlay_dataset)}"
        )
    overlay_metrics = run_gli_validation(
        diffusion,
        overlay_dataset,
        device=device,
        batch_size=int(cfg.validation.batch_size),
        num_workers=int(cfg.validation.num_workers),
        seed=int(cfg.validation.seed),
        prefix="val/overlay/ema",
    )
    payload = {
        "passed": True,
        "route": str(cfg.variant),
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights_key": "ema",
        "real_val_matches_early_stopping": True,
        "real_val_absolute_error": abs(actual - expected),
        "overlay_contract": str(overlay_dataset.mask_overlay_contract_path),
        "overlay_contract_sha256": sha256_file(overlay_dataset.mask_overlay_contract_path),
        "real": real_metrics,
        "overlay": overlay_metrics,
    }
    _atomic_json(results.parent / "best_validation" / "metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    del diffusion, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
