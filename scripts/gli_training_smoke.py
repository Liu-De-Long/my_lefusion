"""Reusable real-data smoke check for GLI lesion-aware diffusion training."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "LeFusion"))

from ddpm import prepare_training_batch  # noqa: E402
from get_dataset.get_dataset import get_train_dataset  # noqa: E402
from train.train import (  # noqa: E402
    build_model_and_diffusion,
    initialize_wandb,
    validate_training_config,
)
from checkpointing import GLI_VALIDATION_CHECKPOINT_SCHEMA  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(
    config_path="../LeFusion/train/config",
    config_name="base_cfg",
    version_base=None,
)
def run(cfg: DictConfig) -> None:
    data_type, spatial_shape = validate_training_config(cfg)
    if data_type != "gli":
        raise ValueError("gli_training_smoke.py only accepts dataset.data_type=gli")
    if not torch.cuda.is_available():
        raise RuntimeError("GLI training smoke requires CUDA")

    torch.cuda.set_device(int(cfg.model.gpus))
    device = torch.device("cuda", int(cfg.model.gpus))
    seed = int(cfg.smoke.seed)
    steps = int(cfg.smoke.steps)
    if steps <= 0:
        raise ValueError(f"smoke.steps must be positive, got {steps}")
    _set_seed(seed)

    dataset, _ = get_train_dataset(cfg)
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=int(cfg.model.batch_size),
                shuffle=False,
                num_workers=int(cfg.model.num_workers),
                pin_memory=True,
            )
        )
    )
    data, mask, hist = prepare_training_batch(batch, device, data_type)
    expected_shape = (int(cfg.model.batch_size), 4, *spatial_shape)
    if tuple(data.shape) != expected_shape or tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"real batch shape mismatch: data={tuple(data.shape)}, mask={tuple(mask.shape)}, "
            f"expected={expected_shape}"
        )
    if hist is None or tuple(hist.shape) != (int(cfg.model.batch_size), 64):
        raise ValueError(f"histogram shape mismatch: {None if hist is None else tuple(hist.shape)}")

    diffusion = build_model_and_diffusion(cfg, device, use_data_parallel=False)
    diffusion.train()
    optimizer = Adam(diffusion.parameters(), lr=float(cfg.model.train_lr))
    scaler = GradScaler(enabled=bool(cfg.model.amp))
    timestep = torch.full(
        (data.shape[0],),
        min(int(cfg.smoke.fixed_timestep), diffusion.num_timesteps - 1),
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn_like(data)
    torch.cuda.reset_peak_memory_stats(device)

    wandb_run = initialize_wandb(cfg)
    if wandb_run is None:
        raise RuntimeError("W&B must be enabled for the GLI training smoke")
    run_url = getattr(wandb_run, "url", None)
    losses: list[float] = []
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg.model.amp)):
                loss = diffusion(x=(data, hist), mask=mask, t=timestep, noise=noise)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at smoke step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            if step == 0:
                finite_nonzero_grad = any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    and bool(parameter.grad.abs().sum() > 0)
                    for parameter in diffusion.parameters()
                )
                if not finite_nonzero_grad:
                    raise RuntimeError("no finite non-zero gradient was produced")
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            wandb_run.log({"smoke/loss": loss_value, "smoke/step": step})

        if steps >= 10:
            first_mean = float(np.mean(losses[:5]))
            last_mean = float(np.mean(losses[-5:]))
            if not last_mean < first_mean:
                raise RuntimeError(
                    f"fixed-batch overfit did not improve: first5={first_mean}, last5={last_mean}"
                )
        else:
            first_mean = losses[0]
            last_mean = losses[-1]

        peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        checkpoint_path = None
        if bool(cfg.smoke.get("save_checkpoint", False)):
            checkpoint_path = Path(str(cfg.smoke.checkpoint_path)).expanduser()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            git_commit = os.environ.get("GIT_COMMIT", "unknown")
            base_dim = cfg.model.get("base_dim") or cfg.model.diffusion_img_size
            metadata = {
                "checkpoint_purpose": "validation-smoke-only",
                "data_type": data_type,
                "diffusion_num_channels": int(cfg.model.diffusion_num_channels),
                "cond_dim": int(cfg.model.cond_dim),
                "base_dim": int(base_dim),
                "spatial_shape_dhw": list(spatial_shape),
                "timesteps": int(cfg.model.timesteps),
                "temporal_max_distance": int(cfg.model.get("temporal_max_distance", 32)),
                "seed": seed,
                "steps": steps,
                "git_commit": git_commit,
                "ema_is_model_copy": True,
            }
            model_state = diffusion.state_dict()
            torch.save(
                {
                    "schema_version": GLI_VALIDATION_CHECKPOINT_SCHEMA,
                    "step": steps,
                    "model": model_state,
                    "ema": model_state,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "metadata": metadata,
                    "resolved_config": OmegaConf.to_container(cfg, resolve=True),
                },
                checkpoint_path,
            )
        summary = {
            "experiment_id": cfg.experiment_id,
            "variant": cfg.variant,
            "batch_shape": list(data.shape),
            "sample_shape": list(diffusion.sample_shape(data.shape[0])),
            "steps": steps,
            "first_loss_mean": first_mean,
            "last_loss_mean": last_mean,
            "peak_memory_mib": peak_memory_mib,
            "wandb_url": run_url,
            "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
            "validation_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        }
        for key, value in summary.items():
            wandb_run.summary[key] = value
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    run()
