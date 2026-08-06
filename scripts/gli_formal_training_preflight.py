"""Run an isolated, non-training preflight for a formal GLI configuration.

This utility deliberately performs one fixed-batch forward/backward pass without
an optimizer update, followed by full deterministic validation and checkpoint
reload verification.  It must use a dedicated W&B run ID and must never reuse
the formal training run ID.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast


ROOT = Path(__file__).parents[1]
import sys

sys.path.insert(0, str(ROOT / "LeFusion"))

from ddpm import prepare_gli_spatial_condition, prepare_training_batch  # noqa: E402
from get_dataset.get_dataset import get_train_dataset, get_validation_dataset  # noqa: E402
from train.train import (  # noqa: E402
    build_checkpoint_metadata,
    build_model_and_diffusion,
    set_global_seed,
    validate_training_config,
)
from train.tracking import initialize_wandb  # noqa: E402
from train.validation import run_gli_validation  # noqa: E402


def build_preflight_cfg(cfg: DictConfig) -> DictConfig:
    """Clone formal config and replace only runtime output/tracking identities."""
    preflight = cfg.get("preflight")
    if preflight is None:
        raise ValueError("formal preflight requires a preflight config section")
    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    if "preflight" not in str(preflight.run_id).lower():
        raise ValueError("preflight.run_id must be a dedicated preflight ID")
    if str(preflight.run_id) == str(cfg.wandb.run_id):
        raise ValueError("preflight W&B run ID must not equal the formal training run ID")
    cloned.wandb.run_id = str(preflight.run_id)
    cloned.wandb.run_name = str(preflight.run_name)
    cloned.wandb.resume = "never"
    cloned.wandb.dir = str(Path(str(preflight.output_dir)) / "wandb")
    cloned.model.results_folder = str(Path(str(preflight.output_dir)) / "checkpoints")
    return cloned


def _write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _create_trainer(cfg: DictConfig, device: torch.device):
    _, spatial_shape = validate_training_config(cfg)
    train_dataset, train_sampler = get_train_dataset(cfg)
    validation_dataset = get_validation_dataset(cfg)
    resolved_config, metadata = build_checkpoint_metadata(cfg, train_dataset, spatial_shape)
    diffusion = build_model_and_diffusion(cfg, device, use_data_parallel=False)
    from ddpm import Trainer

    trainer = Trainer(
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
        checkpoint_metadata=metadata,
        resolved_config=resolved_config,
    )
    return trainer


@hydra.main(
    config_path="../LeFusion/train/config",
    config_name="base_cfg",
    version_base=None,
)
def run(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal GLI preflight requires CUDA")
    if cfg.dataset.data_type != "gli" or not bool(cfg.formal_training.enabled):
        raise ValueError("this preflight only accepts a formal GLI training config")

    preflight_cfg = build_preflight_cfg(cfg)
    validate_training_config(preflight_cfg)
    torch.cuda.set_device(int(preflight_cfg.model.gpus))
    device = torch.device("cuda", int(preflight_cfg.model.gpus))
    set_global_seed(int(preflight_cfg.seed))
    output_dir = Path(str(preflight_cfg.preflight.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoints" / "resume_preflight.pt"
    metrics_path = output_dir / "metrics.json"
    started = time.time()
    wandb_run = initialize_wandb(preflight_cfg)
    trainer = None
    resumed = None
    try:
        trainer = _create_trainer(preflight_cfg, device)
        batch = trainer._next_train_batch()
        data, mask, hist = prepare_training_batch(batch, device, "gli")
        spatial_condition = prepare_gli_spatial_condition(batch, device)
        expected = (int(preflight_cfg.model.batch_size), 4, *trainer.spatial_shape)
        if tuple(data.shape) != expected or tuple(mask.shape) != expected:
            raise ValueError(
                f"preflight batch shape mismatch: data={tuple(data.shape)}, "
                f"mask={tuple(mask.shape)}, expected={expected}"
            )
        if hist is None or tuple(hist.shape) != (expected[0], 64):
            raise ValueError(f"preflight histogram shape mismatch: {None if hist is None else tuple(hist.shape)}")
        expected_spatial_condition = (expected[0], 5, *trainer.spatial_shape)
        if tuple(spatial_condition.shape) != expected_spatial_condition:
            raise ValueError(
                "preflight spatial condition shape mismatch: "
                f"{tuple(spatial_condition.shape)} != {expected_spatial_condition}"
            )

        torch.cuda.reset_peak_memory_stats(device)
        trainer.model.train()
        trainer.opt.zero_grad(set_to_none=True)
        with autocast(enabled=bool(preflight_cfg.model.amp)):
            loss = trainer.model(
                x=(data, hist), mask=mask, spatial_condition=spatial_condition
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"preflight loss is not finite: {float(loss.detach().cpu())}")
        trainer.scaler.scale(loss).backward()
        trainer.scaler.unscale_(trainer.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), float(preflight_cfg.model.max_grad_norm)
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("preflight gradient norm is not finite")
        trainer.opt.zero_grad(set_to_none=True)
        backward_peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

        # No optimizer/scaler step is performed: this is an allocation and
        # gradient-validity check, not a training step.
        trainer.save_checkpoint(checkpoint_path, kind="preflight-resume")
        saved_step = trainer.step
        saved_epoch = trainer.train_epoch
        saved_batch_offset = trainer.batch_in_epoch
        del trainer
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()

        resumed = _create_trainer(preflight_cfg, device)
        resumed.load(checkpoint_path)
        if (resumed.step, resumed.train_epoch, resumed.batch_in_epoch) != (
            saved_step,
            saved_epoch,
            saved_batch_offset,
        ):
            raise RuntimeError("checkpoint resume did not restore optimizer/epoch/batch state")
        resumed._next_train_batch()

        torch.cuda.reset_peak_memory_stats(device)
        validation_metrics = run_gli_validation(
            resumed.ema_model,
            resumed.validation_dataset,
            device=device,
            batch_size=int(preflight_cfg.validation.batch_size),
            num_workers=int(preflight_cfg.validation.num_workers),
            seed=int(preflight_cfg.validation.seed),
        )
        validation_peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        metrics = {
            "experiment_id": str(preflight_cfg.experiment_id),
            "variant": str(preflight_cfg.variant),
            "preflight_only": True,
            "optimizer_updates": 0,
            "batch_shape": list(data.shape),
            "preflight_loss": float(loss.detach().cpu()),
            "preflight_grad_norm": float(grad_norm.detach().cpu()),
            "backward_peak_mib": backward_peak_mib,
            "validation_peak_mib": validation_peak_mib,
            "resume_restored": True,
            "wandb_run_id": str(wandb_run.id),
            "wandb_url": getattr(wandb_run, "url", None),
            "elapsed_seconds": time.time() - started,
            **validation_metrics,
        }
        _write_metrics(metrics_path, metrics)
        wandb_run.log({"preflight/loss": metrics["preflight_loss"], "preflight/grad_norm": metrics["preflight_grad_norm"]})
        wandb_run.log(validation_metrics)
        for key, value in metrics.items():
            wandb_run.summary[key] = value
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        if trainer is not None:
            del trainer
        if resumed is not None:
            del resumed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # The checkpoint is an ephemeral resume gate, not a model artifact.
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    run()
