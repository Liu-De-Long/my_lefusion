"""Deterministic supervised validation for formal GLI training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch.utils.data import DataLoader

from ddpm import prepare_training_batch


CHANNEL_NAMES = ("netc", "snfh", "et", "rc")


@dataclass
class EarlyStopping:
    patience: int
    relative_min_delta: float
    warmup_steps: int
    best: float | None = None
    bad_validations: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("early-stopping patience must be positive")
        if not 0 <= self.relative_min_delta < 1:
            raise ValueError("relative_min_delta must be in [0, 1)")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

    def update(self, metric: float, step: int) -> tuple[bool, bool]:
        if not math.isfinite(metric):
            raise ValueError(f"early-stopping metric is not finite: {metric}")
        improved = self.best is None or metric < self.best * (1.0 - self.relative_min_delta)
        if improved:
            self.best = metric
            self.bad_validations = 0
        elif step >= self.warmup_steps:
            self.bad_validations += 1
        should_stop = step >= self.warmup_steps and self.bad_validations >= self.patience
        return improved, should_stop

    def state_dict(self) -> dict[str, float | int | None]:
        return {
            "patience": self.patience,
            "relative_min_delta": self.relative_min_delta,
            "warmup_steps": self.warmup_steps,
            "best": self.best,
            "bad_validations": self.bad_validations,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["patience"]) != self.patience:
            raise ValueError("early-stopping patience does not match checkpoint")
        if float(state["relative_min_delta"]) != self.relative_min_delta:
            raise ValueError("early-stopping min delta does not match checkpoint")
        if int(state["warmup_steps"]) != self.warmup_steps:
            raise ValueError("early-stopping warmup does not match checkpoint")
        best = state.get("best")
        self.best = None if best is None else float(best)
        self.bad_validations = int(state["bad_validations"])


def run_gli_validation(
    diffusion,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    prefix: str = "val/ema",
) -> dict[str, float | int]:
    """Evaluate all val patches with a fixed timestep/noise stream."""
    if dataset is None:
        raise ValueError("validation dataset is required")
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    total_loss_sum = 0.0
    effective_units = 0
    channel_loss_sums = torch.zeros(4, dtype=torch.float64)
    channel_effective_units = torch.zeros(4, dtype=torch.int64)
    subject_ids: set[str] = set()
    anchor_labels: set[int] = set()
    patch_count = 0

    was_training = diffusion.training
    diffusion.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                data, mask, hist = prepare_training_batch(batch, device, "gli")
                batch_size_actual = data.shape[0]
                timestep = torch.randint(
                    0,
                    diffusion.num_timesteps,
                    (batch_size_actual,),
                    generator=generator,
                    dtype=torch.long,
                ).to(device)
                noise = torch.randn(
                    tuple(data.shape), generator=generator, dtype=torch.float32
                ).to(device)
                details = diffusion.gli_validation_loss_details(
                    data, mask, hist, t=timestep, noise=noise
                )
                total_loss_sum += float(details["total_loss_sum"].detach().cpu())
                effective_units += int(details["effective_units"].detach().cpu())
                channel_loss_sums += details["channel_loss_sums"].detach().cpu().double()
                channel_effective_units += (
                    details["channel_effective_units"].detach().cpu().long()
                )
                patch_count += batch_size_actual
                subject_ids.update(str(value) for value in batch["subject_id"])
                anchor_labels.update(int(value) for value in batch["anchor_label"].tolist())
    finally:
        diffusion.train(was_training)

    if effective_units <= 0:
        raise RuntimeError("validation produced no effective lesion units")
    metrics: dict[str, float | int] = {
        f"{prefix}/total_loss": total_loss_sum / effective_units,
        f"{prefix}/effective_units": effective_units,
        f"{prefix}/patches": patch_count,
        f"{prefix}/subjects": len(subject_ids),
        f"{prefix}/anchor_label_coverage": len(anchor_labels),
    }
    for index, name in enumerate(CHANNEL_NAMES):
        count = int(channel_effective_units[index])
        metrics[f"{prefix}/{name}_effective_units"] = count
        metrics[f"{prefix}/{name}_loss"] = (
            float(channel_loss_sums[index]) / count if count else float("nan")
        )
    return metrics
