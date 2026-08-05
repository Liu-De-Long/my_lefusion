"""Fail-closed W&B integration without credential handling."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def validate_wandb_config(cfg: DictConfig) -> dict[str, object] | None:
    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
        return None
    entity = wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY")
    payload: dict[str, object] = {
        "project": str(wandb_cfg.get("project", "lefusion-brats2024-gli")),
        "entity": None if entity is None else str(entity),
        "name": wandb_cfg.get("run_name"),
        "mode": str(wandb_cfg.get("mode", "online")),
        "id": wandb_cfg.get("run_id"),
        "resume": str(wandb_cfg.get("resume", "never")),
        "dir": str(wandb_cfg.get("dir", cfg.model.results_folder)),
    }
    if bool(wandb_cfg.get("fail_closed", False)):
        if payload["mode"] != "online":
            raise ValueError("formal W&B fail-closed mode requires mode='online'")
        for field in ("project", "entity", "name", "id"):
            if not payload[field]:
                raise ValueError(f"formal W&B config requires {field}")
        if payload["resume"] not in {"never", "must"}:
            raise ValueError("formal W&B resume must be 'never' or 'must'")
    return payload


def initialize_wandb(cfg: DictConfig):
    payload = validate_wandb_config(cfg)
    if payload is None:
        return None
    Path(str(payload["dir"])).expanduser().mkdir(parents=True, exist_ok=True)
    try:
        import wandb

        run = wandb.init(
            project=payload["project"],
            entity=payload["entity"],
            name=payload["name"],
            mode=payload["mode"],
            id=payload["id"],
            resume=payload["resume"],
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=payload["dir"],
        )
    except Exception as exc:
        raise RuntimeError("W&B online initialization failed before training") from exc
    expected_id = payload.get("id")
    if expected_id and str(run.id) != str(expected_id):
        run.finish(exit_code=1)
        raise RuntimeError(f"W&B returned unexpected run ID: {run.id!r} != {expected_id!r}")
    return run
