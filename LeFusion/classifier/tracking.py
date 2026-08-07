"""Fail-closed online W&B tracking for formal classifier training."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def validate_classifier_wandb_config(
    config: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    section = config.get("wandb")
    if not isinstance(section, Mapping):
        raise ValueError("formal classifier training requires a wandb section")
    if not bool(section.get("enabled", False)):
        raise ValueError("formal classifier training requires wandb.enabled=true")
    if not bool(section.get("fail_closed", False)):
        raise ValueError("formal classifier training requires wandb.fail_closed=true")
    if str(section.get("mode", "")) != "online":
        raise ValueError("formal classifier training requires wandb.mode=online")
    required = ("entity", "project", "run_id", "run_name", "dir")
    missing = [key for key in required if not str(section.get(key, "")).strip()]
    if missing:
        raise ValueError(f"wandb config missing fields: {missing}")
    return {
        "entity": str(section["entity"]),
        "project": str(section["project"]),
        "id": str(section["run_id"]),
        "name": str(section["run_name"]),
        "dir": str(section["dir"]),
        "mode": "online",
        "resume": "must" if resume else "never",
    }


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    for key in list(payload):
        if str(key).startswith("_"):
            payload.pop(key)
    return payload


@contextmanager
def classifier_wandb_run(
    config: Mapping[str, Any], *, resume: bool
) -> Iterator[Any]:
    payload = validate_classifier_wandb_config(config, resume=resume)
    Path(payload["dir"]).mkdir(parents=True, exist_ok=True)
    try:
        import wandb
    except Exception as error:  # pragma: no cover - depends on remote runtime
        raise RuntimeError("wandb import failed before formal classifier training") from error
    run = None
    exit_code = 1
    try:
        run = wandb.init(**payload, config=_public_config(config))
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        yield run
        exit_code = 0
    except Exception as error:
        raise RuntimeError("formal classifier W&B online run failed closed") from error
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)
