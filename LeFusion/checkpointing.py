"""Versioned checkpoint helpers shared by GLI training smoke and inference."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


GLI_VALIDATION_CHECKPOINT_SCHEMA = 1
GLI_TRAINING_CHECKPOINT_SCHEMA = 2
MODEL_METADATA_FIELDS = (
    "data_type",
    "diffusion_num_channels",
    "cond_dim",
    "base_dim",
    "spatial_shape_dhw",
    "timesteps",
    "temporal_max_distance",
)


def normalized_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in state_dict.items():
        normalized = key
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        normalized = normalized.replace("denoise_fn.module.", "denoise_fn.")
        result[normalized] = value
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Hash semantic config while excluding resume-only transport fields."""
    payload = json.loads(json.dumps(config))
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("resume_from", None)
    wandb = payload.get("wandb")
    if isinstance(wandb, dict):
        wandb.pop("resume", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path).expanduser(), map_location=map_location)


def validate_checkpoint_metadata(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field in MODEL_METADATA_FIELDS:
        if field not in metadata:
            raise ValueError(f"checkpoint metadata missing {field}")
        actual = metadata[field]
        wanted = expected[field]
        if field == "spatial_shape_dhw":
            actual = tuple(int(value) for value in actual)
            wanted = tuple(int(value) for value in wanted)
        if actual != wanted:
            raise ValueError(f"checkpoint metadata mismatch for {field}: {actual!r} != {wanted!r}")


def load_diffusion_checkpoint(
    diffusion: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    weights_key: str,
    expected_metadata: Mapping[str, Any],
) -> dict:
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("schema_version", -1)) != GLI_VALIDATION_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported checkpoint schema: {checkpoint.get('schema_version')!r}"
        )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint has no metadata mapping")
    validate_checkpoint_metadata(metadata, expected_metadata)
    if weights_key not in {"model", "ema"}:
        raise ValueError(f"weights_key must be model or ema, got {weights_key!r}")
    state_dict = checkpoint.get(weights_key)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"checkpoint has no {weights_key!r} state dict")
    diffusion.load_state_dict(normalized_state_dict(state_dict), strict=True)
    return checkpoint


def validate_training_checkpoint_metadata(
    metadata: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    required = (
        "experiment_id",
        "git_sha",
        "config_hash",
        "manifest_hash",
        "split_hash",
        "data_type",
        "patch_size_xyz",
        "spatial_shape_dhw",
        "sampler_name",
        "wandb_run_id",
    )
    for field in required:
        if field not in metadata:
            raise ValueError(f"training checkpoint metadata missing {field}")
        actual = metadata[field]
        wanted = expected[field]
        if field in {"patch_size_xyz", "spatial_shape_dhw"}:
            actual = tuple(int(value) for value in actual)
            wanted = tuple(int(value) for value in wanted)
        if actual != wanted:
            raise ValueError(
                f"training checkpoint metadata mismatch for {field}: {actual!r} != {wanted!r}"
            )


def load_training_checkpoint(
    checkpoint_path: str | Path, *, expected_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("schema_version", -1)) != GLI_TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported training checkpoint schema: {checkpoint.get('schema_version')!r}"
        )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("training checkpoint has no metadata mapping")
    validate_training_checkpoint_metadata(metadata, expected_metadata)
    for field in (
        "model",
        "ema",
        "optimizer",
        "scaler",
        "rng_state",
        "train_state",
        "early_stopping",
        "resolved_config",
    ):
        if field not in checkpoint:
            raise ValueError(f"training checkpoint missing {field}")
    return checkpoint
