"""Versioned checkpoint helpers shared by GLI training smoke and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


GLI_VALIDATION_CHECKPOINT_SCHEMA = 1
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
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
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
