"""Small voxel MLPs and a lightweight 3D CNN control model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .features import FEATURE_CHANNELS


class VoxelMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"VoxelMLP expects [N,C], got {tuple(features.shape)}")
        return self.network(features)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class Light3DClassifier(nn.Module):
    """Full-resolution receptive-field control with no encoder downsampling."""

    def __init__(self, channels: int = 24, dilations: Sequence[int] = (1, 2, 4, 2, 1)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(2, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResidualDilatedBlock(channels, int(dilation)) for dilation in dilations)
        )
        self.head = nn.Conv3d(channels, 4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 2:
            raise ValueError(f"Light3DClassifier expects [B,2,D,H,W], got {tuple(inputs.shape)}")
        return self.head(self.blocks(self.stem(inputs)))


def build_classifier(
    kind: str,
    *,
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.1,
    cnn_channels: int = 24,
) -> nn.Module:
    kind = str(kind).lower()
    if kind in FEATURE_CHANNELS:
        return VoxelMLP(FEATURE_CHANNELS[kind], hidden_dims, dropout)
    if kind == "c0":
        return Light3DClassifier(channels=int(cnn_channels))
    raise ValueError(f"unknown classifier kind: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
