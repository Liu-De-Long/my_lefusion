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


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock3D(nn.Module):
    """Two-convolution residual block that remains stable for batch size one."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.skip(inputs) + self.body(inputs))


class MultiScaleContext3D(nn.Module):
    """Residual parallel dilated context at the U-Net bottleneck."""

    def __init__(self, channels: int, dilations: Sequence[int] = (1, 2, 4)) -> None:
        super().__init__()
        branch_channels = max(8, channels // len(dilations))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        channels,
                        branch_channels,
                        3,
                        padding=int(dilation),
                        dilation=int(dilation),
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(branch_channels), branch_channels),
                    nn.GELU(),
                )
                for dilation in dilations
            ]
        )
        merged_channels = branch_channels * len(dilations)
        self.fuse = nn.Sequential(
            nn.Conv3d(merged_channels, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([branch(inputs) for branch in self.branches], dim=1)
        return self.activation(inputs + self.fuse(merged))


class ResidualMultiScaleUNet3D(nn.Module):
    """Two-level residual 3D U-Net for complete 32x64x64 p64 patches."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        base = int(base_channels)
        self.encoder0 = ResidualConvBlock3D(2, base)
        self.down0 = nn.Conv3d(base, base * 2, 2, stride=2, bias=False)
        self.encoder1 = ResidualConvBlock3D(base * 2, base * 2)
        self.down1 = nn.Conv3d(base * 2, base * 4, 2, stride=2, bias=False)
        self.bottleneck = nn.Sequential(
            ResidualConvBlock3D(base * 4, base * 4),
            MultiScaleContext3D(base * 4),
            ResidualConvBlock3D(base * 4, base * 4),
        )
        self.up1 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2, bias=False)
        self.decoder1 = ResidualConvBlock3D(base * 4, base * 2)
        self.up0 = nn.ConvTranspose3d(base * 2, base, 2, stride=2, bias=False)
        self.decoder0 = ResidualConvBlock3D(base * 2, base)
        self.head = nn.Conv3d(base, 4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 2:
            raise ValueError(
                f"ResidualMultiScaleUNet3D expects [B,2,D,H,W], got {tuple(inputs.shape)}"
            )
        skip0 = self.encoder0(inputs)
        skip1 = self.encoder1(self.down0(skip0))
        hidden = self.bottleneck(self.down1(skip1))
        hidden = self.decoder1(torch.cat((self.up1(hidden), skip1), dim=1))
        hidden = self.decoder0(torch.cat((self.up0(hidden), skip0), dim=1))
        return self.head(hidden)


def build_classifier(
    kind: str,
    *,
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.1,
    cnn_channels: int = 24,
    unet_base_channels: int = 24,
) -> nn.Module:
    kind = str(kind).lower()
    if kind in FEATURE_CHANNELS:
        return VoxelMLP(FEATURE_CHANNELS[kind], hidden_dims, dropout)
    if kind == "c0":
        return Light3DClassifier(channels=int(cnn_channels))
    if kind == "unet3d":
        return ResidualMultiScaleUNet3D(base_channels=int(unet_base_channels))
    raise ValueError(f"unknown classifier kind: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
