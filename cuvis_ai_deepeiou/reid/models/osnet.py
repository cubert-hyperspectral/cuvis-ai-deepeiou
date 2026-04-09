"""Vendored OSNet architecture from deep-person-reid.

Omni-Scale Feature Learning for Person Re-Identification (ICCV 2019).
Zhou et al., https://arxiv.org/abs/1905.00953

Copyright (c) 2018 Kaiyang Zhou — MIT License.

Only ``osnet_x1_0`` is exposed; other variants are omitted for brevity.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvLayer(nn.Module):
    """Conv + BN + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1×1 conv + BN + optional ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, relu: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(self.conv(x))
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1×1 conv + BN (no activation)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Lightweight 3×3 depthwise-separable convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            groups=in_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


# ---------------------------------------------------------------------------
# Channel gate
# ---------------------------------------------------------------------------


class ChannelGate(nn.Module):
    """Channel-wise attention gate."""

    def __init__(
        self,
        in_channels: int,
        num_gates: int | None = None,
        return_gates: bool = False,
        gate_activation: str = "sigmoid",
        reduction: int = 16,
    ) -> None:
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True)
        self.norm1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, 1, bias=True)
        if gate_activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == "linear":
            self.gate_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x: Tensor) -> Tensor:
        inp = x
        x = self.global_avgpool(x)
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.gate_activation(self.fc2(x))
        if self.return_gates:
            return x
        return inp * x


# ---------------------------------------------------------------------------
# OSBlock — omni-scale feature learning block
# ---------------------------------------------------------------------------


class OSBlock(nn.Module):
    """Omni-scale feature learning block with 4 parallel streams."""

    def __init__(
        self, in_channels: int, out_channels: int, reduction: int = 4, T: int = 4, **kwargs: Any
    ) -> None:
        super().__init__()
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)

        # Build T streams of stacked LightConv3x3
        self.streams = nn.ModuleList()
        for t in range(1, T + 1):
            stream = nn.Sequential(*[LightConv3x3(mid_channels, mid_channels) for _ in range(t)])
            self.streams.append(stream)

        self.gate = ChannelGate(mid_channels)
        self.conv_out = Conv1x1Linear(mid_channels, out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)

        # Multi-scale streams with unified gating
        fused = torch.zeros_like(x)
        for stream in self.streams:
            fused = fused + stream(x)

        x = self.gate(fused)
        x = self.conv_out(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(x + identity)


# ---------------------------------------------------------------------------
# OSNet backbone
# ---------------------------------------------------------------------------


class OSNet(nn.Module):
    """Omni-Scale Network.

    Parameters
    ----------
    num_classes : int
        Number of identity classes.
    blocks : list[nn.Module]
        Block type for each stage.
    layers : list[int]
        Number of blocks per stage.
    channels : list[int]
        Channel widths per stage.
    loss : str
        ``softmax`` or ``triplet``.
    feature_dim : int
        Final embedding dimension (default 512).
    """

    def __init__(
        self,
        num_classes: int,
        blocks: list[type[nn.Module]],
        layers: list[int],
        channels: list[int],
        loss: str = "softmax",
        feature_dim: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.feature_dim = feature_dim
        num_blocks = len(blocks)

        # Stem (conv1)
        self.conv1 = nn.Sequential(
            ConvLayer(3, channels[0], 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Build stages (conv2 .. conv5)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))

        self.conv3 = self._make_layer(
            blocks[min(1, num_blocks - 1)], layers[1], channels[1], channels[2]
        )
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))

        self.conv4 = self._make_layer(
            blocks[min(2, num_blocks - 1)], layers[2], channels[2], channels[3]
        )

        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Fully connected
        self.fc = self._construct_fc_layer(feature_dim, channels[3])
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_params()

    @staticmethod
    def _make_layer(
        block: type[nn.Module], num_blocks: int, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        layers = [block(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def _construct_fc_layer(fc_dims: int, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, fc_dims),
            nn.BatchNorm1d(fc_dims),
            nn.ReLU(inplace=True),
        )

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.featuremaps(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)
        if self.loss == "softmax":
            return y
        elif self.loss == "triplet":
            return y, v
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def osnet_x1_0(num_classes: int = 1, loss: str = "softmax", **kwargs: Any) -> OSNet:
    """OSNet x1.0 (512-dim features)."""
    return OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs,
    )
