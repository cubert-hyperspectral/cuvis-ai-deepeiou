"""Vendored ResNet architecture from deep-person-reid.

Based on "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016).
Original source: torchvision + deep-person-reid adaptations.

Copyright (c) 2018 Kaiyang Zhou — MIT License.

Only ``resnet50`` is exposed; other variants are omitted for brevity.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor


class Bottleneck(nn.Module):
    """ResNet bottleneck block (expansion=4)."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    """ResNet backbone for person re-identification.

    Parameters
    ----------
    num_classes : int
        Number of identity classes.
    loss : str
        ``softmax`` or ``triplet``.
    block : type
        Block type (``Bottleneck``).
    layers : list[int]
        Number of blocks per stage.
    last_stride : int
        Stride of the last conv stage (1 or 2).
    fc_dims : list[int] | None
        Optional FC layer dimensions before classifier.
    feature_dim : int
        Embedding dimension after global average pooling.
    """

    def __init__(
        self,
        num_classes: int,
        loss: str,
        block: type[nn.Module],
        layers: list[int],
        last_stride: int = 2,
        fc_dims: list[int] | None = None,
        feature_dim: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.inplanes = 64
        self.feature_dim = feature_dim or block.expansion * 512  # type: ignore[attr-defined]

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Optional FC layers
        self.fc = self._construct_fc_layer(fc_dims, block.expansion * 512)  # type: ignore[attr-defined]

        out_features = fc_dims[-1] if fc_dims else block.expansion * 512  # type: ignore[attr-defined]
        self.feature_dim = out_features
        self.classifier = nn.Linear(out_features, num_classes)

        self._init_params()

    def _make_layer(
        self,
        block: type,
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def _construct_fc_layer(fc_dims: list[int] | None, input_dim: int) -> nn.Sequential | None:
        if fc_dims is None or len(fc_dims) == 0:
            return None
        layers: list[nn.Module] = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim
        return nn.Sequential(*layers)

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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
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


def resnet50(num_classes: int = 1, loss: str = "softmax", **kwargs: Any) -> ResNet:
    """ResNet-50 backbone (2048-dim features)."""
    return ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        **kwargs,
    )
