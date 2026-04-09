"""Vendored TorchReID model factory.

Copyright (c) 2018 Kaiyang Zhou — MIT License.
"""

from __future__ import annotations

import torch.nn as nn

from cuvis_ai_deepeiou.reid.models.osnet import osnet_x1_0
from cuvis_ai_deepeiou.reid.models.resnet import resnet50

_MODEL_REGISTRY: dict[str, type] = {
    "osnet_x1_0": osnet_x1_0,
    "resnet50": resnet50,
}


def build_model(name: str, num_classes: int = 1, loss: str = "softmax", **kwargs) -> nn.Module:
    """Build a TorchReID backbone by name.

    Parameters
    ----------
    name : str
        Model name (``osnet_x1_0`` or ``resnet50``).
    num_classes : int
        Number of identity classes (only affects the classifier head,
        which is unused at inference time).
    loss : str
        Loss type (``softmax`` or ``triplet``).

    Returns
    -------
    nn.Module
        The instantiated model.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    factory = _MODEL_REGISTRY[name]
    return factory(num_classes=num_classes, loss=loss, **kwargs)
