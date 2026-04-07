"""Vendored weight-loading utility from deep-person-reid.

Copyright (c) 2018 Kaiyang Zhou — MIT License.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from loguru import logger


def load_pretrained_weights(model: nn.Module, weight_path: str) -> None:
    """Load pretrained weights into a model.

    - Incompatible layers (unmatched in name or size) are silently ignored.
    - Keys prefixed with ``module.`` (from DataParallel) are stripped.

    Parameters
    ----------
    model : nn.Module
        The target model.
    weight_path : str
        Path to a ``.pth.tar`` or ``.pth`` checkpoint file.
    """
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    matched_layers: list[str] = []
    discarded_layers: list[str] = []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f'Pretrained weights "{weight_path}" could not be loaded '
            "(no matching layers). Check key names.",
            stacklevel=2,
        )
    else:
        logger.info(
            "Loaded pretrained weights from '{}' ({} layers)", weight_path, len(matched_layers)
        )
        if discarded_layers:
            logger.debug("Discarded {} layers with unmatched keys/sizes", len(discarded_layers))
