"""Vendored TorchReID model definitions and utilities.

Minimal subset (~400 LOC) from deep-person-reid by Kaiyang Zhou.
Only OSNet x1.0 and ResNet-50 backbones are included.

License: MIT — see LICENSE in this directory.
"""

from cuvis_ai_deepeiou.reid.models import build_model
from cuvis_ai_deepeiou.reid.utils import load_pretrained_weights

__all__ = ["build_model", "load_pretrained_weights"]
