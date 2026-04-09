"""cuvis_ai_deepeiou: DeepEIoU wrapper and cuvis.ai plugin package.

This package lives inside the forked DeepEIoU repository and provides:

- Access to the renamed ``deep_eiou_tracker`` package (upstream ``tracker/``
  renamed and properly packaged — no ``sys.path`` manipulation needed).
- A cuvis.ai-compatible Node for multi-object tracking
  (see :mod:`cuvis_ai_deepeiou.node`).
"""

# ── Compatibility shims (MUST run before any tracker import) ─────────────────

import sys
from types import ModuleType

import numpy as np

# Shim 1: np.float was removed in NumPy 1.24.
# Upstream basetrack.py / Deep_EIoU.py may use np.float.
if not hasattr(np, "float"):
    np.float = np.float64

# Shim 2: cython_bbox is a Cython extension that may not compile on all
# platforms. Provide a pure-numpy fallback that matching.py can import.
try:
    import cython_bbox  # noqa: F401
except ImportError:

    def _bbox_overlaps_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Pure-numpy IoU between two sets of boxes [N,4] and [M,4]."""
        x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
        y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
        x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
        y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        return inter / np.maximum(union, 1e-7)

    _fallback = ModuleType("cython_bbox")
    _fallback.bbox_overlaps = _bbox_overlaps_numpy
    sys.modules["cython_bbox"] = _fallback

# ── End shims ────────────────────────────────────────────────────────────────

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cuvis-ai-deepeiou")
except PackageNotFoundError:
    __version__ = "dev"


def register_all_nodes() -> int:
    """Register all cuvis_ai_deepeiou nodes in the cuvis.ai NodeRegistry.

    Returns
    -------
    int
        The number of node classes that were registered.
    """
    package_name = "cuvis_ai_deepeiou.node"
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    registry = NodeRegistry()
    return registry.auto_register_package(package_name)


__all__ = [
    "__version__",
    "register_all_nodes",
]
