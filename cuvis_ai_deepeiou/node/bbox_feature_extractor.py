"""Abstract base class for TorchReID backbone feature extraction."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger


class BBoxFeatureExtractor(Node):
    """Abstract base for TorchReID backbone feature extraction.

    Receives pre-processed NCHW crops (already cropped, resized, and
    channel-normalized), runs the backbone in eval mode, and L2-normalizes
    the output embeddings.

    Subclasses set ``MODEL_NAME`` and ``FEATURE_DIM`` as class variables.

    Parameters
    ----------
    model_path : str
        Path to a ``.pth.tar`` weights file for the TorchReID backbone.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({
        NodeTag.RGB,
        NodeTag.IMAGE,
        NodeTag.EMBEDDING,
        NodeTag.INFERENCE,
        NodeTag.LEARNABLE,
        NodeTag.BATCHED,
        NodeTag.TORCH,
    })

    MODEL_NAME: str  # e.g. "osnet_x1_0"
    FEATURE_DIM: int  # e.g. 512
    HF_WEIGHTS: ClassVar[
        dict[str, str]
    ] = {}  # subclasses override: {"repo_id": ..., "filename": ...}

    INPUT_SPECS = {
        "crops": PortSpec(
            dtype=torch.float32,
            shape=(-1, 3, -1, -1),
            description="Normalized crops [N, 3, crop_h, crop_w] in NCHW.",
        ),
    }

    OUTPUT_SPECS = {
        "embeddings": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="L2-normalised embeddings [B, N, D].",
        ),
    }

    @classmethod
    def _resolve_weights(cls, model_path: str) -> str:
        """Return *model_path* if it exists locally, otherwise download from HuggingFace Hub.

        When the file is missing and ``HF_WEIGHTS`` is set on the subclass, the
        weights are fetched via ``huggingface_hub.hf_hub_download`` and copied to
        the user-specified *model_path* so subsequent runs find them locally.
        """
        p = Path(model_path)
        if p.is_file():
            return model_path
        if not cls.HF_WEIGHTS:
            raise FileNotFoundError(
                f"Weights not found at '{model_path}' and no HuggingFace source configured "
                f"for {cls.__name__}."
            )
        from huggingface_hub import hf_hub_download

        repo_id = cls.HF_WEIGHTS["repo_id"]
        filename = cls.HF_WEIGHTS["filename"]
        logger.info("Downloading {} weights from HF '{}' → {}", cls.MODEL_NAME, repo_id, p)
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, p)
        logger.info("Saved weights to {}", p)
        return model_path

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self.model_path = model_path
        # Must call super().__init__() before assigning nn.Module submodules,
        # because PyTorch's __setattr__ requires _modules to be initialized.
        super().__init__(model_path=model_path, **kwargs)

        resolved_path = self._resolve_weights(model_path)

        from cuvis_ai_deepeiou.reid.models import build_model
        from cuvis_ai_deepeiou.reid.utils import load_pretrained_weights

        # Eager-load the backbone so pipeline.to(device) propagates correctly.
        self._model: nn.Module = build_model(self.MODEL_NAME, num_classes=1, loss="softmax")
        load_pretrained_weights(self._model, resolved_path)
        self._model.eval()

        logger.info(
            "{}: loaded {} backbone ({}D) from '{}'",
            type(self).__name__,
            self.MODEL_NAME,
            self.FEATURE_DIM,
            model_path,
        )

    @torch.no_grad()
    def forward(self, crops: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Extract L2-normalized embeddings from pre-processed crops.

        Parameters
        ----------
        crops : Tensor
            ``[N, 3, crop_h, crop_w]`` normalized image crops.

        Returns
        -------
        dict
            ``{"embeddings": Tensor [1, N, D]}``
        """
        n = crops.shape[0]

        if n == 0:
            return {
                "embeddings": torch.zeros(
                    1,
                    0,
                    self.FEATURE_DIM,
                    dtype=torch.float32,
                    device=crops.device,
                ),
            }

        # Forward through backbone → [N, D]
        emb = self._model(crops)
        # L2 normalize
        emb = F.normalize(emb, p=2, dim=-1)
        # Restore batch dim → [1, N, D]
        emb = emb.unsqueeze(0)

        return {"embeddings": emb}
