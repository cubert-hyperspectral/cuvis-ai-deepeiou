"""Tests for OSNetExtractor and ResNetExtractor nodes."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn as nn

from cuvis_ai_deepeiou.node.bbox_feature_extractor import BBoxFeatureExtractor
from cuvis_ai_deepeiou.node.osnet_extractor import OSNetExtractor
from cuvis_ai_deepeiou.node.resnet_extractor import ResNetExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockBackbone(nn.Module):
    """Fake backbone that returns random features of the right dim."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # Need at least one parameter so .to(device) works
        self._dummy = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return torch.randn(N, self.feature_dim)

    def eval(self) -> _MockBackbone:
        return super().eval()


def _patch_build_and_load(feature_dim: int):
    """Context manager that patches build_model and load_pretrained_weights.

    The imports are deferred (inside __init__), so we patch at the source modules.
    """
    mock_model = _MockBackbone(feature_dim)
    return (
        patch(
            "cuvis_ai_deepeiou.reid.models.build_model",
            return_value=mock_model,
        ),
        patch(
            "cuvis_ai_deepeiou.reid.utils.load_pretrained_weights",
        ),
    )


def _make_extractor(cls: type[BBoxFeatureExtractor]) -> BBoxFeatureExtractor:
    """Instantiate an extractor with mocked model loading."""
    p1, p2 = _patch_build_and_load(cls.FEATURE_DIM)
    with (
        p1,
        p2,
        patch.object(cls, "_resolve_weights", return_value="/fake/weights.pth.tar"),
    ):
        return cls(model_path="/fake/weights.pth.tar", name="test_extractor")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_osnet_extractor_port_specs() -> None:
    """INPUT/OUTPUT_SPECS shapes and dtypes for OSNet."""
    assert OSNetExtractor.INPUT_SPECS["crops"].dtype == torch.float32
    assert OSNetExtractor.INPUT_SPECS["crops"].shape == (-1, 3, -1, -1)
    assert OSNetExtractor.OUTPUT_SPECS["embeddings"].dtype == torch.float32
    assert OSNetExtractor.OUTPUT_SPECS["embeddings"].shape == (-1, -1, -1)
    assert OSNetExtractor.FEATURE_DIM == 512
    assert OSNetExtractor.MODEL_NAME == "osnet_x1_0"


def test_resnet_extractor_port_specs() -> None:
    """INPUT/OUTPUT_SPECS shapes and dtypes for ResNet."""
    assert ResNetExtractor.INPUT_SPECS["crops"].dtype == torch.float32
    assert ResNetExtractor.OUTPUT_SPECS["embeddings"].dtype == torch.float32
    assert ResNetExtractor.FEATURE_DIM == 2048
    assert ResNetExtractor.MODEL_NAME == "resnet50"


def test_forward_output_shape() -> None:
    """Mock model → verify [1, N, 512] output for OSNet."""
    node = _make_extractor(OSNetExtractor)
    crops = torch.randn(5, 3, 256, 128)
    result = node.forward(crops=crops)
    emb = result["embeddings"]
    assert emb.shape == (1, 5, 512)


def test_forward_zero_detections() -> None:
    """Empty crops → [1, 0, D]."""
    node = _make_extractor(OSNetExtractor)
    crops = torch.empty(0, 3, 256, 128)
    result = node.forward(crops=crops)
    emb = result["embeddings"]
    assert emb.shape == (1, 0, 512)


def test_embeddings_l2_normalized() -> None:
    """‖emb‖₂ ≈ 1.0 for each detection."""
    node = _make_extractor(OSNetExtractor)
    crops = torch.randn(4, 3, 256, 128)
    result = node.forward(crops=crops)
    emb = result["embeddings"].squeeze(0)  # [N, D]
    norms = torch.linalg.vector_norm(emb, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_registration() -> None:
    """Plugin register_all_nodes() discovers both extractors."""
    from cuvis_ai_deepeiou.node import OSNetExtractor as OS
    from cuvis_ai_deepeiou.node import ResNetExtractor as RS

    assert OS is OSNetExtractor
    assert RS is ResNetExtractor


def test_constructor_params_passed_to_super() -> None:
    """model_path is stored and accessible."""
    node = _make_extractor(OSNetExtractor)
    assert node.model_path == "/fake/weights.pth.tar"


def test_resolve_weights_existing_file(tmp_path) -> None:
    """_resolve_weights returns path as-is when file exists."""
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake")
    result = OSNetExtractor._resolve_weights(str(weights))
    assert result == str(weights)


def test_resolve_weights_downloads_from_hf(tmp_path) -> None:
    """_resolve_weights calls hf_hub_download and copies to target when file missing."""
    target = tmp_path / "subdir" / "osnet.pt"
    cached = tmp_path / "cached_model.pt"
    cached.write_bytes(b"fake-weights-data")

    with patch(
        "huggingface_hub.hf_hub_download",
        return_value=str(cached),
    ) as mock_dl:
        result = OSNetExtractor._resolve_weights(str(target))

    mock_dl.assert_called_once_with(
        repo_id="kadirnar/osnet_x1_0_imagenet",
        filename="osnet_x1_0_imagenet.pt",
    )
    assert result == str(target)
    assert target.exists()
    assert target.read_bytes() == b"fake-weights-data"


def test_resolve_weights_no_hf_config_raises() -> None:
    """_resolve_weights raises FileNotFoundError when HF_WEIGHTS is empty."""
    import pytest

    with pytest.raises(FileNotFoundError, match="no HuggingFace source"):
        ResNetExtractor._resolve_weights("/nonexistent/weights.pth.tar")
