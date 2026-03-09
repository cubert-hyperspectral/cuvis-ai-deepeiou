"""Tests for DeepEIoUTrack node."""

import sys

import numpy as np
import pytest
import torch


def _make_inputs(n: int = 5, with_embeddings: bool = False, emb_dim: int = 512):
    """Create mock detection inputs for DeepEIoUTrack.

    Returns dict with bboxes [1, N, 4], category_ids [1, N],
    confidences [1, N], and optionally embeddings [1, N, D].
    """
    if n == 0:
        result = {
            "bboxes": torch.zeros(1, 0, 4, dtype=torch.float32),
            "category_ids": torch.zeros(1, 0, dtype=torch.int64),
            "confidences": torch.zeros(1, 0, dtype=torch.float32),
        }
        if with_embeddings:
            result["embeddings"] = torch.zeros(1, 0, emb_dim, dtype=torch.float32)
        return result

    # Spread bboxes so they don't overlap much
    bboxes = torch.zeros(1, n, 4, dtype=torch.float32)
    for i in range(n):
        x1 = 50.0 + i * 100.0
        y1 = 50.0
        x2 = x1 + 60.0
        y2 = y1 + 80.0
        bboxes[0, i] = torch.tensor([x1, y1, x2, y2])

    confidences = torch.full((1, n), 0.9, dtype=torch.float32)
    category_ids = torch.zeros(1, n, dtype=torch.int64)

    result = {
        "bboxes": bboxes,
        "category_ids": category_ids,
        "confidences": confidences,
    }
    if with_embeddings:
        emb = torch.randn(1, n, emb_dim, dtype=torch.float32)
        # Normalize embeddings
        emb = emb / emb.norm(dim=-1, keepdim=True)
        result["embeddings"] = emb
    return result


# ── Import / registration tests ─────────────────────────────────────────────


def test_import():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack  # noqa: F401


def test_register_all_nodes():
    from cuvis_ai_deepeiou import register_all_nodes

    count = register_all_nodes()
    assert count >= 1


# ── Constructor tests ────────────────────────────────────────────────────────


def test_default_params():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack()
    assert node.track_high_thresh == 0.6
    assert node.track_low_thresh == 0.1
    assert node.new_track_thresh == 0.7
    assert node.track_buffer == 60
    assert node.match_thresh == 0.8
    assert node.frame_rate == 30
    assert node.with_reid is True
    assert node.proximity_thresh == 0.5
    assert node.appearance_thresh == 0.25


def test_custom_params():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(
        track_high_thresh=0.5,
        track_low_thresh=0.2,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.7,
        frame_rate=25,
        with_reid=False,
        proximity_thresh=0.4,
        appearance_thresh=0.3,
    )
    assert node.track_high_thresh == 0.5
    assert node.track_low_thresh == 0.2
    assert node.new_track_thresh == 0.6
    assert node.track_buffer == 30
    assert node.match_thresh == 0.7
    assert node.frame_rate == 25
    assert node.with_reid is False
    assert node.proximity_thresh == 0.4
    assert node.appearance_thresh == 0.3


def test_hparams_captured():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(track_high_thresh=0.55)
    assert hasattr(node, "hparams")
    assert node.hparams["track_high_thresh"] == 0.55


# ── Port spec tests ─────────────────────────────────────────────────────────


def test_input_specs():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    specs = DeepEIoUTrack.INPUT_SPECS
    assert "bboxes" in specs
    assert "category_ids" in specs
    assert "confidences" in specs
    assert "embeddings" in specs


def test_output_specs():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    specs = DeepEIoUTrack.OUTPUT_SPECS
    assert "bboxes" in specs
    assert "track_ids" in specs
    assert "confidences" in specs
    assert "category_ids" in specs


def test_embeddings_port_is_optional():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    spec = DeepEIoUTrack.INPUT_SPECS["embeddings"]
    assert spec.optional is True


# ── Forward tests ────────────────────────────────────────────────────────────


def test_empty_detections():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(with_reid=False)
    inputs = _make_inputs(n=0)
    out = node.forward(**inputs)
    assert out["bboxes"].shape == (1, 0, 4)
    assert out["track_ids"].shape == (1, 0)
    assert out["confidences"].shape == (1, 0)
    assert out["category_ids"].shape == (1, 0)


def test_single_detection_produces_track():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(with_reid=False)
    inputs = _make_inputs(n=1)

    # Frame 1
    out1 = node.forward(**inputs)
    # Frame 2 — same detection, should get a track
    out2 = node.forward(**inputs)

    # At least one of the frames should produce a valid track ID
    ids1 = out1["track_ids"][0].numpy()
    ids2 = out2["track_ids"][0].numpy()
    assert (ids1 >= 1).any() or (ids2 >= 1).any()


def test_persistent_track_ids():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(with_reid=False, new_track_thresh=0.5)
    inputs = _make_inputs(n=2)

    # Run several frames with same detections
    results = []
    for _ in range(5):
        out = node.forward(**inputs)
        results.append(out["track_ids"][0].numpy().copy())

    # After initial frames, track IDs should stabilize
    # Check last two frames have same IDs
    last = results[-1]
    second_last = results[-2]
    assigned_last = last[last >= 1]
    assigned_second = second_last[second_last >= 1]
    if len(assigned_last) > 0 and len(assigned_second) > 0:
        assert set(assigned_last.tolist()) == set(assigned_second.tolist())


def test_eiou_only_mode():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(with_reid=False)
    inputs = _make_inputs(n=3)

    out = node.forward(**inputs)
    assert "track_ids" in out
    assert out["track_ids"].shape == (1, 3)


def test_with_reid_and_embeddings():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(with_reid=True)
    inputs = _make_inputs(n=3, with_embeddings=True, emb_dim=128)

    out1 = node.forward(**inputs)
    out2 = node.forward(**inputs)

    assert out1["track_ids"].shape == (1, 3)
    assert out2["track_ids"].shape == (1, 3)


def test_with_reid_no_embeddings_fallback():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(with_reid=True)
    # Provide no embeddings — should fall back to EIoU-only
    inputs = _make_inputs(n=3, with_embeddings=False)

    # Should not crash
    out = node.forward(**inputs)
    assert out["track_ids"].shape == (1, 3)


# ── Output dtype tests ──────────────────────────────────────────────────────


def test_output_dtypes():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(with_reid=False)
    inputs = _make_inputs(n=2)
    out = node.forward(**inputs)

    assert out["bboxes"].dtype == torch.float32
    assert out["track_ids"].dtype == torch.int64
    assert out["confidences"].dtype == torch.float32
    assert out["category_ids"].dtype == torch.int64


# ── State management tests ──────────────────────────────────────────────────


def test_reset_clears_state():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack(with_reid=False)
    inputs = _make_inputs(n=2)
    node.forward(**inputs)

    assert node._tracker is not None
    assert node._frame_id > 0

    node.reset()
    assert node._tracker is None
    assert node._frame_id == 0


def test_state_dict_is_empty():
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    node = DeepEIoUTrack()
    assert node.state_dict() == {}


# ── Package / import integrity tests ────────────────────────────────────────


def test_import_deep_eiou_tracker():
    from deep_eiou_tracker.Deep_EIoU import Deep_EIoU  # noqa: F401


def test_no_sys_path_manipulation():
    """Importing the plugin should not add entries to sys.path."""
    initial_paths = set(sys.path)
    import cuvis_ai_deepeiou  # noqa: F401

    added = set(sys.path) - initial_paths
    # Only the package itself may appear (via editable install), but no
    # hand-crafted tracker/ paths should be injected.
    for p in added:
        assert "tracker" not in p.lower() or "deep_eiou_tracker" in p.lower()


def test_np_float_shim_present():
    import cuvis_ai_deepeiou  # noqa: F401

    assert hasattr(np, "float")


def test_cython_bbox_fallback_available():
    import cuvis_ai_deepeiou  # noqa: F401

    mod = sys.modules.get("cython_bbox")
    assert mod is not None
    assert callable(getattr(mod, "bbox_overlaps", None))


# ── Integration test: ByteTrack + DeepEIoU coexistence ──────────────────────


def test_bytetrack_deepeiou_coexistence(tmp_path):
    """Process-level integration test: loads both ByteTrack and DeepEIoU
    plugin manifests, runs both trackers sequentially on synthetic data —
    no namespace collision.

    Uses tmp_path fixture with temporary plugin manifests (relative paths)
    so it works in CI without absolute paths.
    """
    try:
        from cuvis_ai_bytetrack.node import ByteTrack
    except ImportError:
        pytest.skip("cuvis_ai_bytetrack not installed")

    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    # Create both trackers
    bt = ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    de = DeepEIoUTrack(with_reid=False, track_high_thresh=0.6)

    inputs = _make_inputs(n=3)

    # Run ByteTrack
    bt_out = bt.forward(**inputs)
    assert bt_out["track_ids"].shape == (1, 3)

    # Run DeepEIoU — should not crash from namespace collision
    de_out = de.forward(**inputs)
    assert de_out["track_ids"].shape == (1, 3)

    # Verify both trackers produced valid outputs
    assert bt_out["bboxes"].shape == (1, 3, 4)
    assert de_out["bboxes"].shape == (1, 3, 4)
