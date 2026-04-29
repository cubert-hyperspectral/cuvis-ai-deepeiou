"""DeepEIoU multi-object tracker node for cuvis.ai."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger


class DeepEIoUTrack(Node):
    """DeepEIoU multi-object tracker node.

    Consumes per-frame detections (bounding boxes, class IDs, confidence
    scores) and optionally per-detection ReID embeddings, producing tracked
    bounding boxes with persistent track IDs using the Deep_EIoU algorithm.

    When ``with_reid=True``, the embedding source should be wired to the
    ``embeddings`` input port. If embeddings are missing on individual frames
    (e.g. upstream node produces ``None``), the node sets
    ``args.with_reid = False`` for that frame, falling back to EIoU-only mode
    (equivalent to improved ByteTrack with expansion-IoU).

    Parameters
    ----------
    track_high_thresh : float
        High-confidence threshold for first association (default 0.6).
    track_low_thresh : float
        Lowest detection threshold valid for tracking (default 0.1).
    new_track_thresh : float
        Threshold to initialize new tracks (default 0.7).
    track_buffer : int
        Frames to keep lost tracks before removal (default 60).
    match_thresh : float
        Matching threshold for Hungarian assignment (default 0.8).
    frame_rate : int
        Video frame rate, affects track buffer duration (default 30).
    with_reid : bool
        Enable ReID embedding-based association (default True).
    proximity_thresh : float
        EIoU distance above which ReID match is ignored (default 0.5).
    appearance_thresh : float
        Embedding distance above which appearance match is rejected (default 0.25).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({
        NodeTag.BBOX,
        NodeTag.TRACKING,
        NodeTag.INFERENCE,
        NodeTag.STATEFUL,
        NodeTag.TORCH,
    })

    INPUT_SPECS = {
        "bboxes": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, 4),
            description="Detection bounding boxes [1, N, 4] xyxy pixel coordinates.",
        ),
        "category_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Detection category IDs [1, N].",
        ),
        "confidences": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Detection confidence scores [1, N].",
        ),
        "embeddings": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1),
            description=(
                "Per-detection ReID embeddings [1, N, D]. "
                "Optional; if absent, tracker uses EIoU-only mode."
            ),
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "bboxes": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, 4),
            description="Input bboxes pass-through [1, N, 4] xyxy pixel coordinates.",
        ),
        "track_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description=(
                "Track IDs aligned with input detections [1, N]. "
                "-1 for detections not assigned to any track."
            ),
        ),
        "confidences": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Input confidences pass-through [1, N].",
        ),
        "category_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Input category_ids pass-through [1, N].",
        ),
    }

    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        track_buffer: int = 60,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        with_reid: bool = True,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        **kwargs: Any,
    ) -> None:
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.frame_rate = int(frame_rate)
        self.with_reid = bool(with_reid)
        self.proximity_thresh = float(proximity_thresh)
        self.appearance_thresh = float(appearance_thresh)

        super().__init__(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
            with_reid=with_reid,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
            **kwargs,
        )

        self._tracker = None  # Deep_EIoU, lazy initialized
        self._frame_id: int = 0

    def reset(self) -> None:
        """Reset tracker state for a new video sequence."""
        self._tracker = None
        self._frame_id = 0

    def state_dict(self) -> dict:
        """Return empty dict — no learned weights."""
        return {}

    def _ensure_tracker(self) -> None:
        """Lazily create the Deep_EIoU tracker on first forward() call."""
        if self._tracker is not None:
            return

        from deep_eiou_tracker.Deep_EIoU import Deep_EIoU

        args = SimpleNamespace(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            mot20=False,
            with_reid=self.with_reid,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
        )
        self._tracker = Deep_EIoU(args, frame_rate=self.frame_rate)
        logger.info(
            "[DeepEIoU] Tracker initialized: "
            "track_high_thresh={}, track_buffer={}, match_thresh={}, "
            "with_reid={}, frame_rate={}",
            self.track_high_thresh,
            self.track_buffer,
            self.match_thresh,
            self.with_reid,
            self.frame_rate,
        )

    @torch.no_grad()
    def forward(
        self,
        bboxes: torch.Tensor,
        category_ids: torch.Tensor,
        confidences: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Run DeepEIoU association on per-frame detections.

        Parameters
        ----------
        bboxes : torch.Tensor
            Detection boxes ``[1, N, 4]`` float32 xyxy pixel coordinates.
        category_ids : torch.Tensor
            Detection category IDs ``[1, N]`` int64.
        confidences : torch.Tensor
            Detection confidence scores ``[1, N]`` float32.
        embeddings : torch.Tensor or None
            Per-detection ReID embeddings ``[1, N, D]`` float32. If None,
            tracker falls back to EIoU-only mode for this frame.
        context : Context or None
            Pipeline execution context (unused).

        Returns
        -------
        dict[str, torch.Tensor]
            ``bboxes``       : input bboxes pass-through ``[1, N, 4]``
            ``track_ids``    : track ID per detection ``[1, N]``, ``-1`` if unassigned
            ``confidences``  : input confidences pass-through ``[1, N]``
            ``category_ids`` : input category_ids pass-through ``[1, N]``
        """
        self._ensure_tracker()
        device = bboxes.device
        N = bboxes.shape[1]

        bboxes_np = bboxes[0].cpu().numpy()  # [N, 4]
        confs_np = confidences[0].cpu().numpy()  # [N]

        if N == 0:
            output_results = np.empty((0, 5), dtype=np.float32)
            emb_np = None
        else:
            output_results = np.column_stack([bboxes_np, confs_np]).astype(np.float32)
            if embeddings is not None:
                emb_np = embeddings[0].cpu().numpy()  # [N, D]
            else:
                emb_np = None

        # Upstream Deep_EIoU.update() indexes embedding[...] whenever
        # args.with_reid is True — passing None will crash.
        # Explicitly toggle per-frame to match actual embedding availability.
        self._tracker.args.with_reid = self.with_reid and emb_np is not None

        if self.with_reid and emb_np is None:
            logger.debug(
                "[DeepEIoU] frame={} with_reid=True but no embeddings; EIoU-only for this frame",
                self._frame_id,
            )

        online_targets = self._tracker.update(output_results, emb_np)

        self._frame_id += 1

        # Build N-aligned track_ids using det_idx stored on each STrack (see §6.5).
        track_ids_np = np.full(N, -1, dtype=np.int64)
        for track in online_targets:
            if track.det_idx is not None:
                track_ids_np[track.det_idx] = track.track_id

        logger.trace(
            "[DeepEIoU] frame={} dets={} assigned={}",
            self._frame_id - 1,
            N,
            int((track_ids_np != -1).sum()),
        )

        return {
            "bboxes": bboxes,  # passthrough
            "track_ids": torch.from_numpy(track_ids_np).unsqueeze(0).to(device),
            "confidences": confidences,  # passthrough
            "category_ids": category_ids,  # passthrough
        }
