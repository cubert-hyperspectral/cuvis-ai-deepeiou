"""Phase 1.5 offline example: JSON → DeepEIoU → tracking JSON.

Reads pre-computed COCO detections, runs DeepEIoU tracking, and writes
tracking output with persistent track IDs.

Usage:
    uv run python examples/deepeiou/deepeiou_from_detection_json.py \
        --input-json path/to/detections.json \
        --output-json path/to/tracking_output.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

# Ensure plugin packages are importable when running from the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cuvis_ai_deepeiou  # noqa: E402 — triggers shims


@click.command()
@click.option("--input-json", required=True, type=click.Path(exists=True), help="COCO detection JSON")
@click.option("--output-json", required=True, type=click.Path(), help="Output tracking JSON")
@click.option("--track-high-thresh", default=0.6, type=float, help="High-confidence threshold")
@click.option("--track-low-thresh", default=0.1, type=float, help="Low-confidence threshold")
@click.option("--new-track-thresh", default=0.7, type=float, help="New track threshold")
@click.option("--track-buffer", default=60, type=int, help="Lost-track buffer (frames)")
@click.option("--match-thresh", default=0.8, type=float, help="Match threshold")
@click.option("--frame-rate", default=30, type=int, help="Video frame rate")
@click.option("--with-reid/--no-reid", default=False, help="Enable/disable ReID (no embeddings in JSON)")
def main(
    input_json: str,
    output_json: str,
    track_high_thresh: float,
    track_low_thresh: float,
    new_track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    frame_rate: int,
    with_reid: bool,
) -> None:
    """Run DeepEIoU tracking on pre-computed detections."""
    from cuvis_ai.node.json_reader import DetectionJsonReader
    from cuvis_ai.node.json_writer import ByteTrackCocoJson
    from cuvis_ai_deepeiou.node import DeepEIoUTrack

    logger.info("Reading detections from {}", input_json)
    reader = DetectionJsonReader(json_path=input_json)

    tracker = DeepEIoUTrack(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        frame_rate=frame_rate,
        with_reid=with_reid,
    )

    writer = ByteTrackCocoJson(
        output_json_path=output_json,
        category_id_to_name={},
        flush_interval=1,
    )

    frame_count = 0
    try:
        while True:
            det = reader.forward()
            track_out = tracker.forward(
                bboxes=det["bboxes"],
                category_ids=det["category_ids"],
                confidences=det["confidences"],
            )
            writer.forward(
                frame_id=det["frame_id"],
                bboxes=track_out["bboxes"],
                category_ids=track_out["category_ids"],
                confidences=track_out["confidences"],
                track_ids=track_out["track_ids"],
                orig_hw=det["orig_hw"],
            )
            frame_count += 1
    except StopIteration:
        pass

    writer.close()
    logger.info("Processed {} frames → {}", frame_count, output_json)


if __name__ == "__main__":
    main()
