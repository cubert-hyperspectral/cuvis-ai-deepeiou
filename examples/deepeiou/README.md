# DeepEIoU Offline Tracker Example

Run DeepEIoU tracking on pre-computed COCO detection JSON files.
This is a **Phase 1.5** example — no live model inference, just tracker logic on existing detections.

## Prerequisites

Install the plugin in editable mode from the repo root:

```powershell
cd D:\code-repos\cuvis-ai-deepeiou\deepeiou-init
uv pip install -e .
```

`cuvis-ai-tracking` must also be installed (provides `DetectionJsonReader` and `ByteTrackCocoJson`).

## Quick Start — Busstation RGB Baseline (no ReID)

```powershell
uv run python examples/deepeiou/deepeiou_from_detection_json.py `
    --input-json "D:/data/XMR_notarget_Busstation/20260226/tracker/bytetrack_rgb_test/detection_results.json" `
    --output-json "D:/experiments/deepeiou/baseline_wo_features_rgb/tracking_results.json"
```

This reads the ByteTrack-generated detections, re-tracks them with DeepEIoU (EIoU-only, no ReID features), and writes the tracking output.

### Input

| File | Description |
|------|-------------|
| `D:/data/XMR_notarget_Busstation/20260226/Auto_013+01-trustimulus.mp4` | Source video (RGB tristimulus) |
| `D:/data/XMR_notarget_Busstation/20260226/tracker/bytetrack_rgb_test/detection_results.json` | COCO-format detections from YOLO + ByteTrack pipeline |

### Output

Results are written to `D:/experiments/deepeiou/baseline_wo_features_rgb/`:

| File | Description |
|------|-------------|
| `tracking_results.json` | COCO-format tracking output with `track_id` per annotation |

## CLI Options

```
--input-json         COCO detection JSON (required)
--output-json        Output tracking JSON path (required)
--track-high-thresh  High-confidence detection threshold [default: 0.6]
--track-low-thresh   Low-confidence detection threshold [default: 0.1]
--new-track-thresh   Minimum score to initialize a new track [default: 0.7]
--track-buffer       Frames to keep a lost track alive [default: 60]
--match-thresh       IoU matching threshold [default: 0.8]
--frame-rate         Video frame rate (affects track buffer) [default: 30]
--with-reid/--no-reid  Enable ReID embeddings [default: --no-reid]
```

## ReID Embeddings

`DeepEIoUTrack` has an optional `embeddings` input port `[1, N, D]` float32.
When provided, the tracker fuses EIoU spatial distance with cosine appearance
distance for stronger association. When absent, it falls back to EIoU-only mode.

### Option A: BBoxSpectralExtractor (HSI pipelines — ready now)

For hyperspectral data, `BBoxSpectralExtractor` (in cuvis-ai-tracking,
`cuvis_ai/node/spectral_extractor.py`) produces L2-normalized spectral
signatures `[1, N, C]` that map directly to the `embeddings` port:

```python
spectral = BBoxSpectralExtractor(l2_normalize=True)
spec_out = spectral.forward(cube=cube, bboxes=track_out["bboxes"])

track_out = tracker.forward(
    bboxes=det["bboxes"],
    category_ids=det["category_ids"],
    confidences=det["confidences"],
    embeddings=spec_out["spectral_signatures"],  # [1, N, C] -> embeddings port
)
```

See `examples/object_tracking/deepeiou/yolo_deepeiou_hsi.py` for the full
CU3S pipeline with `--with-reid` / `--no-reid` flags.

### Option B: torchreid FeatureExtractor (RGB — future Phase 2)

The upstream `Deep-EIoU/reid/torchreid/` contains an OSNet model that produces
512-dim appearance embeddings per bbox crop. This is **not yet wrapped** as a
cuvis.ai Node. To use it manually:

```python
from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path="path/to/sports_model.pth.tar-60",
    device="cuda",
)
# Crop each bbox from the RGB frame, resize to 256x128
crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in bboxes_np]
features = extractor(crops)  # [N, 512]
embeddings = features.unsqueeze(0)  # [1, N, 512]
```

A proper `OSNetReIDNode` wrapping this as a cuvis.ai Node is planned for Phase 2.

### Offline JSON limitation

The offline JSON example (`deepeiou_from_detection_json.py`) has **no embedding
source** — COCO detection JSON does not store per-detection embeddings. The
`--with-reid` flag exists but will have no effect unless the script is extended
to load embeddings from a separate file.

## Examples

### Tuning tracker thresholds

Lower `new-track-thresh` to pick up more tracks, increase `track-buffer` for longer occlusions:

```powershell
uv run python examples/deepeiou/deepeiou_from_detection_json.py `
    --input-json "D:/data/XMR_notarget_Busstation/20260226/tracker/bytetrack_rgb_test/detection_results.json" `
    --output-json "D:/experiments/deepeiou/baseline_wo_features_rgb/tracking_results.json" `
    --new-track-thresh 0.5 `
    --track-buffer 90
```
