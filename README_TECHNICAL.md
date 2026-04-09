# cuvis-ai-deepeiou

`cuvis-ai-deepeiou` packages the cuvis.ai plugin nodes that wrap the DeepEIoU
multi-object tracker and its optional ReID feature extractors for use inside
`NodeRegistry` pipelines.

The repository exposes three plugin-facing node classes:

- `cuvis_ai_deepeiou.node.DeepEIoUTrack`
- `cuvis_ai_deepeiou.node.OSNetExtractor`
- `cuvis_ai_deepeiou.node.ResNetExtractor`

## Release Manifest

Use the released plugin from `cuvis-ai-tracking` with a selective manifest:

```yaml
plugins:
  deepeiou:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_deepeiou.node.DeepEIoUTrack
      - cuvis_ai_deepeiou.node.OSNetExtractor
      - cuvis_ai_deepeiou.node.ResNetExtractor
```

## Local Development Manifest

Use a local checkout while iterating on the plugin:

```yaml
plugins:
  deepeiou:
    path: "../../../../cuvis-ai-deepeiou/deepeiou-init"
    provides:
      - cuvis_ai_deepeiou.node.DeepEIoUTrack
      - cuvis_ai_deepeiou.node.OSNetExtractor
      - cuvis_ai_deepeiou.node.ResNetExtractor
```

## Run Through cuvis-ai-tracking

From `cuvis-ai-tracking`, this plugin is exercised by the YOLO + DeepEIoU
tracking example:

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
  --video-path "D:\experiments\20260331\video_creation\tristimulus\XMR_50mm_ObjectTracking\20260331\12_39_03\Auto_002.mp4" `
  --no-reid `
  --output-dir "D:\experiments\20260407\deepeiou" `
  --out-basename "v0_1_0_smoke" `
  --end-frame 60
```

## Local Validation

```powershell
uv run --no-sources --extra dev pytest tests/cuvis_ai_deepeiou -v
uv run --no-sources --extra dev ruff format --check cuvis_ai_deepeiou tests/cuvis_ai_deepeiou
uv run --no-sources --extra dev ruff check cuvis_ai_deepeiou tests/cuvis_ai_deepeiou
uv build --no-sources
uv run --no-sources --with twine twine check dist/*
```

## Notes

- The tracker implementation is derived from Deep-EIoU by Hsiang-Wei Huang et
  al. Cubert GmbH distributes the plugin with permission from the original
  authors.
- `cuvis_ai_deepeiou/reid/` vendors ReID model code under the MIT license in
  `cuvis_ai_deepeiou/reid/LICENSE`.
- The repository-level distribution notice is in [LICENSE](LICENSE).
