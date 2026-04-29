# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.1.1 - 2026-04-29

- Annotated `BBoxFeatureExtractor` (abstract base), `OSNetExtractor`, and `ResNetExtractor` with `_category = NodeCategory.MODEL` and `_tags = {RGB, IMAGE, EMBEDDING, INFERENCE, LEARNABLE, BATCHED, TORCH}`; `DeepEIoUTrack` with `_category = NodeCategory.TRANSFORM` and `_tags = {BBOX, TRACKING, INFERENCE, STATEFUL, TORCH}` (ALL-5187 Phase 6).
- Added `cuvis-ai-schemas>=0.4.0` to dependencies (`NodeCategory` / `NodeTag` enums live there).
- Stripped `hash` fields from `torch` / `torchvision` wheel entries in `uv.lock`.

## 0.1.0 - 2026-04-07

- Added `cuvis_ai_deepeiou` plugin package with `DeepEIoUTrack`, `OSNetExtractor`, and `ResNetExtractor` node classes.
- Added plugin scaffolding with `pyproject.toml`, `setuptools-scm` versioning, LICENSE, and README.
- Added offline JSON + video ReID tracking example workflow.
- Added CI (`ci.yml`) and tag-driven GitHub release (`release.yml`) workflows.
- Added security scanning job (pip-audit, detect-secrets, bandit).
- Added standard `.gitignore` and removed upstream Deep-EIoU files from tracking.
- Fixed extractor tests for Linux CI compatibility.
