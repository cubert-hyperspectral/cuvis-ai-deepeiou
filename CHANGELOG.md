# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.1.0 - 2026-04-07

- Added `cuvis_ai_deepeiou` plugin package with `DeepEIoUTrack`, `OSNetExtractor`, and `ResNetExtractor` node classes.
- Added plugin scaffolding with `pyproject.toml`, `setuptools-scm` versioning, LICENSE, and README.
- Added offline JSON + video ReID tracking example workflow.
- Added CI (`ci.yml`) and tag-driven GitHub release (`release.yml`) workflows.
- Added standard `.gitignore` and removed upstream Deep-EIoU files from tracking.
- Fixed extractor tests for Linux CI compatibility.
