# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added a `no-local-sources` CI workflow that fails if `pyproject.toml` declares a local `[tool.uv.sources]` path entry (a machine-specific path must not ship in a release).

- CI: add a detect-secrets secret-scan job (git-tracked files only).

## 0.2.0 - 2026-06-23

- Migrated the example plugin manifest (`examples/deepeiou/plugins.yaml`) to the bare `capabilities:` shape required by cuvis-ai-schemas 0.6.0.
- Require `cuvis-ai-core>=0.10.0` and `cuvis-ai-schemas>=0.7.0`, adopting the released framework versions.

## 0.1.2 - 2026-06-10

- Require `cuvis-ai-core>=0.7.1` and `cuvis-ai-schemas>=0.5.2` (inherits the upstream security floors transitively).
- Updated `examples/deepeiou/plugins.yaml` `provides` to the `CatalogNodeEntry` `class_name:` form required by cuvis-ai-schemas 0.5.2.
- Added the `cuvis_ai_compat.yml` dependency-compatibility workflow (audits the plugin's deps against the cuvis-ai-core lock).
- Removed the PyPI/TestPyPI release workflow; the plugin is distributed via git tags referenced from cuvis-ai plugin manifests.
- Stripped `torch` / `torchvision` wheel hashes from `uv.lock`.

## 0.1.1 - 2026-04-29

- Annotated `BBoxFeatureExtractor` (abstract base), `OSNetExtractor`, and `ResNetExtractor` with `_category = NodeCategory.MODEL` and `_tags = {RGB, IMAGE, EMBEDDING, INFERENCE, LEARNABLE, BATCHED, TORCH}`; `DeepEIoUTrack` with `_category = NodeCategory.TRANSFORM` and `_tags = {BBOX, TRACKING, INFERENCE, STATEFUL, TORCH}`.
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
