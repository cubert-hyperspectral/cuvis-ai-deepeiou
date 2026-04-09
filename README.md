![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# CUVIS.AI DeepEIoU

This repository provides a port of the DeepEIoU multi-object tracker and its ReID feature extractors as a cuvis.ai plugin. It is maintained by Cubert GmbH as part of the cuvis.ai ecosystem.

## Platform

cuvis.ai is split across multiple repositories:

| Repository | Role |
|---|---|
| [cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai-core) | Framework — base `Node` class, pipeline orchestration, services, and plugin system |
| [cuvis-ai-schemas](https://github.com/cubert-hyperspectral/cuvis-ai-schemas) | Shared schema definitions and generated types |
| [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai) | Node catalog and end-user pipeline examples |
| **cuvis-ai-deepeiou** (this repo) | DeepEIoU plugin — cuvis.ai nodes for multi-object tracking with optional ReID |

## Nodes

| Node | Description |
|---|---|
| `DeepEIoUTrack` | Multi-object tracker using EIoU association with optional deep feature ReID |
| `OSNetExtractor` | OSNet-based appearance feature extractor for ReID |
| `ResNetExtractor` | ResNet-based appearance feature extractor for ReID |

## Quick Start

For local development in this repository:

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou.git
cd cuvis-ai-deepeiou
uv sync --all-extras
```

For cuvis.ai usage examples, see the DeepEIoU tracking pipelines in [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai/tree/main/examples/object_tracking/deepeiou).

For the original upstream Deep-EIoU repository, research background, and technical details, see [README_TECHNICAL.md](README_TECHNICAL.md).

## Links

- **Documentation:** https://docs.cuvis.ai/latest/
- **Website:** https://www.cubert-hyperspectral.com/
- **Support:** http://support.cubert-hyperspectral.com/
- **Issues:** https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou/issues
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Technical README:** [README_TECHNICAL.md](README_TECHNICAL.md)

---

See [LICENSE](LICENSE) for repository licensing details.
