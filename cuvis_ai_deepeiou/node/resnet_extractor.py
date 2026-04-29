"""ResNet-50 feature extractor node."""

from cuvis_ai_schemas.enums import NodeCategory, NodeTag

from cuvis_ai_deepeiou.node.bbox_feature_extractor import BBoxFeatureExtractor


class ResNetExtractor(BBoxFeatureExtractor):
    """ResNet-50 feature extractor (2048-dim embeddings)."""

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.RGB,
            NodeTag.IMAGE,
            NodeTag.EMBEDDING,
            NodeTag.INFERENCE,
            NodeTag.LEARNABLE,
            NodeTag.BATCHED,
            NodeTag.TORCH,
        }
    )

    MODEL_NAME = "resnet50"
    FEATURE_DIM = 2048
