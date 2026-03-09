"""ResNet-50 feature extractor node."""

from cuvis_ai_deepeiou.node.bbox_feature_extractor import BBoxFeatureExtractor


class ResNetExtractor(BBoxFeatureExtractor):
    """ResNet-50 feature extractor (2048-dim embeddings)."""

    MODEL_NAME = "resnet50"
    FEATURE_DIM = 2048
