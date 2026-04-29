"""OSNet x1.0 feature extractor node."""

from cuvis_ai_schemas.enums import NodeCategory, NodeTag

from cuvis_ai_deepeiou.node.bbox_feature_extractor import BBoxFeatureExtractor


class OSNetExtractor(BBoxFeatureExtractor):
    """OSNet x1.0 feature extractor (512-dim embeddings)."""

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

    MODEL_NAME = "osnet_x1_0"
    FEATURE_DIM = 512
    HF_WEIGHTS = {"repo_id": "kadirnar/osnet_x1_0_imagenet", "filename": "osnet_x1_0_imagenet.pt"}
