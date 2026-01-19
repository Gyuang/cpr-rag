"""CTDoc Models - Vision encoders, VLMs, and classifiers."""

# Vision backbones
from .vision import (
    build_vision_backbone,
    MedicalVisionTower,
    RadFMBackbone,
    M3DBackbone,
    CT2RepBackbone,
)

# Vision-Language Models
from .vlm import (
    VisualLlamaConfig,
    VisualLlamaModel,
)

# Classifiers
from .classifiers import (
    OrganContrastiveModel,
    OrganMLPOnlyModel,
)

# Legacy aliases for backwards compatibility
CT2RepLlamaConfig = VisualLlamaConfig
CT2RepLlamaModel = VisualLlamaModel

__all__ = [
    # Vision
    "build_vision_backbone",
    "MedicalVisionTower",
    "RadFMBackbone",
    "M3DBackbone",
    "CT2RepBackbone",
    # VLM
    "VisualLlamaConfig",
    "VisualLlamaModel",
    # Classifiers
    "OrganContrastiveModel",
    "OrganMLPOnlyModel",
    # Legacy aliases
    "CT2RepLlamaConfig",
    "CT2RepLlamaModel",
]
