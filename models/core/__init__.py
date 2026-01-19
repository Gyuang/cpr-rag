"""Core model components - VLM, vision backbones, and utilities."""
from .components import PerceiverResampler, ViT3D
from .visual_llama import VisualLlamaConfig, VisualLlamaModel
from .vision_backbones import build_vision_backbone

# Backwards compatibility aliases
CT2RepLlamaConfig = VisualLlamaConfig
CT2RepLlamaModel = VisualLlamaModel

__all__ = [
    "PerceiverResampler",
    "ViT3D",
    "build_vision_backbone",
    "VisualLlamaConfig",
    "VisualLlamaModel",
    # Legacy aliases
    "CT2RepLlamaConfig",
    "CT2RepLlamaModel",
]
