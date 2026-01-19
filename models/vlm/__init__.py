"""Vision-Language Models."""
from ..core.visual_llama import VisualLlamaConfig, VisualLlamaModel

# Backwards compatibility aliases
CT2RepLlamaConfig = VisualLlamaConfig
CT2RepLlamaModel = VisualLlamaModel

__all__ = [
    "VisualLlamaConfig",
    "VisualLlamaModel",
    # Legacy aliases
    "CT2RepLlamaConfig",
    "CT2RepLlamaModel",
]
