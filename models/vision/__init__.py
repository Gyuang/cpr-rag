"""Vision encoder module."""
from typing import Optional

from torch import nn

from .base import (
    MedicalVisionTower,
    PerceiverAggregator,
    PoolingAggregator,
    VisionProjector,
)
from .radfm import RadFMEncoder, RadFMBackbone
from .m3d import M3DEncoder, M3DBackbone
from .ct2rep import CT2RepEncoder, CT2RepBackbone


def build_vision_backbone(
    name: str,
    *,
    hidden_dim: int,
    num_tokens: int,
    ct2rep_ckpt: Optional[str] = None,
    radfm_ckpt: Optional[str] = None,
    m3d_ckpt: Optional[str] = None,
) -> nn.Module:
    """Build vision backbone by name.

    Note: Reg2RG is a complete VLM, not just a vision backbone.
    Use models.vlm.reg2rg.Reg2RGModel for the full Reg2RG model.
    """
    if name == "radfm":
        return RadFMBackbone(hidden_dim=hidden_dim, num_tokens=num_tokens, checkpoint=radfm_ckpt)
    if name == "ct2rep":
        return CT2RepBackbone(hidden_dim=hidden_dim, num_tokens=num_tokens, checkpoint=ct2rep_ckpt)
    if name == "m3d":
        return M3DBackbone(hidden_dim=hidden_dim, num_tokens=num_tokens, checkpoint=m3d_ckpt)
    raise ValueError(f"Unsupported vision backbone '{name}'.")


__all__ = [
    # Base classes
    "MedicalVisionTower",
    "PerceiverAggregator",
    "PoolingAggregator",
    "VisionProjector",
    # Encoders
    "RadFMEncoder",
    "RadFMBackbone",
    "M3DEncoder",
    "M3DBackbone",
    "CT2RepEncoder",
    "CT2RepBackbone",
    # Factory
    "build_vision_backbone",
]
