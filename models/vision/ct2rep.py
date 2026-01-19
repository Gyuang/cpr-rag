"""CT2Rep vision encoder."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .base import MedicalVisionTower, PerceiverAggregator, VisionProjector
from .utils import load_checkpoint_into
from ..vit3d_encoder import CTViTBackbone


_CT2REP_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("encoder.vit.vit.", "encoder.vit."),
    ("encoder.vit.", "encoder.vit."),
    ("encoder.", "encoder.vit."),
    ("perceiver.", "aggregator.perceiver."),
    ("project.", "projector.linear."),
    ("norm.", "projector.norm."),
)


class CT2RepEncoder(nn.Module):
    """Lightweight ViT-style encoder producing spatiotemporal tokens."""

    def __init__(self, hidden_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.target_hw = 256
        self.target_depth = 64
        self.target_shape = (self.target_hw, self.target_hw, self.target_depth)
        self.vit = CTViTBackbone(
            dim=hidden_dim,
            image_size=self.target_hw,
            patch_size=16,
            temporal_patch_size=2,
            spatial_depth=4,
            temporal_depth=4,
            num_visual_tokens=num_tokens,
            channels=1,
        )

    def _prepare_volume(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.shape[2:5] != self.target_shape:
            volume = F.interpolate(volume, size=self.target_shape, mode="trilinear", align_corners=False)
        # convert [B, C, H, W, D] -> [B, C, D, H, W]
        volume = volume.permute(0, 1, 4, 2, 3)
        return volume

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        x = self._prepare_volume(volume)
        target_dtype = next(self.vit.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        return self.vit(x)  # [B, T, H, W, hidden_dim]


class CT2RepBackbone(MedicalVisionTower):
    """CT2Rep encoder + Perceiver aggregator + projector."""

    def __init__(self, hidden_dim: int, num_tokens: int, checkpoint: Optional[str] = None) -> None:
        encoder = CT2RepEncoder(hidden_dim=hidden_dim, num_tokens=num_tokens)
        aggregator = PerceiverAggregator(input_dim=hidden_dim, num_tokens=num_tokens, depth=2)
        projector = VisionProjector(in_dim=hidden_dim, out_dim=hidden_dim)
        super().__init__(encoder, aggregator, projector)
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        load_checkpoint_into(self, checkpoint, tag="ct2rep-backbone", replacements=_CT2REP_REPLACEMENTS, strict=False)
