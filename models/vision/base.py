"""Base classes for vision encoders."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from einops import rearrange

from .utils import extract_sequence


class PerceiverAggregator(nn.Module):
    """Wrapper around PerceiverResampler to compress vision tokens."""

    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        depth: int = 2,
        max_num_media: Optional[int] = None,
        max_num_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        from ..CT2Rep.radfm_components import PerceiverResampler

        self.perceiver = PerceiverResampler(
            dim=input_dim,
            depth=depth,
            num_latents=num_tokens,
            max_num_media=max_num_media,
            max_num_frames=max_num_frames,
        )

    def forward(self, features: object) -> torch.Tensor:
        seq, token_shape = extract_sequence(features)
        perceiver_input = self._reshape(seq, token_shape)
        latents = self.perceiver(perceiver_input)
        return latents[:, 0]

    def _reshape(self, seq: torch.Tensor, token_shape: Optional[Tuple[int, ...]]) -> torch.Tensor:
        if seq.ndim == 5:  # [B, T, H, W, D]
            return rearrange(seq, "b t h w d -> b 1 t (h w) d")
        if seq.ndim == 4:  # [B, H, W, D]
            return rearrange(seq, "b h w d -> b 1 1 (h w) d")
        if seq.ndim != 3:
            raise ValueError(f"PerceiverAggregator expects a tensor with 3-5 dims, got {seq.shape}.")
        if token_shape and len(token_shape) == 3 and token_shape[0] * token_shape[1] * token_shape[2] == seq.size(1):
            th, tw, td = token_shape
            return rearrange(seq, "b (th tw td) d -> b 1 1 (th tw td) d", th=th, tw=tw, td=td)
        return seq[:, None, None, :, :]


class PoolingAggregator(nn.Module):
    """Adaptive average pooling over token dimension."""

    def __init__(self, num_tokens: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, features: object) -> torch.Tensor:
        seq, _ = extract_sequence(features)
        if seq.ndim == 5:
            seq = rearrange(seq, "b t h w d -> b (t h w) d")
        elif seq.ndim == 4:
            seq = rearrange(seq, "b h w d -> b (h w) d")
        elif seq.ndim != 3:
            raise ValueError(f"PoolingAggregator expects a tensor with 3-5 dims, got {seq.shape}.")
        pooled = self.pool(seq.transpose(1, 2)).transpose(1, 2)
        return pooled


class VisionProjector(nn.Module):
    """Linear projection into the LLM embedding space with optional LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, *, use_norm: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.linear(tokens)
        return self.norm(projected)


class MedicalVisionTower(nn.Module):
    """Composable encoder -> aggregator -> projector pipeline."""

    def __init__(self, encoder: nn.Module, aggregator: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.projector = projector

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        features = self.encoder(volume)
        tokens = self.aggregator(features)
        return self.projector(tokens)
