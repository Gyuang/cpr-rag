"""Subset of the RadFM vision stack used for CT2Rep training.

These components are adapted from https://github.com/chaoyi-wu/RadFM (MIT License)
so the pretrained checkpoints released with RadFM can be reused inside CTDoc.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import einsum, nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many


# ---------------------------------------------------------------------------
# Positional embeddings


class PositionEmbeddingLearned3d(nn.Module):
    """Absolute learnable 3D positional embedding."""

    def __init__(self, num_pos_feats: int = 256, h_patch_num: int = 16, w_patch_num: int = 16, d_patch_num: int = 128) -> None:
        super().__init__()
        self.h_patch_num = h_patch_num
        self.w_patch_num = w_patch_num
        self.d_patch_num = d_patch_num
        self.row_embed = nn.Embedding(h_patch_num, num_pos_feats)
        self.col_embed = nn.Embedding(w_patch_num, num_pos_feats)
        self.dep_embed = nn.Embedding(d_patch_num, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.dep_embed.weight)

    def forward(self, batch: int, h: int, w: int, d: int, x: torch.Tensor) -> torch.Tensor:
        h_idx = (torch.arange(h, device=x.device) + 1) * (self.h_patch_num // h) - 1
        w_idx = (torch.arange(w, device=x.device) + 1) * (self.w_patch_num // w) - 1
        d_idx = (torch.arange(d, device=x.device) + 1) * (self.d_patch_num // d) - 1
        row = self.row_embed(h_idx).unsqueeze(1).unsqueeze(2).repeat(1, w, d, 1)
        col = self.col_embed(w_idx).unsqueeze(0).unsqueeze(2).repeat(h, 1, d, 1)
        dep = self.dep_embed(d_idx).unsqueeze(0).unsqueeze(1).repeat(h, w, 1, 1)
        pos = torch.cat([row, col, dep], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        return rearrange(pos, "b h w d c -> b (h w d) c")


# ---------------------------------------------------------------------------
# ViT-style encoder borrowed from RadFM


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if heads > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT3D(nn.Module):
    """Minimal 3D ViT used by RadFM."""

    def __init__(
        self,
        *,
        image_size: int,
        image_patch_size: int,
        frames: int,
        frame_patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(image_patch_size)
        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")
        if frames % frame_patch_size != 0:
            raise ValueError("Frame dimension must be divisible by frame_patch_size.")

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frame_patch = frame_patch_size
        num_tokens = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)", p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = PositionEmbeddingLearned3d(dim // 3, image_height // patch_height, image_width // patch_width, frames // frame_patch_size)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.num_tokens = num_tokens

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w, d = video.shape
        tokens = self.to_patch_embedding(video)
        pos = self.pos_embedding(b, h // self.patch_height, w // self.patch_width, d // self.frame_patch, tokens)
        tokens = self.dropout(tokens + pos)
        encoded = self.transformer(tokens)
        return encoded, pos


# ---------------------------------------------------------------------------
# Perceiver resampler (taken from flamingo-pytorch)


def exists(value) -> bool:
    return value is not None


def _ffn(dim: int, mult: int = 4) -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        sim = einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 32,
        max_num_media: int | None = None,
        max_num_frames: int | None = None,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if exists(max_num_frames) else None
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        _ffn(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, media, frames = x.shape[:3]
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:frames], "f d -> b m f v d", b=b, m=media, v=x.size(3))
            x = x + frame_embs
        x = rearrange(x, "b m f v d -> b m (f v) d")
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:media]

        latents = repeat(self.latents, "n d -> b m n d", b=b, m=media)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)
