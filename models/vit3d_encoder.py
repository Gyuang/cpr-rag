"""Optional 3D ViT-style visual encoder."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViT3DEncoder(nn.Module):
    """Lightweight convolutional encoder that mimics ViT3D behaviour."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 512,
        patch_size: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = nn.Parameter(torch.randn(1, 1024, embed_dim))

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.positional[:, : tokens.size(1)]
        encoded = self.encoder(tokens)
        return {"sequence": encoded, "pooled": encoded.mean(dim=1)}

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from typing import Tuple

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def l2norm(t):
    return F.normalize(t, dim = -1)

# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# PEG - position generating module

class PEG(nn.Module):
    def __init__(self, dim, causal = False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups = dim)

    def forward(self, x, shape):
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape

        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, 'b ... d -> b d ...')

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value = 0.)
        x = self.dsconv(x)

        x = rearrange(x, 'b d ... -> b ... d')

        if needs_shape:
            x = rearrange(x, 'b ... d -> b (...) d')

        return x.reshape(orig_shape)

# attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        num_null_kv = 0,
        norm_context = True,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads = heads)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        if num_null_kv > 0:
            self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))
        else:
            self.register_parameter('null_kv', None)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        attn_bias = None
    ):
        batch, device, dtype = x.shape[0], x.device, x.dtype
        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if self.num_null_kv > 0 and exists(self.null_kv):
            nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b = batch, r = 2).unbind(dim = -2)
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value = 0.)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# alibi positional bias for extrapolation

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]
        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 2, # 2 for images, 3 for video
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, *dimensions, device = torch.device('cpu')):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        dtype = self.net[0][0].weight.dtype
        rel_pos = self.rel_pos.to(dtype)

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        peg = False,
        peg_causal = False,
        attn_num_null_kv = 2,
        has_cross_attn = False,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PEG(dim = dim, causal = peg_causal) if peg else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout),
                Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv, dropout = attn_dropout) if has_cross_attn else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm_out = LayerNorm(dim)

    def forward(
        self,
        x,
        video_shape = None,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None
    ):

        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape = video_shape) + x

            x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x

            x = ff(x) + x

        return self.norm_out(x)


class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        # x: [b, n, d] (Image Features)
        # latents: [b, num_latents, d] (Queries)
        
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(self, dim, depth=2, num_latents=32, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim, dim_head=dim_head, heads=heads),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim)
                )
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        b = x.shape[0]
        latents = repeat(self.latents, 'n d -> b n d', b=b)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        return self.norm(latents)


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class CTViTBackbone(nn.Module):
    def __init__(
        self,
        dim=512,                # Hidden Dimension
        image_size=256,         # H, W
        patch_size=16,          # Patch H, W
        temporal_patch_size=8,  # Patch D (Time)
        spatial_depth=4,        # Spatial Transformer Layers
        temporal_depth=4,       # Temporal Transformer Layers
        num_visual_tokens=32,   # Final Output Tokens (Perceiver Latents)
        channels=1,
        dim_head=32,
        heads=8,
        attn_dropout=0.,
        ff_dropout=0.
    ):
        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)',
                      p1=patch_height, p2=patch_width, pt=temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)

        self.dim = dim
        self.num_visual_tokens = num_visual_tokens

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def forward(self, video):
        """
        Args:
            video: [Batch, Channel, Depth, Height, Width]
        Returns:
            tokens: [Batch, t, h, w, Dim]
        """
        tokens = self.to_patch_emb(video)
        b, t, h, w, d = tokens.shape

        # Spatial encoding
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        if attn_bias.dtype != tokens.dtype:
            attn_bias = attn_bias.to(tokens.dtype)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=(b, t, h, w))
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, t=t, h=h, w=w)

        # Temporal encoding
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.enc_temporal_transformer(tokens, video_shape=(b, h, w, t))
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        return tokens