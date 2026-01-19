"""Graph-Guided Query Interaction (G-QIA).

Applies self-attention between query tokens with an additive attention bias derived
from a data-driven clinical graph (conditional co-occurrence matrix).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


def build_organ_graph_from_disease_matrix(
    disease_M: torch.Tensor,
    *,
    organ_label_mapping: Dict[str, Iterable[int]],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aggregate 18x18 disease matrix into an OxO organ matrix.

    organ_graph[a, b] = mean_{i in A, j in B} M[i, j]
    """
    M = disease_M.float()
    organs = list(organ_label_mapping.keys())
    O = len(organs)
    organ_graph = torch.zeros((O, O), dtype=torch.float32)
    for a, organ_a in enumerate(organs):
        ia = [int(i) for i in organ_label_mapping[organ_a]]
        for b, organ_b in enumerate(organs):
            ib = [int(i) for i in organ_label_mapping[organ_b]]
            block = M.index_select(0, torch.tensor(ia)).index_select(1, torch.tensor(ib))
            organ_graph[a, b] = float(block.mean().item())
    organ_graph = organ_graph.clamp(min=0.0, max=1.0)
    # Ensure diagonals allow intra-organ interaction.
    for i in range(O):
        organ_graph[i, i] = 1.0
    # Avoid exactly 0.
    organ_graph = organ_graph.clamp(min=float(eps))
    return organ_graph


def build_query_attn_bias(
    organ_graph: torch.Tensor,
    *,
    num_organ_tokens: int,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Build additive attention bias for query-query self-attention.

    Returns:
        bias: [L, L] where L = num_organs * num_organ_tokens
    """
    if num_organ_tokens < 1:
        raise ValueError(f"num_organ_tokens must be >= 1, got {num_organ_tokens}")
    O = int(organ_graph.shape[0])
    if organ_graph.shape != (O, O):
        raise ValueError(f"organ_graph must be square, got {tuple(organ_graph.shape)}")

    L = O * int(num_organ_tokens)
    bias = torch.empty((L, L), dtype=torch.float32)

    # log(prob) is <= 0; higher prob -> closer to 0 (less penalty).
    log_g = torch.log(organ_graph.float().clamp(min=float(eps)))

    for a in range(O):
        for b in range(O):
            v = log_g[a, b]
            ra = slice(a * num_organ_tokens, (a + 1) * num_organ_tokens)
            rb = slice(b * num_organ_tokens, (b + 1) * num_organ_tokens)
            bias[ra, rb] = v
    return bias


def load_disease_graph_matrix(graph_path: str | Path) -> torch.Tensor:
    """Load 18x18 disease conditional probability matrix from checkpoint."""
    ckpt = torch.load(graph_path, map_location="cpu", weights_only=False)
    M = ckpt["prob_matrix"]
    if not torch.is_tensor(M):
        M = torch.tensor(np.asarray(M), dtype=torch.float32)
    M = M.float()
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"prob_matrix must be square, got {tuple(M.shape)}")
    return M


class GraphGuidedQueryInteraction(nn.Module):
    """Graph-masked self-attention for query tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        attn_bias: torch.Tensor,  # [L, L]
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")

        self.query_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.register_buffer("attn_bias", attn_bias.float(), persistent=True)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """queries: [B, L, D]."""
        attn_out, _ = self.query_self_attn(queries, queries, queries, attn_mask=self.attn_bias)
        out = queries + torch.sigmoid(self.gate) * attn_out
        return self.norm(out)

