"""Organ-specific contrastive models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrganContrastiveModel(nn.Module):
    """Organ-specific contrastive model built on shared vision backbone."""

    ALIASES = {"airways": "trachea and bronchie"}
    DEFAULT_ORGANS = (
        "lung",
        "heart",
        "mediastinum",
        "pleura",
        "trachea and bronchie",
    )
    IGNORE_ORGANS = {"bone", "abdomen"}

    def __init__(
        self,
        vision_cfg,
        hidden_dim: int,
        embed_dim: int = 128,
        freeze_encoder: bool = True,
        organs: tuple[str, ...] | list[str] | None = None,
    ):
        super().__init__()
        from ..vlm_factory import build_vision_backbone_module

        self.encoder = build_vision_backbone_module(vision_cfg, hidden_dim=hidden_dim)
        organ_keys = tuple(organs) if organs is not None else self.DEFAULT_ORGANS
        self.heads = nn.ModuleDict(
            {self._canonical(name, allow_unknown=True): self._build_head(hidden_dim, embed_dim) for name in organ_keys}
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _build_head(in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, crops: dict) -> dict:
        outputs = {}
        for organ, img in crops.items():
            canon = self._canonical(organ)
            if canon not in self.heads:
                if canon in self.IGNORE_ORGANS:
                    continue
                raise KeyError(f"Organ '{organ}' (canonical '{canon}') not found in OrganContrastiveModel heads.")
            feats = self.encoder(img)
            if feats.dim() == 3:
                feats = feats.mean(dim=1) + feats.max(dim=1)[0]
            emb = self.heads[canon](feats)
            outputs[organ] = F.normalize(emb, dim=1)
        return outputs

    @classmethod
    def _canonical(cls, name: str, allow_unknown: bool = False) -> str:
        canon = cls.ALIASES.get(name, name)
        if not allow_unknown and canon not in cls.DEFAULT_ORGANS:
            raise KeyError(f"Organ '{name}' not recognized.")
        return canon


class OrganMLPOnlyModel(nn.Module):
    """Head-only model with Input Normalization."""

    DEFAULT_ORGANS = (
        "lung",
        "heart",
        "mediastinum",
        "pleura",
        "trachea and bronchie",
    )
    ALIASES = {"airways": "trachea and bronchie"}
    IGNORE_ORGANS = {"bone", "abdomen"}

    def __init__(self, input_dim: int, embed_dim: int = 128, organs: tuple[str, ...] | list[str] | None = None):
        super().__init__()

        def _canonical(name: str) -> str:
            return self.ALIASES.get(name, name)

        base_organs = tuple(organs) if organs is not None else self.DEFAULT_ORGANS
        organ_keys: list[str] = []
        for name in base_organs:
            canon = _canonical(name)
            if canon not in organ_keys:
                organ_keys.append(canon)

        self.input_norms = nn.ModuleDict({org: nn.BatchNorm1d(input_dim) for org in organ_keys})
        self.heads = nn.ModuleDict({org: self._build_head(input_dim, embed_dim) for org in organ_keys})
        self._canonical = _canonical

    @staticmethod
    def _build_head(in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, embeddings: dict) -> dict:
        outputs = {}
        for organ, feat in embeddings.items():
            canon = self._canonical(organ)
            if canon not in self.heads:
                if canon in self.IGNORE_ORGANS:
                    continue
                raise KeyError(
                    f"Organ '{organ}' (canonical '{canon}') not found in OrganMLPOnlyModel heads. "
                    f"Available: {list(self.heads.keys())}"
                )
            target_dtype = self.heads[canon][0].weight.dtype
            if feat.dtype != target_dtype:
                feat = feat.to(dtype=target_dtype)
            emb = self.heads[canon](feat)
            outputs[organ] = emb
        return outputs
