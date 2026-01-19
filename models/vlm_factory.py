"""Helpers to build modular vision encoders, language decoders, and their composed VLMs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn

from .core.visual_llama import VisualLlamaConfig, VisualLlamaModel
from .core.vision_backbones import build_vision_backbone

VisionBuilder = Callable[[int, int], nn.Module]


@dataclass
class VisionBackboneConfig:
    """Configuration for instantiating a vision encoder/perceiver stack."""

    name: str = "radfm"
    num_visual_tokens: int = 32
    ct2rep_ckpt: Optional[str] = None
    radfm_ckpt: Optional[str] = None
    m3d_ckpt: Optional[str] = None

    def checkpoint_hint(self) -> Optional[str]:
        if self.name == "ct2rep":
            return self.ct2rep_ckpt
        if self.name == "radfm":
            return self.radfm_ckpt
        if self.name == "m3d":
            return self.m3d_ckpt
        return self.ct2rep_ckpt or self.radfm_ckpt or self.m3d_ckpt

    def create_builder(self) -> VisionBuilder:
        def builder(hidden_dim: int, num_tokens: int) -> nn.Module:
            return build_vision_backbone(
                self.name,
                hidden_dim=hidden_dim,
                num_tokens=num_tokens,
                ct2rep_ckpt=self.ct2rep_ckpt,
                radfm_ckpt=self.radfm_ckpt,
                m3d_ckpt=self.m3d_ckpt,
            )
        return builder

    def build_module(self, hidden_dim: int) -> nn.Module:
        """Instantiate the actual vision module (encoder + pooling/perceiver) directly."""
        return self.create_builder()(hidden_dim, self.num_visual_tokens)


@dataclass
class DecoderConfig:
    """Configuration for language decoders (e.g., LLaMA + LoRA)."""

    llama_path: str
    tokenizer_path: Optional[str] = None
    freeze_llama: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def build(
        self,
        *,
        num_visual_tokens: int,
        vision_builder: Optional[VisionBuilder],
        vision_checkpoint_hint: Optional[str] = None,
        use_precomputed_vision: bool = False,
    ) -> VisualLlamaModel:
        config = VisualLlamaConfig(
            llama_path=self.llama_path,
            tokenizer_path=self.tokenizer_path,
            vision_checkpoint=vision_checkpoint_hint,
            freeze_llama=self.freeze_llama,
            num_visual_tokens=num_visual_tokens,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            use_precomputed_vision=use_precomputed_vision,
        )
        return VisualLlamaModel(config, vision_builder=vision_builder)


def build_vision_backbone_module(vision_cfg: VisionBackboneConfig, hidden_dim: int) -> nn.Module:
    """Utility to grab just the vision module (useful for RAG or standalone encoders)."""
    return vision_cfg.build_module(hidden_dim)


def build_decoder_model(
    decoder_cfg: DecoderConfig,
    vision_builder: Optional[VisionBuilder],
    *,
    num_visual_tokens: int,
    vision_checkpoint_hint: Optional[str] = None,
    use_precomputed_vision: bool = False,
) -> VisualLlamaModel:
    """Instantiate a language decoder and attach the provided vision builder."""
    return decoder_cfg.build(
        num_visual_tokens=num_visual_tokens,
        vision_builder=vision_builder,
        vision_checkpoint_hint=vision_checkpoint_hint,
        use_precomputed_vision=use_precomputed_vision,
    )


def build_vlm_model(
    vision_cfg: VisionBackboneConfig,
    decoder_cfg: DecoderConfig,
    *,
    use_precomputed_vision: bool = False,
) -> VisualLlamaModel:
    """Compose the requested vision encoder and decoder into a full VLM."""
    vision_builder = vision_cfg.create_builder()
    return build_decoder_model(
        decoder_cfg,
        vision_builder,
        num_visual_tokens=vision_cfg.num_visual_tokens,
        vision_checkpoint_hint=vision_cfg.checkpoint_hint(),
        use_precomputed_vision=use_precomputed_vision,
    )
