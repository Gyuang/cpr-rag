"""LLaMA wrapper that prepends visual tokens for vision-language models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


@dataclass
class VisualLlamaConfig:
    """Configuration for visual LLaMA model."""
    llama_path: str
    tokenizer_path: Optional[str] = None
    vision_checkpoint: Optional[str] = None
    freeze_llama: bool = False
    num_visual_tokens: int = 32
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_precomputed_vision: bool = False


class VisualLlamaModel(nn.Module):
    """Visual-prompted LLaMA model that prepends vision tokens to text."""

    def __init__(self, config: VisualLlamaConfig, vision_builder: Optional[Callable[[int, int], nn.Module]]) -> None:
        super().__init__()
        self.config = config
        tokenizer_path = config.tokenizer_path or config.llama_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.language_model = AutoModelForCausalLM.from_pretrained(config.llama_path, torch_dtype=torch_dtype)
        self.hidden_dim = self.language_model.config.hidden_size
        self.num_visual_tokens = config.num_visual_tokens
        self.visual_mask_token = nn.Parameter(torch.zeros(1, self.num_visual_tokens, self.hidden_dim))

        if config.freeze_llama:
            self.language_model.requires_grad_(False)

        if config.lora_r and config.lora_r > 0:
            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            self.language_model = get_peft_model(self.language_model, lora_cfg)

        self.use_precomputed_vision = config.use_precomputed_vision
        self.vision_backbone: Optional[nn.Module] = None
        # Always create vision_backbone if builder is provided (needed for aggregator/projector even with precomputed)
        if vision_builder is not None:
            self.vision_backbone = vision_builder(self.hidden_dim, config.num_visual_tokens)
            self.vision_backbone = self.vision_backbone.to(dtype=torch_dtype)
        elif not self.use_precomputed_vision:
            raise ValueError("vision_builder must be provided when precomputed vision is disabled.")

        with torch.no_grad():
            # Initialize to a reasonable scale so "blindfold" replacement is numerically stable.
            self.visual_mask_token.normal_(mean=0.0, std=0.02)

    def _prepare_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        vision_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed_tokens = self.language_model.get_input_embeddings()
        text_embeds = embed_tokens(input_ids)
        if vision_tokens.dtype != text_embeds.dtype:
            vision_tokens = vision_tokens.to(text_embeds.dtype)
        inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
        batch_size = input_ids.size(0)
        # Use actual visual token count from tensor, not config (supports variable token counts like raw 8192 tokens)
        num_visual = vision_tokens.size(1)
        vision_mask = torch.ones(batch_size, num_visual, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([vision_mask, attention_mask], dim=1)
        ignore = labels.new_full((batch_size, num_visual), -100)
        full_labels = torch.cat([ignore, labels], dim=1)
        return inputs_embeds, full_mask, full_labels

    def _encode_visual_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.vision_backbone is None:
            raise ValueError("Vision backbone is not initialized.")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"visual_features must be a torch.Tensor (got {type(features)})")
        if features.ndim != 3:
            raise ValueError(f"visual_features must be [B, N, D] (got {tuple(features.shape)})")
        if not torch.isfinite(features).all():
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            ref_param = next(self.vision_backbone.parameters())
            ref_device = ref_param.device
            ref_dtype = ref_param.dtype
        except StopIteration:
            ref_device = features.device
            ref_dtype = features.dtype
        if features.device != ref_device:
            features = features.to(ref_device)
        if features.dtype != ref_dtype:
            features = features.to(ref_dtype)
        if features.size(1) == self.num_visual_tokens:
            tokens = features
        else:
            tokens = self.vision_backbone.aggregator(features)
        projected = self.vision_backbone.projector(tokens)
        # Apply norm if available (important for Reg2RG scale matching)
        if hasattr(self.vision_backbone, 'norm') and self.vision_backbone.norm is not None:
            projected = self.vision_backbone.norm(projected)
        return projected

    def forward(self, batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        if self.use_precomputed_vision:
            # Precomputed raw embeddings still need aggregation + projection
            visual_embeds = batch["visual_embeds"]
            vision_tokens = self._encode_visual_features(visual_embeds)
        else:
            if self.vision_backbone is None:
                raise ValueError("Vision backbone is not initialized.")
            visual_features = batch.get("visual_features")
            if visual_features is not None:
                vision_tokens = self._encode_visual_features(visual_features)
            else:
                volume = batch.get("volume")
                if volume is None:
                    raise ValueError("Batch is missing 'volume' tensor required for vision encoding.")
                # Pass mask to vision backbone if available (for Reg2RGFullBackbone)
                mask = batch.get("mask") or batch.get("organ_mask")
                if mask is not None and hasattr(self.vision_backbone, 'forward'):
                    import inspect
                    sig = inspect.signature(self.vision_backbone.forward)
                    if 'mask' in sig.parameters:
                        vision_tokens = self.vision_backbone(volume, mask=mask)
                    else:
                        vision_tokens = self.vision_backbone(volume)
                else:
                    vision_tokens = self.vision_backbone(volume)
        inputs_embeds, full_mask, full_labels = self._prepare_embeddings(input_ids, attention_mask, labels, vision_tokens)
        return self.language_model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt_ids: torch.Tensor,
        volume: Optional[torch.Tensor] = None,
        visual_embeds: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **generation_kwargs,
    ) -> torch.Tensor:
        self.eval()
        if self.use_precomputed_vision:
            if visual_embeds is None:
                raise ValueError("visual_embeds must be provided for generation when using precomputed vision tokens.")
            # Precomputed raw embeddings still need aggregation + projection
            vision_tokens = self._encode_visual_features(visual_embeds)
        else:
            if self.vision_backbone is None:
                raise ValueError("Vision backbone is not initialized.")
            if visual_features is not None:
                vision_tokens = self._encode_visual_features(visual_features)
            else:
                if volume is None:
                    raise ValueError("Volume tensor is required for generation when using the vision backbone.")
                # Pass mask to vision backbone if available (for Reg2RGFullBackbone)
                if mask is not None and hasattr(self.vision_backbone, 'forward'):
                    import inspect
                    sig = inspect.signature(self.vision_backbone.forward)
                    if 'mask' in sig.parameters:
                        vision_tokens = self.vision_backbone(volume, mask=mask)
                    else:
                        vision_tokens = self.vision_backbone(volume)
                else:
                    vision_tokens = self.vision_backbone(volume)
        embed_tokens = self.language_model.get_input_embeddings()
        prompt_embeds = embed_tokens(prompt_ids)
        if vision_tokens.dtype != prompt_embeds.dtype:
            vision_tokens = vision_tokens.to(prompt_embeds.dtype)
        inputs_embeds = torch.cat([vision_tokens, prompt_embeds], dim=1)
        batch_size = prompt_ids.size(0)
        # Use actual visual token count from tensor
        num_visual = vision_tokens.size(1)
        attention_mask = torch.ones(batch_size, inputs_embeds.size(1), dtype=torch.long, device=prompt_ids.device)
        prompt_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
        attention_mask[:, num_visual:] = prompt_mask

        prefix_ids = torch.full(
            (batch_size, num_visual),
            self.tokenizer.pad_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        input_ids = torch.cat([prefix_ids, prompt_ids], dim=1)
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generation_kwargs,
        )
        # Remove input tokens (visual + prompt), return only generated tokens
        input_len = inputs_embeds.size(1)
        return outputs[:, input_len:]
