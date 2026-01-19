"""M3D-CLIP vision encoder."""
from __future__ import annotations

import importlib
import json
import sys
import types
import uuid
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .base import MedicalVisionTower, PoolingAggregator, VisionProjector
from .utils import (
    unwrap_state_dict_container,
    prepare_state_dict,
    apply_state_dict,
    load_checkpoint_into,
)


_M3D_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("vision_encoder.", "encoder.vision_encoder."),
    ("pool.", "aggregator.pool."),
    ("project.", "projector.linear."),
    ("norm.", "projector.norm."),
)


class M3DEncoder(nn.Module):
    """Wrapper around the published M3D-CLIP vision encoder that returns raw features."""

    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint)
        self.finetuned_checkpoint: Optional[Path] = None
        base_dir: Optional[Path] = None

        if checkpoint_path.is_dir():
            base_dir = checkpoint_path
        elif checkpoint_path.is_file():
            self.finetuned_checkpoint = checkpoint_path
            base_dir = self._infer_m3d_base_dir(checkpoint_path)

        if base_dir is not None and base_dir.is_dir():
            model = self._load_local_m3d_clip_model(base_dir)
        else:
            if self.finetuned_checkpoint is None:
                raise FileNotFoundError(f"M3D-CLIP checkpoint directory not found: {checkpoint_path}")
            print(
                f"[M3DEncoder] M3D-CLIP base model directory not found for '{checkpoint_path}'. "
                "Falling back to HuggingFace model 'GoodBaiBai88/M3D-CLIP'."
            )
            model = self._load_hf_m3d_clip_model()

        self.vision_encoder = model.vision_encoder
        self.output_dim = getattr(self.vision_encoder, "hidden_size", model.config.hidden_size)
        img_size = getattr(model.config, "img_size", (32, 256, 256))
        self.target_shape = tuple(img_size)
        if hasattr(model, "language_encoder"):
            del model.language_encoder
        del model

    @staticmethod
    def _infer_m3d_base_dir(checkpoint_file: Path) -> Optional[Path]:
        try:
            state = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise FileNotFoundError(f"Failed to read M3D checkpoint metadata from {checkpoint_file}") from exc
        config = state.get("config")
        if isinstance(config, dict):
            for key in ("m3d_ckpt", "vision_ckpt"):
                base = config.get(key)
                if base:
                    base_path = Path(base)
                    if base_path.is_dir():
                        return base_path
        companion_json = checkpoint_file.with_suffix(".json")
        if companion_json.exists():
            try:
                meta = json.loads(companion_json.read_text())
                cfg = meta.get("config") if isinstance(meta, dict) else None
                if isinstance(cfg, dict):
                    base = cfg.get("m3d_ckpt") or cfg.get("vision_ckpt")
                    if base:
                        base_path = Path(base)
                        if base_path.is_dir():
                            return base_path
            except Exception:
                pass
        return None

    @staticmethod
    def _load_hf_m3d_clip_model() -> PreTrainedModel:
        from transformers import AutoModel

        model = AutoModel.from_pretrained("GoodBaiBai88/M3D-CLIP", trust_remote_code=True)
        model.eval()
        return model

    def _load_local_m3d_clip_model(self, checkpoint_path: Path) -> PreTrainedModel:
        config_path = checkpoint_path / "config.json"
        model_file = checkpoint_path / "modeling_m3d_clip.py"
        cfg_file = checkpoint_path / "configuration_m3d_clip.py"
        weight_path = checkpoint_path / "pretrained_ViT.bin"

        if not config_path.exists() or not model_file.exists() or not cfg_file.exists():
            raise FileNotFoundError(f"Expected config/model files in {checkpoint_path}")

        package_name = f"_m3d_clip_local_{uuid.uuid4().hex}"
        package = types.ModuleType(package_name)
        package.__path__ = [str(checkpoint_path)]
        sys.modules[package_name] = package

        try:
            cfg_module = importlib.import_module(f"{package_name}.configuration_m3d_clip")
            model_module = importlib.import_module(f"{package_name}.modeling_m3d_clip")

            config_dict = json.loads(config_path.read_text())
            if "add_cross_attention" in config_dict:
                config_dict["add_cross_attention"] = False

            config = cfg_module.M3DCLIPConfig(**config_dict)
            model_class = getattr(model_module, "M3DCLIP")

            state_dict = torch.load(weight_path, map_location="cpu")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = f"vision_encoder.{k}" if not k.startswith("vision_encoder.") else k
                new_state_dict[new_k] = v

            model = model_class(config)
            try:
                from monai.networks.blocks import PatchEmbeddingBlock

                if (
                    hasattr(model, "vision_encoder")
                    and hasattr(model.vision_encoder, "patch_embedding")
                    and getattr(config, "pos_embed", None) is not None
                ):
                    model.vision_encoder.patch_embedding = PatchEmbeddingBlock(
                        in_channels=config.in_channels,
                        img_size=config.img_size,
                        patch_size=config.patch_size,
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        dropout_rate=config.dropout_rate,
                        spatial_dims=config.spatial_dims,
                        pos_embed=config.pos_embed,
                        proj_type=config.pos_embed,
                        pos_embed_type="learnable",
                    )
            except Exception:
                pass
            missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
            print(f"[M3D Fix] Loaded vision-only checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            model.eval()
            return model
        finally:
            sys.modules.pop(package_name, None)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        target_hw = self.target_shape[1]
        target_w = self.target_shape[2]
        target_depth = self.target_shape[0]
        if volume.shape[2:5] != (target_hw, target_w, target_depth):
            volume = F.interpolate(volume, size=(target_hw, target_w, target_depth), mode="trilinear", align_corners=False)
        vision_ready = volume.permute(0, 1, 4, 2, 3)
        dtype = next(self.vision_encoder.parameters()).dtype
        if vision_ready.dtype != dtype:
            vision_ready = vision_ready.to(dtype)
        features, _ = self.vision_encoder(vision_ready)
        if hasattr(self.vision_encoder, "cls_token"):
            features = features[:, 1:]
        return features


class M3DBackbone(MedicalVisionTower):
    """Wrapper around the published M3D-CLIP vision encoder with pooled tokens."""

    def __init__(self, hidden_dim: int, num_tokens: int, checkpoint: Optional[str] = None) -> None:
        if checkpoint is None:
            raise ValueError("An --m3d-ckpt path pointing to a GoodBaiBai88/M3D-CLIP checkpoint directory is required.")
        encoder = M3DEncoder(checkpoint=checkpoint)
        aggregator = PoolingAggregator(num_tokens=num_tokens)
        projector = VisionProjector(in_dim=encoder.output_dim, out_dim=hidden_dim)
        super().__init__(encoder, aggregator, projector)
        if encoder.finetuned_checkpoint is not None:
            self.load_checkpoint(encoder.finetuned_checkpoint)

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        if path.is_file() and path.name.startswith("vision_best"):
            proj_path = path.parent / path.name.replace("vision", "projector")
            if proj_path.exists():
                print(f"[M3DBackbone] Loading split checkpoints: {path.name} and {proj_path.name}")
                vision_state = torch.load(str(path), map_location="cpu", weights_only=False)
                proj_state = torch.load(str(proj_path), map_location="cpu", weights_only=False)
                self._load_split_m3d(vision_state, proj_state)
                return
        if path.is_file() and path.name.startswith("projector_best"):
            vision_path = path.parent / path.name.replace("projector", "vision")
            if vision_path.exists():
                print(f"[M3DBackbone] Loading split checkpoints: {vision_path.name} and {path.name}")
                vision_state = torch.load(str(vision_path), map_location="cpu", weights_only=False)
                proj_state = torch.load(str(path), map_location="cpu", weights_only=False)
                self._load_split_m3d(vision_state, proj_state)
                return
        load_checkpoint_into(self, checkpoint, tag="m3d-backbone", replacements=_M3D_REPLACEMENTS)

    def _load_split_m3d(self, vision_state: dict, projector_state: dict) -> None:
        vision_state_unwrapped = unwrap_state_dict_container(vision_state)
        vision_prepared = prepare_state_dict(
            vision_state_unwrapped,
            replacements=tuple(r for r in _M3D_REPLACEMENTS if r[1].startswith(("encoder.", "aggregator."))),
        )
        allowed_prefixes = ("encoder.", "aggregator.")
        filtered_vision = {k: v for k, v in vision_prepared.items() if any(str(k).startswith(p) for p in allowed_prefixes)}
        vision_report = apply_state_dict(self, filtered_vision, tag="m3d-vision", strict=False)

        projector_state_unwrapped = unwrap_state_dict_container(projector_state)
        projector_prepared = prepare_state_dict(
            projector_state_unwrapped,
            replacements=tuple(r for r in _M3D_REPLACEMENTS if r[1].startswith("projector.")),
        )
        projector_final = {k.replace("projector.", "", 1) if k.startswith("projector.") else k: v for k, v in projector_prepared.items()}
        proj_report = apply_state_dict(self.projector, projector_final, tag="m3d-projector", strict=False)

        print(f"[M3D Split Load] Vision Missing: {vision_report['missing']}, Unexpected: {vision_report['unexpected']}")
        print(f"[M3D Split Load] Projector Missing: {proj_report['missing']}, Unexpected: {proj_report['unexpected']}")
