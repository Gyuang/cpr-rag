"""Factory helpers for building vision encoders."""
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
from einops import rearrange
from transformers import PreTrainedModel

from ..vit3d_encoder import ViT3DEncoder, CTViTBackbone
from .components import PerceiverResampler, ViT3D


DEFAULT_CHECKPOINT_PREFIXES = (
    "vision_backbone.",
    "module.vision_backbone.",
    "vision_adapter.",
    "module.vision_adapter.",
    "vision_encoder.",
    "module.vision_encoder.",
    "backbone.",
    "module.backbone.",
)


def _unwrap_state_dict_container(state: object) -> object:
    """Extract the actual model state dict from common checkpoint wrappers."""
    model_state = state
    if isinstance(state, dict):
        for key in ("model", "state_dict"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                model_state = candidate
                break
    return model_state


def _strip_prefixes(model_state: Dict[str, torch.Tensor], extra_prefixes: Sequence[str] = ()) -> Dict[str, torch.Tensor]:
    """Strip common prefixes so checkpoints saved under wrappers can still load."""
    prefixes = tuple(extra_prefixes) + DEFAULT_CHECKPOINT_PREFIXES
    filtered: Optional[Dict[str, torch.Tensor]] = None
    for prefix in prefixes:
        subset = {
            key.replace(prefix, "", 1): value
            for key, value in model_state.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if subset:
            filtered = subset
            break
    if filtered is None and any(isinstance(key, str) and key.startswith("module.") for key in model_state.keys()):
        filtered = {
            key.replace("module.", "", 1): value
            for key, value in model_state.items()
            if isinstance(key, str)
        }
    return filtered or model_state


def _remap_state_keys(model_state: Dict[str, torch.Tensor], replacements: Sequence[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
    """Rename checkpoint keys to match the refactored module layout."""
    if not replacements:
        return dict(model_state)
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        new_key = key
        for source, target in replacements:
            if isinstance(key, str) and key.startswith(source):
                new_key = key.replace(source, target, 1)
                break
        remapped[new_key] = value
    return remapped


def _apply_state_dict(
    module: nn.Module,
    model_state: Dict[str, torch.Tensor],
    *,
    tag: str,
    strict: bool = True,
) -> Dict[str, Sequence[str]]:
    """Load state dict with strict shape checking."""
    missing, unexpected = module.load_state_dict(model_state, strict=strict)
    if missing and strict:
        print(f"[{tag}] missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected and strict:
        print(f"[{tag}] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
        
    return {"missing": missing, "unexpected": unexpected}


def _prepare_state_dict(
    model_state: Dict[str, torch.Tensor],
    *,
    extra_prefixes: Sequence[str] = (),
    replacements: Sequence[Tuple[str, str]] = (),
) -> Dict[str, torch.Tensor]:
    stripped = _strip_prefixes(model_state, extra_prefixes=extra_prefixes)
    return _remap_state_keys(stripped, replacements)


def _load_checkpoint_into(
    module: nn.Module,
    checkpoint: str | Path,
    *,
    tag: str,
    extra_prefixes: Sequence[str] = (),
    replacements: Sequence[Tuple[str, str]] = (),
    strict: bool = True,
) -> Dict[str, Sequence[str]]:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model_state = _unwrap_state_dict_container(state)
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint at {checkpoint} does not contain a state_dict-like object.")
    prepared = _prepare_state_dict(model_state, extra_prefixes=extra_prefixes, replacements=replacements)
    return _apply_state_dict(module, prepared, tag=tag, strict=strict)


def _extract_sequence(features: object) -> Tuple[torch.Tensor, Optional[Tuple[int, ...]]]:
    """Normalize encoder outputs into a tensor plus optional token shape metadata."""
    token_shape: Optional[Tuple[int, ...]] = None
    seq = features
    if isinstance(features, dict):
        token_shape = features.get("token_shape")
        for key in ("sequence", "tokens", "features"):
            candidate = features.get(key)
            if isinstance(candidate, torch.Tensor):
                seq = candidate
                break
    elif isinstance(features, tuple) and len(features) == 2 and torch.is_tensor(features[0]):
        seq, token_shape = features
    if not isinstance(seq, torch.Tensor):
        raise TypeError(f"Unsupported feature payload type: {type(features)}")
    return seq, token_shape


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
        self.perceiver = PerceiverResampler(
            dim=input_dim,
            depth=depth,
            num_latents=num_tokens,
            max_num_media=max_num_media,
            max_num_frames=max_num_frames,
        )

    def forward(self, features: object) -> torch.Tensor:
        seq, token_shape = _extract_sequence(features)
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
        seq, _ = _extract_sequence(features)
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


_CT2REP_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("encoder.vit.vit.", "encoder.vit."),  # handle double-prefixed vit.* keys from some checkpoints
    ("encoder.vit.", "encoder.vit."),      # guard against re-prefixing already-correct keys
    ("encoder.", "encoder.vit."),
    ("perceiver.", "aggregator.perceiver."),
    ("project.", "projector.linear."),
    ("norm.", "projector.norm."),
)


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
        # Some checkpoints (e.g., LoRA-only saves) may omit non-trainable buffers like beta.
        # Use non-strict loading so missing buffers fall back to model defaults.
        _load_checkpoint_into(self, checkpoint, tag="ct2rep-backbone", replacements=_CT2REP_REPLACEMENTS, strict=False)


_RADFM_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("vision_backbone.vit.", "encoder.vit."),
    ("vision_backbone.perceiver.", "aggregator.perceiver."),
    ("vision_backbone.project.", "projector.linear."),
    ("vision_backbone.norm.", "projector.norm."),
    ("embedding_layer.vision_encoder.", "encoder.vit."),
    ("embedding_layer.perceiver.", "aggregator.perceiver."),
    ("embedding_layer.fc.", "projector.linear."),
    ("embedding_layer.norm.", "projector.norm."),
    ("vit.", "encoder.vit."),
    ("perceiver.", "aggregator.perceiver."),
    ("project.", "projector.linear."),
    ("norm.", "projector.norm."),
)


class RadFMEncoder(nn.Module):
    """ViT3D encoder used by RadFM."""

    def __init__(self, image_patch_size: int = 32, frame_patch_size: int = 4) -> None:
        super().__init__()
        self.vit = ViT3D(
            image_size=512,
            image_patch_size=image_patch_size,
            frames=512,
            frame_patch_size=frame_patch_size,
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            channels=3,
            dropout=0.1,
            emb_dropout=0.1,
        )

    def _prepare_volume(self, volume: torch.Tensor) -> torch.Tensor:
        # Expect [B, 1, H, W, D]; repeat to 3 channels for ViT weights trained on RGB-like inputs.
        if volume.size(1) == 1:
            volume = volume.repeat(1, 3, 1, 1, 1)
        patch_h = self.vit.patch_height
        patch_w = self.vit.patch_width
        patch_d = self.vit.frame_patch
        h, w, d = volume.shape[2:5]
        target_h = ((h + patch_h - 1) // patch_h) * patch_h
        target_w = ((w + patch_w - 1) // patch_w) * patch_w
        target_d = ((d + patch_d - 1) // patch_d) * patch_d
        if (h, w, d) != (target_h, target_w, target_d):
            volume = F.interpolate(volume, size=(target_h, target_w, target_d), mode="trilinear", align_corners=False)
        return volume

    def forward(self, volume: torch.Tensor) -> Dict[str, object]:
        x = self._prepare_volume(volume)
        target_dtype = self.vit.to_patch_embedding[1].weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        encoded, _ = self.vit(x)
        _, _, h, w, d = x.shape
        h_tokens = h // self.vit.patch_height
        w_tokens = w // self.vit.patch_width
        d_tokens = d // self.vit.frame_patch
        return {"sequence": encoded, "token_shape": (h_tokens, w_tokens, d_tokens)}


class RadFMBackbone(MedicalVisionTower):
    """RadFM vision encoder + Perceiver resampler to produce LLaMA-compatible tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int,
        checkpoint: Optional[str] = None,
        image_patch_size: int = 32,
        frame_patch_size: int = 4,
    ) -> None:
        encoder = RadFMEncoder(image_patch_size=image_patch_size, frame_patch_size=frame_patch_size)
        aggregator = PerceiverAggregator(input_dim=768, num_tokens=num_tokens, depth=6, max_num_media=None, max_num_frames=None)
        projector = VisionProjector(in_dim=768, out_dim=hidden_dim)
        super().__init__(encoder, aggregator, projector)
        self.last_checkpoint_key_report: Dict[str, Dict[str, Sequence[str]]] = {}
        if checkpoint and num_tokens != 32:
            raise ValueError("RadFM checkpoints expect num_visual_tokens=32.")
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def _disable_projector_norm(self) -> None:
        if not isinstance(self.projector.norm, nn.Identity):
            self.projector.norm = nn.Identity()

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)

        # Handle vision_best.pt -> infer projector_best.pt
        if path.is_file() and path.name.startswith("vision_best"):
            proj_path = path.parent / path.name.replace("vision", "projector")
            if proj_path.exists():
                print(f"[RadFMBackbone] Loading split checkpoints: {path.name} and {proj_path.name}")
                vit_state = torch.load(str(path), map_location="cpu", weights_only=False)
                head_state = torch.load(str(proj_path), map_location="cpu", weights_only=False)
                self._load_split_radfm(vit_state, head_state)
                return

        # Handle projector_best.pt -> infer vision_best.pt
        if path.is_file() and path.name.startswith("projector_best"):
            vision_path = path.parent / path.name.replace("projector", "vision")
            if vision_path.exists():
                print(f"[RadFMBackbone] Loading split checkpoints: {vision_path.name} and {path.name}")
                vit_state = torch.load(str(vision_path), map_location="cpu", weights_only=False)
                head_state = torch.load(str(path), map_location="cpu", weights_only=False)
                self._load_split_radfm(vit_state, head_state)
                return

        if path.is_dir():
            vit_path = path / "RadFM_vit3d.pth"
            head_path = path / "RadFM_perceiver_fc.pth"
            if not vit_path.exists() or not head_path.exists():
                raise FileNotFoundError(f"Expected RadFM split checkpoints at {path} (RadFM_vit3d.pth + RadFM_perceiver_fc.pth).")
            vit_state = torch.load(vit_path, map_location="cpu", weights_only=False)
            head_state = torch.load(head_path, map_location="cpu", weights_only=False)
            self._load_split_radfm(vit_state, head_state)
            return

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        model_state = _unwrap_state_dict_container(state)
        if not isinstance(model_state, dict):
            raise ValueError(f"Unsupported RadFM checkpoint format at {path}. Provide either a combined .bin/.pt or split .pth files.")
        representative_keys = {"to_patch_embedding.1.weight", "transformer.layers.0.0.norm.weight"}
        if representative_keys <= set(model_state.keys()):
            head_path = path.with_name("RadFM_perceiver_fc.pth")
            if not head_path.exists():
                raise FileNotFoundError(
                    f"Found ViT weights at {path} but missing companion 'RadFM_perceiver_fc.pth' in the same directory."
                )
            head_state = torch.load(head_path, map_location="cpu", weights_only=False)
            self._load_split_radfm(model_state, head_state)
            return

        if any(str(key).startswith(("vision_backbone.", "embedding_layer.", "vit.", "perceiver.", "project.", "norm.")) for key in model_state):
            prepared = _prepare_state_dict(model_state, replacements=_RADFM_REPLACEMENTS)
            allowed_prefixes = ("encoder.", "aggregator.", "projector.")
            filtered = {k: v for k, v in prepared.items() if any(str(k).startswith(p) for p in allowed_prefixes)}
            has_norm_weights = {"projector.norm.weight", "projector.norm.bias"} <= set(filtered.keys())
            if not has_norm_weights:
                filtered = {k: v for k, v in filtered.items() if not str(k).startswith("projector.norm")}
                self._disable_projector_norm()
            report = _apply_state_dict(self, filtered, tag="radfm-backbone")
            self.last_checkpoint_key_report = {"backbone": report}
            return

        raise ValueError(f"Unsupported RadFM checkpoint format at {path}. Provide either a combined .bin/.pt or split .pth files.")

    def _load_split_radfm(self, vit_state: dict, head_state: dict) -> None:
        # Handle full RadFMBackbone checkpoint format (encoder.vit.*, aggregator.*)
        if any(str(k).startswith("encoder.vit.") for k in vit_state):
            # This is a full backbone checkpoint - extract and load directly
            vit_only = {str(k).replace("encoder.vit.", "", 1): v for k, v in vit_state.items() if str(k).startswith("encoder.vit.")}
            vit_report = _apply_state_dict(self.encoder.vit, vit_only, tag="radfm-vit")
            # Also load aggregator if present
            agg_state = {str(k).replace("aggregator.", "", 1): v for k, v in vit_state.items() if str(k).startswith("aggregator.")}
            if agg_state:
                perc_report = _apply_state_dict(self.aggregator, agg_state, tag="radfm-perceiver")
            else:
                perc_report = {}
            self.last_checkpoint_key_report = {"vit": vit_report, "perceiver": perc_report, "projector": {}}
            return
        # Accept both prefixed ("vit.to_patch_embedding...") and unprefixed keys.
        if any(str(k).startswith("vit.") for k in vit_state):
            vit_state = {str(k).replace("vit.", "", 1): v for k, v in vit_state.items()}
        vit_report = _apply_state_dict(self.encoder.vit, dict(vit_state), tag="radfm-vit")
        perceiver_state = head_state.get("perceiver", {}) if isinstance(head_state, dict) else {}
        if perceiver_state and not any(str(k).startswith("perceiver.") for k in perceiver_state):
            perceiver_state = {f"perceiver.{k}": v for k, v in perceiver_state.items()}
        perc_report = _apply_state_dict(self.aggregator, perceiver_state, tag="radfm-perceiver")

        projector_state: Dict[str, torch.Tensor] = {}
        fc_state = head_state.get("fc", {}) if isinstance(head_state, dict) else {}
        if isinstance(fc_state, dict):
            if "weight" in fc_state:
                projector_state["linear.weight"] = fc_state["weight"]
            if "bias" in fc_state:
                projector_state["linear.bias"] = fc_state["bias"]
        norm_state = head_state.get("norm", {}) if isinstance(head_state, dict) else {}
        if isinstance(norm_state, dict):
            for name in ("weight", "bias"):
                if name in norm_state:
                    projector_state[f"norm.{name}"] = norm_state[name]
        has_norm_weights = {"norm.weight", "norm.bias"} <= set(projector_state.keys())
        if not has_norm_weights:
            projector_state = {k: v for k, v in projector_state.items() if not str(k).startswith("norm.")}
            self._disable_projector_norm()
        proj_report = _apply_state_dict(self.projector, projector_state, tag="radfm-projector")
        self.last_checkpoint_key_report = {
            "vit": vit_report,
            "perceiver": perc_report,
            "projector": proj_report,
        }
class M3DEncoder(nn.Module):
    """Wrapper around the published M3D-CLIP vision encoder that returns raw features."""

    def __init__(
        self,
        checkpoint: str,
    ) -> None:
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
        # Drop references to the text encoder to avoid keeping unnecessary weights around.
        if hasattr(model, "language_encoder"):
            del model.language_encoder
        del model

    @staticmethod
    def _infer_m3d_base_dir(checkpoint_file: Path) -> Optional[Path]:
        """Infer the original M3D-CLIP directory from a CTDoc checkpoint."""
        try:
            state = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise FileNotFoundError(
                f"Failed to read M3D checkpoint metadata from {checkpoint_file}"
            ) from exc
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
        """Hugging Face에서 기본 M3D-CLIP 모델을 로드하는 fallback."""
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "GoodBaiBai88/M3D-CLIP",
            trust_remote_code=True,
        )
        model.eval()
        return model

    def _load_local_m3d_clip_model(self, checkpoint_path: Path) -> PreTrainedModel:
        config_path = checkpoint_path / "config.json"
        model_file = checkpoint_path / "modeling_m3d_clip.py"
        cfg_file = checkpoint_path / "configuration_m3d_clip.py"
        weight_path = checkpoint_path / "pretrained_ViT.bin"

        if not config_path.exists() or not model_file.exists() or not cfg_file.exists():
            raise FileNotFoundError(
                f"Expected config/model files in {checkpoint_path}"
            )
        
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
                # Original weights are vision-only and lack the vision_encoder prefix expected by HF modules.
                new_k = f"vision_encoder.{k}" if not k.startswith("vision_encoder.") else k
                new_state_dict[new_k] = v

            model = model_class(config)
            try:
                from monai.networks.blocks import PatchEmbeddingBlock  # type: ignore
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
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
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
            volume = F.interpolate(
                volume,
                size=(target_hw, target_w, target_depth),
                mode="trilinear",
                align_corners=False,
            )
        vision_ready = volume.permute(0, 1, 4, 2, 3)
        dtype = next(self.vision_encoder.parameters()).dtype
        if vision_ready.dtype != dtype:
            vision_ready = vision_ready.to(dtype)
        features, _ = self.vision_encoder(vision_ready)
        if hasattr(self.vision_encoder, "cls_token"):
            features = features[:, 1:]
        return features


_M3D_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("vision_encoder.", "encoder.vision_encoder."),
    ("pool.", "aggregator.pool."),
    ("project.", "projector.linear."),
    ("norm.", "projector.norm."),
)


# Reg2RG checkpoints in this repo typically already match the `Reg2RGBackbone` module
# layout (or are wrapped under common prefixes handled by DEFAULT_CHECKPOINT_PREFIXES),
# so no additional key remapping is required.
_REG2RG_REPLACEMENTS: Tuple[Tuple[str, str], ...] = ()


class M3DBackbone(MedicalVisionTower):
    """Wrapper around the published M3D-CLIP vision encoder with pooled tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int,
        checkpoint: Optional[str] = None,
    ) -> None:
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

        # Handle vision_best.pt -> infer projector_best.pt
        if path.is_file() and path.name.startswith("vision_best"):
            proj_path = path.parent / path.name.replace("vision", "projector")
            if proj_path.exists():
                print(f"[M3DBackbone] Loading split checkpoints: {path.name} and {proj_path.name}")
                vision_state = torch.load(str(path), map_location="cpu", weights_only=False)
                proj_state = torch.load(str(proj_path), map_location="cpu", weights_only=False)
                self._load_split_m3d(vision_state, proj_state)
                return

        # Handle projector_best.pt -> infer vision_best.pt
        if path.is_file() and path.name.startswith("projector_best"):
            vision_path = path.parent / path.name.replace("projector", "vision")
            if vision_path.exists():
                print(f"[M3DBackbone] Loading split checkpoints: {vision_path.name} and {path.name}")
                vision_state = torch.load(str(vision_path), map_location="cpu", weights_only=False)
                proj_state = torch.load(str(path), map_location="cpu", weights_only=False)
                self._load_split_m3d(vision_state, proj_state)
                return

        _load_checkpoint_into(self, checkpoint, tag="m3d-backbone", replacements=_M3D_REPLACEMENTS)

    def _load_split_m3d(self, vision_state: dict, projector_state: dict) -> None:
        """Load split M3D checkpoint (vision + projector)."""
        vision_state_unwrapped = _unwrap_state_dict_container(vision_state)
        vision_prepared = _prepare_state_dict(
            vision_state_unwrapped,
            replacements=tuple(r for r in _M3D_REPLACEMENTS if r[1].startswith(("encoder.", "aggregator.")))
        )

        allowed_prefixes = ("encoder.", "aggregator.")
        filtered_vision = {k: v for k, v in vision_prepared.items() if any(str(k).startswith(p) for p in allowed_prefixes)}
        vision_report = _apply_state_dict(self, filtered_vision, tag="m3d-vision", strict=False)

        projector_state_unwrapped = _unwrap_state_dict_container(projector_state)
        projector_prepared = _prepare_state_dict(
            projector_state_unwrapped,
            replacements=tuple(r for r in _M3D_REPLACEMENTS if r[1].startswith("projector."))
        )
        projector_final = {
            k.replace("projector.", "", 1) if k.startswith("projector.") else k: v
            for k, v in projector_prepared.items()
        }
        proj_report = _apply_state_dict(self.projector, projector_final, tag="m3d-projector", strict=False)
        
        print(f"[M3D Split Load] Vision Missing: {vision_report['missing']}, Unexpected: {vision_report['unexpected']}")
        print(f"[M3D Split Load] Projector Missing: {proj_report['missing']}, Unexpected: {proj_report['unexpected']}")


class Reg2RGEncoder(nn.Module):
    """ViT3D-style encoder with a larger receptive field for Reg2RG inputs."""

    def __init__(self, hidden_dim: int, target_shape: Sequence[int] = (64, 64, 64)) -> None:
        super().__init__()
        embed_dim = max(384, hidden_dim // 2)
        self.output_dim = embed_dim
        self.target_shape = tuple(target_shape)
        self.backbone = ViT3DEncoder(
            in_channels=1,
            embed_dim=embed_dim,
            patch_size=4,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        target_dtype = self.backbone.proj.weight.dtype
        if volume.dtype != target_dtype:
            volume = volume.to(target_dtype)
        if volume.shape[2:5] != tuple(self.target_shape):
            volume = F.interpolate(volume, size=self.target_shape, mode="trilinear", align_corners=False)
        feats = self.backbone(volume)["sequence"]
        return feats


class Reg2RGBackbone(MedicalVisionTower):
    """Reg2RG encoder + pooling aggregator + projector."""

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int,
        checkpoint: Optional[str] = None,
        target_shape: Sequence[int] = (64, 64, 64),
    ) -> None:
        encoder = Reg2RGEncoder(hidden_dim=hidden_dim, target_shape=target_shape)
        aggregator = PoolingAggregator(num_tokens=num_tokens)
        projector = VisionProjector(in_dim=encoder.output_dim, out_dim=hidden_dim)
        super().__init__(encoder, aggregator, projector)
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        _load_checkpoint_into(
            self,
            checkpoint,
            tag="reg2rg-backbone",
            extra_prefixes=("reg2rg_backbone.", "module.reg2rg_backbone."),
            replacements=_REG2RG_REPLACEMENTS,
        )


class Reg2RGFullBackbone(nn.Module):
    """Full Reg2RG backbone with vision encoder + mask encoder.

    This follows the original Reg2RG architecture:
    - Vision encoder: ViT3D (256x256x64, patch_size=32, depth=12)
    - Mask encoder: ViT3D (256x256x64, patch_size=32, depth=3, single channel)
    - Perceiver: 32 tokens
    - Projector: Linear to LLM hidden dim

    The forward method takes both CT volume and region mask as inputs.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int = 32,
        checkpoint: Optional[str] = None,
        pretrained_visual_encoder: Optional[str] = None,
        pretrained_adapter: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.vis_dim = 768  # Same as RadFM

        # Vision encoder (same as RadFM ViT3D - must match pretrained checkpoint)
        self.vision_encoder = ViT3D(
            image_size=512,  # RadFM uses 512
            image_patch_size=32,
            frames=512,  # RadFM uses 512
            frame_patch_size=4,
            dim=self.vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            channels=3,  # Will repeat single channel to 3
            dropout=0.1,
            emb_dropout=0.1,
        )

        # Mask encoder (smaller ViT3D for region masks)
        self.mask_encoder = ViT3D(
            image_size=256,
            image_patch_size=32,
            frames=64,
            frame_patch_size=16,  # Larger temporal patch for masks
            dim=255,  # Original Reg2RG uses 255
            depth=3,
            heads=8,
            mlp_dim=512,
            channels=1,  # Single channel masks
            dropout=0.1,
            emb_dropout=0.1,
        )

        # Perceiver for token compression
        self.perceiver = PerceiverResampler(
            dim=self.vis_dim,
            depth=6,
            num_latents=num_tokens,
        )

        # Projectors
        self.fc = nn.Linear(self.vis_dim, hidden_dim)  # Vision projector
        self.mask_fc = nn.Linear(255, hidden_dim)  # Mask projector
        self.norm = nn.LayerNorm(hidden_dim)

        # Aggregator wrapper for flat precomputed embeddings
        # ViT3D outputs 32768 tokens = 128 frames × 256 spatial (16×16)
        self._perceiver_frames = 128
        self._perceiver_spatial = 256
        self.aggregator = self._aggregate_precomputed
        self.projector = self.fc

        # Load pretrained weights
        if pretrained_visual_encoder:
            self._load_pretrained_vision(pretrained_visual_encoder)
        if pretrained_adapter:
            self._load_pretrained_adapter(pretrained_adapter)
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def _aggregate_precomputed(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate precomputed embeddings via perceiver.

        Input: [B, 1024, 768] (flat tokens)
        Output: [B, num_tokens, 768] (compressed tokens)
        """
        # Check if already in correct shape for perceiver
        if x.dim() == 5:
            # Already [b, m, f, v, d] format
            out = self.perceiver(x)
            return out[:, 0] if out.dim() == 4 else out

        # Reshape flat tokens to perceiver expected format [b, m, f, v, d]
        b = x.size(0)
        # [B, 1024, 768] -> [B, 1, 16, 64, 768]
        x = x.view(b, 1, self._perceiver_frames, self._perceiver_spatial, -1)
        out = self.perceiver(x)  # [B, 1, num_tokens, 768]
        return out[:, 0]  # [B, num_tokens, 768]

    def _load_pretrained_vision(self, path: str) -> None:
        """Load pretrained RadFM vision encoder weights."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = self.vision_encoder.load_state_dict(state, strict=False)
        print(f"[Reg2RGFullBackbone] Loaded vision encoder: Missing={len(missing)}, Unexpected={len(unexpected)}")

    def _load_pretrained_adapter(self, path: str) -> None:
        """Load pretrained perceiver adapter weights."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "perceiver" in state:
            perceiver_state = state["perceiver"]
            # Add perceiver. prefix if needed
            if not any(k.startswith("perceiver.") for k in perceiver_state):
                perceiver_state = {f"perceiver.{k}": v for k, v in perceiver_state.items()}
            missing, unexpected = self.perceiver.load_state_dict(perceiver_state, strict=False)
            print(f"[Reg2RGFullBackbone] Loaded perceiver: Missing={len(missing)}, Unexpected={len(unexpected)}")

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        """Load full Reg2RG checkpoint."""
        path = Path(checkpoint)

        # Handle split checkpoint loading
        if path.is_file() and path.name.startswith("vision_best"):
            proj_path = path.parent / path.name.replace("vision", "projector")
            if proj_path.exists():
                print(f"[Reg2RGFullBackbone] Loading split checkpoints: {path.name} and {proj_path.name}")
                vision_state = torch.load(str(path), map_location="cpu", weights_only=False)
                proj_state = torch.load(str(proj_path), map_location="cpu", weights_only=False)
                self._load_split_checkpoint(vision_state, proj_state)
                return

        # Single checkpoint
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        model_state = _unwrap_state_dict_container(state)
        if isinstance(model_state, dict):
            # Handle native Reg2RG checkpoint format (embedding_layer.* prefix)
            if any(k.startswith("embedding_layer.") for k in model_state.keys()):
                model_state = self._remap_native_reg2rg_keys(model_state)
            missing, unexpected = self.load_state_dict(model_state, strict=False)
            print(f"[Reg2RGFullBackbone] Loaded checkpoint: Missing={len(missing)}, Unexpected={len(unexpected)}")

    def _remap_native_reg2rg_keys(self, state: dict, skip_encoders: bool = True) -> dict:
        """Remap native Reg2RG checkpoint keys to CTDoc format.

        Args:
            state: Original state dict
            skip_encoders: If True, skip vision_encoder and mask_encoder keys (for precomputed embeddings)
        """
        remapped = {}
        skipped_vision = 0
        skipped_mask = 0
        for k, v in state.items():
            # Skip non-vision keys (lang_model, etc.)
            if not k.startswith("embedding_layer."):
                continue
            # Strip embedding_layer. prefix
            new_key = k.replace("embedding_layer.", "")
            # Skip vision/mask encoder keys when using precomputed (avoid shape mismatch)
            if skip_encoders:
                if new_key.startswith("vision_encoder."):
                    skipped_vision += 1
                    continue
                if new_key.startswith("mask_encoder."):
                    skipped_mask += 1
                    continue
            # Map fc -> projector
            if new_key.startswith("fc."):
                new_key = new_key.replace("fc.", "projector.")
            remapped[new_key] = v
        print(f"[Reg2RGFullBackbone] Remapped {len(remapped)} keys from native format")
        if skip_encoders:
            print(f"[Reg2RGFullBackbone] Skipped {skipped_vision} vision_encoder + {skipped_mask} mask_encoder keys (using precomputed)")
        return remapped

    def _load_split_checkpoint(self, vision_state: dict, proj_state: dict) -> None:
        """Load from split vision/projector checkpoints."""
        # Vision state contains vision_encoder, mask_encoder, perceiver
        vision_state = _unwrap_state_dict_container(vision_state)
        proj_state = _unwrap_state_dict_container(proj_state)

        # Merge and load
        combined = {}
        combined.update(vision_state)
        combined.update(proj_state)
        missing, unexpected = self.load_state_dict(combined, strict=False)
        print(f"[Reg2RGFullBackbone] Loaded split: Missing={len(missing)}, Unexpected={len(unexpected)}")

    def _prepare_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Prepare CT volume for vision encoder.

        Note: RadFM ViT3D expects up to 512x512x512, but will handle dynamic sizes
        via positional embedding interpolation. We resize to a reasonable size
        that's divisible by patch sizes (32 for spatial, 4 for depth).
        """
        # Repeat single channel to 3 channels
        if volume.size(1) == 1:
            volume = volume.repeat(1, 3, 1, 1, 1)

        # Get current shape
        h, w, d = volume.shape[2:5]

        # Ensure dimensions are divisible by patch sizes
        patch_h = 32
        patch_w = 32
        patch_d = 4

        target_h = ((h + patch_h - 1) // patch_h) * patch_h
        target_w = ((w + patch_w - 1) // patch_w) * patch_w
        target_d = ((d + patch_d - 1) // patch_d) * patch_d

        # Cap at 512 to match positional embedding size
        target_h = min(target_h, 512)
        target_w = min(target_w, 512)
        target_d = min(target_d, 512)

        if (h, w, d) != (target_h, target_w, target_d):
            volume = F.interpolate(volume, size=(target_h, target_w, target_d), mode="trilinear", align_corners=False)

        return volume

    def _prepare_single_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Prepare a single region mask for mask encoder.

        Args:
            mask: Single mask [B, 1, H, W, D]

        Returns:
            Resized mask [B, 1, 256, 256, 64]
        """
        # Resize to 256x256x64 if needed
        target_shape = (256, 256, 64)
        if mask.shape[2:5] != target_shape:
            mask = F.interpolate(mask, size=target_shape, mode="trilinear", align_corners=False)
        return mask

    def forward(
        self,
        volume: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass (Original Reg2RG approach).

        Each mask channel is processed separately by mask_encoder, following the
        original Reg2RG architecture where each region gets its own mask token.

        Args:
            volume: CT volume [B, C, H, W, D]
            mask: Optional region masks [B, num_masks, H, W, D]. Each mask is encoded separately.

        Returns:
            Projected tokens:
            - If mask is None: [B, num_tokens, hidden_dim]
            - If mask provided: [B, num_tokens + num_masks, hidden_dim]
        """
        # Prepare and encode volume
        x = self._prepare_volume(volume)
        target_dtype = self.vision_encoder.to_patch_embedding[1].weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Vision encoding + perceiver
        vision_tokens, _ = self.vision_encoder(x)  # [B, num_patches, vis_dim]
        vision_tokens = vision_tokens[:, None, None, :, :]  # [B, 1, 1, num_patches, vis_dim]
        vision_tokens = self.perceiver(vision_tokens)[:, 0]  # [B, num_tokens, vis_dim]
        vision_tokens = self.fc(vision_tokens)  # [B, num_tokens, hidden_dim]
        vision_tokens = self.norm(vision_tokens)

        if mask is None:
            return vision_tokens

        # Encode each mask separately (original Reg2RG approach)
        B, num_masks = mask.shape[:2]
        mask_tokens_list = []

        for i in range(num_masks):
            single_mask = mask[:, i:i+1]  # [B, 1, H, W, D]
            m = self._prepare_single_mask(single_mask)
            if m.dtype != target_dtype:
                m = m.to(target_dtype)

            tokens, _ = self.mask_encoder(m)  # [B, num_patches, 255]
            token = torch.mean(tokens, dim=1)  # [B, 255]
            token = self.mask_fc(token)  # [B, hidden_dim]
            mask_tokens_list.append(token)

        # Stack all mask tokens: [B, num_masks, hidden_dim]
        all_mask_tokens = torch.stack(mask_tokens_list, dim=1)

        # Concatenate vision and all mask tokens
        output = torch.cat([vision_tokens, all_mask_tokens], dim=1)  # [B, num_tokens + num_masks, hidden_dim]
        return output


def build_vision_backbone(
    name: str,
    *,
    hidden_dim: int,
    num_tokens: int,
    ct2rep_ckpt: Optional[str] = None,
    radfm_ckpt: Optional[str] = None,
    m3d_ckpt: Optional[str] = None,
    reg2rg_ckpt: Optional[str] = None,
    reg2rg_pretrained_visual_encoder: Optional[str] = None,
    reg2rg_pretrained_adapter: Optional[str] = None,
) -> nn.Module:
    if name == "radfm":
        return RadFMBackbone(hidden_dim=hidden_dim, num_tokens=num_tokens, checkpoint=radfm_ckpt)
    if name == "ct2rep":
        return CT2RepBackbone(hidden_dim=hidden_dim, num_tokens=num_tokens, checkpoint=ct2rep_ckpt)
    if name == "m3d":
        return M3DBackbone(hidden_dim=hidden_dim, num_tokens=num_tokens, checkpoint=m3d_ckpt)
    if name == "reg2rg":
        return Reg2RGBackbone(hidden_dim=hidden_dim, num_tokens=num_tokens, checkpoint=reg2rg_ckpt)
    if name == "reg2rg_full":
        return Reg2RGFullBackbone(
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            checkpoint=reg2rg_ckpt,
            pretrained_visual_encoder=reg2rg_pretrained_visual_encoder,
            pretrained_adapter=reg2rg_pretrained_adapter,
        )
    raise ValueError(f"Unsupported vision backbone '{name}'.")
