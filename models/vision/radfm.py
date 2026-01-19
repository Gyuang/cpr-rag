"""RadFM vision encoder."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .base import MedicalVisionTower, PerceiverAggregator, VisionProjector
from .utils import (
    unwrap_state_dict_container,
    prepare_state_dict,
    apply_state_dict,
)


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
        from ..CT2Rep.radfm_components import ViT3D

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
        aggregator = PerceiverAggregator(
            input_dim=768, num_tokens=num_tokens, depth=6, max_num_media=None, max_num_frames=None
        )
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

        # Split checkpoint loading (vision_best.pt + projector_best.pt)
        if path.is_file() and path.name.startswith("vision_best"):
            proj_path = path.parent / path.name.replace("vision", "projector")
            if proj_path.exists():
                print(f"[RadFMBackbone] Loading split checkpoints: {path.name} and {proj_path.name}")
                vit_state = torch.load(str(path), map_location="cpu", weights_only=False)
                head_state = torch.load(str(proj_path), map_location="cpu", weights_only=False)
                self._load_split_radfm(vit_state, head_state)
                return

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
                raise FileNotFoundError(
                    f"Expected RadFM split checkpoints at {path} (RadFM_vit3d.pth + RadFM_perceiver_fc.pth)."
                )
            vit_state = torch.load(vit_path, map_location="cpu", weights_only=False)
            head_state = torch.load(head_path, map_location="cpu", weights_only=False)
            self._load_split_radfm(vit_state, head_state)
            return

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        model_state = unwrap_state_dict_container(state)
        if not isinstance(model_state, dict):
            raise ValueError(f"Unsupported RadFM checkpoint format at {path}.")

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

        if any(
            str(key).startswith(("vision_backbone.", "embedding_layer.", "vit.", "perceiver.", "project.", "norm."))
            for key in model_state
        ):
            prepared = prepare_state_dict(model_state, replacements=_RADFM_REPLACEMENTS)
            allowed_prefixes = ("encoder.", "aggregator.", "projector.")
            filtered = {k: v for k, v in prepared.items() if any(str(k).startswith(p) for p in allowed_prefixes)}
            has_norm_weights = {"projector.norm.weight", "projector.norm.bias"} <= set(filtered.keys())
            if not has_norm_weights:
                filtered = {k: v for k, v in filtered.items() if not str(k).startswith("projector.norm")}
                self._disable_projector_norm()
            report = apply_state_dict(self, filtered, tag="radfm-backbone")
            self.last_checkpoint_key_report = {"backbone": report}
            return

        raise ValueError(f"Unsupported RadFM checkpoint format at {path}.")

    def _load_split_radfm(self, vit_state: dict, head_state: dict) -> None:
        if any(str(k).startswith("encoder.vit.") for k in vit_state):
            vit_only = {
                str(k).replace("encoder.vit.", "", 1): v for k, v in vit_state.items() if str(k).startswith("encoder.vit.")
            }
            vit_report = apply_state_dict(self.encoder.vit, vit_only, tag="radfm-vit")
            agg_state = {str(k).replace("aggregator.", "", 1): v for k, v in vit_state.items() if str(k).startswith("aggregator.")}
            if agg_state:
                perc_report = apply_state_dict(self.aggregator, agg_state, tag="radfm-perceiver")
            else:
                perc_report = {}
            self.last_checkpoint_key_report = {"vit": vit_report, "perceiver": perc_report, "projector": {}}
            return

        if any(str(k).startswith("vit.") for k in vit_state):
            vit_state = {str(k).replace("vit.", "", 1): v for k, v in vit_state.items()}
        vit_report = apply_state_dict(self.encoder.vit, dict(vit_state), tag="radfm-vit")
        perceiver_state = head_state.get("perceiver", {}) if isinstance(head_state, dict) else {}
        if perceiver_state and not any(str(k).startswith("perceiver.") for k in perceiver_state):
            perceiver_state = {f"perceiver.{k}": v for k, v in perceiver_state.items()}
        perc_report = apply_state_dict(self.aggregator, perceiver_state, tag="radfm-perceiver")

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
        proj_report = apply_state_dict(self.projector, projector_state, tag="radfm-projector")
        self.last_checkpoint_key_report = {"vit": vit_report, "perceiver": perc_report, "projector": proj_report}
