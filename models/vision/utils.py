"""Checkpoint loading utilities for vision backbones."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


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


def unwrap_state_dict_container(state: object) -> object:
    """Extract the actual model state dict from common checkpoint wrappers."""
    model_state = state
    if isinstance(state, dict):
        for key in ("model", "state_dict"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                model_state = candidate
                break
    return model_state


def strip_prefixes(
    model_state: Dict[str, torch.Tensor],
    extra_prefixes: Sequence[str] = (),
) -> Dict[str, torch.Tensor]:
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
    if filtered is None and any(
        isinstance(key, str) and key.startswith("module.") for key in model_state.keys()
    ):
        filtered = {
            key.replace("module.", "", 1): value
            for key, value in model_state.items()
            if isinstance(key, str)
        }
    return filtered or model_state


def remap_state_keys(
    model_state: Dict[str, torch.Tensor],
    replacements: Sequence[Tuple[str, str]],
) -> Dict[str, torch.Tensor]:
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


def apply_state_dict(
    module: nn.Module,
    model_state: Dict[str, torch.Tensor],
    *,
    tag: str,
    strict: bool = True,
) -> Dict[str, Sequence[str]]:
    """Load state dict into module with logging."""
    missing, unexpected = module.load_state_dict(model_state, strict=strict)
    if missing and strict:
        print(f"[{tag}] missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected and strict:
        print(f"[{tag}] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    return {"missing": missing, "unexpected": unexpected}


def prepare_state_dict(
    model_state: Dict[str, torch.Tensor],
    *,
    extra_prefixes: Sequence[str] = (),
    replacements: Sequence[Tuple[str, str]] = (),
) -> Dict[str, torch.Tensor]:
    stripped = strip_prefixes(model_state, extra_prefixes=extra_prefixes)
    return remap_state_keys(stripped, replacements)


def load_checkpoint_into(
    module: nn.Module,
    checkpoint: str | Path,
    *,
    tag: str,
    extra_prefixes: Sequence[str] = (),
    replacements: Sequence[Tuple[str, str]] = (),
    strict: bool = True,
) -> Dict[str, Sequence[str]]:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model_state = unwrap_state_dict_container(state)
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint at {checkpoint} does not contain a state_dict-like object.")
    prepared = prepare_state_dict(model_state, extra_prefixes=extra_prefixes, replacements=replacements)
    return apply_state_dict(module, prepared, tag=tag, strict=strict)


def extract_sequence(features: object) -> Tuple[torch.Tensor, Optional[Tuple[int, ...]]]:
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
