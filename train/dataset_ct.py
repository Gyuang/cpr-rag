"""Dataset utilities for the CT-RATE corpus."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


LOGGER = logging.getLogger(__name__)

TARGET_HW = 256
TARGET_DEPTH = 64


def _resolve_precomputed_tensor_path(split_dir: Path, volume_name: str) -> Path:
    path = split_dir / f"{volume_name}.pt"
    if path.exists():
        return path
    stem = str(volume_name)
    if stem.endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    elif stem.endswith(".nii"):
        stem = stem[: -len(".nii")]
    alt = split_dir / f"{stem}.pt"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Missing precomputed vision tensor: {path}")


def _load_precomputed_tensor(split_dir: Path, volume_name: str) -> torch.Tensor:
    path = _resolve_precomputed_tensor_path(split_dir, volume_name)
    data = torch.load(path, map_location="cpu")
    tensor: Optional[torch.Tensor] = None
    if isinstance(data, dict):
        for key in ("visual_embeds", "embedding", "visual_tokens"):
            candidate = data.get(key)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
                break
        if tensor is None:
            tensor = next((v for v in data.values() if isinstance(v, torch.Tensor)), None)
    elif isinstance(data, torch.Tensor):
        tensor = data
    if tensor is None:
        raise ValueError(f"Invalid precomputed tensor payload at {path}")
    if tensor.ndim == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Precomputed tensor at {path} must be 2D (got {tuple(tensor.shape)})")
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def _validate_label_coverage(
    samples: Sequence[Dict[str, Any]],
    label_map: Optional[Dict[str, List[float]]],
    label_csv: Optional[Path],
) -> None:
    """Raise or warn when the manifest and label CSV do not line up."""
    if not samples or not label_map or label_csv is None:
        return
    total = len(samples)
    covered = sum(1 for sample in samples if sample["volume_name"] in label_map)
    if covered == 0:
        raise ValueError(
            f"No labels from '{label_csv}' match any manifest entries. "
            "Double-check that the VolumeName column aligns with the manifest split."
        )
    if covered < total:
        LOGGER.warning(
            "Labels missing for %d/%d samples in %s; missing volumes will use all-zero targets.",
            total - covered,
            total,
            label_csv,
        )


class ProcessedCTRateDataset(Dataset):
    """Loads processed CT volumes (.npz) together with tokenised reports."""

    def __init__(
        self,
        manifest_path: str | Path,
        reports_csv: str | Path,
        tokenizer,
        split: Optional[str] = None,
        report_fields: Sequence[str] = ("Findings_EN", "Impressions_EN"),
        max_tokens: int = 512,
        load_into_memory: bool = False,
        load_volumes: bool = True,
        max_samples: Optional[int] = None,
        label_csv: Optional[str] = None,
        precomputed_vision_dir: Optional[str | Path] = None,
        precomputed_features_dir: Optional[str | Path] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.reports_csv = Path(reports_csv)
        self.tokenizer = tokenizer
        self.report_fields = tuple(report_fields)
        self.max_tokens = max_tokens
        self.load_into_memory = load_into_memory
        self.load_volumes = load_volumes
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        self.pad_token_id = int(pad_id)
        self._volume_cache: Dict[str, torch.Tensor] = {}
        self._meta_cache: Dict[str, Dict[str, Any]] = {}
        self.label_csv = Path(label_csv) if label_csv else None
        self.label_names: Optional[Tuple[str, ...]] = None
        self._label_map: Optional[Dict[str, List[float]]] = None
        self._empty_label: Optional[torch.Tensor] = None
        self.precomputed_vision_dir = Path(precomputed_vision_dir) if precomputed_vision_dir else None
        self.precomputed_features_dir = Path(precomputed_features_dir) if precomputed_features_dir else None
        if self.precomputed_vision_dir is not None and self.precomputed_features_dir is not None:
            raise ValueError("precomputed_vision_dir and precomputed_features_dir are mutually exclusive.")
        self.precomputed_split_dir = (
            (self.precomputed_vision_dir / split) if self.precomputed_vision_dir and split else self.precomputed_vision_dir
        )
        self.precomputed_features_split_dir = (
            (self.precomputed_features_dir / split) if self.precomputed_features_dir and split else self.precomputed_features_dir
        )
        self.reports = self._load_reports(self.reports_csv)
        if self.label_csv:
            self._label_map, self.label_names = self._load_labels(self.label_csv)
            if self.label_names:
                self._empty_label = torch.zeros(len(self.label_names), dtype=torch.float32)
        self.samples = self._load_manifest(self.manifest_path, split)
        if max_samples is not None:
            self.samples = self.samples[: max_samples]
        if not self.samples:
            raise ValueError(
                f"No usable samples found in manifest '{self.manifest_path}' "
                f"for split '{split}'."
            )
        _validate_label_coverage(self.samples, self._label_map, self.label_csv)

    @staticmethod
    def _load_reports(path: Path) -> Dict[str, Dict[str, str]]:
        reports: Dict[str, Dict[str, str]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "VolumeName" not in reader.fieldnames:
                raise ValueError(f"'VolumeName' column missing in {path}")
            for row in reader:
                key = row["VolumeName"]
                if not key:
                    continue
                reports[key] = row
        if not reports:
            raise ValueError(f"No report rows found in {path}")
        return reports

    @staticmethod
    def _load_labels(path: Path) -> Tuple[Dict[str, List[float]], Tuple[str, ...]]:
        label_map: Dict[str, List[float]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "VolumeName" not in reader.fieldnames:
                raise ValueError(f"'VolumeName' column missing in {path}")
            label_names = tuple(name for name in reader.fieldnames if name != "VolumeName")
            if not label_names:
                raise ValueError(f"No label columns found in {path}")
            for row in reader:
                key = row.get("VolumeName")
                if not key:
                    continue
                label_map[key] = [float(row.get(name, 0)) for name in label_names]
        if not label_map:
            raise ValueError(f"No labels loaded from {path}")
        return label_map, label_names

    def _get_label_tensor(self, volume_name: str) -> Optional[torch.Tensor]:
        if self._label_map is None or self.label_names is None:
            return None
        values = self._label_map.get(volume_name)
        if values is None:
            if self._empty_label is None:
                return None
            return self._empty_label.clone()
        return torch.tensor(values, dtype=torch.float32)

    def _load_manifest(
        self,
        manifest_path: Path,
        split: Optional[str],
    ) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source_path", "output_path", "status", "split"}
            if missing := required - set(reader.fieldnames or []):
                raise ValueError(
                    f"Manifest '{manifest_path}' missing columns: {sorted(missing)}"
                )
            for row in reader:
                if row["status"] != "ok":
                    continue
                if split is not None and row["split"] != split:
                    continue
                volume_name = Path(row["source_path"]).name
                npz_path = Path(row["output_path"])
                if not npz_path.is_absolute():
                    npz_path = manifest_path.parent / npz_path
                if volume_name not in self.reports:
                    LOGGER.warning(
                        "No report entry for %s (source=%s)", volume_name, row["source_path"]
                    )
                    continue
                samples.append(
                    {
                        "npz_path": npz_path,
                        "volume_name": volume_name,
                        "manifest": row,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _parse_metadata(value: Any) -> Dict[str, Any]:
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)
        return json.loads(text)

    def _compose_report(self, volume_name: str) -> str:
        row = self.reports[volume_name]
        parts: List[str] = []
        for field in self.report_fields:
            text = row.get(field)
            if text:
                stripped = text.strip()
                if stripped:
                    parts.append(stripped)
        if not parts:
            # fall back to any non-empty column
            for value in row.values():
                if value and value.strip():
                    parts.append(value.strip())
                    break
        if not parts:
            raise ValueError(f"No textual content available for {volume_name}")
        return "\n\n".join(parts)

    def _load_volume(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        key = sample["npz_path"].as_posix()
        if key in self._volume_cache:
            volume = self._volume_cache[key]
            metadata = self._meta_cache[key]
            return volume, metadata

        with np.load(sample["npz_path"]) as data:
            volume_arr = data["volume"]
            metadata = self._parse_metadata(data["metadata"])
        volume_tensor = torch.from_numpy(volume_arr).float().unsqueeze(0)
        volume_tensor = _normalize_volume_zero_to_one(volume_tensor)
        volume_tensor = _resize_volume_to_target(volume_tensor, target_hw=TARGET_HW, target_depth=TARGET_DEPTH)

        if self.load_into_memory:
            self._volume_cache[key] = volume_tensor
            self._meta_cache[key] = metadata
        return volume_tensor, metadata

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if self.load_volumes:
            volume_tensor, metadata = self._load_volume(sample)
        else:
            volume_tensor = torch.empty(0)
            metadata = {}
        report_text = self._compose_report(sample["volume_name"])
        encoded = self.tokenizer(
            report_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.tokenizer.pad_token_id
        item = {
            "volume": volume_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "meta": {
                "volume_name": sample["volume_name"],
                "manifest": sample["manifest"],
                "npz_metadata": metadata,
                "report_text": report_text,
            },
        }
        item["pad_token_id"] = self.pad_token_id
        cls_labels = self._get_label_tensor(sample["volume_name"])
        if cls_labels is not None:
            item["cls_labels"] = cls_labels
            item["meta"]["label_names"] = list(self.label_names or [])
        if self.precomputed_split_dir is not None:
            item["visual_embeds"] = _load_precomputed_tensor(self.precomputed_split_dir, sample["volume_name"])
        elif self.precomputed_features_split_dir is not None:
            item["visual_features"] = _load_precomputed_tensor(self.precomputed_features_split_dir, sample["volume_name"])
        return item


class RadGenomeCTDataset(Dataset):
    """Loads RadGenome Chest CT NIfTI volumes together with grounded reports."""

    def __init__(
        self,
        manifest_path: str | Path,
        reports_csv: str | Path,
        tokenizer,
        split: Optional[str] = None,
        report_fields: Sequence[str] = ("Report",),
        max_tokens: int = 512,
        load_into_memory: bool = False,
        load_volumes: bool = True,
        max_samples: Optional[int] = None,
        load_masks: bool = False,
        mask_names: Optional[Sequence[str]] = None,
        label_csv: Optional[str] = None,
        precomputed_vision_dir: Optional[str | Path] = None,
        precomputed_features_dir: Optional[str | Path] = None,
        abnormal_only: bool = False,
        abnormal_threshold: float = 0.5,
        abnormal_exclude_labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.reports_csv = Path(reports_csv)
        self.tokenizer = tokenizer
        self.report_fields = tuple(report_fields)
        self.max_tokens = max_tokens
        self.load_into_memory = load_into_memory
        self.load_volumes = load_volumes
        self.load_masks = load_masks
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        self.pad_token_id = int(pad_id)
        if mask_names is None:
            mask_names = ("lung", "heart", "mediastinum", "pleura", "trachea and bronchie")
        normalized_names: List[str] = []
        for name in mask_names:
            if name.lower().endswith(".nii") or name.lower().endswith(".nii.gz"):
                normalized_names.append(name)
            else:
                normalized_names.append(f"{name}.nii.gz")
        self.mask_names = tuple(normalized_names)
        self._volume_cache: Dict[str, torch.Tensor] = {}
        self._meta_cache: Dict[str, Dict[str, Any]] = {}
        self._mask_cache: Dict[str, torch.Tensor] = {}
        self.label_csv = Path(label_csv) if label_csv else None
        self.label_names: Optional[Tuple[str, ...]] = None
        self._label_map: Optional[Dict[str, List[float]]] = None
        self._empty_label: Optional[torch.Tensor] = None
        self.precomputed_vision_dir = Path(precomputed_vision_dir) if precomputed_vision_dir else None
        self.precomputed_features_dir = Path(precomputed_features_dir) if precomputed_features_dir else None
        if self.precomputed_vision_dir is not None and self.precomputed_features_dir is not None:
            raise ValueError("precomputed_vision_dir and precomputed_features_dir are mutually exclusive.")
        self.precomputed_split_dir = (
            (self.precomputed_vision_dir / split) if self.precomputed_vision_dir and split else self.precomputed_vision_dir
        )
        self.precomputed_features_split_dir = (
            (self.precomputed_features_dir / split) if self.precomputed_features_dir and split else self.precomputed_features_dir
        )
        self.abnormal_only = bool(abnormal_only)
        self.abnormal_threshold = float(abnormal_threshold)
        self.abnormal_exclude_labels = tuple(
            str(n).strip() for n in (abnormal_exclude_labels or []) if str(n).strip()
        )

        self.reports = ProcessedCTRateDataset._load_reports(self.reports_csv)
        if self.label_csv:
            self._label_map, self.label_names = ProcessedCTRateDataset._load_labels(self.label_csv)
            if self.label_names:
                self._empty_label = torch.zeros(len(self.label_names), dtype=torch.float32)
        self.samples = self._load_manifest(self.manifest_path, split)
        if self.abnormal_only:
            _validate_label_coverage(self.samples, self._label_map, self.label_csv)
            if self._label_map is None or self.label_names is None:
                raise ValueError("abnormal_only=True requires label_csv with aligned labels.")
            exclude = set(self.abnormal_exclude_labels)
            include_indices = [i for i, name in enumerate(self.label_names) if name not in exclude]
            if not include_indices:
                raise ValueError(
                    "abnormal_only=True removed all labels (abnormal_exclude_labels excluded everything)."
                )
            before = len(self.samples)
            missing_labels = 0
            kept: List[Dict[str, Any]] = []
            for sample in self.samples:
                vol = str(sample.get("volume_name") or "")
                values = self._label_map.get(vol)
                if values is None:
                    missing_labels += 1
                    continue
                if any(float(values[i]) >= self.abnormal_threshold for i in include_indices):
                    kept.append(sample)
            self.samples = kept
            LOGGER.warning(
                "abnormal_only=True filtered %s split: kept %d/%d samples (missing labels for %d).",
                split,
                len(self.samples),
                before,
                missing_labels,
            )
        if max_samples is not None:
            self.samples = self.samples[: max_samples]
        if not self.samples:
            raise ValueError(
                f"No usable samples found in manifest '{self.manifest_path}' "
                f"for split '{split}'."
            )
        if not self.abnormal_only:
            _validate_label_coverage(self.samples, self._label_map, self.label_csv)

    def _load_manifest(
        self,
        manifest_path: Path,
        split: Optional[str],
    ) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"volume_name", "image_path", "split"}
            if missing := required - set(reader.fieldnames or []):
                raise ValueError(
                    f"RadGenome manifest '{manifest_path}' missing columns: {sorted(missing)}"
                )
            has_mask_path = "mask_path" in (reader.fieldnames or [])
            has_mask_dir = "mask_dir" in (reader.fieldnames or [])
            for row in reader:
                if split is not None and row["split"] != split:
                    continue
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = manifest_path.parent / image_path
                volume_name = row.get("volume_name") or image_path.name
                if volume_name not in self.reports:
                    LOGGER.warning(
                        "No RadGenome report entry for %s (path=%s)", volume_name, image_path
                    )
                    continue
                mask_path = None
                if has_mask_path:
                    raw_mask = row.get("mask_path")
                    if raw_mask:
                        mask_path = Path(raw_mask)
                        if not mask_path.is_absolute():
                            mask_path = manifest_path.parent / mask_path
                mask_dir = None
                if has_mask_dir:
                    raw_dir = row.get("mask_dir")
                    if raw_dir:
                        mask_dir = Path(raw_dir)
                        if not mask_dir.is_absolute():
                            mask_dir = manifest_path.parent / mask_dir
                samples.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "mask_dir": mask_dir,
                        "volume_name": volume_name,
                        "manifest": row,
                    }
                )
        return samples

    def _get_label_tensor(self, volume_name: str) -> Optional[torch.Tensor]:
        if self._label_map is None or self.label_names is None:
            return None
        values = self._label_map.get(volume_name)
        if values is None:
            if self._empty_label is None:
                return None
            return self._empty_label.clone()
        return torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _compose_report(self, volume_name: str) -> str:
            row = self.reports.get(volume_name, {})
            parts = []
            for field in self.report_fields:
                text = row.get(field)
                if text and text.strip():
                    clean_text = text.split('\n')[0].strip()
                    

                    parts.append(clean_text)
            
            return "\n\n".join(parts) if parts else "No report available."

    def _load_volume(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        key = sample["image_path"].as_posix()
        if key in self._volume_cache:
            return self._volume_cache[key], self._meta_cache[key]

        image_path = sample["image_path"]
        image = nib.load(str(image_path))
        volume_arr = np.asarray(image.get_fdata(), dtype=np.float32)
        if not np.isfinite(volume_arr).all():
            # Some NIfTI volumes contain NaN/Inf voxels (e.g., empty slices or corrupted conversions).
            # Downstream vision encoders + LM loss are not robust to non-finite inputs, so sanitize here.
            volume_arr = np.nan_to_num(volume_arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
        volume_arr = np.clip(volume_arr, -1000, 1000)
        volume_tensor = torch.from_numpy(volume_arr).float()
        if volume_tensor.ndim == 4 and volume_tensor.shape[0] == 1:
            volume_tensor = volume_tensor.squeeze(0)
        if volume_tensor.ndim != 3:
            raise ValueError(f"Unexpected tensor shape {tuple(volume_tensor.shape)} for {image_path}")
        volume_tensor = volume_tensor.unsqueeze(0)
        volume_tensor = _normalize_volume_zero_to_one(volume_tensor)
        volume_tensor = _resize_volume_to_target(volume_tensor, target_hw=TARGET_HW, target_depth=TARGET_DEPTH)

        metadata = {
            "affine": image.affine.tolist(),
            "zooms": tuple(float(z) for z in (image.header.get_zooms() or ())),
            "shape": tuple(int(dim) for dim in volume_arr.shape),
        }
        if self.load_into_memory:
            self._volume_cache[key] = volume_tensor
            self._meta_cache[key] = metadata
        return volume_tensor, metadata

    def _resolve_mask_path(self, sample: Dict[str, Any], mask_filename: str) -> Optional[Path]:
        mask_dir = sample.get("mask_dir")
        if mask_dir is None:
            return None
        candidate = mask_dir / mask_filename
        if candidate.exists():
            return candidate
        normalized = mask_filename.lower().replace(".nii.gz", "").replace(".nii", "")
        for file in mask_dir.glob("*.nii*"):
            stem = file.name.lower().replace(".nii.gz", "").replace(".nii", "")
            if stem == normalized:
                return file
        return None

    def _load_mask_tensor(self, sample: Dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.load_masks or not self.mask_names:
            return None
        mask_tensors: List[torch.Tensor] = []
        for mask_name in self.mask_names:
            resolved = self._resolve_mask_path(sample, mask_name)
            if resolved is None:
                zeros = torch.zeros(
                    (1, TARGET_HW, TARGET_HW, TARGET_DEPTH), dtype=torch.float32
                )
                mask_tensors.append(zeros)
                continue
            key = resolved.as_posix()
            if key in self._mask_cache:
                mask_tensors.append(self._mask_cache[key])
                continue
            mask_img = nib.load(str(resolved))
            mask_arr = np.asarray(mask_img.get_fdata(), dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_arr).float()
            if mask_tensor.ndim == 4 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.ndim != 3:
                raise ValueError(f"Unexpected mask tensor shape {tuple(mask_tensor.shape)} for {resolved}")
            mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = _resize_volume_to_target(
                mask_tensor,
                target_hw=TARGET_HW,
                target_depth=TARGET_DEPTH,
                mode="nearest",
            )
            mask_tensor = mask_tensor.round().clamp_min(0)
            if self.load_into_memory:
                self._mask_cache[key] = mask_tensor
            mask_tensors.append(mask_tensor)
        if not mask_tensors:
            return None
        combined = torch.cat(mask_tensors, dim=0).long()
        return combined

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if self.load_volumes:
            volume_tensor, metadata = self._load_volume(sample)
        else:
            volume_tensor = torch.empty(0)
            metadata = {}
        mask_tensor = self._load_mask_tensor(sample)
        report_text = self._compose_report(sample["volume_name"])
        encoded = self.tokenizer(
            report_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.tokenizer.pad_token_id
        item = {
            "volume": volume_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "meta": {
                "volume_name": sample["volume_name"],
                "manifest": sample["manifest"],
                "npz_metadata": metadata,
                "report_text": report_text,
            },
        }
        item["pad_token_id"] = self.pad_token_id
        if mask_tensor is not None:
            item["organ_mask"] = mask_tensor
            item["meta"]["mask_names"] = list(self.mask_names)
        cls_labels = self._get_label_tensor(sample["volume_name"])
        if cls_labels is not None:
            item["cls_labels"] = cls_labels
            item["meta"]["label_names"] = list(self.label_names or [])
        if self.precomputed_split_dir is not None:
            item["visual_embeds"] = _load_precomputed_tensor(self.precomputed_split_dir, sample["volume_name"])
        elif self.precomputed_features_split_dir is not None:
            item["visual_features"] = _load_precomputed_tensor(self.precomputed_features_split_dir, sample["volume_name"])
        return item


def _pad_and_stack(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(seq.size(0) for seq in sequences)
    if all(seq.size(0) == max_len for seq in sequences):
        return torch.stack(sequences)
    batch_size = len(sequences)
    result = sequences[0].new_full((batch_size, max_len), pad_value)
    for idx, seq in enumerate(sequences):
        result[idx, : seq.size(0)] = seq
    return result


def collate_processed_ct_rate(
    batch: List[Dict[str, Any]]
) -> Dict[str, Any]:
    volumes_list = [item["volume"] for item in batch]
    volumes = None
    if volumes_list and not all(vol.numel() == 0 for vol in volumes_list):
        volumes = torch.stack(volumes_list)
    pad_ids = [item.pop("pad_token_id", None) for item in batch]
    pad_token_id = next((pid for pid in pad_ids if pid is not None), 0)
    input_ids = _pad_and_stack([item["input_ids"] for item in batch], pad_token_id)
    attention_mask = _pad_and_stack([item["attention_mask"] for item in batch], 0)
    labels = _pad_and_stack([item["labels"] for item in batch], -100)
    meta = [item["meta"] for item in batch]
    organ_masks = [item.get("organ_mask") for item in batch]
    cls_labels = [item.get("cls_labels") for item in batch]
    batched_masks: Optional[torch.Tensor] = None
    batched_labels: Optional[torch.Tensor] = None
    if any(mask is not None for mask in organ_masks):
        first_mask = next(mask for mask in organ_masks if mask is not None)
        zeros = torch.zeros_like(first_mask)
        batched_masks = torch.stack(
            [mask if mask is not None else zeros.clone() for mask in organ_masks]
        )
    if any(label is not None for label in cls_labels):
        first_label = next(label for label in cls_labels if label is not None)
        zeros = torch.zeros_like(first_label)
        batched_labels = torch.stack(
            [label if label is not None else zeros.clone() for label in cls_labels]
        )
    visual_embeds = [item.get("visual_embeds") for item in batch]
    stacked_visual: Optional[torch.Tensor] = None
    if any(embed is not None for embed in visual_embeds):
        first = next(embed for embed in visual_embeds if embed is not None)
        zeros = torch.zeros_like(first)
        stacked_visual = torch.stack([embed if embed is not None else zeros.clone() for embed in visual_embeds])
    visual_features = [item.get("visual_features") for item in batch]
    stacked_features: Optional[torch.Tensor] = None
    if any(feat is not None for feat in visual_features):
        first = next(feat for feat in visual_features if feat is not None)
        zeros = torch.zeros_like(first)
        stacked_features = torch.stack([feat if feat is not None else zeros.clone() for feat in visual_features])
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "meta": meta,
    }
    if volumes is not None:
        result["volume"] = volumes
    if stacked_visual is not None:
        result["visual_embeds"] = stacked_visual
    if stacked_features is not None:
        result["visual_features"] = stacked_features
    if batched_masks is not None:
        result["organ_mask"] = batched_masks
    if batched_labels is not None:
        result["cls_labels"] = batched_labels
    return result


def build_processed_ct_rate_dataloader(
    manifest_path: str | Path,
    reports_csv: str | Path,
    tokenizer,
    split: Optional[str],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_tokens: int = 256,
    report_fields: Sequence[str] = ("Findings_EN", "Impressions_EN"),
    load_into_memory: bool = False,
    max_samples: Optional[int] = None,
    label_csv: Optional[str] = None,
    precomputed_vision_dir: Optional[str | Path] = None,
    precomputed_features_dir: Optional[str | Path] = None,
) -> DataLoader:
    dataset = ProcessedCTRateDataset(
        manifest_path=manifest_path,
        reports_csv=reports_csv,
        tokenizer=tokenizer,
        split=split,
        report_fields=report_fields,
        max_tokens=max_tokens,
        load_into_memory=load_into_memory,
        load_volumes=precomputed_vision_dir is None and precomputed_features_dir is None,
        max_samples=max_samples,
        label_csv=label_csv,
        precomputed_vision_dir=precomputed_vision_dir,
        precomputed_features_dir=precomputed_features_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_processed_ct_rate,
    )


def build_radgenome_dataloader(
    manifest_path: str | Path,
    reports_csv: str | Path,
    tokenizer,
    split: Optional[str],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_tokens: int = 512,
    report_fields: Sequence[str] = ("Report",),
    load_into_memory: bool = False,
    max_samples: Optional[int] = None,
    load_masks: bool = False,
    mask_names: Optional[Sequence[str]] = None,
    label_csv: Optional[str] = None,
    precomputed_vision_dir: Optional[str | Path] = None,
    precomputed_features_dir: Optional[str | Path] = None,
) -> DataLoader:
    dataset = RadGenomeCTDataset(
        manifest_path=manifest_path,
        reports_csv=reports_csv,
        tokenizer=tokenizer,
        split=split,
        report_fields=report_fields,
        max_tokens=max_tokens,
        load_into_memory=load_into_memory,
        load_volumes=precomputed_vision_dir is None and precomputed_features_dir is None,
        max_samples=max_samples,
        load_masks=load_masks,
        mask_names=mask_names,
        label_csv=label_csv,
        precomputed_vision_dir=precomputed_vision_dir,
        precomputed_features_dir=precomputed_features_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_processed_ct_rate,
    )


def _normalize_volume_zero_to_one(volume: torch.Tensor) -> torch.Tensor:
    """Clip to [-1000, 1000] HU and scale to [0, 1]."""
    volume = volume.clamp(-1000, 1000)
    return (volume + 1000.0) / 2000.0


def _resize_volume_to_target(
    volume: torch.Tensor,
    target_hw: int = TARGET_HW,
    target_depth: int = TARGET_DEPTH,
    mode: str = "trilinear",
) -> torch.Tensor:
    """Resize [1, H, W, D] tensor to [1, target_hw, target_hw, target_depth]."""
    if volume.shape[1:4] == (target_hw, target_hw, target_depth):
        return volume
    vol = volume.unsqueeze(0)  # -> [1, 1, H, W, D]
    interpolate_kwargs = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = False
    vol = F.interpolate(
        vol,
        size=(target_hw, target_hw, target_depth),
        mode=mode,
        **interpolate_kwargs,
    )
    return vol.squeeze(0)
