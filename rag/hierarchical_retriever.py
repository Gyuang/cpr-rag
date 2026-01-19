#!/usr/bin/env python3
"""Hierarchical RAG Retriever with Binary Gatekeeper + FAISS Feature Search.

Pipeline:
1. Binary Classifier: Normal (prob > 0.5) → retrieve from Normal DB
2. Multi-label Feature Extractor: Abnormal → FAISS search in Abnormal Index

Uses:
- GlobalBinaryClassifier for normal/abnormal gating
- OrganClassifierModel for penultimate feature extraction
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss
except ImportError:
    faiss = None

from rag.clinical_graph_reranker import ClinicalGraphReranker

# Organ mapping (same as train_organ_classifier.py)
# Labels: 0=Medical material, 1=Arterial wall calcification, 2=Cardiomegaly, 3=Pericardial effusion,
#         4=Coronary artery wall calcification, 5=Hiatal hernia, 6=Lymphadenopathy, 7=Emphysema,
#         8=Atelectasis, 9=Lung nodule, 10=Lung opacity, 11=Pulmonary fibrotic sequela,
#         12=Pleural effusion, 13=Mosaic attenuation pattern, 14=Peribronchial thickening,
#         15=Consolidation, 16=Bronchiectasis, 17=Interlobular septal thickening
ORGAN_LABEL_MAPPING = {
    "lung": [7, 8, 9, 10, 11, 13, 15, 17],  # 8 labels (removed 14, 16 - airway specific)
    "trachea and bronchie": [14, 16],  # 2 labels (Peribronchial thickening, Bronchiectasis)
    "heart": [2, 3, 4],  # 3 labels
    "mediastinum": [1, 5, 6],  # 3 labels (removed 0=Medical material)
    "pleura": [12],  # 1 label
}


@dataclass
class RetrievalResult:
    """Result from hierarchical retrieval."""
    is_normal: bool
    normal_prob: float
    retrieved_reports: Dict[str, List[str]]  # organ → list of reports
    retrieved_volume_names: Dict[str, List[str]]  # organ → list of volume names
    query_features: Optional[Dict[str, np.ndarray]] = None  # organ → feature vector


class GlobalBinaryClassifier(nn.Module):
    """Binary classifier: Normal vs Abnormal."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_tokens: int = 1024,
        dropout: float = 0.1,
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.use_attention_pooling = use_attention_pooling
        self.input_proj = None  # Optional projection for dimension mismatch

        if use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def add_input_projection(self, actual_input_dim: int):
        """Add a projection layer to handle dimension mismatch."""
        if actual_input_dim != self.input_dim:
            self.input_proj = nn.Linear(actual_input_dim, self.input_dim)
            # Initialize with small weights
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        if x.dim() == 3:
            if self.use_attention_pooling:
                attn_weights = self.attention(x)
                attn_weights = F.softmax(attn_weights, dim=1)
                x = (x * attn_weights).sum(dim=1)
            else:
                x = x.mean(dim=1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class OrganClassifierModel(nn.Module):
    """Multi-head classifier with feature extraction capability."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = nn.ModuleDict()

        for organ, indices in ORGAN_LABEL_MAPPING.items():
            num_classes = len(indices)
            self.heads[organ] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )
        self.organ_indices = ORGAN_LABEL_MAPPING

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        outputs = {}
        for organ, head in self.heads.items():
            outputs[organ] = head(x)
        return outputs

    def extract_penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract concatenated penultimate layer features from all heads.

        Returns: [B, num_organs * hidden_dim] feature vector
        """
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        features = []
        for organ, head in self.heads.items():
            # Get output before final linear layer
            # head[0]: Linear, head[1]: LayerNorm, head[2]: ReLU, head[3]: Dropout, head[4]: final Linear
            penultimate = head[3](head[2](head[1](head[0](x))))  # After Dropout
            features.append(penultimate)

        return torch.cat(features, dim=-1)  # [B, num_organs * hidden_dim]

    def extract_shared_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature from first organ head (simpler approach).

        Returns: [B, hidden_dim] feature vector
        """
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        # Use lung head as representative (largest label set)
        head = self.heads["lung"]
        feature = head[3](head[2](head[1](head[0](x))))  # After Dropout
        return feature

    def extract_organ_feature(self, x: torch.Tensor, organ: str) -> torch.Tensor:
        """Extract feature from a specific organ head.

        Args:
            x: [B, input_dim] input embedding
            organ: Organ name (lung, heart, mediastinum, pleura, trachea and bronchie)

        Returns: [B, hidden_dim] feature vector for the specified organ
        """
        if organ not in self.heads:
            raise ValueError(f"Unknown organ: {organ}. Available: {list(self.heads.keys())}")

        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        head = self.heads[organ]
        # head[0]: Linear, head[1]: LayerNorm, head[2]: ReLU, head[3]: Dropout, head[4]: final Linear
        feature = head[3](head[2](head[1](head[0](x))))  # After Dropout, before final Linear
        return feature

    def extract_all_organ_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from all organ heads.

        Args:
            x: [B, input_dim] input embedding

        Returns: Dict mapping organ name to [B, hidden_dim] feature vector
        """
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        features = {}
        for organ, head in self.heads.items():
            features[organ] = head[3](head[2](head[1](head[0](x))))
        return features


class HierarchicalRetriever:
    """Hierarchical RAG Retriever with Binary Gatekeeper and Organ-Specific Retrieval.

    Pipeline:
    1. Query → Binary Classifier → Normal/Abnormal decision
    2. If Normal (prob > threshold): Random sample from Normal DB
    3. If Abnormal: For each organ, extract organ-specific feature → FAISS search in organ-specific index
    """

    def __init__(
        self,
        multilabel_checkpoint: Union[str, Path],
        normal_index_path: Optional[Union[str, Path]],
        abnormal_index_dir: Union[str, Path],  # Directory containing organ-specific indices
        device: str = "cuda",
        normal_threshold: float = 0.5,
        embedding_dim: int = 4096,  # Actual input embedding dimension
        binary_checkpoint: Optional[Union[str, Path]] = None,
        graph_path: Optional[Union[str, Path]] = None,
        graph_alpha: float = 0.3,
        graph_pool_multiplier: int = 5,
        graph_pool_size: Optional[int] = None,
    ):
        self.device = device
        self.normal_threshold = normal_threshold
        self.organs = list(ORGAN_LABEL_MAPPING.keys())
        self.embedding_dim = embedding_dim
        self.graph_alpha = float(graph_alpha)
        self.graph_pool_multiplier = int(graph_pool_multiplier)
        if graph_pool_size is None:
            self.graph_pool_size: Optional[int] = None
        else:
            graph_pool_size = int(graph_pool_size)
            self.graph_pool_size = graph_pool_size if graph_pool_size > 0 else None

        # Load models
        self.binary_model = None
        if binary_checkpoint is not None:
            self.binary_model = self._load_binary_model(binary_checkpoint, embedding_dim)
        self.multilabel_model, self._multilabel_expects_tokens = self._load_multilabel_model(multi_label_path=multilabel_checkpoint, embedding_dim=embedding_dim)

        # Load indices
        self.normal_db = None
        if normal_index_path is not None and Path(normal_index_path).exists():
            self.normal_db = self._load_normal_index(normal_index_path)
        self.organ_dbs, self.organ_faiss_indices = self._load_organ_indices(abnormal_index_dir)

        self.graph_reranker: Optional[ClinicalGraphReranker] = None
        if graph_path is not None and Path(graph_path).exists():
            print(f"[HierarchicalRetriever] Loading clinical graph: {graph_path}")
            self.graph_reranker = ClinicalGraphReranker.load(graph_path, device=self.device)

        print(f"[HierarchicalRetriever] Loaded:")
        if self.binary_model is not None:
            print(f"  - Binary model: {binary_checkpoint}")
        print(f"  - Multi-label model: {multilabel_checkpoint}")
        if self.normal_db is not None:
            print(f"  - Normal DB: {len(self.normal_db['volume_names'])} samples")
        else:
            print(f"  - Normal DB: (disabled)")
        for organ in self.organs:
            if organ in self.organ_dbs:
                print(f"  - {organ} DB: {len(self.organ_dbs[organ]['volume_names'])} samples")
        if self.graph_reranker is not None:
            if self.graph_pool_size is not None:
                print(f"  - Graph rerank pool: fixed size={self.graph_pool_size}")
            else:
                print(f"  - Graph rerank pool: multiplier={self.graph_pool_multiplier}")

    def _load_binary_model(self, checkpoint_path: Union[str, Path], embedding_dim: int) -> GlobalBinaryClassifier:
        """Load binary classifier from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Get config
        config = ckpt.get("config", {})
        model_input_dim = config.get("input_dim", 768)
        model = GlobalBinaryClassifier(
            input_dim=model_input_dim,
            hidden_dim=config.get("hidden_dim", 256),
            num_tokens=config.get("num_tokens", 1024),
            dropout=config.get("dropout", 0.1),
            use_attention_pooling=config.get("use_attention_pooling", True),
        )

        # Load state dict (support both "model_state_dict" and "model" keys)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict)

        # Add projection if embedding dimension doesn't match model's input dimension
        if embedding_dim != model_input_dim:
            print(f"  [BinaryClassifier] Adding input projection: {embedding_dim} -> {model_input_dim}")
            model.add_input_projection(embedding_dim)

        model.to(self.device)
        model.eval()
        return model

    def _load_multilabel_model(self, *, multi_label_path: Union[str, Path], embedding_dim: int) -> Tuple[nn.Module, bool]:
        """Load multi-label classifier from checkpoint.

        Automatically detects model type from checkpoint structure:
        - OrganAttentionClassifier: has input_proj, organ_queries, cross_attention keys
        - OrganClassifierModel: simple MLP heads
        """
        ckpt = torch.load(multi_label_path, map_location=self.device, weights_only=False)

        # Get config and state_dict
        config = ckpt.get("config", {})
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        task = ckpt.get("task", config.get("task", None))
        model_type = ckpt.get("model_type", config.get("model_type", None))

        # Use embedding_dim if config doesn't specify input_dim
        model_input_dim = config.get("input_dim", embedding_dim)

        if task == "organ_binary":
            from train.train_organ_binary_classifier import (
                OrganBinaryAttentionClassifier,
                OrganBinaryMLP,
            )

            if model_type in {"cross_attention", "attention"}:
                model = OrganBinaryAttentionClassifier(
                    input_dim=model_input_dim,
                    hidden_dim=config.get("hidden_dim", 512),
                    num_heads=config.get("num_heads", 8),
                    num_layers=config.get("num_layers", 2),
                    dropout=config.get("dropout", 0.1),
                    prompt_type=config.get("prompt_type", "none"),
                    num_prompt_tokens=config.get("num_prompt_tokens", 8),
                    num_visual_tokens=config.get("num_visual_tokens", 32),
                )
                expects_tokens = True
            else:
                model = OrganBinaryMLP(
                    input_dim=model_input_dim,
                    hidden_dim=config.get("hidden_dim", 1024),
                    dropout=config.get("dropout", 0.1),
                )
                expects_tokens = False
        elif model_type is not None:
            # Prefer explicit model_type if present (more robust than key heuristics).
            from train.train_organ_classifier import (
                OrganAttentionClassifier,
                OrganClassifierModel,
            )

            if model_type == "mlp":
                model = OrganClassifierModel(
                    input_dim=model_input_dim,
                    hidden_dim=config.get("hidden_dim", 1024),
                )
                expects_tokens = False
            elif model_type in {"cross_attention", "attention"}:
                model = OrganAttentionClassifier(
                    input_dim=model_input_dim,
                    hidden_dim=config.get("hidden_dim", 512),
                    num_heads=config.get("num_heads", 8),
                    num_layers=config.get("num_layers", 2),
                    dropout=config.get("dropout", 0.1),
                    num_organ_tokens=config.get("num_organ_tokens", 1),
                    query_interaction=bool(config.get("query_interaction", False)),
                    query_interaction_graph_path=Path(config.get("query_interaction_graph_path") or config.get("graph_path")) if (config.get("query_interaction_graph_path") or config.get("graph_path")) else None,
                    query_interaction_gate_init=float(config.get("query_interaction_gate_init", 0.0)),
                    prompt_type=config.get("prompt_type", "none"),
                    num_prompt_tokens=config.get("num_prompt_tokens", 8),
                    num_visual_tokens=config.get("num_visual_tokens", 32),
                )
                expects_tokens = True
            else:
                raise ValueError(f"Unknown model_type in checkpoint: {model_type}")
        else:
            # Backward-compat heuristic: infer from state_dict keys.
            state_dict_keys = set(state_dict.keys())
            is_attention_model = any(
                k.startswith("input_proj") or k.startswith("cross_attention") or k.startswith("organ_queries")
                for k in state_dict_keys
            )
            if is_attention_model:
                from train.train_organ_classifier import OrganAttentionClassifier

                model = OrganAttentionClassifier(
                    input_dim=model_input_dim,
                    hidden_dim=config.get("hidden_dim", 512),
                    num_heads=config.get("num_heads", 8),
                    num_layers=config.get("num_layers", 2),
                    dropout=config.get("dropout", 0.1),
                    prompt_type=config.get("prompt_type", "none"),
                    num_prompt_tokens=config.get("num_prompt_tokens", 8),
                    num_visual_tokens=config.get("num_visual_tokens", 32),
                )
                expects_tokens = True
            else:
                from train.train_organ_classifier import OrganClassifierModel

                model = OrganClassifierModel(
                    input_dim=model_input_dim,
                    hidden_dim=config.get("hidden_dim", 1024),
                )
                expects_tokens = False

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model, expects_tokens

    def _load_normal_index(self, index_path: Union[str, Path]) -> Dict:
        """Load normal index (volume names + reports + optional features)."""
        data = torch.load(index_path, map_location="cpu", weights_only=False)
        result = {
            "volume_names": data["volume_names"],
            "reports": data["reports"],
        }
        # Load features if available, create FAISS index
        if "features" in data:
            features = data["features"]
            if not isinstance(features, np.ndarray):
                features = features.numpy()
            features = features.astype(np.float32)
            faiss.normalize_L2(features)
            result["features"] = features
            # Create FAISS index
            dim = features.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            index.add(features)
            result["faiss_index"] = index
            print(f"  - Normal DB FAISS index created: {features.shape}")
        return result

    def _load_organ_indices(
        self, index_dir: Union[str, Path]
    ) -> Tuple[Dict[str, Dict], Dict[str, "faiss.Index"]]:
        """Load organ-specific indices.

        Args:
            index_dir: Directory containing organ-specific index files (e.g., lung_index.pt, heart_index.pt)

        Returns:
            (organ_dbs, organ_faiss_indices): Dict mapping organ name to DB and FAISS index
        """
        if faiss is None:
            raise ImportError("faiss is required for abnormal retrieval. Install with: pip install faiss-cpu")

        index_dir = Path(index_dir)
        organ_dbs = {}
        organ_faiss_indices = {}

        for organ in self.organs:
            # Try different filename formats
            safe_organ = organ.replace(" ", "_").replace("and", "").strip("_")
            index_path = None
            for candidate in [f"{organ}_index.pt", f"{safe_organ}_index.pt"]:
                if (index_dir / candidate).exists():
                    index_path = index_dir / candidate
                    break

            if index_path is None:
                print(f"  [Warning] Index not found for organ: {organ}")
                continue

            data = torch.load(index_path, map_location="cpu", weights_only=False)

            organ_dbs[organ] = {
                "volume_names": data["volume_names"],
                "reports": data["reports"],  # organ-specific reports
                "features": data["features"],  # [N, D] numpy array
                "labels": data.get("labels", None),  # [N, 18] (optional)
                "label_cols": data.get("label_cols", None),  # list[str] (optional)
                "label_indices": data.get("label_indices", ORGAN_LABEL_MAPPING[organ]),
            }

            # Build or load FAISS index
            if "faiss_index" in data:
                organ_faiss_indices[organ] = faiss.deserialize_index(data["faiss_index"])
            else:
                features = data["features"].astype(np.float32)
                dim = features.shape[1]
                index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(features)
                index.add(features)
                organ_faiss_indices[organ] = index

        return organ_dbs, organ_faiss_indices

    def _predict_probs18(self, query_embedding: torch.Tensor, pooled: torch.Tensor) -> Optional[torch.Tensor]:
        """Predict 18-disease probabilities from the (multi-label) organ classifier.

        Returns None if the loaded checkpoint is an organ-binary model (5 logits).
        """
        ml_input = query_embedding if self._multilabel_expects_tokens else pooled
        organ_logits = self.multilabel_model(ml_input)  # {organ: [B, num_classes]}
        if not organ_logits:
            return None

        # Detect organ-binary: every organ head has exactly 1 logit.
        if all(v.shape[-1] == 1 for v in organ_logits.values()):
            return None

        probs18 = torch.zeros(18, device=self.device, dtype=torch.float32)
        for organ, indices in ORGAN_LABEL_MAPPING.items():
            if organ not in organ_logits:
                continue
            probs = torch.sigmoid(organ_logits[organ]).squeeze(0).detach().float()  # [len(indices)]
            for local_j, label_idx in enumerate(indices):
                if 0 <= label_idx < 18 and local_j < probs.numel():
                    probs18[label_idx] = probs[local_j]
        return probs18

    @torch.no_grad()
    def predict_probs18(self, query_embedding: torch.Tensor) -> Optional[torch.Tensor]:
        """Predict 18-label CT-RATE probabilities for a query embedding.

        This is used for positive-only RAG gating/filtering without consulting any label CSV.
        Returns a CPU float32 tensor of shape [18], or None if the checkpoint is organ-binary.
        """
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(0)  # [1, num_tokens, dim]

        query_embedding = query_embedding.to(self.device)
        pooled = query_embedding.mean(dim=1) if query_embedding.dim() == 3 else query_embedding
        probs18 = self._predict_probs18(query_embedding, pooled)
        if probs18 is None:
            return None
        return probs18.detach().float().cpu()

    def _rerank_indices_with_graph(
        self,
        *,
        organ: str,
        distances: np.ndarray,  # [K]
        indices: np.ndarray,  # [K]
        context: torch.Tensor,  # [18] (graph label space)
        k: int,
    ) -> np.ndarray:
        db = self.organ_dbs[organ]
        labels = db.get("labels", None)
        if labels is None or self.graph_reranker is None:
            return indices[:k]

        labels = np.asarray(labels)
        if labels.ndim != 2:
            return indices[:k]

        label_cols = db.get("label_cols", None)
        boosts = []
        for idx in indices:
            if 0 <= idx < labels.shape[0]:
                boosts.append(
                    self.graph_reranker.score_boost(
                        context=context,
                        candidate_labels=labels[idx],
                        target_indices=db.get("label_indices", ORGAN_LABEL_MAPPING[organ]),
                        index_label_names=label_cols if isinstance(label_cols, list) else None,
                    )
                )
            else:
                boosts.append(0.0)

        boosts = np.asarray(boosts, dtype=np.float32)
        scores = distances.astype(np.float32) + (self.graph_alpha * boosts)
        order = np.argsort(-scores)
        return indices[order][:k]

    @torch.no_grad()
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 3,
    ) -> RetrievalResult:
        """Retrieve similar reports for a query embedding.

        Args:
            query_embedding: [num_tokens, dim] or [dim] visual embedding
            k: Number of reports to retrieve per organ

        Returns:
            RetrievalResult with organ-specific retrieved reports
        """
        # Ensure batch dimension
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(0)  # [1, num_tokens, dim]

        query_embedding = query_embedding.to(self.device)

        # Step 1: Normal/Abnormal gating (binary classifier if available).
        if query_embedding.dim() == 3:
            pooled = query_embedding.mean(dim=1)  # [B, dim]
        else:
            pooled = query_embedding  # [B, dim]

        if self.binary_model is not None:
            with torch.no_grad():
                normal_probs = self.binary_model.predict_proba(query_embedding if query_embedding.dim() == 3 else pooled)
            normal_prob = float(normal_probs.squeeze().item())
            is_normal = normal_prob > self.normal_threshold
            if self.normal_db is None:
                is_normal = False
        else:
            # Fallback: infer abnormality from multilabel logits.
            with torch.no_grad():
                ml_input = query_embedding if self._multilabel_expects_tokens else pooled
                organ_logits = self.multilabel_model(ml_input)  # {organ: [B, num_classes]}
            max_abnormal_prob = 0.0
            for logits in organ_logits.values():
                probs = torch.sigmoid(logits)
                max_abnormal_prob = max(max_abnormal_prob, float(probs.max().item()))
            normal_prob = 1.0 - max_abnormal_prob
            is_normal = normal_prob > self.normal_threshold
            if self.normal_db is None:
                is_normal = False

        if is_normal and self.normal_db is not None and len(self.normal_db.get("volume_names", [])) > 0:
            # Normal case: similarity search in Normal DB if FAISS index available
            if "faiss_index" in self.normal_db:
                # Extract global feature for query (concatenate all organ features)
                with torch.no_grad():
                    ml_input = query_embedding if self._multilabel_expects_tokens else pooled
                    organ_features = self.multilabel_model.extract_all_organ_features(ml_input)
                    feat_list = [organ_features[org] for org in sorted(organ_features.keys())]
                    query_feat = torch.cat(feat_list, dim=-1)  # [1, 5*hidden_dim]
                    query_np = query_feat.cpu().numpy().astype(np.float32)
                    faiss.normalize_L2(query_np)

                # FAISS search
                distances, indices = self.normal_db["faiss_index"].search(query_np, k)
                indices = indices[0]  # Remove batch dimension
            else:
                # Fallback: random sample
                indices = random.sample(
                    range(len(self.normal_db["volume_names"])),
                    min(k, len(self.normal_db["volume_names"]))
                )

            full_reports = [self.normal_db["reports"][i] for i in indices]
            full_volumes = [self.normal_db["volume_names"][i] for i in indices]

            # Return as dict with special key "full_report"
            return RetrievalResult(
                is_normal=True,
                normal_prob=normal_prob,
                retrieved_reports={"full_report": full_reports},
                retrieved_volume_names={"full_report": full_volumes},
                query_features=None,
            )
        else:
            # Abnormal case: Organ-specific retrieval
            # (pooled already computed above)

            # Extract organ-specific features and search
            retrieved_reports: Dict[str, List[str]] = {}
            retrieved_volumes: Dict[str, List[str]] = {}
            query_features: Dict[str, np.ndarray] = {}

            context = None
            if self.graph_reranker is not None:
                probs18 = self._predict_probs18(query_embedding, pooled)
                if probs18 is not None:
                    context = self.graph_reranker.context_vector(probs18)

            for organ in self.organs:
                if organ not in self.organ_faiss_indices:
                    continue

                # Extract organ-specific feature
                ml_input = query_embedding if self._multilabel_expects_tokens else pooled
                organ_feature = self.multilabel_model.extract_organ_feature(ml_input, organ)

                # Convert to numpy for FAISS
                organ_np = organ_feature.cpu().numpy().astype(np.float32)
                faiss.normalize_L2(organ_np)

                # FAISS search in organ-specific index
                pool_k = k
                if context is not None:
                    desired_pool = max(k * self.graph_pool_multiplier, k)
                    if self.graph_pool_size is not None:
                        desired_pool = max(self.graph_pool_size, k)
                    pool_k = min(desired_pool, len(self.organ_dbs[organ]["volume_names"]))
                distances, indices = self.organ_faiss_indices[organ].search(organ_np, pool_k)
                distances = distances[0]
                indices = indices[0]
                if context is not None:
                    indices = self._rerank_indices_with_graph(
                        organ=organ,
                        distances=distances,
                        indices=indices,
                        context=context,
                        k=k,
                    )
                else:
                    indices = indices[:k]

                # Get organ-specific reports
                retrieved_reports[organ] = [self.organ_dbs[organ]["reports"][i] for i in indices]
                retrieved_volumes[organ] = [self.organ_dbs[organ]["volume_names"][i] for i in indices]
                query_features[organ] = organ_np[0]

            return RetrievalResult(
                is_normal=False,
                normal_prob=normal_prob,
                retrieved_reports=retrieved_reports,
                retrieved_volume_names=retrieved_volumes,
                query_features=query_features,
            )

    def retrieve_batch(
        self,
        query_embeddings: torch.Tensor,
        k: int = 3,
    ) -> List[RetrievalResult]:
        """Retrieve for a batch of query embeddings.

        Args:
            query_embeddings: [B, num_tokens, dim] visual embeddings
            k: Number of reports to retrieve per query

        Returns:
            List of RetrievalResult for each query
        """
        results = []
        for i in range(query_embeddings.shape[0]):
            result = self.retrieve(query_embeddings[i], k=k)
            results.append(result)
        return results


def load_hierarchical_retriever(
    binary_checkpoint: str = "/workspace/CTDoc/outputs/classifiers/global_binary/global_classifier_best_f1.pt",
    multilabel_checkpoint: str = "/workspace/CTDoc/outputs/organ_classifier_radfm_mlp.pt",
    normal_index_path: str = "/workspace/CTDoc/outputs/rag_indices/normal_index.pt",
    abnormal_index_dir: str = "/workspace/CTDoc/outputs/rag_indices/organ_indices",
    device: str = "cuda",
    normal_threshold: float = 0.5,
) -> HierarchicalRetriever:
    """Factory function to load the hierarchical retriever.

    Args:
        binary_checkpoint: Path to binary classifier checkpoint
        multilabel_checkpoint: Path to multi-label classifier checkpoint
        normal_index_path: Path to normal index file
        abnormal_index_dir: Directory containing organ-specific indices (lung_index.pt, heart_index.pt, etc.)
        device: Device to run on
        normal_threshold: Threshold for normal/abnormal classification

    Returns:
        HierarchicalRetriever instance
    """
    return HierarchicalRetriever(
        binary_checkpoint=binary_checkpoint,
        multilabel_checkpoint=multilabel_checkpoint,
        normal_index_path=normal_index_path,
        abnormal_index_dir=abnormal_index_dir,
        device=device,
        normal_threshold=normal_threshold,
    )
