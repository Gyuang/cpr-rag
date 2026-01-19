#!/usr/bin/env python3
"""Unified retriever interface for all RAG retrieval methods.

Supports:
1. "classifier" - OrganClassifierRetriever (label-space similarity with 4096-dim embeddings)
2. "hybrid" - HybridRAGRetriever (FAISS + organ classifier with 768-dim raw tokens)
3. "hierarchical" - HierarchicalRetriever (binary gating + organ FAISS search)
4. "faiss" - OrganFaissRetriever (FAISS-only organ embedding search)

Usage:
    from rag.unified_retriever import create_retriever, UnifiedRetrieverConfig

    config = UnifiedRetrieverConfig(
        retriever_type="classifier",
        index_path="/path/to/index.pkl",
        classifier_ckpt="/path/to/classifier.pt",
    )
    retriever = create_retriever(config)

    # Unified interface - returns Dict[organ, List[str]] reports
    reports = retriever.retrieve(embedding, k=3, volume_name="sample_001")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, Any
import sys

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class UnifiedRetrieverConfig:
    """Configuration for unified retriever."""

    retriever_type: str = "classifier"  # "classifier", "hybrid", "hierarchical", "faiss"

    # Common settings
    device: str = "cuda:0"
    top_k: int = 3
    top_per_organ: int = 1

    # Classifier retriever settings
    index_path: Optional[str] = None  # .pkl for classifier, directory for others
    classifier_ckpt: Optional[str] = None
    threshold: float = 0.5
    use_embedding_similarity: bool = True
    similarity_weight: float = 0.3

    # Hybrid retriever settings
    hybrid_approach: str = "hybrid"  # "baseline", "organ", "hybrid", "rerank"
    embedding_weight: float = 0.5

    # Hierarchical retriever settings
    binary_checkpoint: Optional[str] = None
    multilabel_checkpoint: Optional[str] = None
    normal_index_path: Optional[str] = None
    abnormal_index_dir: Optional[str] = None
    normal_threshold: float = 0.5
    embedding_dim: int = 4096  # Input embedding dimension
    graph_path: Optional[str] = None
    graph_alpha: float = 0.3
    graph_pool_multiplier: int = 5
    graph_pool_size: Optional[int] = None

    # FAISS retriever settings
    faiss_index_dir: Optional[str] = None

    # Test mode: force classifier predictions
    force_classifier: bool = False


@dataclass
class OrganReport:
    """Single organ report from retrieval."""
    organ: str
    report: str
    volume_id: str
    score: float = 0.0
    matched_labels: List[int] = field(default_factory=list)


class BaseUnifiedRetriever:
    """Base class for unified retriever interface."""

    def __init__(self, config: UnifiedRetrieverConfig):
        self.config = config
        self.device = config.device

    def retrieve(
        self,
        embedding: Tensor,
        k: int = 3,
        volume_name: Optional[str] = None,
        force_classifier: bool = False,
    ) -> Dict[str, List[OrganReport]]:
        """
        Retrieve similar reports for a query embedding.

        Args:
            embedding: Query embedding (format depends on retriever type)
            k: Number of results per organ
            volume_name: Optional volume name to exclude from results
            force_classifier: Force classifier predictions (for test inference)

        Returns:
            Dict[organ, List[OrganReport]] - organ-specific retrieval results
        """
        raise NotImplementedError

    def retrieve_formatted(
        self,
        embedding: Tensor,
        k: int = 3,
        volume_name: Optional[str] = None,
        force_classifier: bool = False,
    ) -> Sequence[str]:
        """
        Retrieve and format reports as strings for prompt injection.

        Returns:
            List of formatted strings like "[LUNG]: report text"
        """
        results = self.retrieve(embedding, k, volume_name, force_classifier)

        formatted = []
        for organ, reports in results.items():
            for r in reports[:self.config.top_per_organ]:
                if r.report.strip():
                    formatted.append(f"[{organ.upper()}]: {r.report}")

        return formatted[:k]

    def predict_probs18(self, embedding: Tensor) -> Optional[Tensor]:
        """Optional: predict 18-label CT-RATE probabilities for a query.

        Subclasses that have an organ multi-label classifier (e.g. hierarchical retriever)
        can override this to support positive-only gating without label CSV lookup.
        """
        return None


class ClassifierRetriever(BaseUnifiedRetriever):
    """Wrapper for OrganClassifierRetriever."""

    def __init__(self, config: UnifiedRetrieverConfig):
        super().__init__(config)

        from rag.retriever import OrganClassifierRetriever

        self.retriever = OrganClassifierRetriever(
            index_path=config.index_path,
            classifier_ckpt=config.classifier_ckpt,
            threshold=config.threshold,
            device=config.device,
        )

    def retrieve(
        self,
        embedding: Tensor,
        k: int = 3,
        volume_name: Optional[str] = None,
        force_classifier: bool = False,
    ) -> Dict[str, List[OrganReport]]:
        # OrganClassifierRetriever expects [B, 32, 4096] or [B, 4096]
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)

        results = self.retriever.retrieve(
            query_embedding=embedding,
            k=k + 5,  # Extra candidates for filtering
            use_embedding_similarity=self.config.use_embedding_similarity,
            similarity_weight=self.config.similarity_weight,
            query_volume_name=volume_name,
            force_classifier=force_classifier or self.config.force_classifier,
        )

        # Convert to unified format
        unified: Dict[str, List[OrganReport]] = {}
        for organ, organ_results in results.items():
            unified[organ] = []
            for batch_results in organ_results:
                for hit in batch_results:
                    if str(hit.id) == str(volume_name):
                        continue
                    unified[organ].append(OrganReport(
                        organ=organ,
                        report=hit.report,
                        volume_id=hit.id,
                        score=1 - hit.distance,  # Convert distance to similarity
                        matched_labels=hit.matched_labels,
                    ))
                    if len(unified[organ]) >= k:
                        break

        return unified


class HybridRetriever(BaseUnifiedRetriever):
    """Wrapper for HybridRAGRetriever."""

    def __init__(self, config: UnifiedRetrieverConfig):
        super().__init__(config)

        from rag.hybrid_retriever import HybridRAGRetriever

        self.retriever = HybridRAGRetriever(
            index_dir=config.index_path or config.faiss_index_dir,
            classifier_ckpt=config.classifier_ckpt,
            approach=config.hybrid_approach,
            device=config.device,
            embedding_weight=config.embedding_weight,
        )

    def retrieve(
        self,
        embedding: Tensor,
        k: int = 3,
        volume_name: Optional[str] = None,
        force_classifier: bool = False,
    ) -> Dict[str, List[OrganReport]]:
        # HybridRAGRetriever expects [num_tokens, 768] raw encoder tokens
        if embedding.dim() == 3:
            embedding = embedding.squeeze(0)

        output = self.retriever.retrieve(embedding, k=k + 5)

        # Convert to unified format (HybridRAGRetriever returns flat list)
        unified: Dict[str, List[OrganReport]] = {}
        for result in output.results:
            if str(result.id) == str(volume_name):
                continue

            organ = result.organ or "general"
            if organ not in unified:
                unified[organ] = []

            unified[organ].append(OrganReport(
                organ=organ,
                report=result.report,
                volume_id=result.id,
                score=result.score,
                matched_labels=result.matched_labels,
            ))

        # Limit per organ
        for organ in unified:
            unified[organ] = unified[organ][:k]

        return unified


class HierarchicalRetrieverWrapper(BaseUnifiedRetriever):
    """Wrapper for HierarchicalRetriever."""

    def __init__(self, config: UnifiedRetrieverConfig):
        super().__init__(config)

        from rag.hierarchical_retriever import HierarchicalRetriever

        self.retriever = HierarchicalRetriever(
            binary_checkpoint=config.binary_checkpoint,
            multilabel_checkpoint=config.multilabel_checkpoint,
            normal_index_path=config.normal_index_path,
            abnormal_index_dir=config.abnormal_index_dir,
            device=config.device,
            normal_threshold=config.normal_threshold,
            embedding_dim=config.embedding_dim,
            graph_path=config.graph_path,
            graph_alpha=config.graph_alpha,
            graph_pool_multiplier=config.graph_pool_multiplier,
            graph_pool_size=config.graph_pool_size,
        )
        self._organ_volume_to_row: Dict[str, Dict[str, int]] = {}
        for organ, db in getattr(self.retriever, "organ_dbs", {}).items():
            vols = db.get("volume_names") or []
            self._organ_volume_to_row[str(organ)] = {str(v): int(i) for i, v in enumerate(vols)}

    def retrieve(
        self,
        embedding: Tensor,
        k: int = 3,
        volume_name: Optional[str] = None,
        force_classifier: bool = False,
    ) -> Dict[str, List[OrganReport]]:
        # HierarchicalRetriever expects [num_tokens, dim] or [dim]
        result = self.retriever.retrieve(embedding, k=k)

        # Convert to unified format
        unified: Dict[str, List[OrganReport]] = {}

        if result.is_normal:
            # Normal case - returns full_report
            reports = result.retrieved_reports.get("full_report", [])
            volumes = result.retrieved_volume_names.get("full_report", [])
            unified["full_report"] = [
                OrganReport(
                    organ="full_report",
                    report=r,
                    volume_id=v,
                    score=result.normal_prob,
                )
                for r, v in zip(reports, volumes)
                if str(v) != str(volume_name)
            ][:k]
        else:
            # Abnormal case - organ-specific reports
            for organ, reports in result.retrieved_reports.items():
                volumes = result.retrieved_volume_names.get(organ, [])
                unified[organ] = []
                for r, v in zip(reports, volumes):
                    if str(v) == str(volume_name):
                        continue
                    matched_labels: List[int] = []
                    db = getattr(self.retriever, "organ_dbs", {}).get(organ) or {}
                    labels = db.get("labels")
                    label_indices = db.get("label_indices")
                    if labels is not None and label_indices:
                        row = self._organ_volume_to_row.get(str(organ), {}).get(str(v))
                        if row is not None and 0 <= int(row) < len(labels):
                            row_labels = labels[int(row)]
                            try:
                                matched_labels = [
                                    int(idx)
                                    for idx in label_indices
                                    if 0 <= int(idx) < 18 and float(row_labels[int(idx)]) >= float(self.config.threshold)
                                ]
                            except Exception:
                                matched_labels = []
                    unified[organ].append(OrganReport(
                        organ=organ,
                        report=r,
                        volume_id=v,
                        score=1 - result.normal_prob,  # Higher score for abnormal
                        matched_labels=matched_labels,
                    ))
                    if len(unified[organ]) >= k:
                        break

        return unified

    def predict_probs18(self, embedding: Tensor) -> Optional[Tensor]:
        return self.retriever.predict_probs18(embedding)


class FaissRetriever(BaseUnifiedRetriever):
    """Wrapper for OrganFaissRetriever."""

    def __init__(self, config: UnifiedRetrieverConfig):
        super().__init__(config)

        from rag.retriever import OrganFaissRetriever

        self.retriever = OrganFaissRetriever(
            index_dir=config.faiss_index_dir or config.index_path,
            device=config.device,
        )

    def retrieve(
        self,
        embedding: Tensor,
        k: int = 3,
        volume_name: Optional[str] = None,
        force_classifier: bool = False,
    ) -> Dict[str, List[OrganReport]]:
        # OrganFaissRetriever expects Dict[organ, Tensor[B, dim]]
        # Need to convert from global embedding to organ embeddings
        # This requires pre-computed organ embeddings

        if isinstance(embedding, dict):
            organ_embeddings = embedding
        else:
            # Assume it's a dict-like or has organ keys
            raise ValueError(
                "FaissRetriever requires pre-computed organ embeddings as Dict[str, Tensor]. "
                "Use ClassifierRetriever or HybridRetriever for global embeddings."
            )

        results = self.retriever.retrieve(organ_embeddings, k=k + 5)

        # Convert to unified format
        unified: Dict[str, List[OrganReport]] = {}
        for organ, organ_results in results.items():
            unified[organ] = []
            for batch_results in organ_results:
                for hit in batch_results:
                    if str(hit.id) == str(volume_name):
                        continue
                    unified[organ].append(OrganReport(
                        organ=organ,
                        report=hit.report,
                        volume_id=hit.id,
                        score=hit.distance,
                    ))
                    if len(unified[organ]) >= k:
                        break

        return unified


def create_retriever(config: UnifiedRetrieverConfig) -> BaseUnifiedRetriever:
    """Factory function to create the appropriate retriever."""

    retriever_type = config.retriever_type.lower()
    builders = {
        "hybrid": HybridRetriever,
        "hierarchical": HierarchicalRetrieverWrapper,
    }
    builder = builders.get(retriever_type)
    if builder is None:
        choices = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unknown retriever_type: {retriever_type}. Choose from: {choices}")
    return builder(config)


def create_retriever_from_yaml(config_dict: Dict[str, Any]) -> BaseUnifiedRetriever:
    """Create retriever from YAML config dict.

    Expected config structure:
    ```yaml
    retriever:
      type: "hybrid"  # or "hierarchical"
      index_path: "/path/to/index.pkl"
      classifier_ckpt: "/path/to/classifier.pt"
      # ... other settings
    ```
    """
    retriever_config = config_dict.get("retriever", config_dict)

    unified_config = UnifiedRetrieverConfig(
        retriever_type=retriever_config.get("type", "hybrid"),
        device=retriever_config.get("device", "cuda:0"),
        top_k=retriever_config.get("top_k", 3),
        top_per_organ=retriever_config.get("top_per_organ", 1),
        index_path=retriever_config.get("index_path"),
        classifier_ckpt=retriever_config.get("classifier_ckpt"),
        threshold=retriever_config.get("threshold", 0.5),
        use_embedding_similarity=retriever_config.get("use_embedding_similarity", True),
        similarity_weight=retriever_config.get("similarity_weight", 0.3),
        hybrid_approach=retriever_config.get("hybrid_approach", "hybrid"),
        embedding_weight=retriever_config.get("embedding_weight", 0.5),
        binary_checkpoint=retriever_config.get("binary_checkpoint"),
        multilabel_checkpoint=retriever_config.get("multilabel_checkpoint"),
        normal_index_path=retriever_config.get("normal_index_path"),
        abnormal_index_dir=retriever_config.get("abnormal_index_dir"),
        normal_threshold=retriever_config.get("normal_threshold", 0.5),
        graph_path=retriever_config.get("graph_path"),
        graph_alpha=retriever_config.get("graph_alpha", 0.3),
        graph_pool_multiplier=retriever_config.get("graph_pool_multiplier", 5),
        graph_pool_size=retriever_config.get("graph_pool_size"),
        faiss_index_dir=retriever_config.get("faiss_index_dir"),
        force_classifier=retriever_config.get("force_classifier", False),
    )

    return create_retriever(unified_config)


# Convenience type alias
RetrieverType = Union[HybridRetriever, HierarchicalRetrieverWrapper]
