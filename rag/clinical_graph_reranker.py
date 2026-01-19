"""Data-driven clinical graph reranker.

Builds a conditional co-occurrence matrix M_ij = P(D_j | D_i) from training labels
and uses it as a Bayesian prior to calibrate retrieval scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ClinicalGraph:
    label_names: list[str]
    prob_matrix: torch.Tensor  # [C, C], float32

    def to(self, device: torch.device | str) -> "ClinicalGraph":
        return ClinicalGraph(self.label_names, self.prob_matrix.to(device))


class ClinicalGraphReranker:
    def __init__(self, graph: ClinicalGraph) -> None:
        self.graph = graph
        if self.graph.prob_matrix.dim() != 2 or self.graph.prob_matrix.shape[0] != self.graph.prob_matrix.shape[1]:
            raise ValueError(f"prob_matrix must be square, got {tuple(self.graph.prob_matrix.shape)}")

    @staticmethod
    def build_prob_matrix(labels01: np.ndarray, *, alpha: float = 1.0) -> torch.Tensor:
        """Compute M_ij = P(D_j | D_i) from binary labels [N, C].

        Uses Laplace smoothing:
        M_ij = (Count(f_i, f_j) + α) / (Count(f_i) + α|L|)
        """
        y = torch.tensor(labels01, dtype=torch.float32)
        num_labels = y.shape[1]  # |L|
        co = y.T @ y  # [C, C] - Count(f_i, f_j)
        counts = y.sum(dim=0).unsqueeze(1)  # [C, 1] - Count(f_i)
        # Laplace smoothing
        m = (co + alpha) / (counts + alpha * num_labels)
        m.fill_diagonal_(0)
        return m

    @classmethod
    def from_labels(
        cls,
        *,
        label_names: list[str],
        labels: np.ndarray,
        threshold: float = 0.5,
        alpha: float = 1.0,
    ) -> "ClinicalGraphReranker":
        """Build reranker from training labels with Laplace smoothing."""
        labels01 = (labels > threshold).astype(np.float32)
        m = cls.build_prob_matrix(labels01, alpha=alpha)
        return cls(ClinicalGraph(label_names=label_names, prob_matrix=m))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "label_names": self.graph.label_names,
                "prob_matrix": self.graph.prob_matrix.detach().cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: Optional[str | torch.device] = None) -> "ClinicalGraphReranker":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        label_names = list(ckpt["label_names"])
        prob_matrix = ckpt["prob_matrix"].float()
        graph = ClinicalGraph(label_names=label_names, prob_matrix=prob_matrix)
        if device is not None:
            graph = graph.to(device)
        return cls(graph)

    def context_vector(self, predicted_probs: torch.Tensor) -> torch.Tensor:
        """Compute context vector: p @ M -> [C]."""
        if predicted_probs.dim() != 1:
            predicted_probs = predicted_probs.view(-1)
        p = predicted_probs.to(dtype=self.graph.prob_matrix.dtype, device=self.graph.prob_matrix.device)
        return p @ self.graph.prob_matrix

    @staticmethod
    def align_labels(
        *,
        src_label_names: list[str],
        dst_label_names: list[str],
    ) -> list[int]:
        """Return indices into src that match each dst label name, or -1 if missing."""
        src_index = {n: i for i, n in enumerate(src_label_names)}
        return [src_index.get(n, -1) for n in dst_label_names]

    def score_boost(
        self,
        *,
        context: torch.Tensor,  # [C_graph]
        candidate_labels: np.ndarray,  # [C_idx]
        target_indices: Iterable[int],
        index_label_names: Optional[list[str]] = None,
    ) -> float:
        """Compute candidate-specific boost for an organ.

        Uses candidate label vector to pick which diseases are present.
        """
        target_indices = list(target_indices)
        if not target_indices:
            return 0.0

        if index_label_names is None:
            # Assume candidate_labels is already aligned with graph labels.
            aligned_context = context
            aligned_labels = candidate_labels
        else:
            # Align graph labels -> index labels.
            mapping = self.align_labels(src_label_names=self.graph.label_names, dst_label_names=index_label_names)
            # Build aligned context in index label space.
            ctx = torch.zeros(len(index_label_names), dtype=context.dtype, device=context.device)
            for j, src_i in enumerate(mapping):
                if src_i >= 0:
                    ctx[j] = context[src_i]
            aligned_context = ctx
            aligned_labels = candidate_labels

        # Candidate label presence (binary-ish), use >0.5 as present.
        y = (aligned_labels > 0.5).astype(np.float32)
        idx = [i for i in target_indices if 0 <= i < y.shape[0]]
        if not idx:
            return 0.0

        c = aligned_context.detach().float().cpu().numpy()
        return float((c[idx] * y[idx]).mean())

