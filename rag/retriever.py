"""Label-space retriever for multi-classification RAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


# Organ label mapping (same as train_organ_classifier.py)
ORGAN_LABEL_MAPPING = {
    "lung": [7, 8, 9, 10, 11, 13, 15, 17],  # 8 labels (removed 14, 16 - airway specific)
    "trachea and bronchie": [14, 16],  # 2 labels (Peribronchial thickening, Bronchiectasis)
    "heart": [2, 3, 4],  # 3 labels
    "mediastinum": [1, 5, 6],  # 3 labels (removed 0=Medical material)
    "pleura": [12],  # 1 label
}


@dataclass
class RetrievalResult:
    id: str
    distance: float
    report: str
    organ: str = ""
    matched_labels: List[int] = field(default_factory=list)


class ConceptRetriever:
    """Retrieves closest training reports in the label probability space."""

    def __init__(self, db_path: str | torch.Tensor, device: str = "cpu") -> None:
        from pathlib import Path

        payload = (
            torch.load(str(db_path), map_location="cpu")
            if isinstance(db_path, (str, bytes, Path))
            else db_path
        )
        self.db_labels: Tensor = payload["labels"].to(device)
        self.reports: List[str] = payload["reports"]
        self.ids: List[str] = payload.get("ids") or list(range(len(self.reports)))
        self.label_names: Optional[List[str]] = payload.get("label_names")
        self.device = device

    def to(self, device: str) -> "ConceptRetriever":
        self.db_labels = self.db_labels.to(device)
        self.device = device
        return self

    def retrieve(self, query_logits: Tensor, k: int = 3) -> List[List[RetrievalResult]]:
        if not torch.is_tensor(query_logits):
            raise TypeError("query_logits must be a torch.Tensor")
        tensor = query_logits.to(self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        probs = torch.sigmoid(tensor)
        distances = torch.cdist(probs, self.db_labels, p=2)
        k = min(k, distances.size(1))
        dists, indices = torch.topk(distances, k, largest=False)
        results: List[List[RetrievalResult]] = []
        for row in range(dists.size(0)):
            hits: List[RetrievalResult] = []
            for dist, idx in zip(dists[row], indices[row]):
                hit_id = self.ids[int(idx)]
                hits.append(
                    RetrievalResult(
                        id=str(hit_id),
                        distance=float(dist.item()),
                        report=self.reports[int(idx)],
                    )
                )
            results.append(hits)
        return results


class OrganFaissRetriever:
    """FAISS-based retriever over organ-specific embeddings."""

    def __init__(self, index_dir: str | Path, device: str = "cuda:0") -> None:
        if faiss is None:
            raise ImportError("faiss is required for OrganFaissRetriever. Install faiss-cpu or faiss-gpu.")
        self.index_dir = Path(index_dir)
        import pickle

        meta_path = self.index_dir / "metadata.pkl"
        with meta_path.open("rb") as f:
            payload = pickle.load(f)
        self.meta: List[dict] = payload.get("meta") or []
        self.reports: List[str] = payload.get("reports") or []
        self.ids: List[str] = [m.get("volume_name") or m.get("StudyInstanceUID") or str(i) for i, m in enumerate(self.meta)]
        self.organs = ["lung", "heart", "mediastinum", "pleura", "trachea and bronchie"]
        alias = {"trachea and bronchie": "airways"}  # fallback to legacy index filename
        self.index: Dict[str, faiss.Index] = {}
        for organ in self.organs:
            idx_path = self.index_dir / f"{organ}.index"
            if not idx_path.exists() and organ in alias:
                # backward compatibility for old indices saved under "airways"
                idx_path = self.index_dir / f"{alias[organ]}.index"
            if idx_path.exists():
                self.index[organ] = faiss.read_index(str(idx_path))
        self.device = device

    def retrieve(self, organ_embeddings: Dict[str, Tensor], k: int = 5) -> Dict[str, List[List[RetrievalResult]]]:
        alias_map = {"airways": "trachea and bronchie"}
        results: Dict[str, List[List[RetrievalResult]]] = {}
        for organ_raw, emb in organ_embeddings.items():
            organ = alias_map.get(organ_raw, organ_raw)
            idx = self.index.get(organ)
            if idx is None:
                continue
            vecs = emb.detach().cpu().float()
            if vecs.dim() == 1:
                vecs = vecs.unsqueeze(0)
            # ensure L2 normalized for IP similarity
            faiss.normalize_L2(vecs.numpy())
            D, I = idx.search(vecs.numpy(), min(k, idx.ntotal))
            organ_hits: List[List[RetrievalResult]] = []
            for row_d, row_i in zip(D, I):
                hits: List[RetrievalResult] = []
                for dist, idx_val in zip(row_d, row_i):
                    meta_idx = int(idx_val)
                    rid = self.ids[meta_idx] if meta_idx < len(self.ids) else str(meta_idx)
                    report = self.reports[meta_idx] if meta_idx < len(self.reports) else ""
                    hits.append(RetrievalResult(id=rid, distance=float(dist), report=report))
                organ_hits.append(hits)
            results[organ_raw] = organ_hits
        return results


class OrganClassifierRetriever:
    """
    Classification-based retriever for organ-specific RAG.

    Instead of embedding similarity, this retriever:
    1. Uses OrganClassifierModel to predict disease labels per organ
    2. Finds volumes with similar disease patterns (Jaccard similarity on labels)
    3. Retrieves organ-specific report text from matched volumes
    """

    def __init__(
        self,
        index_path: str | Path,
        classifier_ckpt: Optional[str | Path] = None,
        threshold: float = 0.5,
        device: str = "cuda:0",
    ) -> None:
        """
        Args:
            index_path: Path to the prebuilt index (.pkl) containing:
                - volume_names: List[str]
                - labels: Tensor [N, 18] - full labels for all volumes
                - organ_reports: Dict[volume_name, Dict[organ, report_text]]
                - global_embeddings: Optional Tensor [N, 32, 4096] for embedding similarity
            classifier_ckpt: Path to trained OrganClassifierModel checkpoint
            threshold: Probability threshold for binary label prediction
            device: Device to run classifier on
        """
        self.index_path = Path(index_path)
        self.threshold = threshold
        self.device = device
        self.organs = list(ORGAN_LABEL_MAPPING.keys())

        # Load index
        with self.index_path.open("rb") as f:
            index_data = pickle.load(f)

        self.volume_names: List[str] = index_data["volume_names"]
        self.labels: Tensor = index_data["labels"].float()  # [N, 18]
        self.organ_reports: Dict[str, Dict[str, str]] = index_data.get("organ_reports", {})
        self.global_embeddings: Optional[Tensor] = index_data.get("global_embeddings")  # [N, 32, 4096]

        # Build per-organ label tensors for efficient retrieval
        self.organ_labels: Dict[str, Tensor] = {}
        for organ, indices in ORGAN_LABEL_MAPPING.items():
            self.organ_labels[organ] = self.labels[:, indices]  # [N, num_labels_for_organ]

        # Load classifier if provided
        self.classifier = None
        if classifier_ckpt is not None:
            self._load_classifier(classifier_ckpt)

    def _load_classifier(self, ckpt_path: str | Path) -> None:
        """Load the trained classifier (supports MLP and Attention models with Visual Prompt Tuning)."""
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from train.train_organ_classifier import (
            OrganClassifierModel,
            OrganAttentionClassifier,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        model_type = ckpt.get("model_type", config.get("model_type", "mlp"))
        input_dim = config.get("input_dim", 4096)
        hidden_dim = config.get("hidden_dim", 768)

        if model_type == "mlp":
            self.classifier = OrganClassifierModel(input_dim=input_dim, hidden_dim=hidden_dim)
            self.use_full_tokens = False
        elif model_type == "cross_attention":
            num_heads = config.get("num_heads", 8)
            num_layers = config.get("num_layers", 2)
            dropout = config.get("dropout", 0.1)
            num_organ_tokens = config.get("num_organ_tokens", 1)
            query_interaction = bool(config.get("query_interaction", False))
            qi_graph_path = config.get("query_interaction_graph_path") or config.get("graph_path")
            query_interaction_graph_path = Path(qi_graph_path) if qi_graph_path else None
            query_interaction_gate_init = float(config.get("query_interaction_gate_init", 0.0))
            # Visual Prompt Tuning support
            prompt_type = config.get("prompt_type", "none")
            num_prompt_tokens = config.get("num_prompt_tokens", 8)
            num_visual_tokens = config.get("num_visual_tokens", 32)
            self.classifier = OrganAttentionClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                num_organ_tokens=num_organ_tokens,
                query_interaction=query_interaction,
                query_interaction_graph_path=query_interaction_graph_path,
                query_interaction_gate_init=query_interaction_gate_init,
                prompt_type=prompt_type,
                num_prompt_tokens=num_prompt_tokens,
                num_visual_tokens=num_visual_tokens,
            )
            self.use_full_tokens = True
            self.prompt_type = prompt_type
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.classifier.load_state_dict(ckpt["model"])
        self.classifier.to(self.device)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        prompt_info = ""
        if hasattr(self, "prompt_type") and self.prompt_type != "none":
            prompt_info = f" + {self.prompt_type} visual prompt"
        print(f"[OrganClassifierRetriever] Loaded {model_type} classifier{prompt_info} from {ckpt_path}")

    def predict_labels(
        self,
        embedding: Tensor,  # [B, 32, 4096] or [B, 4096] (mean pooled)
    ) -> Dict[str, Tensor]:
        """
        Predict disease labels per organ from global embedding.

        Returns:
            Dict[organ, Tensor[B, num_labels]] - predicted probabilities per organ
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded. Provide classifier_ckpt.")

        # Handle input shape based on model type
        use_full = getattr(self, "use_full_tokens", False)

        if use_full:
            # Attention models expect [B, num_tokens, dim]
            if embedding.dim() == 2:
                embedding = embedding.unsqueeze(0)  # [num_tokens, dim] -> [1, num_tokens, dim]
        else:
            # MLP model expects [B, dim]
            if embedding.dim() == 3:
                embedding = embedding.mean(dim=1)  # [B, 32, 4096] -> [B, 4096]

        embedding = embedding.to(self.device)

        with torch.no_grad():
            outputs = self.classifier(embedding)  # {organ: [B, num_labels]}

        # Convert logits to probabilities
        return {organ: torch.sigmoid(logits) for organ, logits in outputs.items()}

    def retrieve(
        self,
        query_embedding: Tensor,  # [B, 32, 4096] or [B, 4096]
        k: int = 3,
        use_embedding_similarity: bool = True,  # Default: hybrid mode ON
        similarity_weight: float = 0.3,  # Embedding weight (0.3 label + 0.7 embedding when no binary match)
        query_labels: Optional[Tensor] = None,  # [B, 18] - optional pre-computed labels
        query_volume_name: Optional[str] = None,  # Optional: lookup labels from index by volume name
        force_classifier: bool = False,  # Force classifier instead of index lookup (for test inference)
    ) -> Dict[str, List[List[RetrievalResult]]]:
        """
        Retrieve similar volumes based on disease label similarity.

        Args:
            query_embedding: Query volume's global embedding
            k: Number of results per organ
            use_embedding_similarity: Also consider embedding similarity (hybrid mode)
            similarity_weight: Weight for embedding similarity when hybrid mode is on
            query_labels: Optional pre-computed labels [B, 18]. If provided, skip classifier.
            query_volume_name: Optional volume name to lookup labels from index.
            force_classifier: If True, skip volume_name lookup and use classifier directly.
                              Use this for test inference to ensure classifier predictions are used.

        Returns:
            Dict[organ, List[List[RetrievalResult]]] - results per organ per batch item
        """
        # Get predicted labels - priority: query_labels > volume_name lookup (unless force_classifier) > classifier
        if query_labels is not None:
            # Use provided labels directly
            pred_probs = {}
            for organ, indices in ORGAN_LABEL_MAPPING.items():
                pred_probs[organ] = query_labels[:, indices].float()
        elif query_volume_name is not None and query_volume_name in self.volume_names and not force_classifier:
            # Lookup from index (skipped when force_classifier=True for test inference)
            idx = self.volume_names.index(query_volume_name)
            full_labels = self.labels[idx].unsqueeze(0)  # [1, 18]
            pred_probs = {}
            for organ, indices in ORGAN_LABEL_MAPPING.items():
                pred_probs[organ] = full_labels[:, indices].float()
        elif self.classifier is not None:
            pred_probs = self.predict_labels(query_embedding)
        else:
            # Fallback: use embedding similarity only (uniform label probs)
            batch_size = query_embedding.size(0) if query_embedding.dim() >= 2 else 1
            pred_probs = {}
            for organ, indices in ORGAN_LABEL_MAPPING.items():
                # Use 0.5 as neutral probability
                pred_probs[organ] = torch.full((batch_size, len(indices)), 0.5, device=query_embedding.device)

        batch_size = query_embedding.size(0)
        results: Dict[str, List[List[RetrievalResult]]] = {}

        # Precompute embedding similarities if needed
        emb_similarities = None
        device = query_embedding.device
        if use_embedding_similarity and self.global_embeddings is not None:
            # query_embedding: [B, 32, 4096] or [B, 4096]
            if query_embedding.dim() == 3:
                q_emb = query_embedding.mean(dim=1)  # [B, 4096]
            else:
                q_emb = query_embedding
            q_emb = q_emb.to(device)
            db_emb = self.global_embeddings.mean(dim=1).to(device)  # [N, 4096]
            q_norm = F.normalize(q_emb, dim=1)  # [B, 4096]
            db_norm = F.normalize(db_emb, dim=1)  # [N, 4096]
            emb_similarities = q_norm @ db_norm.T  # [B, N]

        for organ in self.organs:
            organ_probs = pred_probs[organ].to(device)  # [B, num_labels]
            organ_indices = ORGAN_LABEL_MAPPING[organ]
            db_labels = self.organ_labels[organ].to(device)  # [N, num_labels]

            organ_results: List[List[RetrievalResult]] = []

            for b in range(batch_size):
                q_probs = organ_probs[b]  # [num_labels]

                # 1) Soft Jaccard: use probabilities instead of binary
                # soft_intersection = sum(min(p_q, p_db))
                # soft_union = sum(max(p_q, p_db))
                q_expanded = q_probs.unsqueeze(0)  # [1, num_labels]
                soft_intersection = torch.minimum(q_expanded, db_labels).sum(dim=1)  # [N]
                soft_union = torch.maximum(q_expanded, db_labels).sum(dim=1)  # [N]
                soft_jaccard = soft_intersection / (soft_union + 1e-8)  # [N]

                # 2) Combine with embedding similarity
                if emb_similarities is not None:
                    cos_sim = emb_similarities[b]  # [N]
                    # Adaptive weighting: if soft_jaccard is low, rely more on embedding
                    max_jaccard = soft_jaccard.max().item()
                    if max_jaccard < 0.1:
                        # No good label match, rely heavily on embedding
                        adaptive_weight = 0.8
                    else:
                        adaptive_weight = similarity_weight
                    score = (1 - adaptive_weight) * soft_jaccard + adaptive_weight * cos_sim
                else:
                    score = soft_jaccard

                # 3) Get top-k (no strict score > 0 filter for soft jaccard)
                topk_scores, topk_indices = torch.topk(score, min(k + 10, len(score)))

                hits: List[RetrievalResult] = []
                for score_val, idx in zip(topk_scores, topk_indices):
                    if score_val <= 0:
                        continue

                    vol_name = self.volume_names[idx]
                    report = self._get_organ_report(vol_name, organ)

                    if not report:
                        continue

                    # Find which labels matched (using threshold for reporting)
                    q_binary = (q_probs > self.threshold).float()
                    db_binary = (db_labels[idx] > self.threshold).float()
                    matched = (q_binary * db_binary > 0).nonzero(as_tuple=True)[0].tolist()
                    matched_global = [organ_indices[i] for i in matched]

                    hits.append(RetrievalResult(
                        id=vol_name,
                        distance=float(1 - score_val),  # Convert similarity to distance
                        report=report,
                        organ=organ,
                        matched_labels=matched_global,
                    ))

                    if len(hits) >= k:
                        break

                organ_results.append(hits)

            results[organ] = organ_results

        return results

    def _get_organ_report(self, volume_name: str, organ: str) -> str:
        """Get organ-specific report text for a volume."""
        # Try original name first
        vol_reports = self.organ_reports.get(volume_name, {})
        if vol_reports:
            return vol_reports.get(organ, "")

        # Try with .nii.gz suffix
        vol_reports = self.organ_reports.get(f"{volume_name}.nii.gz", {})
        if vol_reports:
            return vol_reports.get(organ, "")

        # Try without .nii.gz suffix
        name_no_suffix = volume_name.replace(".nii.gz", "").replace(".nii", "")
        vol_reports = self.organ_reports.get(name_no_suffix, {})
        return vol_reports.get(organ, "")

    @classmethod
    def build_index(
        cls,
        embeddings_dir: Path,
        label_csv: Path,
        organ_report_csvs: List[Path],
        output_path: Path,
        global_embeddings_dir: Optional[Path] = None,
    ) -> None:
        """
        Build the retrieval index from precomputed embeddings and labels.

        Args:
            embeddings_dir: Directory with per-volume .pt files (organ embeddings)
            label_csv: CSV with disease labels
            organ_report_csvs: CSVs with organ-specific report text
            output_path: Where to save the index .pkl
            global_embeddings_dir: Optional directory with global embeddings for hybrid mode
        """
        import pandas as pd
        import csv

        # Load labels
        label_df = pd.read_csv(label_csv)
        if 'VolumeName' in label_df.columns:
            label_df = label_df.set_index('VolumeName')
        label_cols = [c for c in label_df.columns if c not in ['VolumeName', 'split']][:18]

        # Load organ reports
        organ_reports: Dict[str, Dict[str, str]] = {}
        for csv_path in organ_report_csvs:
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vol = row.get("Volumename") or row.get("VolumeName") or row.get("volume_name")
                    sentence = (row.get("Sentence") or "").strip()
                    anatomy = (row.get("Anatomy") or "").strip().lower()
                    if not vol or not sentence:
                        continue

                    # Normalize organ name
                    organ = None
                    if "lung" in anatomy:
                        organ = "lung"
                    elif "pleur" in anatomy:
                        organ = "pleura"
                    elif "mediast" in anatomy:
                        organ = "mediastinum"
                    elif "trache" in anatomy or "bronch" in anatomy or "airway" in anatomy:
                        organ = "trachea and bronchie"
                    elif "heart" in anatomy or "card" in anatomy or "pericard" in anatomy:
                        organ = "heart"

                    if organ:
                        if vol not in organ_reports:
                            organ_reports[vol] = {}
                        if organ not in organ_reports[vol]:
                            organ_reports[vol][organ] = sentence
                        else:
                            organ_reports[vol][organ] += "\n" + sentence

        # Collect volume names and labels
        volume_names = []
        labels_list = []
        global_embeddings_list = []

        for pt_file in sorted(embeddings_dir.glob("**/*.pt")):
            vol_name = pt_file.stem
            # Get labels - try multiple name formats
            label_row = None
            for name_variant in [vol_name, f"{vol_name}.nii.gz", f"{vol_name}.nii"]:
                if name_variant in label_df.index:
                    label_row = label_df.loc[name_variant]
                    break
            if label_row is None:
                continue  # Skip volumes without labels
            label_vec = torch.tensor([float(label_row[c]) for c in label_cols], dtype=torch.float32)

            volume_names.append(vol_name)
            labels_list.append(label_vec)

            # Load global embedding if available
            if global_embeddings_dir:
                global_path = global_embeddings_dir / f"{vol_name}.pt"
                if global_path.exists():
                    data = torch.load(global_path, map_location="cpu", weights_only=False)
                    if isinstance(data, torch.Tensor):
                        global_embeddings_list.append(data.float())
                    elif isinstance(data, dict):
                        emb = data.get("visual_embeds", data.get("embedding"))
                        if emb is not None:
                            global_embeddings_list.append(emb.float())

        # Stack tensors
        labels_tensor = torch.stack(labels_list)
        global_embeddings_tensor = None
        if global_embeddings_list and len(global_embeddings_list) == len(volume_names):
            global_embeddings_tensor = torch.stack(global_embeddings_list)

        # Save index
        index_data = {
            "volume_names": volume_names,
            "labels": labels_tensor,
            "organ_reports": organ_reports,
            "global_embeddings": global_embeddings_tensor,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(index_data, f)

        print(f"Built index with {len(volume_names)} volumes, saved to {output_path}")
