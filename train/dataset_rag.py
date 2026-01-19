"""RAG-enabled dataset that injects retrieved reports into the prompt."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# avoid shadowing by train/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPT_DIR]

from train.dataset_ct import RadGenomeCTDataset, collate_processed_ct_rate
from rag.retriever import OrganFaissRetriever, OrganClassifierRetriever, ORGAN_LABEL_MAPPING as _ORGAN_LABEL_MAPPING
from rag.unified_retriever import BaseUnifiedRetriever
from models.classifiers.organ import OrganMLPOnlyModel


_CT_RATE_LABEL_KEYWORDS: Dict[str, Sequence[str]] = {
    "Medical material": (
        "catheter",
        "port catheter",
        "port-a-cath",
        "pacemaker",
        "sternotomy",
        "suture material",
        "endotracheal",
        "tracheal cannula",
        "prosthesis",
        "valve replacement",
        "stent",
    ),
    "Arterial wall calcification": (
        "atherosclerotic",
        "atheroma",
        "atheromatous",
        "calcific",
        "calcification",
        "calcified plaque",
        "calcific plaque",
    ),
    "Cardiomegaly": (
        "cardiomegaly",
        "heart size increased",
        "heart is larger",
        "cardiothoracic",
        "cardiothoracic ratio",
    ),
    "Pericardial effusion": ("pericardial effusion",),
    "Coronary artery wall calcification": (
        "coronary calcification",
        "coronary artery calcification",
        "coronary arteries",
    ),
    "Hiatal hernia": ("hiatal hernia",),
    "Lymphadenopathy": (
        "lymphadenopathy",
        "lymph node",
        "lymph nodes",
        "lap",
    ),
    "Emphysema": ("emphysema", "emphysematous"),
    "Atelectasis": ("atelectasis", "volume loss", "collapse"),
    "Lung nodule": ("nodule", "nodules"),
    "Lung opacity": (
        "opacity",
        "ground glass",
        "ground-glass",
        "infiltrate",
        "infiltration",
        "density increase",
        "airspace",
    ),
    "Pulmonary fibrotic sequela": (
        "fibrotic",
        "fibrosis",
        "sequela",
        "scar",
        "scarring",
        "fibroatelectatic",
        "fibroatelectasis",
    ),
    "Pleural effusion": ("pleural effusion",),
    "Mosaic attenuation pattern": ("mosaic attenuation", "mosaic atteniation", "mosaic pattern"),
    "Peribronchial thickening": ("peribronchial thickening", "bronchial wall thickening", "peribronchial"),
    "Consolidation": ("consolidation", "air bronchogram", "air bronchogram"),
    "Bronchiectasis": ("bronchiectasis", "bronchiectatic"),
    "Interlobular septal thickening": ("interlobular septal", "septal thickening", "interlobular septa"),
}

_CT_RATE_LABEL_NAMES: tuple[str, ...] = (
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
)


_NEGATION_RE = re.compile(
    r"\bno\b|\bwithout\b|\babsent\b|\babsence\b|free of|negative for|no evidence|"
    r"not\s+(?:seen|observed|detected|demonstrated|identified|present)",
    re.IGNORECASE,
)


_TAGGED_RETRIEVAL_RE = re.compile(r"^(\[[^\]]+\]:)\s*(.*)$")


def _split_tagged_retrieval(text: str) -> tuple[str, str]:
    """Split a retrieved snippet like "[LUNG]: ..." into (tag, body)."""
    stripped = str(text).strip()
    if not stripped:
        return "", ""
    match = _TAGGED_RETRIEVAL_RE.match(stripped)
    if match:
        return match.group(1), match.group(2).strip()
    return "", stripped


def _filter_to_positive_label_sentences(text: str, positive_labels: Sequence[str]) -> str:
    """Keep only sentences that match positive label keywords and look affirmative.

    This is a heuristic for "abnormal-only" RAG context when using oracle label signatures.
    """
    normalized = " ".join(str(text).split())
    if not normalized or not positive_labels:
        return ""

    keywords: set[str] = set()
    for label in positive_labels:
        label = str(label).strip()
        if not label:
            continue
        for kw in _CT_RATE_LABEL_KEYWORDS.get(label, ()):
            kw = str(kw).strip().lower()
            if kw:
                keywords.add(kw)
        keywords.add(label.lower())

    if not keywords:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    kept: List[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        low = sent.lower()
        if not any(kw in low for kw in keywords):
            continue
        if _NEGATION_RE.search(low):
            if "no change" not in low and "no significant change" not in low and "no interval change" not in low:
                continue
        kept.append(sent)
    return " ".join(kept)


class _RetrievalBackend:
    def retrieve(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        raise NotImplementedError


class _NoopBackend(_RetrievalBackend):
    def retrieve(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        return []


class _UnifiedBackend(_RetrievalBackend):
    def __init__(self, dataset: "RAGDataset") -> None:
        self.dataset = dataset

    def retrieve(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        return self.dataset._retrieve_with_unified(volume_name, target_text)


class _ClassifierBackend(_RetrievalBackend):
    def __init__(self, dataset: "RAGDataset") -> None:
        self.dataset = dataset

    def retrieve(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        return self.dataset._retrieve_with_classifier(volume_name, target_text)


class _FaissBackend(_RetrievalBackend):
    def __init__(self, dataset: "RAGDataset") -> None:
        self.dataset = dataset

    def retrieve(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        dataset = self.dataset
        query_embs = dataset._load_query_embeddings(volume_name)
        if query_embs is None:
            return []
        positive_labels: List[str] = []
        if getattr(dataset, "rag_positive_labels_only", False):
            global_emb = dataset._load_global_embedding(volume_name)
            if global_emb is None:
                return []
            positive_labels = dataset._get_query_positive_labels_from_classifier(global_emb)
            if not positive_labels:
                return []
        if dataset.organ_mlp is not None:
            with torch.no_grad():
                refined = dataset.organ_mlp({k: v for k, v in query_embs.items()})
        else:
            refined = query_embs
        hits = dataset.retriever.retrieve(refined, k=dataset.top_k + 5)
        candidates = []
        for organ_name, organ_hits in hits.items():
            if not organ_hits:
                continue

            count = 0
            for h in organ_hits[0]:
                if str(h.id) == str(volume_name):
                    continue
                text = dataset._select_text(h, organ_name)
                if text:
                    if getattr(dataset, "rag_positive_labels_only", False):
                        tag, body = _split_tagged_retrieval(text)
                        body = _filter_to_positive_label_sentences(body, positive_labels)
                        if not body:
                            continue
                        body = dataset._compact_retrieved_text(body)
                        if not body:
                            continue
                        text = f"{tag} {body}" if tag else body
                    candidates.append(text)
                    count += 1
                if count >= dataset.top_per_organ:
                    break
        return dataset._dedupe_candidates(candidates, target_text)


class _OracleExactLabelBackend(_RetrievalBackend):
    """Oracle retrieval using exact per-organ CT-RATE label signature matching.

    This ignores embeddings and retrieves a report whose label signature matches the query
    within each organ group.
    """

    def __init__(self, dataset: "RAGDataset", index_dir: Path) -> None:
        self.dataset = dataset
        self.index_dir = Path(index_dir)
        self._tables = self._load_tables()

    def _stable_choice_indices(self, volume_name: str, organ: str, pool_size: int, count: int) -> List[int]:
        if pool_size <= 0 or count <= 0:
            return []
        seed = hashlib.sha1(f"{volume_name}|{organ}".encode("utf-8")).hexdigest()
        base = int(seed[:12], 16)
        return [int((base + i) % pool_size) for i in range(count)]

    def _load_tables(self) -> Dict[str, Dict[str, Any]]:
        tables: Dict[str, Dict[str, Any]] = {}
        for pt in sorted(self.index_dir.glob("*_index.pt")):
            data = torch.load(pt, map_location="cpu", weights_only=False)
            organ = str(data.get("organ") or pt.stem.replace("_index", "")).strip()
            vols = [str(v) for v in (data.get("volume_names") or [])]
            reports = [str(r) for r in (data.get("reports") or [])]
            labels = data.get("labels")
            label_cols = [str(c) for c in (data.get("label_cols") or [])]
            label_indices = [int(i) for i in (data.get("label_indices") or [])]
            if labels is None or not vols or not reports or not label_indices or not label_cols:
                continue
            labels_np = np.asarray(labels, dtype=np.float32)

            # Align dataset label order -> index label order by name.
            query_names = list(getattr(self.dataset, "label_names", []) or [])
            if not query_names:
                raise ValueError("oracle_exact_label requires a label_csv (label_names) loaded in the dataset.")
            query_pos = {n: i for i, n in enumerate(query_names)}
            col_to_query_pos: List[int] = []
            for name in label_cols:
                if name not in query_pos:
                    raise ValueError(
                        f"Label column '{name}' not found in dataset label_names; cannot align oracle labels."
                    )
                col_to_query_pos.append(int(query_pos[name]))

            sig_to_rows: Dict[tuple, List[int]] = {}
            for i in range(len(vols)):
                sig = tuple(1 if float(labels_np[i, j]) >= 0.5 else 0 for j in label_indices)
                sig_to_rows.setdefault(sig, []).append(i)

            tables[organ] = {
                "volume_names": vols,
                "reports": reports,
                "label_indices": label_indices,
                "col_to_query_pos": col_to_query_pos,
                "sig_to_rows": sig_to_rows,
                "label_cols": label_cols,
            }

        if not tables:
            raise ValueError(f"No *_index.pt files loaded from {self.index_dir}")
        return tables

    def retrieve(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        ds = self.dataset
        q = ds._get_label_tensor(str(volume_name))
        if q is None:
            return []
        q = q.detach().float().cpu().numpy()

        ordered_organs = ["lung", "trachea and bronchie", "heart", "mediastinum", "pleura"]
        candidates: List[str] = []

        for organ in ordered_organs:
            table = self._tables.get(organ)
            if table is None:
                continue
            q_reordered = q[np.asarray(table["col_to_query_pos"], dtype=np.int64)]
            sig = tuple(1 if float(q_reordered[j]) >= 0.5 else 0 for j in table["label_indices"])
            positive_labels: List[str] = []
            filter_positive = bool(
                getattr(ds, "oracle_positive_labels_only", False) or getattr(ds, "rag_positive_labels_only", False)
            )
            if filter_positive:
                positive_labels = [
                    str(table["label_cols"][j])
                    for j in table["label_indices"]
                    if float(q_reordered[j]) >= 0.5
                ]
                if not positive_labels:
                    continue
            elif getattr(ds, "oracle_abnormal_only", False) and sum(sig) == 0:
                continue
            rows = table["sig_to_rows"].get(sig, [])
            if not rows:
                continue
            picks = self._stable_choice_indices(str(volume_name), organ, len(rows), ds.top_per_organ)
            for pick in picks:
                ridx = rows[pick]
                rid = table["volume_names"][ridx]
                if str(rid) == str(volume_name):
                    continue
                organ_text = ""
                per_vol = ds.organ_reports.get(str(rid)) if getattr(ds, "organ_reports", None) else None
                if per_vol and isinstance(per_vol, dict):
                    organ_text = (per_vol.get(organ) or "").strip()
                text = organ_text or table["reports"][ridx].strip()
                if filter_positive:
                    text = _filter_to_positive_label_sentences(text, positive_labels)
                if not text:
                    continue
                text = ds._compact_retrieved_text(text)
                if text:
                    candidates.append(f"[{organ.upper()}]: {text}")

        return ds._dedupe_candidates(candidates[: ds.top_k], target_text)


class RAGDataset(RadGenomeCTDataset):
    """RadGenome dataset that performs organ-level retrieval to augment the prompt."""

    _RETRIEVAL_MAX_CHARS: int = 500
    _RETRIEVAL_MAX_SENTENCES: int = 2

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        reports_csv: str | Path,
        tokenizer,
        split: Optional[str],
        label_csv: str | Path,
        embeddings_dir: str | Path,
        organ_mlp: Optional[OrganMLPOnlyModel] = None,
        retriever: Optional[OrganFaissRetriever] = None,
        classifier_retriever: Optional[OrganClassifierRetriever] = None,
        unified_retriever: Optional[BaseUnifiedRetriever] = None,
        organ_classifier: Optional[Any] = None,  # OrganAttentionClassifier for FAISS query extraction
        retriever_type: str = "classifier",  # "classifier", "hybrid", "hierarchical", "faiss"
        organ_reports: Optional[Dict[str, Dict[str, str]]] = None,
        precomputed_vision_dir: Optional[str | Path] = None,
        top_k: int = 2,
        top_per_organ: int = 1,
        rag_dropout: float = 0.0,
        max_tokens: int = 512,  # RAG prompts are long; 512+ is recommended
        prompt_style: str = "llama2",  # "llama2" or "llama3"
        max_samples: Optional[int] = None,
        use_classifier_retrieval: bool = False,  # Use classifier-based retrieval
        force_classifier: bool = False,  # Force classifier predictions (skip GT lookup) for test
        oracle_exact_label: bool = False,  # Oracle retrieval (exact label match; label leakage / upper bound)
        oracle_exact_label_index_dir: Optional[str | Path] = None,  # organ indices directory
        oracle_abnormal_only: bool = False,  # If true, skip organs whose query organ-signature is all-zero.
        oracle_positive_labels_only: bool = False,  # If true, keep only sentences matching positive labels (1s).
        rag_positive_labels_only: bool = False,  # If true, keep only sentences matching positive labels (1s).
        rag_positive_labels_from_retrieved: bool = False,  # If true, filter using retrieved item's labels (not query).
        **kwargs: Any,
    ) -> None:
        base_embeddings_dir = Path(embeddings_dir)
        split_dir = base_embeddings_dir / split if split else None
        if split_dir is not None and split_dir.exists():
            self.embeddings_dir = split_dir
        else:
            self.embeddings_dir = base_embeddings_dir
        self.organ_mlp = organ_mlp
        self.retriever = retriever
        self.classifier_retriever = classifier_retriever
        self.unified_retriever = unified_retriever
        self.organ_classifier = organ_classifier
        self.retriever_type = retriever_type
        self.use_classifier_retrieval = use_classifier_retrieval
        self.force_classifier = force_classifier
        self.oracle_exact_label = bool(oracle_exact_label)
        self.oracle_exact_label_index_dir = (
            Path(oracle_exact_label_index_dir) if oracle_exact_label_index_dir else None
        )
        self.oracle_abnormal_only = bool(oracle_abnormal_only)
        self.oracle_positive_labels_only = bool(oracle_positive_labels_only)
        self.rag_positive_labels_only = bool(rag_positive_labels_only)
        self.rag_positive_labels_from_retrieved = bool(rag_positive_labels_from_retrieved)
        self.organ_reports = organ_reports or {}
        self.precomputed_vision_dir = Path(precomputed_vision_dir) if precomputed_vision_dir else None
        if self.precomputed_vision_dir is not None:
            self.precomputed_split_dir = self.precomputed_vision_dir / split if split else self.precomputed_vision_dir
        else:
            self.precomputed_split_dir = None
        self.top_k = top_k
        self.top_per_organ = max(1, top_per_organ)
        self.rag_dropout = max(0.0, min(1.0, rag_dropout))
        self.max_tokens = max_tokens
        self.prompt_style = prompt_style.lower()
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        self.pad_token_id = int(pad_id)
        super().__init__(
            manifest_path=manifest_path,
            reports_csv=reports_csv,
            tokenizer=tokenizer,
            split=split,
            label_csv=label_csv,
            max_tokens=max_tokens,
            max_samples=max_samples,
            load_volumes=self.precomputed_vision_dir is None,
            precomputed_vision_dir=precomputed_vision_dir,
            **kwargs,
        )
        self.split = split or ""
        if self.organ_mlp is not None:
            self.organ_mlp.eval()
            for p in self.organ_mlp.parameters():
                p.requires_grad = False
        self._retrieval_backend: _RetrievalBackend = self._build_retrieval_backend()


    def _get_query_positive_labels_from_classifier(
        self,
        query_embedding: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> List[str]:

        if query_embedding is None:
            return []

        probs18 = None
        if self.unified_retriever is not None:
            probs18 = self.unified_retriever.predict_probs18(query_embedding)

        if probs18 is None and self.classifier_retriever is not None and getattr(self.classifier_retriever, "classifier", None) is not None:
            emb = query_embedding
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)  # [1, num_tokens, dim] for attention models
            probs_by_organ = self.classifier_retriever.predict_labels(emb)
            probs18 = torch.zeros(18, dtype=torch.float32)
            for organ, indices in _ORGAN_LABEL_MAPPING.items():
                organ_probs = probs_by_organ.get(organ)
                if organ_probs is None or organ_probs.numel() == 0:
                    continue
                organ_probs = organ_probs[0].detach().float().cpu()
                for local_j, label_idx in enumerate(indices):
                    if 0 <= label_idx < 18 and local_j < organ_probs.numel():
                        probs18[label_idx] = float(organ_probs[local_j])

        if probs18 is None:
            return []
        values = probs18.detach().float().cpu().tolist()
        positive: List[str] = []
        for name, val in zip(_CT_RATE_LABEL_NAMES, values):
            if float(val) >= float(threshold):
                positive.append(str(name))
        return positive

    def _build_retrieval_backend(self) -> _RetrievalBackend:
        if self.top_k <= 0:
            return _NoopBackend()
        if self.oracle_exact_label:
            if self.oracle_exact_label_index_dir is None:
                raise ValueError("oracle_exact_label_index_dir is required when oracle_exact_label is enabled.")
            return _OracleExactLabelBackend(self, self.oracle_exact_label_index_dir)
        if self.unified_retriever is not None:
            return _UnifiedBackend(self)
        if self.use_classifier_retrieval and self.classifier_retriever is not None:
            return _ClassifierBackend(self)
        if self.retriever is not None:
            return _FaissBackend(self)
        return _NoopBackend()

    def _load_query_embeddings(self, volume_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load organ-specific embeddings for FAISS retrieval.

        If embeddings are in raw format (e.g., [8192, 4096]) and organ_classifier is available,
        extract organ-specific embeddings using the classifier.
        """
        path = self._resolve_embedding_path(volume_name)
        if path is None:
            return None
        data = torch.load(path, map_location="cpu")

        # Case 1: Pre-extracted organ embeddings
        if isinstance(data, dict) and "embeddings" in data:
            embs = data["embeddings"]
            return {k: (v if v.dim() == 2 else v.unsqueeze(0)).float() for k, v in embs.items()}

        # Case 2: Raw embeddings - use classifier to extract organ features
        if self.organ_classifier is not None:
            if isinstance(data, torch.Tensor):
                raw_emb = data
            elif isinstance(data, dict):
                raw_emb = data.get("visual_embeds", data.get("embedding"))
            else:
                return None

            if raw_emb is None:
                return None

            # Convert to float and add batch dimension
            raw_emb = raw_emb.float()
            if raw_emb.dim() == 2:
                raw_emb = raw_emb.unsqueeze(0)  # [1, num_tokens, dim]

            # Extract organ features using classifier
            device = next(self.organ_classifier.parameters()).device
            with torch.no_grad():
                organ_feats = self.organ_classifier.extract_all_organ_features(raw_emb.to(device))

            return {k: v[0:1].cpu().float() for k, v in organ_feats.items()}

        return None

    def _load_global_embedding(self, volume_name: str) -> Optional[torch.Tensor]:
        """Load global embedding for classifier retrieval."""
        path = self._resolve_embedding_path(volume_name)
        if path is None:
            return None

        data = torch.load(path, map_location="cpu")
        if isinstance(data, torch.Tensor):
            return data.float()
        if isinstance(data, dict):
            emb = data.get("visual_embeds", data.get("embedding"))
            if emb is None:
                return None
            return emb.float()
        raise TypeError(f"Unexpected embedding payload type at {path}: {type(data)}")

    def _resolve_embedding_path(self, volume_name: str) -> Optional[Path]:
        """Resolve the embedding path for a given volume name.

        Some embedding dumps are stored without the `.nii.gz` suffix (e.g. `train_123.pt`),
        while manifests/reports often use the full filename (e.g. `train_123.nii.gz`).
        """
        candidates = [f"{volume_name}.pt"]
        stem = str(volume_name)
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        elif stem.endswith(".nii"):
            stem = stem[: -len(".nii")]
        if stem != volume_name:
            candidates.append(f"{stem}.pt")

        for name in candidates:
            path = self.embeddings_dir / name
            if path.exists():
                return path
        return None

    def _retrieve_with_unified(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        """Retrieve reports using unified retriever interface."""
        global_emb = self._load_global_embedding(volume_name)
        if global_emb is None:
            return []
        positive_labels: List[str] = []
        if self.rag_positive_labels_only and not self.rag_positive_labels_from_retrieved:
            positive_labels = self._get_query_positive_labels_from_classifier(global_emb)
            if not positive_labels:
                return []

        results = self.unified_retriever.retrieve(
            embedding=global_emb,
            k=self.top_k + 5,  # Extra candidates for filtering
            volume_name=volume_name,
            force_classifier=self.force_classifier,
        )

        candidates: List[str] = []
        for organ, reports in results.items():
            if not reports:
                continue
            count = 0
            for r in reports:
                # Prefer organ-specific sentence banks (region reports) when available.
                organ_text = ""
                per_vol = self.organ_reports.get(str(r.volume_id)) if self.organ_reports else None
                if per_vol and isinstance(per_vol, dict):
                    organ_text = (per_vol.get(organ) or "").strip()
                text = organ_text or (r.report or "").strip()
                chosen = text
                if not organ_text and text:
                    # Fallback text may contain long multi-line metadata blocks; prefer the top-level
                    # "organ:" line when present (ignore nested "organ/..." paths), otherwise keep only
                    # the first non-empty line.
                    wanted = f"{organ.lower()}:"
                    chosen = ""
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        low = ln.lower()
                        if low.startswith(wanted):
                            chosen = ln.split(":", 1)[1].strip()
                            break
                    if not chosen:
                        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                        chosen = lines[0] if lines else ""
                text = chosen
                if text and self.rag_positive_labels_only:
                    if self.rag_positive_labels_from_retrieved:
                        per_item_labels = [
                            _CT_RATE_LABEL_NAMES[int(i)]
                            for i in (getattr(r, "matched_labels", None) or [])
                            if 0 <= int(i) < len(_CT_RATE_LABEL_NAMES)
                        ]
                        text = _filter_to_positive_label_sentences(text, per_item_labels)
                    else:
                        text = _filter_to_positive_label_sentences(text, positive_labels)
                if text:
                    text = self._compact_retrieved_text(text)
                if text:
                    candidates.append(f"[{organ.upper()}]: {text}")
                    count += 1
                if count >= self.top_per_organ:
                    break

        return self._dedupe_candidates(candidates, target_text)

    def _retrieve_reports(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        return self._retrieval_backend.retrieve(volume_name, target_text)

    def _retrieve_with_classifier(self, volume_name: str, target_text: Optional[str] = None) -> Sequence[str]:
        """Retrieve reports using classifier-based label similarity."""
        global_emb = self._load_global_embedding(volume_name)
        if global_emb is None:
            return []
        positive_labels: List[str] = []
        if self.rag_positive_labels_only:
            positive_labels = self._get_query_positive_labels_from_classifier(global_emb)
            if not positive_labels:
                return []

        # Add batch dimension
        if global_emb.dim() == 2:
            global_emb = global_emb.unsqueeze(0)  # [1, 32, 4096]

        # Pass volume_name for label lookup (works even without classifier)
        # Use force_classifier=True for test to ensure classifier predictions are used
        hits = self.classifier_retriever.retrieve(
            global_emb, k=self.top_k + 5, query_volume_name=volume_name,
            force_classifier=self.force_classifier
        )

        candidates = []
        for organ_name, organ_hits in hits.items():
            if not organ_hits or not organ_hits[0]:  # organ_hits[0] is batch index 0
                continue

            count = 0
            for h in organ_hits[0]:
                # Skip the sample itself
                if str(h.id) == str(volume_name):
                    continue

                # Use the report from the retrieval result (already organ-specific)
                text = h.report.strip()
                if text:
                    if self.rag_positive_labels_only:
                        text = _filter_to_positive_label_sentences(text, positive_labels)
                    if not text:
                        continue
                    text = self._compact_retrieved_text(text)
                    # Format with organ name
                    formatted = f"[{organ_name.upper()}]: {text}"
                    candidates.append(formatted)
                    count += 1

                if count >= self.top_per_organ:
                    break

        return self._dedupe_candidates(candidates, target_text)

    def _compact_retrieved_text(self, text: str) -> str:
        """Make retrieved snippets short and single-line to reduce prompt noise."""
        text = " ".join(text.split())
        if not text:
            return ""

        # Keep only the first N sentences (heuristic).
        sentences: List[str] = []
        start = 0
        for i, ch in enumerate(text):
            if ch in ".!?":
                sent = text[start : i + 1].strip()
                if sent:
                    sentences.append(sent)
                start = i + 1
                if len(sentences) >= self._RETRIEVAL_MAX_SENTENCES:
                    break
        if sentences:
            text = " ".join(sentences)

        if len(text) > self._RETRIEVAL_MAX_CHARS:
            text = text[: self._RETRIEVAL_MAX_CHARS].rstrip()
            # Avoid ending mid-token too awkwardly.
            if text and text[-1].isalnum():
                text = text.rsplit(" ", 1)[0] or text
            text = f"{text}..."
        return text

    def _dedupe_candidates(self, candidates: List[str], target_text: Optional[str] = None) -> List[str]:
        """Remove duplicates."""
        deduped: List[str] = []
        seen: set[str] = set()
        for txt in candidates:
            normalized = " ".join(txt.split()).lower()
            # De-duplicate by content (ignore leading "[ORGAN]:" tag).
            if normalized.startswith("["):
                tag_end = normalized.find("]:")
                if tag_end != -1:
                    normalized = normalized[tag_end + 2 :].strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(txt)
            if len(deduped) >= self.top_k:
                break
        return deduped

    def _load_precomputed_tokens(self, volume_name: str) -> torch.Tensor:
        if self.precomputed_split_dir is None:
            raise ValueError("Precomputed vision directory not configured.")
        path = self.precomputed_split_dir / f"{volume_name}.pt"
        if not path.exists():
            stem = str(volume_name)
            if stem.endswith(".nii.gz"):
                stem = stem[: -len(".nii.gz")]
            elif stem.endswith(".nii"):
                stem = stem[: -len(".nii")]
            alt = self.precomputed_split_dir / f"{stem}.pt"
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(f"Missing precomputed vision tokens: {path}")
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            tensor = data.get("embedding")
            if tensor is None:
                tensor = next(iter(data.values()))
        else:
            tensor = data
        if tensor is None:
            raise ValueError(f"Invalid precomputed vision tensor at {path}")
        tensor = tensor.float()
        if not torch.isfinite(tensor).all():
            # Precomputed tokens can occasionally contain NaN/Inf (corrupted export / numerical overflow).
            # Replace them to avoid NaN loss during training.
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        volume_name = item["meta"]["volume_name"]

        # 1) retrieval with self-filtering
        eos_token = self.tokenizer.eos_token or "<|endoftext|>"
        base_report = (item["meta"]["report_text"] or "").strip()
        if eos_token and not base_report.endswith(eos_token):
            base_report = f"{base_report}{eos_token}"

        references = self._retrieve_reports(volume_name, target_text=base_report)
        if self.split == "train" and self.rag_dropout > 0 and torch.rand(1).item() < self.rag_dropout:
            references = []

        # 2) prompt construction (LLaMA3 style only)
        sys_msg = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are an expert radiologist. Generate a CT finding report.\n"
            "Use the retrieved similar cases as important references for your diagnosis.\n"
            "If similar cases show abnormalities, carefully check if the current scan has similar findings."
            "<|eot_id|>"
        )

        if references:
            ref_text = "\n".join(references)
            user_body = (
                "Similar cases from database:\n"
                f"{ref_text}\n\n"
                "Based on the CT image and the similar cases above, generate the finding report."
            )
        else:
            user_body = "Generate the finding for this CT scan."

        prefix = (
            f"{sys_msg}"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_body}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # 3) tokenize prompt + response separately for accurate masking
        # Ensure there is always room left for at least a small amount of supervised target tokens.
        min_target_tokens = 8
        if self.max_tokens <= min_target_tokens:
            min_target_tokens = 1
        max_prompt_tokens = max(1, self.max_tokens - min_target_tokens)
        prompt_limit = min(max(8, self.max_tokens // 2), max_prompt_tokens)
        prompt_ids = self.tokenizer(
            prefix,
            add_special_tokens=True,
            truncation=True,
            max_length=prompt_limit,
            return_tensors="pt",
        )["input_ids"][0]

        target_ids = self.tokenizer(
            base_report,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )["input_ids"][0]

        input_ids = torch.cat([prompt_ids, target_ids])
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100

        if len(input_ids) > self.max_tokens:
            input_ids = input_ids[: self.max_tokens]
            labels = labels[: self.max_tokens]

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        if self.precomputed_split_dir is not None:
            item["visual_embeds"] = self._load_precomputed_tokens(volume_name)

        item["input_ids"] = input_ids
        item["attention_mask"] = attention_mask
        item["labels"] = labels
        item["pad_token_id"] = self.pad_token_id
        item["meta"]["retrieved_reports"] = list(references)
        return item

    def _select_text(self, hit: RetrievalResult, organ: str) -> str:
        reports = self.organ_reports.get(hit.id)
        if reports:
            organ_text = reports.get(organ)
            if organ_text:
                organ_text = organ_text.strip()
                if organ_text:
                    return f"[{organ.upper()}]: {organ_text}"
        fallback = (hit.report or "").strip()
        if fallback:
            return f"[{organ.upper()}]: {fallback}"
        return ""


def build_rag_dataloader(**kwargs):
    """Convenience wrapper for the RAG dataset."""
    dataset_kwargs = dict(kwargs)
    batch_size = dataset_kwargs.pop("batch_size")
    shuffle = dataset_kwargs.pop("shuffle")
    num_workers = dataset_kwargs.pop("num_workers")
    return DataLoader(
        RAGDataset(**dataset_kwargs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_processed_ct_rate,
    )
