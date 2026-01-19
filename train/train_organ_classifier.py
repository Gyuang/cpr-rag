#!/usr/bin/env python3
"""Train organ-specific multi-label classifiers on global embeddings."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import warnings

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPT_DIR]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm.auto import tqdm
import numpy as np

from models.graph_guided_query_interaction import (
    GraphGuidedQueryInteraction,
    build_organ_graph_from_disease_matrix,
    build_query_attn_bias,
    load_disease_graph_matrix,
)

# Organ mapping (same as dataset_organ.py)
ORGAN_LABEL_MAPPING = {
    "lung": [7, 8, 9, 10, 11, 13, 15, 17],  # 8 labels (removed 14, 16 - airway specific)
    "trachea and bronchie": [14, 16],  # 2 labels (Peribronchial thickening, Bronchiectasis)
    "heart": [2, 3, 4],  # 3 labels
    "mediastinum": [1, 5, 6],  # 3 labels (removed 0=Medical material)
    "pleura": [12],  # 1 label
}

DEFAULT_GRAPH_PATH = Path("/workspace/CTDoc/outputs12-14/graphs/ct_rate_condprob.pt")


def _reorder_square_matrix(
    matrix: torch.Tensor,
    *,
    src_names: list[str],
    dst_names: list[str],
) -> torch.Tensor:
    """Reorder a square matrix from src_names order to dst_names order."""
    src_index = {n: i for i, n in enumerate(src_names)}
    idx = []
    missing = []
    for n in dst_names:
        i = src_index.get(n, None)
        if i is None:
            missing.append(n)
        else:
            idx.append(i)
    if missing:
        raise ValueError(f"Graph label_names missing labels: {missing}")
    idx_t = torch.tensor(idx, dtype=torch.long)
    return matrix.index_select(0, idx_t).index_select(1, idx_t)


def load_graph_matrix_and_weights(
    graph_path: Path,
    *,
    label_cols: list[str],
    device: torch.device,
    col_weight_beta: float,
    col_weight_min: float,
    col_weight_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load M and optional per-label weights aligned to label_cols order."""
    ckpt = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_names = list(ckpt["label_names"])
    M = ckpt["prob_matrix"].float()
    if list(label_cols) != graph_names:
        M = _reorder_square_matrix(M, src_names=graph_names, dst_names=list(label_cols))

    M = M.to(device=device, dtype=torch.float32)
    M.fill_diagonal_(0)

    # Asymmetric loss weighting from incoming edge mass (column sums).
    # Higher column sum -> more common -> lower weight.
    col_sum = M.sum(dim=0)  # [18]
    eps = 1e-8
    beta = float(col_weight_beta)
    if beta <= 0:
        weights = torch.ones_like(col_sum)
    else:
        weights = (col_sum.mean() / (col_sum + eps)).pow(beta)
        weights = weights / (weights.mean() + eps)
        weights = weights.clamp(min=float(col_weight_min), max=float(col_weight_max))
    return M, weights


class GlobalEmbeddingDataset(Dataset):
    """Global embedding + organ-specific labels dataset."""

    def __init__(
        self,
        embed_dir: Path,
        label_csv: Path = None,
        use_full_tokens: bool = False,
        *,
        verbose: bool = True,
    ):
        self.embed_dir = Path(embed_dir)
        self.files = sorted(self.embed_dir.glob("*.pt"))
        self.use_full_tokens = use_full_tokens  # True: return [32, dim], False: return [dim] (mean pooled)
        if not self.files:
            raise ValueError(f"No .pt files in {embed_dir}")
        self.verbose = verbose

        # Load label CSV if provided
        self.labels_df = None
        self.label_cols = []
        if label_csv and Path(label_csv).exists():
            import pandas as pd
            self.labels_df = pd.read_csv(label_csv)
            if 'VolumeName' in self.labels_df.columns:
                self.labels_df = self.labels_df.set_index('VolumeName')
            self.label_cols = [c for c in self.labels_df.columns if c not in ['VolumeName', 'split']][:18]
            if self.verbose:
                print(f"[GlobalEmbeddingDataset] Label columns: {self.label_cols}")

        if self.verbose:
            print(f"[GlobalEmbeddingDataset] Loaded {len(self.files)} samples from {embed_dir} (full_tokens={use_full_tokens})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)

        # Load embedding - handle both dict and tensor formats
        if isinstance(data, torch.Tensor):
            embed = data
            volume_name = self.files[idx].stem
        elif isinstance(data, dict):
            if "visual_embeds" in data:
                embed = data["visual_embeds"]
            elif "embedding" in data:
                embed = data["embedding"]
            else:
                raise KeyError(f"No embedding found in {self.files[idx]}")
            volume_name = data.get("volume_name", self.files[idx].stem)
        else:
            raise TypeError(f"Unexpected data type: {type(data)}")

        embed = embed.float()

        # Handle different shapes
        if self.use_full_tokens:
            # Return full [num_tokens, dim] for attention-based model
            if embed.dim() == 1:
                embed = embed.unsqueeze(0)  # [dim] -> [1, dim]
            # embed is [num_tokens, dim]
        else:
            # Mean pool to [dim] for MLP-based model
            if embed.dim() == 2:
                embed = embed.mean(dim=0)
            # embed is [dim]

        # Label loading logic - try multiple name formats
        labels = torch.zeros(18, dtype=torch.float32)

        if self.labels_df is not None and self.label_cols:
            row = None

            # Try different name formats
            name_variants = [
                volume_name,                                          # As-is
                f"{volume_name}.nii.gz",                             # Add .nii.gz suffix
                str(volume_name).replace(".nii.gz", "").replace(".nii", ""),  # Remove suffix
            ]

            for name in name_variants:
                if name in self.labels_df.index:
                    row = self.labels_df.loc[name]
                    break

            if row is not None:
                for i, col in enumerate(self.label_cols):
                    if i < 18 and col in row.index:
                        labels[i] = float(row[col])
            

        return {
            "embedding": embed,
            "labels": labels,
            "volume_name": volume_name,
        }


def collate_global(batch):
    embeddings = torch.stack([b["embedding"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"embedding": embeddings, "labels": labels}


class OrganClassifierModel(nn.Module):
    """Multi-head classifier: one head per organ."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
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

    def forward(self, x):
        """x: [B, input_dim] -> {organ: [B, num_classes]}"""
        # [FIX] Ensure input dtype matches the model weights (e.g. bfloat16)
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        outputs = {}
        for organ, head in self.heads.items():
            outputs[organ] = head(x)
        return outputs

    def extract_organ_feature(self, x: torch.Tensor, organ: str) -> torch.Tensor:
        """Extract a per-organ embedding before the final classification layer."""
        if organ not in self.heads:
            raise ValueError(f"Unknown organ: {organ}. Available: {list(self.heads.keys())}")
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        head = self.heads[organ]
        # head: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
        return head[3](head[2](head[1](head[0](x))))

    def extract_all_organ_features(self, x: torch.Tensor) -> dict:
        """Extract per-organ embeddings for every organ head."""
        return {organ: self.extract_organ_feature(x, organ) for organ in self.heads}


class OrganAttentionClassifier(nn.Module):
    """
    Attention-based classifier that uses learnable queries per organ to aggregate
    information from multiple visual tokens before classification.

    Instead of mean pooling, each organ has its own learnable query that attends
    to the 32 visual tokens to extract organ-relevant features.

    Supports Visual Prompt Tuning:
    - additive: Learnable prompt added to each visual token
    - prefix: Learnable prefix tokens prepended to visual sequence
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_organ_tokens: int = 1,
        query_interaction: bool = False,
        query_interaction_graph_path: Path | None = None,
        query_interaction_gate_init: float = -3.0,
        # Visual Prompt Tuning options
        prompt_type: str = "none",  # "none", "additive", "prefix"
        num_prompt_tokens: int = 8,  # For prefix prompt
        num_visual_tokens: int = 32,  # Number of visual tokens (for additive)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prompt_type = prompt_type
        self.num_prompt_tokens = num_prompt_tokens
        self.num_organ_tokens = int(num_organ_tokens)
        if self.num_organ_tokens < 1:
            raise ValueError(f"num_organ_tokens must be >= 1, got {self.num_organ_tokens}")
        self.organs = list(ORGAN_LABEL_MAPPING.keys())

        # Visual Prompt Tuning
        if prompt_type == "additive":
            # Learnable prompt added to each of the 32 visual tokens
            self.visual_prompt = nn.Parameter(torch.zeros(1, num_visual_tokens, input_dim))
            nn.init.normal_(self.visual_prompt, std=0.02)
        elif prompt_type == "prefix":
            # Learnable prefix tokens prepended to visual sequence
            self.prefix_tokens = nn.Parameter(torch.randn(1, num_prompt_tokens, input_dim) * 0.02)

        # Project input to hidden dim for attention
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable query tokens per organ
        self.organ_queries = nn.ParameterDict()
        for organ in self.organs:
            # Each organ has N learnable query tokens
            self.organ_queries[organ] = nn.Parameter(torch.randn(1, self.num_organ_tokens, hidden_dim))

        # [NEW] Graph-Guided Query Interaction (G-QIA) among query tokens before cross-attention.
        self.query_interaction = None
        if query_interaction:
            graph_path = query_interaction_graph_path or DEFAULT_GRAPH_PATH
            if graph_path is None or not Path(graph_path).exists():
                raise FileNotFoundError(
                    f"query_interaction enabled but graph not found: {graph_path}. "
                    f"Pass --query-interaction-graph-path."
                )
            disease_M = load_disease_graph_matrix(graph_path)
            organ_graph = build_organ_graph_from_disease_matrix(disease_M, organ_label_mapping=ORGAN_LABEL_MAPPING)
            attn_bias = build_query_attn_bias(organ_graph, num_organ_tokens=self.num_organ_tokens)
            self.query_interaction = GraphGuidedQueryInteraction(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attn_bias=attn_bias,
                gate_init=query_interaction_gate_init,
            )

        # Shared cross-attention layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Classification heads per organ
        self.heads = nn.ModuleDict()
        for organ, indices in ORGAN_LABEL_MAPPING.items():
            num_classes = len(indices)
            self.heads[organ] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        self.organ_indices = ORGAN_LABEL_MAPPING

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, num_tokens, input_dim] - visual tokens (e.g., [B, 32, 4096])

        Returns:
            {organ: [B, num_classes]} - logits per organ
        """
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        B = x.size(0)

        # Apply Visual Prompt Tuning
        if self.prompt_type == "additive":
            # Add learnable prompt to each visual token
            prompt = self.visual_prompt.expand(B, -1, -1)
            # Handle case where num tokens doesn't match
            if x.size(1) == prompt.size(1):
                x = x + prompt
            else:
                # Interpolate or truncate prompt if sizes don't match
                x = x + prompt[:, :x.size(1), :]
        elif self.prompt_type == "prefix":
            # Prepend learnable prefix tokens
            prefix = self.prefix_tokens.expand(B, -1, -1)
            x = torch.cat([prefix, x], dim=1)  # [B, num_prefix + num_tokens, dim]

        # Project to hidden dim: [B, num_tokens, hidden_dim]
        memory = self.input_proj(x)

        # Stack queries across organs, attend to image tokens (Check First),
        # then allow graph-guided interaction between query states (Chat Later).
        q_all = torch.cat([self.organ_queries[o].expand(B, -1, -1) for o in self.organs], dim=1)  # [B, O*T, H]
        visual_ctx = self.cross_attention(q_all, memory)  # [B, O*T, H]
        q_ctx = q_all + visual_ctx
        if self.query_interaction is not None:
            q_ctx = self.query_interaction(q_ctx)
        q_ctx = q_ctx.view(B, len(self.organs), self.num_organ_tokens, self.hidden_dim).mean(dim=2)

        outputs = {}
        for idx, organ in enumerate(self.organs):
            outputs[organ] = self.heads[organ](q_ctx[:, idx, :])

        return outputs

    def _get_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Get projected memory from input tokens (with prompt if applicable)."""
        target_dtype = next(self.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)

        B = x.size(0)

        # Apply Visual Prompt Tuning
        if self.prompt_type == "additive":
            prompt = self.visual_prompt.expand(B, -1, -1)
            if x.size(1) == prompt.size(1):
                x = x + prompt
            else:
                x = x + prompt[:, :x.size(1), :]
        elif self.prompt_type == "prefix":
            prefix = self.prefix_tokens.expand(B, -1, -1)
            x = torch.cat([prefix, x], dim=1)

        return self.input_proj(x)

    def extract_organ_feature(self, x: torch.Tensor, organ: str) -> torch.Tensor:
        """Extract feature from a specific organ's cross-attention output.

        Args:
            x: [B, num_tokens, input_dim] or [B, input_dim] input embedding
            organ: Organ name (lung, heart, mediastinum, pleura, trachea and bronchie)

        Returns: [B, hidden_dim] feature vector for the specified organ
        """
        if organ not in self.organ_queries:
            raise ValueError(f"Unknown organ: {organ}. Available: {list(self.organ_queries.keys())}")

        # Handle 2D input (mean-pooled) by unsqueezing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, input_dim]

        B = x.size(0)
        memory = self._get_memory(x)

        q_all = torch.cat([self.organ_queries[o].expand(B, -1, -1) for o in self.organs], dim=1)
        visual_ctx = self.cross_attention(q_all, memory)
        q_ctx = q_all + visual_ctx
        if self.query_interaction is not None:
            q_ctx = self.query_interaction(q_ctx)
        q_ctx = q_ctx.view(B, len(self.organs), self.num_organ_tokens, self.hidden_dim).mean(dim=2)
        return q_ctx[:, self.organs.index(organ), :]

    def extract_all_organ_features(self, x: torch.Tensor) -> dict:
        """Extract features from all organ heads.

        Args:
            x: [B, num_tokens, input_dim] or [B, input_dim] input embedding

        Returns: Dict mapping organ name to [B, hidden_dim] feature vector
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B = x.size(0)
        memory = self._get_memory(x)

        features = {}
        q_all = torch.cat([self.organ_queries[o].expand(B, -1, -1) for o in self.organs], dim=1)
        visual_ctx = self.cross_attention(q_all, memory)
        q_ctx = q_all + visual_ctx
        if self.query_interaction is not None:
            q_ctx = self.query_interaction(q_ctx)
        q_ctx = q_ctx.view(B, len(self.organs), self.num_organ_tokens, self.hidden_dim).mean(dim=2)
        for idx, organ in enumerate(self.organs):
            features[organ] = q_ctx[:, idx, :]
        return features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--embeddings-dir", type=Path, required=False)
    parser.add_argument("--val-embeddings-dir", type=Path, default=None)
    parser.add_argument("--label-csv", type=Path, default=None)
    parser.add_argument("--val-label-csv", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=Path("outputs12-14/organ_classifier.pt"))
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-freq", type=int, default=1)

    # Attention-based classifier options
    parser.add_argument("--model-type", type=str, default="mlp",
                        choices=["mlp", "cross_attention"],
                        help="Model architecture: mlp (mean pool), cross_attention (organ queries)")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for attention models")
    parser.add_argument("--num-organ-tokens", type=int, default=1,
                        help="Number of learnable query tokens per organ (cross-attention)")
    parser.add_argument("--query-interaction", action="store_true",
                        help="Enable graph-guided query interaction before cross-attention")
    parser.add_argument("--query-interaction-graph-path", type=Path, default=None,
                        help="Graph checkpoint path for query interaction (defaults to outputs12-14 graph)")
    parser.add_argument("--query-interaction-gate-init", type=float, default=0.0,
                        help="Initial gate value (sigmoid(gate) scales interaction residual)")

    # Graph-guided training (optional)
    parser.add_argument("--graph-path", type=Path, default=None,
                        help="Path to conditional probability matrix checkpoint (.pt)")
    parser.add_argument("--graph-soft-alpha", type=float, default=0.0,
                        help="Soft target mixing ratio alpha (0 disables graph-guided label smoothing)")
    parser.add_argument("--graph-colweight-beta", type=float, default=0.0,
                        help="Asymmetric loss weighting strength beta from column-sum(M) (0 disables)")
    parser.add_argument("--graph-colweight-min", type=float, default=0.5)
    parser.add_argument("--graph-colweight-max", type=float, default=2.0)

    # Visual Prompt Tuning options
    parser.add_argument("--prompt-type", type=str, default="none",
                        choices=["none", "additive", "prefix"],
                        help="Visual prompt tuning: none, additive (add to tokens), prefix (prepend tokens)")
    parser.add_argument("--num-prompt-tokens", type=int, default=8,
                        help="Number of prefix tokens (for prefix prompt type)")
    parser.add_argument("--num-visual-tokens", type=int, default=32,
                        help="Number of visual tokens (for additive prompt type)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (lower = higher recall)")

    args = parser.parse_args()

    if args.config is not None:
        with args.config.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                if isinstance(v, str) and (k.endswith("_dir") or k.endswith("_csv") or k == "save_path"):
                    v = Path(v) if v else None
                setattr(args, k, v)

    if isinstance(args.lr, str): args.lr = float(args.lr)
    if isinstance(args.weight_decay, str): args.weight_decay = float(args.weight_decay)
    if isinstance(args.input_dim, str): args.input_dim = int(args.input_dim)
    if isinstance(args.hidden_dim, str): args.hidden_dim = int(args.hidden_dim)
    if isinstance(args.batch_size, str): args.batch_size = int(args.batch_size)
    if isinstance(args.epochs, str): args.epochs = int(args.epochs)
    if isinstance(args.num_heads, str): args.num_heads = int(args.num_heads)
    if isinstance(args.num_layers, str): args.num_layers = int(args.num_layers)
    if isinstance(args.dropout, str): args.dropout = float(args.dropout)
    if isinstance(args.num_organ_tokens, str): args.num_organ_tokens = int(args.num_organ_tokens)
    if isinstance(args.query_interaction_gate_init, str): args.query_interaction_gate_init = float(args.query_interaction_gate_init)
    if isinstance(args.num_prompt_tokens, str): args.num_prompt_tokens = int(args.num_prompt_tokens)
    if isinstance(args.num_visual_tokens, str): args.num_visual_tokens = int(args.num_visual_tokens)
    if isinstance(args.threshold, str): args.threshold = float(args.threshold)
    if isinstance(args.graph_soft_alpha, str): args.graph_soft_alpha = float(args.graph_soft_alpha)
    if isinstance(args.graph_colweight_beta, str): args.graph_colweight_beta = float(args.graph_colweight_beta)
    if isinstance(args.graph_colweight_min, str): args.graph_colweight_min = float(args.graph_colweight_min)
    if isinstance(args.graph_colweight_max, str): args.graph_colweight_max = float(args.graph_colweight_max)

    return args


def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute classification metrics.

    Args:
        preds: [N, num_labels] - predicted probabilities
        targets: [N, num_labels] - ground truth binary labels
        threshold: threshold for binary prediction

    Returns:
        dict with metrics including:
        - macro_f1: Macro-averaged F1 (average F1 across all classes)
        - auprc: Area Under Precision-Recall Curve (mean AP)
        - auroc: Area Under ROC Curve
        - micro_f1: Micro-averaged F1
        - sample_f1: Sample-averaged F1 (legacy)
    """
    preds_np = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    preds_binary = (preds_np > threshold).astype(float)

    # === Macro-F1 (Main Metric) ===
    # Per-class F1, then average
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        targets_np, preds_binary, average="macro", zero_division=0
    )

    # === Micro-F1 ===
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        targets_np, preds_binary, average="micro", zero_division=0
    )

    # === AUPRC (Mean Average Precision) - Robustness Metric ===
    # Per-class AP, then average (handles class imbalance well)
    ap_scores = []
    auc_scores = []
    for i in range(targets_np.shape[1]):
        # Need both positive and negative samples for AP/AUC
        if len(np.unique(targets_np[:, i])) > 1:
            try:
                ap = average_precision_score(targets_np[:, i], preds_np[:, i])
                auc = roc_auc_score(targets_np[:, i], preds_np[:, i])
                ap_scores.append(ap)
                auc_scores.append(auc)
            except Exception:
                pass

    auprc = float(np.mean(ap_scores)) if ap_scores else 0.0
    auroc = float(np.mean(auc_scores)) if auc_scores else 0.0

    # === Sample-level F1 (legacy, for backward compatibility) ===
    preds_binary_t = torch.from_numpy(preds_binary)
    targets_t = torch.from_numpy(targets_np)
    tp = (preds_binary_t * targets_t).sum(dim=1)
    fp = (preds_binary_t * (1 - targets_t)).sum(dim=1)
    fn = ((1 - preds_binary_t) * targets_t).sum(dim=1)

    precision_sample = tp / (tp + fp + 1e-8)
    recall_sample = tp / (tp + fn + 1e-8)
    f1_sample = 2 * precision_sample * recall_sample / (precision_sample + recall_sample + 1e-8)

    has_pos = targets_t.sum(dim=1) > 0
    sample_f1 = f1_sample[has_pos].mean().item() if has_pos.any() else 0.0

    return {
        # Main metrics (use these for model selection)
        "macro_f1": float(macro_f1),
        "auprc": auprc,  # mAP
        "auroc": auroc,
        # Additional metrics
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        # Legacy (sample-level)
        "precision": precision_sample[has_pos].mean().item() if has_pos.any() else 0.0,
        "recall": recall_sample[has_pos].mean().item() if has_pos.any() else 0.0,
        "f1": sample_f1,  # legacy sample F1
    }


def compute_per_label_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_names: list[str],
    threshold: float = 0.5,
) -> dict:
    """Compute per-label F1/Recall/AUROC for multi-label predictions.

    Args:
        preds: [N, C] probabilities in [0, 1]
        targets: [N, C] binary targets in {0, 1}
    """
    preds_np = preds.detach().cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
    targets_np = targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
    preds_bin = (preds_np > threshold).astype(np.float32)

    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    _, recalls, f1s, _ = precision_recall_fscore_support(
        targets_np, preds_bin, average=None, zero_division=0
    )

    aurocs = []
    for i in range(targets_np.shape[1]):
        y = targets_np[:, i]
        if len(np.unique(y)) < 2:
            aurocs.append(float("nan"))
            continue
        try:
            aurocs.append(float(roc_auc_score(y, preds_np[:, i])))
        except Exception:
            aurocs.append(float("nan"))

    rows = []
    for i, name in enumerate(label_names):
        rows.append(
            {
                "index": i,
                "label": name,
                "f1": float(f1s[i]) if i < len(f1s) else float("nan"),
                "recall": float(recalls[i]) if i < len(recalls) else float("nan"),
                "auroc": float(aurocs[i]) if i < len(aurocs) else float("nan"),
                "support_pos": int(targets_np[:, i].sum()) if targets_np.size else 0,
            }
        )

    return {"per_label": rows}


def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Determine if we need full tokens (for attention models)
    use_full_tokens = args.model_type == "cross_attention"

    train_dataset = GlobalEmbeddingDataset(
        args.embeddings_dir,
        args.label_csv,
        use_full_tokens=use_full_tokens,
        verbose=accelerator.is_main_process,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_global,
        drop_last=True,
    )

    val_loader = None
    if args.val_embeddings_dir:
        val_dataset = GlobalEmbeddingDataset(
            args.val_embeddings_dir,
            args.val_label_csv or args.label_csv,
            use_full_tokens=use_full_tokens,
            verbose=accelerator.is_main_process,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_global,
        )

    graph_path = args.graph_path or (DEFAULT_GRAPH_PATH if DEFAULT_GRAPH_PATH.exists() else None)
    graph_M = None
    graph_label_weights = None
    if graph_path is not None and Path(graph_path).exists():
        if args.graph_soft_alpha > 0.0 or args.graph_colweight_beta > 0.0:
            graph_M, graph_label_weights = load_graph_matrix_and_weights(
                Path(graph_path),
                label_cols=train_dataset.label_cols,
                device=accelerator.device,
                col_weight_beta=args.graph_colweight_beta,
                col_weight_min=args.graph_colweight_min,
                col_weight_max=args.graph_colweight_max,
            )
            accelerator.print(
                f"[Graph] Enabled: path={graph_path} | soft_alpha={args.graph_soft_alpha} | "
                f"colweight_beta={args.graph_colweight_beta}"
            )

    # Create model based on model_type
    if args.model_type == "mlp":
        model = OrganClassifierModel(args.input_dim, args.hidden_dim)
        accelerator.print(f"📦 Using MLP Classifier (mean pooling)")
    elif args.model_type == "cross_attention":
        prompt_type = getattr(args, "prompt_type", "none")
        model = OrganAttentionClassifier(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_organ_tokens=getattr(args, "num_organ_tokens", 1),
            query_interaction=bool(getattr(args, "query_interaction", False)),
            query_interaction_graph_path=getattr(args, "query_interaction_graph_path", None),
            query_interaction_gate_init=float(getattr(args, "query_interaction_gate_init", 0.0)),
            prompt_type=prompt_type,
            num_prompt_tokens=args.num_prompt_tokens,
            num_visual_tokens=args.num_visual_tokens,
        )
        prompt_info = f" + {prompt_type} prompt" if prompt_type != "none" else ""
        accelerator.print(f"📦 Using Cross-Attention Classifier (organ queries){prompt_info}")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if val_loader:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )

    best_val_f1 = 0.0

    accelerator.print(f"🚀 Training Organ Classifier | BS={args.batch_size} | LR={args.lr} | Threshold={args.threshold}")
    accelerator.print(f"   Organs: {list(ORGAN_LABEL_MAPPING.keys())}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Ep {epoch}")
        for batch in pbar:
            embeddings = batch["embedding"]
            labels = batch["labels"]

            labels_hard = (labels > 0.5).float()
            soft_targets = labels_hard
            if graph_M is not None and args.graph_soft_alpha > 0.0:
                inferred = labels_hard @ graph_M  # [B, 18]
                soft_targets = ((1.0 - args.graph_soft_alpha) * labels_hard) + (args.graph_soft_alpha * inferred)
                soft_targets = soft_targets.clamp(0.0, 1.0)

            # [Auto Cast] Logic handled inside Model.forward() now, but explicit check here is also fine
            outputs = model(embeddings)

            loss = torch.tensor(0.0, device=accelerator.device)
            for organ, logits in outputs.items():
                indices = ORGAN_LABEL_MAPPING[organ]
                organ_labels_hard = labels_hard[:, indices]
                organ_targets = soft_targets[:, indices]

                pos_weight = (organ_labels_hard.shape[0] - organ_labels_hard.sum(dim=0)) / (organ_labels_hard.sum(dim=0) + 1)
                pos_weight = pos_weight.clamp(max=10.0)

                # Ensure pos_weight matches logits dtype (bfloat16)
                if pos_weight.dtype != logits.dtype:
                    pos_weight = pos_weight.to(dtype=logits.dtype)

                loss_mat = F.binary_cross_entropy_with_logits(
                    logits,
                    organ_targets.to(dtype=logits.dtype),
                    pos_weight=pos_weight,
                    reduction="none",
                )
                if graph_label_weights is not None and args.graph_colweight_beta > 0.0:
                    w = graph_label_weights[indices].to(device=loss_mat.device, dtype=loss_mat.dtype)  # [C_org]
                    loss_mat = loss_mat * w.unsqueeze(0)
                loss = loss + loss_mat.mean()

            loss = loss / len(outputs)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            total_loss += accelerator.reduce(loss.detach(), reduction="mean").item()
            steps += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / max(1, steps)
        accelerator.print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        if val_loader and epoch % args.eval_freq == 0:
            model.eval()
            all_preds = {organ: [] for organ in ORGAN_LABEL_MAPPING}
            all_targets = {organ: [] for organ in ORGAN_LABEL_MAPPING}
            val_total_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    embeddings = batch["embedding"]
                    labels = batch["labels"]
                    labels_hard = (labels > 0.5).float()
                    soft_targets = labels_hard
                    if graph_M is not None and args.graph_soft_alpha > 0.0:
                        inferred = labels_hard @ graph_M
                        soft_targets = ((1.0 - args.graph_soft_alpha) * labels_hard) + (args.graph_soft_alpha * inferred)
                        soft_targets = soft_targets.clamp(0.0, 1.0)

                    # Forward (auto-casting inside model)
                    outputs = model(embeddings)

                    # Calculate validation loss
                    batch_loss = torch.tensor(0.0, device=accelerator.device)
                    for organ, logits in outputs.items():
                        indices = ORGAN_LABEL_MAPPING[organ]
                        organ_labels_hard = labels_hard[:, indices]
                        organ_targets = soft_targets[:, indices]
                        probs = torch.sigmoid(logits)

                        # Loss calculation
                        pos_weight = (organ_labels_hard.shape[0] - organ_labels_hard.sum(dim=0)) / (organ_labels_hard.sum(dim=0) + 1)
                        pos_weight = pos_weight.clamp(max=10.0)
                        if pos_weight.dtype != logits.dtype:
                            pos_weight = pos_weight.to(dtype=logits.dtype)
                        loss_mat = F.binary_cross_entropy_with_logits(
                            logits,
                            organ_targets.to(dtype=logits.dtype),
                            pos_weight=pos_weight,
                            reduction="none",
                        )
                        if graph_label_weights is not None and args.graph_colweight_beta > 0.0:
                            w = graph_label_weights[indices].to(device=loss_mat.device, dtype=loss_mat.dtype)
                            loss_mat = loss_mat * w.unsqueeze(0)
                        batch_loss = batch_loss + loss_mat.mean()

                        all_preds[organ].append(probs.detach())
                        all_targets[organ].append(organ_labels_hard.detach())

                    batch_loss = batch_loss / len(outputs)
                    val_total_loss += accelerator.reduce(batch_loss.detach(), reduction="mean").item()
                    val_steps += 1

            avg_val_loss = val_total_loss / max(1, val_steps)
            accelerator.print(f"  --> Val Loss: {avg_val_loss:.4f}")
            gathered_preds = {}
            gathered_targets = {}
            for organ, indices in ORGAN_LABEL_MAPPING.items():
                num_classes = len(indices)
                preds_local = (
                    torch.cat(all_preds[organ], dim=0)
                    if all_preds[organ]
                    else torch.empty((0, num_classes), device=accelerator.device)
                )
                targets_local = (
                    torch.cat(all_targets[organ], dim=0)
                    if all_targets[organ]
                    else torch.empty((0, num_classes), device=accelerator.device)
                )

                preds_g = accelerator.gather_for_metrics(preds_local).detach().float().cpu()
                targets_g = accelerator.gather_for_metrics(targets_local).detach().float().cpu()
                gathered_preds[organ] = preds_g
                gathered_targets[organ] = targets_g

            avg_macro_f1 = 0.0
            avg_auprc = 0.0
            avg_auroc = 0.0
            avg_recall = 0.0
            if accelerator.is_main_process:
                # Full 18-label view
                label_names = train_dataset.label_cols if train_dataset.label_cols else [f"Label_{i}" for i in range(18)]
                N = 0
                for organ in ORGAN_LABEL_MAPPING:
                    N = max(N, int(gathered_preds[organ].shape[0]))
                full_preds = torch.zeros((N, 18), dtype=torch.float32)
                full_targets = torch.zeros((N, 18), dtype=torch.float32)
                for organ, idxs in ORGAN_LABEL_MAPPING.items():
                    probs = gathered_preds[organ]
                    t = gathered_targets[organ]
                    for local_j, label_idx in enumerate(idxs):
                        if 0 <= label_idx < 18 and local_j < probs.shape[1]:
                            full_preds[:, label_idx] = probs[:, local_j]
                            full_targets[:, label_idx] = t[:, local_j]

                total_macro_f1 = 0.0
                total_auprc = 0.0
                total_auroc = 0.0
                total_recall = 0.0
                for organ in ORGAN_LABEL_MAPPING:
                    metrics = compute_metrics(
                        gathered_preds[organ],
                        gathered_targets[organ],
                        threshold=args.threshold,
                    )
                    accelerator.print(
                        f"    [{organ}] Macro-F1: {metrics['macro_f1']:.4f} | "
                        f"Recall: {metrics['macro_recall']:.4f} | AUPRC: {metrics['auprc']:.4f}"
                    )
                    total_macro_f1 += metrics["macro_f1"]
                    total_auprc += metrics["auprc"]
                    total_auroc += metrics["auroc"]
                    total_recall += metrics["macro_recall"]

                avg_macro_f1 = total_macro_f1 / len(ORGAN_LABEL_MAPPING)
                avg_auprc = total_auprc / len(ORGAN_LABEL_MAPPING)
                avg_auroc = total_auroc / len(ORGAN_LABEL_MAPPING)
                avg_recall = total_recall / len(ORGAN_LABEL_MAPPING)
                accelerator.print(
                    f"    [Average] Macro-F1: {avg_macro_f1:.4f} | "
                    f"Recall: {avg_recall:.4f} | AUPRC: {avg_auprc:.4f}"
                )

                # Print per-label metrics (18 labels)
                per_label = compute_per_label_metrics(
                    full_preds,
                    full_targets,
                    label_names=label_names,
                    threshold=args.threshold,
                )["per_label"]
                accelerator.print("    [Per-Label] F1 / Recall / AUROC")
                for row in per_label:
                    auroc = row["auroc"]
                    auroc_str = f"{auroc:.4f}" if np.isfinite(auroc) else "nan"
                    accelerator.print(
                        f"      [{row['index']:02d}] {row['label']}: "
                        f"F1={row['f1']:.4f} | R={row['recall']:.4f} | AUROC={auroc_str} | Pos={row['support_pos']}"
                    )

            avg_macro_f1 = accelerator.reduce(
                torch.tensor(avg_macro_f1, device=accelerator.device),
                reduction="sum",
            ).item()
            avg_auprc = accelerator.reduce(
                torch.tensor(avg_auprc, device=accelerator.device),
                reduction="sum",
            ).item()
            avg_auroc = accelerator.reduce(
                torch.tensor(avg_auroc, device=accelerator.device),
                reduction="sum",
            ).item()
            avg_recall = accelerator.reduce(
                torch.tensor(avg_recall, device=accelerator.device),
                reduction="sum",
            ).item()

            # Use Macro-F1 as the main metric for model selection
            if avg_macro_f1 > best_val_f1:
                best_val_f1 = avg_macro_f1
                if accelerator.is_main_process:
                    args.save_path.parent.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save({
                        "model": unwrapped.state_dict(),
                        "config": vars(args),
                        "model_type": args.model_type,
                        "best_macro_f1": best_val_f1,
                        "best_recall": avg_recall,
                        "best_auprc": avg_auprc,
                        "best_auroc": avg_auroc,
                    }, args.save_path)
                    accelerator.print(f"  --> Saved Best Model (F1: {best_val_f1:.4f}, Recall: {avg_recall:.4f}) [{args.model_type}]")

    accelerator.print(f"✅ Training Complete! Best Macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
