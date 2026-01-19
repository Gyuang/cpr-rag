#!/usr/bin/env python3
"""Train VLM decoder with RAG retrieval (supports Organ Disease Classification based retrieval)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate import DistributedType
from accelerate.utils import DistributedDataParallelKwargs, set_seed
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# avoid shadowing by train/ dir
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPT_DIR]

from utils.radiology_eval import compute_text_metrics

from train.dataset_rag import build_rag_dataloader
from models.vlm_factory import DecoderConfig as VLMDecoderConfig
from models.vlm_factory import VisionBackboneConfig, build_vlm_model
from models.classifiers.organ import OrganMLPOnlyModel
from rag.retriever import OrganFaissRetriever, OrganClassifierRetriever
from rag.unified_retriever import create_retriever, UnifiedRetrieverConfig

RESULTS_ROOT = Path("/research/04-CT/00-RawData/models/results")
DEFAULT_RAG_RAW_TOKEN_DIR = Path("/workspace/CTDoc/outputs/dataset/radfm_raw_embeddings_bf16")

DEFAULT_KEYWORD_LOSS_KEYWORDS: tuple[str, ...] = (
    "nodule",
    "nodules",
    "mass",
    "masses",
    "effusion",
    "pleural",
    "pericardial",
    "pneumonia",
    "infiltrate",
    "infiltration",
    "consolidation",
    "atelectasis",
    "emphysema",
    "pneumothorax",
    "edema",
)


def _load_config_with_bases(config_path: Path) -> dict:
    """Load a YAML config and recursively merge base configs."""
    def _merge_dicts(base: dict, overrides: dict) -> dict:
        merged = dict(base)
        merged.update(overrides)
        return merged

    def _load(path: Path, stack: set[Path]) -> dict:
        real = path.resolve()
        if real in stack:
            raise ValueError(f"Cyclic base config include detected: {real}")
        stack.add(real)
        with real.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config at {real} must be a mapping.")
        base_field = None
        for key in ("base_configs", "base_config", "base"):
            if key in raw:
                base_field = raw.pop(key)
                break
        merged: dict = {}
        if base_field:
            bases = base_field if isinstance(base_field, (list, tuple)) else [base_field]
            for base in bases:
                base_path = (real.parent / Path(base)).resolve()
                merged = _merge_dicts(merged, _load(base_path, stack))
        merged = _merge_dicts(merged, raw)
        stack.remove(real)
        return merged

    return _load(config_path, set())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VLM decoder with organ-level RAG retrieval.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config to populate arguments.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--reports", type=Path, default=None)
    parser.add_argument("--val-reports", type=Path, default=None)
    parser.add_argument("--test-reports", type=Path, default=None, help="CSV for test set evaluation at end of training.")
    parser.add_argument("--llama-path", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--dataset-format", choices=["processed_ct_rate", "radgenome"], default="radgenome")
    parser.add_argument("--report-max-tokens", type=int, default=256)
    parser.add_argument("--vision-ckpt", type=str, default=None)
    parser.add_argument("--ct2rep-ckpt", type=str, default=None)
    parser.add_argument("--radfm-ckpt", type=str, default=None)
    parser.add_argument("--m3d-ckpt", type=str, default=None)
    parser.add_argument("--vision-backbone", choices=["ct2rep", "radfm", "m3d"], default="radfm")
    parser.add_argument("--num-visual-tokens", type=int, default=32)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="CTreport")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--eval-text-metrics", action="store_true")
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--eval-min-new-tokens",
        type=int,
        default=16,
        help="Force generation to produce at least N tokens (helps avoid empty outputs when EOS is emitted early).",
    )
    parser.add_argument(
        "--visual-dropout-prob",
        type=float,
        default=0.0,
        help=(
            "Blindfold training: randomly zero-out visual tokens for a fraction of training samples to force "
            "reliance on retrieved context. Applied only when retrieved context is non-empty."
        ),
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--test-only", action="store_true", help="Skip training and only run test evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint to load for test-only mode.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load a checkpoint and run evaluation on a split (val/test), saving predictions + metrics.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help="Which split to run when using --eval-only.",
    )

    # Model Freezing Options (For LoRA-only training)
    parser.add_argument("--freeze-llama", action="store_true", help="Freeze the base Llama model.")
    parser.add_argument("--freeze-vision-backbone", action="store_true", help="Freeze the vision encoder.")
    parser.add_argument("--freeze-projector", action="store_true", help="Freeze the vision projector (perceiver/mlp).")

    # LoRA Options
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # RAG Common Options
    parser.add_argument("--organ-embeddings-dir", type=Path, default=None, help="Precomputed organ embeddings for queries.")
    parser.add_argument("--retrieval-top-k", type=int, default=2)
    parser.add_argument("--retrieval-top-per-organ", type=int, default=1)
    parser.add_argument("--rag-dropout", type=float, default=0.25)
    parser.add_argument("--organ-report-csvs", type=Path, nargs="+", default=None)
    parser.add_argument("--precomputed-vision-dir", type=Path, default=None)
    parser.add_argument("--prompt-style", choices=["llama2", "llama3"], default=None)
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=Path("/research/04-CT/00-RawData/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv"),
    )
    parser.add_argument(
        "--val-label-csv",
        type=Path,
        default=Path("/research/04-CT/00-RawData/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv"),
        help="Label CSV for validation set. If not provided, uses --label-csv.",
    )
    parser.add_argument(
        "--test-label-csv",
        type=Path,
        default=Path("/research/04-CT/00-RawData/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"),
        help="Label CSV for test set. If not provided, uses --label-csv.",
    )

    # Classifier Retrieval Options
    parser.add_argument("--use-classifier-retrieval", action="store_true", help="Use organ classifier for retrieval instead of FAISS.")
    parser.add_argument("--organ-classifier-ckpt", type=Path, default=None, help="Path to trained OrganClassifierModel checkpoint.")
    parser.add_argument("--organ-classifier-index", type=Path, default=None, help="Path to the .pkl index built for classifier retrieval.")
    parser.add_argument("--classifier-threshold", type=float, default=0.5, help="Threshold for classifier predictions.")
    parser.add_argument("--use-oracle", action="store_true", help="Use GT labels for retrieval in test (Oracle mode, for comparison).")
    parser.add_argument(
        "--oracle-exact-label",
        action="store_true",
        help=(
            "Oracle retrieval for RAG: per-organ exact CT-RATE label match from the organ index DB "
            "(ignores embeddings). Label-leakage mode for upper-bound / ablations (can be used for training too)."
        ),
    )
    parser.add_argument(
        "--oracle-abnormal-only",
        action="store_true",
        help="When using --oracle-exact-label, only include retrieved organ snippets for organs that are abnormal in the query (skip all-zero organ signatures).",
    )
    parser.add_argument(
        "--oracle-positive-labels-only",
        action="store_true",
        help=(
            "When using --oracle-exact-label, keep only sentences in the retrieved organ snippets that match "
            "the query's positive CT-RATE labels (1s). This is a heuristic to drop normal/negated sentences."
        ),
    )
    parser.add_argument(
        "--rag-positive-labels-only",
        action="store_true",
        help=(
            "Filter retrieved RAG snippets (non-oracle too) to keep only sentences that match the query's "
            "positive CT-RATE labels (1s) from the split label CSV, dropping normal/negated sentences."
        ),
    )

    # Hierarchical Retriever Options
    parser.add_argument("--hierarchical-binary-ckpt", type=Path, default=None,
                        help="Path to trained binary (Normal/Abnormal) classifier checkpoint")
    parser.add_argument("--hierarchical-multilabel-ckpt", type=Path, default=None,
                        help="Path to trained multi-label organ classifier checkpoint")
    parser.add_argument("--hierarchical-normal-index", type=Path, default=None,
                        help="Path to normal cases index file (.pt)")
    parser.add_argument("--hierarchical-abnormal-dir", type=Path, default=None,
                        help="Directory containing organ-specific abnormal indices")
    parser.add_argument("--hierarchical-normal-threshold", type=float, default=0.5,
                        help="Threshold for normal/abnormal classification")
    parser.add_argument("--hierarchical-graph-path", type=Path, default=None,
                        help="Optional clinical graph (.pt) used to rerank FAISS candidates.")
    parser.add_argument("--hierarchical-graph-alpha", type=float, default=0.3,
                        help="Strength of the graph score boost during reranking.")
    parser.add_argument("--hierarchical-graph-pool-multiplier", type=int, default=5,
                        help="Candidate pool multiplier before graph reranking (pool_k = k * multiplier).")
    parser.add_argument(
        "--hierarchical-graph-pool-size",
        type=int,
        default=None,
        help=(
            "Optional absolute candidate pool size before graph reranking (pool_k = max(pool_size, k)). "
            "When set (>0), this overrides --hierarchical-graph-pool-multiplier to keep the pool fixed across top-k."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for the LLaMA decoder to reduce memory usage."
    )
    prelim, _ = parser.parse_known_args()
    if prelim.config:
        try:
            loaded = _load_config_with_bases(prelim.config)
        except Exception as exc:
            parser.error(str(exc))
        parser.set_defaults(**loaded)

    args = parser.parse_args()

    if getattr(args, "oracle_positive_labels_only", False) and not getattr(args, "oracle_exact_label", False):
        parser.error("--oracle-positive-labels-only requires --oracle-exact-label")
    if getattr(args, "oracle_exact_label", False):
        if getattr(args, "retriever_type", "hierarchical") != "hierarchical":
            parser.error("--oracle-exact-label currently requires --retriever-type hierarchical")
        if not getattr(args, "hierarchical_abnormal_dir", None):
            parser.error("--oracle-exact-label requires --hierarchical-abnormal-dir pointing to *_index.pt files")
    if getattr(args, "visual_dropout_prob", 0.0) < 0.0 or getattr(args, "visual_dropout_prob", 0.0) > 1.0:
        parser.error("--visual-dropout-prob must be within [0, 1].")

    # Set default raw-token embedding directory for RAG if not specified.
    # This directory contains vision-encoder tokens like [1024, 768] saved as .pt tensors.
    if args.organ_embeddings_dir is None:
        args.organ_embeddings_dir = DEFAULT_RAG_RAW_TOKEN_DIR

    # Default save_dir to outputs12-14 for new runs (keep test/eval-only strictness).
    if not args.test_only and not args.eval_only and args.save_dir is None:
        run_stub = f"{args.vision_backbone}_{args.config.stem if args.config else 'manual'}"
        args.save_dir = REPO_ROOT / "outputs12-14" / "experiments" / run_stub

    # Type coercion
    for key in {
        "batch_size",
        "grad_accum_steps",
        "epochs",
        "num_workers",
    }:
        val = getattr(args, key, None)
        if val is not None and not isinstance(val, int):
            setattr(args, key, int(val))
    for key in {"lr", "weight_decay", "adam_eps", "classifier_threshold"}:
        val = getattr(args, key, None)
        if val is not None and not isinstance(val, float):
            setattr(args, key, float(val))

    path_fields = [
        "manifest", "reports", "val_reports", "test_reports", "label_csv",
        "val_label_csv", "test_label_csv", "save_dir",
        "organ_embeddings_dir", "resume_checkpoint", "precomputed_vision_dir",
        "organ_classifier_ckpt", "organ_classifier_index", "checkpoint",
        "hierarchical_binary_ckpt", "hierarchical_multilabel_ckpt",
        "hierarchical_normal_index", "hierarchical_abnormal_dir", "hierarchical_graph_path",
    ]
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))

    # Required fields check
    if args.test_only or args.eval_only:
        # For test/eval-only mode, require a split reports CSV and checkpoint (or save_dir with model_best.pt)
        required_fields = ["manifest", "label_csv", "llama_path", "organ_embeddings_dir"]
        if args.test_only:
            required_fields.append("test_reports")
        else:
            required_fields.append("val_reports" if args.eval_split == "val" else "test_reports")
        missing = [field for field in required_fields if getattr(args, field) is None]
        if missing:
            parser.error(f"Missing required arguments for test-only mode: {', '.join(missing)}")
        if not args.checkpoint and not args.save_dir:
            parser.error("--checkpoint or --save-dir (to find model_best.pt) required for test/eval-only mode")
    else:
        required_fields = ["manifest", "reports", "val_reports", "label_csv", "llama_path", "save_dir", "organ_embeddings_dir"]
        missing = [field for field in required_fields if getattr(args, field) is None]
        if missing:
            parser.error(f"Missing required arguments: {', '.join(missing)}")

    # Retrieval Logic Validation
    if args.retrieval_top_k > 0:
        retriever_type = getattr(args, "retriever_type", "hierarchical")
        if retriever_type == "hierarchical":
            if not args.hierarchical_multilabel_ckpt:
                parser.error("Hierarchical retrieval requires --hierarchical-multilabel-ckpt")
            if not args.hierarchical_abnormal_dir:
                parser.error("Hierarchical retrieval requires --hierarchical-abnormal-dir")

    if args.prompt_style is None:
        lp = (args.llama_path or "").lower()
        args.prompt_style = "llama3" if "llama-3" in lp or "llama3" in lp else "llama2"

    if not args.wandb_run_name:
        args.wandb_run_name = f"{args.vision_backbone}_{Path(args.config).stem}" if args.config else f"{args.vision_backbone}_manual"

    return args


def _normalize_anatomy(anatomy: str) -> Optional[str]:
    name = anatomy.strip().lower()
    if not name: return None
    if "lung" in name: return "lung"
    if "pleur" in name: return "pleura"
    if "mediast" in name: return "mediastinum"
    if "trache" in name or "bronch" in name or "airway" in name: return "trachea and bronchie"
    if "heart" in name or "card" in name or "pericard" in name: return "heart"
    return None


def _load_organ_reports(csv_paths: Sequence[Path]) -> Dict[str, Dict[str, str]]:
    # Prefer top-level anatomy rows like "lung", but fall back to hierarchical ones like "lung/lung".
    buckets: Dict[str, Dict[str, Dict[str, list[str]]]] = {}
    for csv_path in csv_paths:
        if not csv_path or not csv_path.exists(): continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                vol = row.get("Volumename") or row.get("VolumeName")
                sent = (row.get("Sentence") or "").strip()
                anat = (row.get("Anatomy") or "").strip()
                # Some sources include multi-line blocks containing multiple anatomy-prefixed lines
                # (e.g., "heart: ...\nlung: ..."). Select the line matching the current row anatomy
                # instead of blindly taking the first line.
                if "\n" in sent and anat:
                    raw_anat = anat.strip().lower()
                    norm = _normalize_anatomy(anat) or ""
                    tail = raw_anat.split("/")[-1] if "/" in raw_anat else raw_anat
                    candidates = [raw_anat, tail]
                    if norm and norm not in candidates:
                        candidates.append(norm)
                    bracket = f"[{norm.upper()}]:" if norm else ""
                    chosen = ""
                    lines = [ln.strip() for ln in sent.splitlines() if ln.strip()]
                    for ln in lines:
                        low = ln.lower()
                        if bracket and ln.startswith(bracket):
                            chosen = ln[len(bracket) :].strip()
                            break
                        for tag in candidates:
                            if tag and (low.startswith(f"{tag}:") or low.startswith(f"{tag} :")):
                                chosen = ln.split(":", 1)[1].strip()
                                break
                        if chosen:
                            break
                    if not chosen:
                        chosen = lines[0] if lines else ""
                    sent = chosen
                if not vol or not sent:
                    continue
                organ = _normalize_anatomy(anat)
                if not organ: continue
                kind = "primary" if "/" not in anat else "fallback"
                organ_map = buckets.setdefault(vol.strip(), {}).setdefault(organ, {}).setdefault(kind, [])
                if sent not in organ_map:
                    organ_map.append(sent)

    # Keep retrieval context compact: use the top-level anatomy sentence if available.
    out: Dict[str, Dict[str, str]] = {}
    for vol, per_organ in buckets.items():
        out[vol] = {}
        for organ, kinds in per_organ.items():
            sentences = (kinds.get("primary") or []) or (kinds.get("fallback") or [])
            out[vol][organ] = " ".join(sentences[:1]).strip()
    return out


def _extract_prompt_ids(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    tokenizer,
) -> torch.Tensor:
    """Extract per-sample prompt ids from a batch.

    Uses the contiguous -100 prefix in `labels` (prompt region) and ignores
    any -100 values introduced by padding at the end of the sequence.
    """
    bs = input_ids.size(0)
    device = input_ids.device

    prompt_ids_list: List[torch.Tensor] = []
    max_prompt_len = 0

    for b in range(bs):
        seq_len = int(attention_mask[b].sum().item()) if attention_mask is not None else int(input_ids.size(1))
        seq_len = max(1, min(seq_len, int(input_ids.size(1))))
        seq_labels = labels[b, :seq_len]

        non_prompt = (seq_labels != -100).nonzero(as_tuple=False)
        if non_prompt.numel() > 0:
            prompt_len = int(non_prompt[0].item())
        else:
            # If everything is -100 (shouldn't happen for normal supervision),
            # fall back to the full (unpadded) sequence.
            prompt_len = seq_len

        if prompt_len <= 0:
            bos_id = tokenizer.bos_token_id or tokenizer.pad_token_id
            prompt_ids = torch.tensor([bos_id], device=device, dtype=input_ids.dtype)
        else:
            prompt_ids = input_ids[b, :prompt_len]

        prompt_ids_list.append(prompt_ids)
        max_prompt_len = max(max_prompt_len, int(prompt_ids.numel()))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    prompt_batch = torch.full(
        (bs, max_prompt_len),
        pad_id,
        dtype=input_ids.dtype,
        device=device,
    )
    for b, p_ids in enumerate(prompt_ids_list):
        # NOTE: Left-pad for decoder-only generation correctness.
        # HF `generate()` reads the last token in `input_ids` as the decoding start;
        # right-padding would make shorter prompts start decoding from PAD/EOS.
        p_len = int(p_ids.numel())
        start = max_prompt_len - p_len
        prompt_batch[b, start : start + p_len] = p_ids

    return prompt_batch


def main() -> None:
    args = parse_args()
    run_config = vars(args)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)
    
    if accelerator.is_main_process and args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    use_precomputed = args.precomputed_vision_dir is not None

    # 1. Model Building
    vision_cfg = VisionBackboneConfig(
        name=args.vision_backbone,
        num_visual_tokens=args.num_visual_tokens,
        ct2rep_ckpt=args.ct2rep_ckpt or args.vision_ckpt,
        radfm_ckpt=args.radfm_ckpt,
        m3d_ckpt=args.m3d_ckpt,
    )
    decoder_cfg = VLMDecoderConfig(
        llama_path=args.llama_path,
        tokenizer_path=args.tokenizer_path,
        freeze_llama=args.freeze_llama,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = build_vlm_model(vision_cfg, decoder_cfg, use_precomputed_vision=use_precomputed)

    # Get LLaMA reference (handles peft/wrapper)
    llama_ref = getattr(model, "llama_model", model)
    if hasattr(llama_ref, "model"):
        llama_ref = llama_ref.model

    # Disable KV cache (conflicts with gradient checkpointing)
    if hasattr(llama_ref, "config"):
        llama_ref.config.use_cache = False

    if args.gradient_checkpointing:
        if hasattr(llama_ref, "gradient_checkpointing_enable"):
            llama_ref.gradient_checkpointing_enable()
        if hasattr(llama_ref, "enable_input_require_grads"):
            llama_ref.enable_input_require_grads()
        if hasattr(llama_ref, "config"):
            setattr(llama_ref.config, "gradient_checkpointing", True)
            setattr(llama_ref.config, "use_cache", False)

        vision = getattr(model, "vision_backbone", None)
        if vision is not None and hasattr(vision, "gradient_checkpointing_enable"):
            vision.gradient_checkpointing_enable()
    else:
        if hasattr(llama_ref, "config"):
            llama_ref.config.use_cache = False


    # 3. Freezing Strategy
    def _set_vision_trainability(module):
        if module is None: return
        for name, param in module.named_parameters():
            is_projector = name.startswith("projector.") or "perceiver" in name or "aggregator" in name
            if is_projector:
                param.requires_grad_(not args.freeze_projector)
            else:
                param.requires_grad_(not args.freeze_vision_backbone)

    if not use_precomputed:
        _set_vision_trainability(getattr(model, "vision_backbone", None))
    
    if args.resume_checkpoint:
        accelerator.print(f"[resume] Loading weights from {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # 4. Retrieval Setup
    unified_retriever = None
    organ_reports_lookup = {}

    if args.retrieval_top_k > 0:
        if args.organ_report_csvs:
            organ_reports_lookup = _load_organ_reports([Path(p) for p in args.organ_report_csvs])

        accelerator.print(f"[RAG] Using Hierarchical Retrieval.")
        accelerator.print(f"  Binary ckpt: {args.hierarchical_binary_ckpt}")
        accelerator.print(f"  Multilabel ckpt: {args.hierarchical_multilabel_ckpt}")
        accelerator.print(f"  Normal index: {args.hierarchical_normal_index}")
        accelerator.print(f"  Abnormal dir: {args.hierarchical_abnormal_dir}")
        if not args.hierarchical_binary_ckpt or not args.hierarchical_normal_index:
            accelerator.print("  [Hierarchical] Running in organ-only mode (no binary gate / no normal DB).")
        unified_config = UnifiedRetrieverConfig(
            retriever_type="hierarchical",
            device=str(accelerator.device),
            binary_checkpoint=str(args.hierarchical_binary_ckpt) if args.hierarchical_binary_ckpt else None,
            multilabel_checkpoint=str(args.hierarchical_multilabel_ckpt) if args.hierarchical_multilabel_ckpt else None,
            normal_index_path=str(args.hierarchical_normal_index) if args.hierarchical_normal_index else None,
            abnormal_index_dir=str(args.hierarchical_abnormal_dir) if args.hierarchical_abnormal_dir else None,
            normal_threshold=args.hierarchical_normal_threshold,
            graph_path=str(args.hierarchical_graph_path) if args.hierarchical_graph_path else None,
            graph_alpha=float(args.hierarchical_graph_alpha),
            graph_pool_multiplier=int(args.hierarchical_graph_pool_multiplier),
            graph_pool_size=(
                int(args.hierarchical_graph_pool_size)
                if getattr(args, "hierarchical_graph_pool_size", None) not in (None, 0)
                and int(args.hierarchical_graph_pool_size) > 0
                else None
            ),
            top_k=args.retrieval_top_k,
            top_per_organ=args.retrieval_top_per_organ,
        )
        unified_retriever = create_retriever(unified_config)

    tokenizer = model.tokenizer

    def _build_loader(split: str, reports_csv: Path, shuffle: bool, max_samples: Optional[int]):
        base_embeddings_dir = Path(args.organ_embeddings_dir)
        split_embeddings_dir = base_embeddings_dir / split
        if not split_embeddings_dir.exists():
            split_embeddings_dir = base_embeddings_dir

        # Select appropriate label CSV for split
        if split == "test" and args.test_label_csv:
            split_label_csv = args.test_label_csv
        elif split in {"validation", "val"} and args.val_label_csv:
            split_label_csv = args.val_label_csv
        else:
            split_label_csv = args.label_csv

        common_kwargs = dict(
            manifest_path=args.manifest,
            reports_csv=reports_csv,
            tokenizer=tokenizer,
            split=split,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            max_samples=max_samples,
            label_csv=split_label_csv,
        )

        if args.dataset_format == "radgenome":
            return build_rag_dataloader(
                **common_kwargs,
                embeddings_dir=split_embeddings_dir,
                unified_retriever=unified_retriever,
                retriever_type="hierarchical",
                force_classifier=(split == "test" and not getattr(args, "use_oracle", False)),
                oracle_exact_label=bool(getattr(args, "oracle_exact_label", False)),
                oracle_abnormal_only=bool(getattr(args, "oracle_abnormal_only", False)),
                oracle_positive_labels_only=bool(getattr(args, "oracle_positive_labels_only", False)),
                rag_positive_labels_only=bool(getattr(args, "rag_positive_labels_only", False)),
                rag_positive_labels_from_retrieved=bool(getattr(args, "rag_positive_labels_from_retrieved", False)),
                oracle_exact_label_index_dir=str(args.hierarchical_abnormal_dir) if args.hierarchical_abnormal_dir else None,
                top_k=args.retrieval_top_k,
                top_per_organ=args.retrieval_top_per_organ,
                rag_dropout=args.rag_dropout,
                organ_reports=organ_reports_lookup,
                max_tokens=args.report_max_tokens,
                precomputed_vision_dir=args.precomputed_vision_dir,
                prompt_style=args.prompt_style,
            )
        else:
            raise ValueError("RAG loader currently supports radgenome format only.")

    # ============ TEST-ONLY MODE ============
    if args.test_only:
        accelerator.print("\n" + "=" * 60)
        accelerator.print("[TEST-ONLY MODE] Loading checkpoint and running test evaluation...")
        accelerator.print("=" * 60)

        # Load checkpoint
        ckpt_path = args.checkpoint or (args.save_dir / "model_best.pt" if args.save_dir else None)
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)
            accelerator.print(f"Loaded checkpoint from {ckpt_path}")
        else:
            accelerator.print(f"Warning: Checkpoint not found at {ckpt_path}")

        results_dir = args.save_dir or (ckpt_path.parent if ckpt_path else Path.cwd())
        if accelerator.is_main_process:
            results_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        # Build test loader only
        test_loader = _build_loader("test", args.test_reports, False, None)
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            # DeepSpeed ZeRO stage 2 requires an optimizer even for eval-only/test-only in `accelerator.prepare`.
            # Use a tiny dummy optimizer to satisfy the engine without allocating optimizer states for the full model.
            if not hasattr(model, "_ds_eval_dummy_param"):
                model._ds_eval_dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            dummy_opt = torch.optim.AdamW([model._ds_eval_dummy_param], lr=0.0)
            model, test_loader, _ = accelerator.prepare(model, test_loader, dummy_opt)
        else:
            model, test_loader = accelerator.prepare(model, test_loader)

        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)

        test_loss = 0.0
        test_count = 0
        all_preds = []
        all_refs = []
        all_vol_names = []
        all_retrieved = []

        from tqdm import tqdm
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                outputs = model(batch)
                loss_mean = accelerator.reduce(outputs.loss.detach(), reduction="mean")
                test_loss += loss_mean.item()
                test_count += 1

                # Generate with RAG context
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                visual_embeds = batch.get("visual_embeds")
                vols = batch.get("volume")
                prompt_ids = _extract_prompt_ids(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=batch.get("attention_mask"),
                    tokenizer=tokenizer,
                )

                gen_kwargs = {
                    "prompt_ids": prompt_ids,
                    "max_new_tokens": args.eval_max_new_tokens,
                    "min_new_tokens": args.eval_min_new_tokens,
                    "repetition_penalty": 1.2,
                }
                if visual_embeds is not None:
                    gen_kwargs["visual_embeds"] = visual_embeds
                elif vols is not None:
                    gen_kwargs["volume"] = vols

                generated = unwrapped_model.generate(**gen_kwargs)
                decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                refs = [meta["report_text"] for meta in batch["meta"]]
                vol_names = [meta.get("volume_name", f"sample_{test_count}_{j}") for j, meta in enumerate(batch["meta"])]
                retrieved = [" ||| ".join(meta.get("retrieved_reports", [])) for meta in batch["meta"]]

                all_preds.extend([p.strip() for p in decoded_preds])
                all_refs.extend([r.strip() for r in refs])
                all_vol_names.extend(vol_names)
                all_retrieved.extend(retrieved)

        test_loss /= max(1, test_count)

        # Gather across processes
        paired = list(zip(all_preds, all_refs, all_vol_names, all_retrieved))
        gathered_pairs = accelerator.gather_for_metrics(paired)

        if accelerator.is_main_process:
            if gathered_pairs:
                flat_preds, flat_refs, flat_names, flat_retrieved = zip(*gathered_pairs)
                flat_preds = list(flat_preds)
                flat_refs = list(flat_refs)
                flat_names = list(flat_names)
                flat_retrieved = list(flat_retrieved)
            else:
                flat_preds, flat_refs, flat_names, flat_retrieved = [], [], [], []

            test_metrics = compute_text_metrics(flat_preds, flat_refs) if flat_preds else {}

            accelerator.print(f"\n[TEST RESULTS] Loss: {test_loss:.4f}")
            accelerator.print(f"[TEST RESULTS] Samples evaluated: {len(flat_preds)}")
            for key, value in test_metrics.items():
                accelerator.print(f"  - {key}: {value:.4f}")

            # Save predictions to CSV
            suffix = "_oracle_exact_label" if getattr(args, "oracle_exact_label", False) else ""
            results_csv_path = results_dir / f"test_predictions{suffix}.csv"
            with results_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["volume_name", "prediction", "reference", "retrieved_reports"])
                for name, pred, ref, retr in zip(flat_names, flat_preds, flat_refs, flat_retrieved):
                    writer.writerow([name, pred, ref, retr])
            accelerator.print(f"[TEST] Saved predictions to {results_csv_path}")

            # Save metrics summary
            metrics_path = results_dir / f"test_metrics{suffix}.yaml"
            metrics_summary = {"test_loss": float(test_loss), "num_samples": len(flat_preds)}
            metrics_summary.update({k: float(v) for k, v in test_metrics.items()})
            with metrics_path.open("w", encoding="utf-8") as f:
                yaml.dump(metrics_summary, f, default_flow_style=False)
            accelerator.print(f"[TEST] Saved metrics to {metrics_path}")

        accelerator.print("\n[TEST-ONLY MODE] Completed.")
        return  # Exit early

    # ============ EVAL-ONLY MODE ============
    if args.eval_only:
        split = args.eval_split
        reports_csv = args.val_reports if split == "val" else args.test_reports
        if reports_csv is None:
            raise ValueError(f"--eval-only requires {'--val-reports' if split == 'val' else '--test-reports'}")

        accelerator.print("\n" + "=" * 60)
        accelerator.print(f"[EVAL-ONLY MODE] Loading checkpoint and running {split} evaluation...")
        accelerator.print("=" * 60)

        ckpt_path = args.checkpoint or (args.save_dir / "model_best.pt" if args.save_dir else None)
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)
            accelerator.print(f"Loaded checkpoint from {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Pass --checkpoint or --save-dir.")

        results_dir = args.save_dir or ckpt_path.parent
        if accelerator.is_main_process:
            results_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        max_samples = args.max_val_samples if split == "val" else None
        loader = _build_loader(split, reports_csv, False, max_samples)
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            if not hasattr(model, "_ds_eval_dummy_param"):
                model._ds_eval_dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            dummy_opt = torch.optim.AdamW([model._ds_eval_dummy_param], lr=0.0)
            model, loader, _ = accelerator.prepare(model, loader, dummy_opt)
        else:
            model, loader = accelerator.prepare(model, loader)

        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)

        total_loss = 0.0
        count = 0
        all_preds: List[str] = []
        all_refs: List[str] = []
        all_vol_names: List[str] = []
        all_retrieved: List[str] = []

        from tqdm import tqdm
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating ({split})"):
                outputs = model(batch)
                loss_mean = accelerator.reduce(outputs.loss.detach(), reduction="mean")
                total_loss += loss_mean.item()
                count += 1

                input_ids = batch["input_ids"]
                labels = batch["labels"]
                visual_embeds = batch.get("visual_embeds")
                vols = batch.get("volume")

                prompt_ids = _extract_prompt_ids(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=batch.get("attention_mask"),
                    tokenizer=tokenizer,
                )

                gen_kwargs = {
                    "prompt_ids": prompt_ids,
                    "max_new_tokens": args.eval_max_new_tokens,
                    "min_new_tokens": args.eval_min_new_tokens,
                    "repetition_penalty": 1.2,
                }
                if visual_embeds is not None:
                    gen_kwargs["visual_embeds"] = visual_embeds
                elif vols is not None:
                    gen_kwargs["volume"] = vols

                generated = unwrapped_model.generate(**gen_kwargs)
                decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

                refs = [meta["report_text"] for meta in batch["meta"]]
                vol_names = [meta.get("volume_name", "") for meta in batch["meta"]]
                retrieved = [" ||| ".join(meta.get("retrieved_reports", [])) for meta in batch["meta"]]

                all_preds.extend([p.strip() for p in decoded_preds])
                all_refs.extend([r.strip() for r in refs])
                all_vol_names.extend(vol_names)
                all_retrieved.extend(retrieved)

        avg_loss = total_loss / max(1, count)

        paired = list(zip(all_preds, all_refs, all_vol_names, all_retrieved))
        gathered_pairs = accelerator.gather_for_metrics(paired)

        if accelerator.is_main_process:
            if gathered_pairs:
                flat_preds, flat_refs, flat_names, flat_retrieved = zip(*gathered_pairs)
                flat_preds = list(flat_preds)
                flat_refs = list(flat_refs)
                flat_names = list(flat_names)
                flat_retrieved = list(flat_retrieved)
            else:
                flat_preds, flat_refs, flat_names, flat_retrieved = [], [], [], []

            metrics = compute_text_metrics(flat_preds, flat_refs, compute_clinical=False) if flat_preds else {}
            accelerator.print(f"\n[EVAL {split.upper()}] Loss: {avg_loss:.4f} | Samples: {len(flat_preds)}")
            for key, value in metrics.items():
                accelerator.print(f"  - {key}: {value:.4f}")

            out_csv = results_dir / f"{split}_predictions_eval.csv"
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["volume_name", "prediction", "reference", "retrieved_reports"])
                for name, pred, ref, retr in zip(flat_names, flat_preds, flat_refs, flat_retrieved):
                    writer.writerow([name, pred, ref, retr])
            accelerator.print(f"[EVAL {split.upper()}] Saved predictions to {out_csv}")

            out_yaml = results_dir / f"{split}_metrics_eval.yaml"
            summary = {f"{split}_loss": float(avg_loss), "num_samples": len(flat_preds)}
            summary.update({k: float(v) for k, v in metrics.items()})
            with out_yaml.open("w", encoding="utf-8") as f:
                yaml.dump(summary, f, default_flow_style=False)
            accelerator.print(f"[EVAL {split.upper()}] Saved metrics to {out_yaml}")

        accelerator.print(f"\n[EVAL-ONLY MODE] Completed ({split}).")
        return

    # ============ NORMAL TRAINING MODE ============
    train_loader = _build_loader("train", args.reports, True, args.max_train_samples)
    val_loader = _build_loader("val", args.val_reports, False, args.max_val_samples)

    # 5. Training Loop Setup
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    train_params_count = sum(p.numel() for p in trainable_params)
    
    if accelerator.is_main_process:
        pct = train_params_count / max(1, total_params)
        accelerator.print(f"[Model Stats] Total: {total_params:,} | Trainable: {train_params_count:,} ({pct:.2%})")
        if train_params_count == 0:
            accelerator.print("[WARNING] No parameters are trainable! Check --lora-r or freeze settings.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(1, total_steps // 10)),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    optimizer.zero_grad(set_to_none=True)

    wandb_run = None
    if accelerator.is_main_process and args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=run_config)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        vd_total = 0
        vd_has_retrieval = 0
        vd_blind = 0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                # Guard: if a batch has no supervised tokens (all labels are -100),
                # HF's CausalLM loss becomes NaN due to 0/0 reduction.
                labels = batch.get("labels")
                if labels is None:
                    raise ValueError("Training batch is missing 'labels'.")
                if not (labels != -100).any().item():
                    metas = batch.get("meta") or []
                    names = [str((m or {}).get("volume_name", "")) for m in metas]
                    raise RuntimeError(
                        "Encountered a batch with no supervised tokens (all labels are -100). "
                        "This typically happens when the prompt consumes the full token budget and "
                        "the target is truncated away. Try increasing --report-max-tokens, reducing "
                        "retrieval verbosity, or using a smaller prompt. "
                        f"Example volume_names: {names[:4]}"
                    )

                # Sanitize rare corrupted inputs (NaNs/Infs) early so they don't poison the forward pass.
                if "visual_embeds" in batch and batch["visual_embeds"] is not None:
                    batch["visual_embeds"] = torch.nan_to_num(
                        batch["visual_embeds"], nan=0.0, posinf=0.0, neginf=0.0
                    )
                if "volume" in batch and batch["volume"] is not None:
                    batch["volume"] = torch.nan_to_num(
                        batch["volume"], nan=0.0, posinf=0.0, neginf=0.0
                    )

                # Blindfold training: remove visual evidence for a fraction of samples so the model must
                # use retrieved context to minimize loss. Only apply when retrieval is present.
                p_blind = float(getattr(args, "visual_dropout_prob", 0.0))
                if p_blind > 0.0:
                    metas = batch.get("meta") or []
                    has_retrieval = torch.tensor(
                        [bool((m or {}).get("retrieved_reports")) for m in metas],
                        dtype=torch.bool,
                        device=accelerator.device,
                    )
                    vd_total += int(has_retrieval.numel())
                    vd_has_retrieval += int(has_retrieval.sum().item())
                    if has_retrieval.any():
                        blind = (torch.rand(has_retrieval.shape[0], device=accelerator.device) < p_blind) & has_retrieval
                        vd_blind += int(blind.sum().item())
                        if blind.any():
                            if "visual_embeds" in batch and batch["visual_embeds"] is not None:
                                ve = batch["visual_embeds"]
                                unwrapped = accelerator.unwrap_model(model)
                                mask_token = getattr(unwrapped, "visual_mask_token", None)
                                # Only use mask_token if shapes match (aggregated tokens)
                                if mask_token is not None and ve.size(1) == mask_token.size(0):
                                    replacement = mask_token.to(device=ve.device, dtype=ve.dtype)
                                    ve = ve.clone()
                                    ve[blind] = replacement.expand(int(blind.sum().item()), -1, -1)
                                    batch["visual_embeds"] = ve
                            elif "volume" in batch and batch["volume"] is not None:
                                vol = batch["volume"].clone()
                                vol[blind] = 0
                                batch["volume"] = vol

                outputs = model(batch)
                loss = outputs.loss
                if not torch.isfinite(loss).all().item():
                    metas = batch.get("meta") or []
                    names = [str((m or {}).get("volume_name", "")) for m in metas]
                    raise RuntimeError(
                        f"Non-finite loss detected (loss={loss.detach().float().item()}). "
                        "This can happen due to invalid inputs (NaNs/Infs) or numerical instability. "
                        f"Example volume_names: {names[:4]}"
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.grad_clip:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    loss_val = loss.detach().item()
                    accelerator.print(f"[Ep {epoch}] step {global_step} loss={loss_val:.4f}")
                    if float(getattr(args, "visual_dropout_prob", 0.0)) > 0.0 and vd_total > 0:
                        retrieval_rate = vd_has_retrieval / max(1, vd_total)
                        blind_rate = vd_blind / max(1, vd_has_retrieval)
                        accelerator.print(
                            f"  [VisionDropout] has_retrieval={vd_has_retrieval}/{vd_total} ({retrieval_rate:.1%}) "
                            f"blind={vd_blind}/{max(1, vd_has_retrieval)} ({blind_rate:.1%})"
                        )
                        vd_total = 0
                        vd_has_retrieval = 0
                        vd_blind = 0
                    if wandb_run:
                        wandb_run.log({"train/loss": loss_val, "step": global_step})
        
        # Validation
        if (epoch % args.val_every == 0) or (epoch == args.epochs):
            model.eval()
            val_loss = 0.0
            count = 0
            local_preds = []
            local_refs = []
            local_vol_names = []
            local_retrieved = []
            unwrapped_model = accelerator.unwrap_model(model)
            do_eval_gen = bool(getattr(args, "eval_text_metrics", False))

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    outputs = model(batch)
                    loss_mean = accelerator.reduce(outputs.loss.detach(), reduction="mean")
                    val_loss += loss_mean.item()
                    count += 1

                    # Compute text metrics on subset of batches
                    if do_eval_gen and i < 50:
                        vols = batch.get("volume")
                        visual_embeds = batch.get("visual_embeds")
                        input_ids = batch["input_ids"]
                        labels = batch["labels"]
                        prompt_ids = _extract_prompt_ids(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=batch.get("attention_mask"),
                            tokenizer=tokenizer,
                        )

                        gen_kwargs = {
                            "prompt_ids": prompt_ids,
                            "max_new_tokens": args.eval_max_new_tokens,
                            "min_new_tokens": args.eval_min_new_tokens,
                            "repetition_penalty": 1.2,
                        }
                        if visual_embeds is not None:
                            gen_kwargs["visual_embeds"] = visual_embeds
                        elif vols is not None:
                            gen_kwargs["volume"] = vols

                        generated = unwrapped_model.generate(**gen_kwargs)
                        decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

                        refs = [meta["report_text"] for meta in batch["meta"]]
                        vol_names = [meta.get("volume_name", "") for meta in batch["meta"]]
                        retrieved = [" ||| ".join(meta.get("retrieved_reports", [])) for meta in batch["meta"]]

                        local_preds.extend([p.strip() for p in decoded_preds])
                        local_refs.extend([r.strip() for r in refs])
                        local_vol_names.extend(vol_names)
                        local_retrieved.extend(retrieved)

            val_loss /= max(1, count)

            # Gather text metrics across processes
            text_metrics = {}
            if do_eval_gen and local_preds:
                paired = list(zip(local_preds, local_refs, local_vol_names, local_retrieved))
                gathered_pairs = accelerator.gather_for_metrics(paired)

                if accelerator.is_main_process and gathered_pairs:
                    flat_preds, flat_refs, flat_names, flat_retrieved = zip(*gathered_pairs)
                    flat_preds = list(flat_preds)
                    flat_refs = list(flat_refs)
                    flat_names = list(flat_names)
                    flat_retrieved = list(flat_retrieved)
                    if getattr(args, "eval_text_metrics", False):
                        # Disable clinical score during validation for speed
                        text_metrics.update(compute_text_metrics(flat_preds, flat_refs, compute_clinical=False))

                    # Save validation predictions to CSV
                    val_csv_path = args.save_dir / f"val_predictions_epoch{epoch}.csv"
                    with val_csv_path.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["volume_name", "prediction", "reference", "retrieved_reports"])
                        for name, pred, ref, retr in zip(flat_names, flat_preds, flat_refs, flat_retrieved):
                            writer.writerow([name, pred, ref, retr])
                    accelerator.print(f"  --> Saved {len(flat_preds)} val predictions to {val_csv_path}")

            if accelerator.is_main_process:
                accelerator.print(f"[Ep {epoch}] val_loss={val_loss:.4f}")
                for key, value in text_metrics.items():
                    accelerator.print(f"  - {key}: {value:.4f}")

                if wandb_run:
                    log_dict = {"val/loss": val_loss, "epoch": epoch}
                    log_dict.update({f"val/{k}": v for k, v in text_metrics.items()})
                    wandb_run.log(log_dict)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Safe save: write to temp file first, then rename
                    temp_path = args.save_dir / "model_best.pt.tmp"
                    final_path = args.save_dir / "model_best.pt"
                    torch.save({"model": unwrapped_model.state_dict(), "epoch": epoch, "val_loss": val_loss}, temp_path)
                    temp_path.rename(final_path)
                    accelerator.print(f"  --> Saved best model (val_loss={val_loss:.4f})")

                # Periodic checkpoint every save_every epochs
                if epoch % args.save_every == 0:
                    ckpt_path = args.save_dir / f"checkpoint_epoch{epoch}.pt"
                    torch.save({
                        "model": unwrapped_model.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": val_loss,
                    }, ckpt_path)
                    accelerator.print(f"  --> Saved checkpoint at epoch {epoch}")

    # ===== Test Set Evaluation =====
    if args.test_reports and args.test_reports.exists():
        accelerator.print("\n" + "=" * 60)
        accelerator.print("[TEST] Starting test set evaluation with best model...")
        accelerator.print("=" * 60)

        # Load best model
        best_ckpt_path = args.save_dir / "model_best.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(best_ckpt.get("model", best_ckpt), strict=False)
            accelerator.print(f"[TEST] Loaded best model from {best_ckpt_path}")
        else:
            accelerator.print("[TEST] No best model found, using final model")

        # Build test loader
        test_loader = _build_loader("test", args.test_reports, False, None)
        test_loader = accelerator.prepare(test_loader)

        model.eval()
        test_loss = 0.0
        test_count = 0
        all_preds = []
        all_refs = []
        all_vol_names = []
        all_retrieved = []

        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch)
                loss_mean = accelerator.reduce(outputs.loss.detach(), reduction="mean")
                test_loss += loss_mean.item()
                test_count += 1

                # Generate predictions with RAG context
                vols = batch.get("volume")
                visual_embeds = batch.get("visual_embeds")
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                prompt_ids = _extract_prompt_ids(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=batch.get("attention_mask"),
                    tokenizer=tokenizer,
                )

                gen_kwargs = {
                    "prompt_ids": prompt_ids,
                    "max_new_tokens": args.eval_max_new_tokens,
                    "min_new_tokens": args.eval_min_new_tokens,
                    "repetition_penalty": 1.2,
                }
                if visual_embeds is not None:
                    gen_kwargs["visual_embeds"] = visual_embeds
                elif vols is not None:
                    gen_kwargs["volume"] = vols

                generated = unwrapped_model.generate(**gen_kwargs)
                decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                refs = [meta["report_text"] for meta in batch["meta"]]
                vol_names = [meta.get("volume_name", f"sample_{test_count}_{j}") for j, meta in enumerate(batch["meta"])]
                retrieved = [" ||| ".join(meta.get("retrieved_reports", [])) for meta in batch["meta"]]

                all_preds.extend([p.strip() for p in decoded_preds])
                all_refs.extend([r.strip() for r in refs])
                all_vol_names.extend(vol_names)
                all_retrieved.extend(retrieved)

        test_loss /= max(1, test_count)

        # Gather all predictions across processes
        paired_data = list(zip(all_preds, all_refs, all_vol_names, all_retrieved))
        gathered_data = accelerator.gather_for_metrics(paired_data)

        if accelerator.is_main_process:
            if gathered_data:
                flat_preds, flat_refs, flat_names, flat_retrieved = zip(*gathered_data)
                flat_preds = list(flat_preds)
                flat_refs = list(flat_refs)
                flat_names = list(flat_names)
                flat_retrieved = list(flat_retrieved)
            else:
                flat_preds, flat_refs, flat_names, flat_retrieved = [], [], [], []

            # Compute metrics
            test_metrics = compute_text_metrics(flat_preds, flat_refs) if flat_preds else {}

            accelerator.print(f"\n[TEST RESULTS] Loss: {test_loss:.4f}")
            accelerator.print(f"[TEST RESULTS] Samples evaluated: {len(flat_preds)}")
            for key, value in test_metrics.items():
                accelerator.print(f"  - {key}: {value:.4f}")

            # Save predictions to CSV
            suffix = "_oracle_exact_label" if getattr(args, "oracle_exact_label", False) else ""
            results_csv_path = args.save_dir / f"test_predictions{suffix}.csv"
            with results_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["volume_name", "prediction", "reference", "retrieved_reports"])
                for name, pred, ref, retr in zip(flat_names, flat_preds, flat_refs, flat_retrieved):
                    writer.writerow([name, pred, ref, retr])
            accelerator.print(f"[TEST] Saved predictions to {results_csv_path}")

            # Save metrics summary
            metrics_path = args.save_dir / f"test_metrics{suffix}.yaml"
            metrics_summary = {"test_loss": float(test_loss), "num_samples": len(flat_preds)}
            metrics_summary.update({k: float(v) for k, v in test_metrics.items()})
            with metrics_path.open("w", encoding="utf-8") as f:
                yaml.dump(metrics_summary, f, default_flow_style=False)
            accelerator.print(f"[TEST] Saved metrics to {metrics_path}")

            if wandb_run:
                test_log = {"test/loss": test_loss, "test/num_samples": len(flat_preds)}
                test_log.update({f"test/{k}": v for k, v in test_metrics.items()})
                wandb_run.log(test_log)

    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    main()
