"""Radiology evaluation helpers (text metrics + clinical proxies).

This module centralizes evaluation logic so training scripts stay readable.

Includes:
- Text metrics: BLEU/ROUGE/METEOR/BERTScore (+ optional RadGraph F1)
- CT-RATE 18-label "clinical efficacy" (CheXbert-style) via a RadBERT classifier
- GREEN6 score (GREEN metric with 6 error categories) via a GREEN evaluator LLM
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    import evaluate
except Exception as exc:  # pragma: no cover
    evaluate = None  # type: ignore
    _EVALUATE_IMPORT_ERROR = exc
else:
    _EVALUATE_IMPORT_ERROR = None

try:
    from radgraph import F1RadGraph

    RADGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    F1RadGraph = None  # type: ignore
    RADGRAPH_AVAILABLE = False

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


DEFAULT_CT_RATE_CE_MODEL_NAME = "zzxslp/RadBERT-RoBERTa-4m"
DEFAULT_CT_RATE_CE_CKPT = Path("/research/04-CT/00-RawData/CT-RATE/models/RadBertClassifier.pth")
DEFAULT_GREEN_MODEL = "StanfordAIMI/GREEN-RadPhi2"


def _require_evaluate() -> None:
    if evaluate is None:  # pragma: no cover
        raise ImportError("Missing dependency 'evaluate'.") from _EVALUATE_IMPORT_ERROR


@lru_cache(maxsize=1)  # type: ignore[misc]
def get_bleu_metric():
    _require_evaluate()
    return evaluate.load("bleu")  # type: ignore[union-attr]


@lru_cache(maxsize=1)  # type: ignore[misc]
def get_rouge_metric():
    _require_evaluate()
    return evaluate.load("rouge")  # type: ignore[union-attr]


@lru_cache(maxsize=1)  # type: ignore[misc]
def get_meteor_metric():
    _require_evaluate()
    return evaluate.load("meteor")  # type: ignore[union-attr]


@lru_cache(maxsize=1)  # type: ignore[misc]
def get_bertscore_metric():
    _require_evaluate()
    return evaluate.load("bertscore")  # type: ignore[union-attr]


@lru_cache(maxsize=1)  # type: ignore[misc]
def get_radgraph_scorer():
    if RADGRAPH_AVAILABLE:
        return F1RadGraph(reward_level="all", model_type="radgraph-xl")  # type: ignore[misc]
    return None


def compute_radgraph_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute RadGraph F1 clinical score."""
    if not RADGRAPH_AVAILABLE:
        return 0.0
    scorer = get_radgraph_scorer()
    if scorer is None:
        return 0.0
    preds = [p.strip() if str(p).strip() else "normal" for p in predictions]
    refs = [r.strip() if str(r).strip() else "normal" for r in references]
    try:
        result = scorer(preds, refs)
        if isinstance(result, tuple) and len(result) >= 1:
            score = result[0]
            if isinstance(score, (list, tuple, np.ndarray)):
                return float(np.mean(score))
            return float(score)
        return float(result)
    except Exception:
        return 0.0


def compute_text_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    compute_clinical: bool = True,
) -> dict:
    if not predictions or not references:
        return {}

    predictions = [str(p).strip() for p in predictions]
    references = [str(r).strip() for r in references]

    bleu_result = get_bleu_metric().compute(predictions=predictions, references=[[r] for r in references])
    rouge_result = get_rouge_metric().compute(predictions=predictions, references=references)
    meteor_result = get_meteor_metric().compute(predictions=predictions, references=references)

    # BERTScore crashes on empty strings; substitute placeholders for empties.
    bert_predictions = [p if p else "<EMPTY>" for p in predictions]
    bert_references = [r if r else "<EMPTY>" for r in references]
    bert_result = get_bertscore_metric().compute(
        predictions=bert_predictions,
        references=bert_references,
        lang="en",
    )

    metrics: Dict[str, float] = {
        "metrics/bleu": float(bleu_result.get("bleu", 0.0)),
        "metrics/meteor": float(meteor_result.get("meteor", 0.0)),
    }
    precisions = bleu_result.get("precisions") or []
    for idx, precision in enumerate(precisions, start=1):
        metrics[f"metrics/bleu{idx}"] = float(precision)

    def _mean(values):
        if not values:
            return 0.0
        return float(sum(float(v) for v in values) / len(values))

    metrics["metrics/bertscore_precision"] = _mean(bert_result.get("precision"))
    metrics["metrics/bertscore_recall"] = _mean(bert_result.get("recall"))
    metrics["metrics/bertscore_f1"] = _mean(bert_result.get("f1"))
    for key, value in rouge_result.items():
        metrics[f"metrics/{key}"] = float(value)

    if compute_clinical and RADGRAPH_AVAILABLE:
        metrics["metrics/radgraph_f1"] = float(compute_radgraph_score(predictions, references))

    return metrics


def _sanitize_metric_key(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "label"


@lru_cache(maxsize=4)  # type: ignore[misc]
def _load_ct_rate_label_names(label_csv: str) -> tuple[str, ...]:
    """Load CT-RATE label column names from a CSV header (first 18 labels)."""
    path = Path(label_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
    if "VolumeName" in fields:
        fields = [f for f in fields if f != "VolumeName"]
    elif "volume_name" in fields:
        fields = [f for f in fields if f != "volume_name"]
    fields = [f for f in fields if f != "split"]
    if len(fields) < 18:
        raise ValueError(f"Expected >=18 label columns in {path}, found {len(fields)}")
    return tuple(fields[:18])


class CtRateClinicalEfficacyLabeler(nn.Module):
    """RoBERTa/RadBERT + linear head for 18 CT-RATE abnormalities (CheXbert-style CE).

    Matches Reg2RG's RadBertClassifier exactly:
    - Uses AutoModel.from_pretrained (not from_config)
    - Uses pooler_output (not last_hidden_state[:, 0])
    - No dropout layer
    """

    def __init__(self, base_model_name: str, num_labels: int = 18, dropout: float = 0.1) -> None:
        super().__init__()
        # Use from_pretrained to match Reg2RG's classifier.py
        self.model = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Reg2RG uses pooler_output (not last_hidden_state[:, 0])
        return self.classifier(outputs.pooler_output)


@lru_cache(maxsize=2)  # type: ignore[misc]
def _load_ct_rate_ce_components(
    *,
    base_model_name: str,
    ckpt_path: str,
    device: str,
) -> tuple[AutoTokenizer, CtRateClinicalEfficacyLabeler]:
    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = CtRateClinicalEfficacyLabeler(base_model_name, num_labels=18)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        maybe = state["model"]
        if any(k.startswith("model.") or k.startswith("classifier.") for k in maybe.keys()):
            state = maybe
    if isinstance(state, dict) and any(str(k).startswith("module.") for k in state.keys()):
        state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    if not isinstance(state, dict) or "classifier.weight" not in state:
        raise ValueError(f"CT-RATE CE checkpoint {ckpt_path} does not look like a RadBERT classifier state_dict.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", missing[:20], len(missing))
    print("unexpected:", unexpected[:20], len(unexpected))

    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device="cuda")
    else:
        model = model.to(device="cpu")
    model.eval()
    return tok, model


def _ct_rate_ce_predict(
    texts: Sequence[str],
    *,
    tokenizer: AutoTokenizer,
    model: CtRateClinicalEfficacyLabeler,
    threshold: float,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    out: List[np.ndarray] = []
    safe_texts = [t.strip() if str(t).strip() else "normal" for t in texts]
    with torch.inference_mode():
        for i in range(0, len(safe_texts), max(1, int(batch_size))):
            chunk = safe_texts[i : i + max(1, int(batch_size))]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length),
            )
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            out.append((probs >= float(threshold)).astype(np.int32))
    if not out:
        return np.zeros((0, 18), dtype=np.int32)
    return np.concatenate(out, axis=0)


def compute_ct_rate_ce_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    label_csv: Optional[Path] = None,
    base_model_name: str = DEFAULT_CT_RATE_CE_MODEL_NAME,
    ckpt_path: Path = DEFAULT_CT_RATE_CE_CKPT,
    device: str = "cpu",
    threshold: float = 0.5,
    max_length: int = 256,
    batch_size: int = 16,
) -> dict:
    """CheXbert-style CE metrics: label(prediction) vs label(reference) via a RadBERT classifier."""
    if not predictions or not references:
        return {}
    n = min(len(predictions), len(references))
    predictions = list(predictions)[:n]
    references = list(references)[:n]

    label_names = (
        _load_ct_rate_label_names(str(label_csv))
        if label_csv is not None
        else tuple(f"label_{i}" for i in range(18))
    )
    tokenizer, model = _load_ct_rate_ce_components(
        base_model_name=str(base_model_name),
        ckpt_path=str(ckpt_path),
        device=str(device),
    )

    y_pred = _ct_rate_ce_predict(
        predictions,
        tokenizer=tokenizer,
        model=model,
        threshold=threshold,
        max_length=max_length,
        batch_size=batch_size,
    )
    y_ref = _ct_rate_ce_predict(
        references,
        tokenizer=tokenizer,
        model=model,
        threshold=threshold,
        max_length=max_length,
        batch_size=batch_size,
    )

    from sklearn.metrics import precision_recall_fscore_support

    # Micro-average in multilabel setting: global TP/FP/FN over the positive class (Reg2RG-style).
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_ref,
        y_pred,
        average="micro",
        zero_division=0,
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_ref,
        y_pred,
        average="macro",
        zero_division=0,
    )
    p_lbl, r_lbl, f_lbl, s_lbl = precision_recall_fscore_support(
        y_ref,
        y_pred,
        average=None,
        zero_division=0,
    )

    metrics: Dict[str, float] = {
        "metrics/ct_rate_ce_precision_micro": float(p_micro),
        "metrics/ct_rate_ce_recall_micro": float(r_micro),
        "metrics/ct_rate_ce_f1_micro": float(f_micro),
        "metrics/ct_rate_ce_precision_macro": float(p_macro),
        "metrics/ct_rate_ce_recall_macro": float(r_macro),
        "metrics/ct_rate_ce_f1_macro": float(f_macro),
    }
    # Common short names (paper-friendly aliases)
    metrics["metrics/ce_precision"] = float(p_micro)
    metrics["metrics/ce_recall"] = float(r_micro)
    metrics["metrics/ce_f1"] = float(f_micro)
    for name, p, r, f, sup in zip(label_names, p_lbl, r_lbl, f_lbl, s_lbl):
        key = _sanitize_metric_key(name)
        metrics[f"metrics/ct_rate_ce_f1/{key}"] = float(f)
        metrics[f"metrics/ct_rate_ce_precision/{key}"] = float(p)
        metrics[f"metrics/ct_rate_ce_recall/{key}"] = float(r)
        metrics[f"metrics/ct_rate_ce_support/{key}"] = float(sup)
    return metrics


def compute_ct_rate_ce_metrics_distributed(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    accelerator,
    label_csv: Optional[Path] = None,
    base_model_name: str = DEFAULT_CT_RATE_CE_MODEL_NAME,
    ckpt_path: Path = DEFAULT_CT_RATE_CE_CKPT,
    device: str = "cpu",
    threshold: float = 0.5,
    max_length: int = 256,
    batch_size: int = 16,
) -> dict:
    """Compute CT-RATE CE metrics across all processes (sums TP/FP/FN via `accelerator.reduce`)."""
    n = min(len(predictions), len(references))
    local_preds = list(predictions)[:n]
    local_refs = list(references)[:n]

    if n > 0:
        tokenizer, model = _load_ct_rate_ce_components(
            base_model_name=str(base_model_name),
            ckpt_path=str(ckpt_path),
            device=str(device),
        )
        y_pred = _ct_rate_ce_predict(
            local_preds,
            tokenizer=tokenizer,
            model=model,
            threshold=threshold,
            max_length=max_length,
            batch_size=batch_size,
        )
        y_ref = _ct_rate_ce_predict(
            local_refs,
            tokenizer=tokenizer,
            model=model,
            threshold=threshold,
            max_length=max_length,
            batch_size=batch_size,
        )
        tp_np = (y_pred & y_ref).sum(axis=0).astype(np.int64)
        fp_np = (y_pred & (1 - y_ref)).sum(axis=0).astype(np.int64)
        fn_np = ((1 - y_pred) & y_ref).sum(axis=0).astype(np.int64)
        support_np = y_ref.sum(axis=0).astype(np.int64)
    else:
        tp_np = np.zeros((18,), dtype=np.int64)
        fp_np = np.zeros((18,), dtype=np.int64)
        fn_np = np.zeros((18,), dtype=np.int64)
        support_np = np.zeros((18,), dtype=np.int64)

    dev = getattr(accelerator, "device", torch.device("cpu"))
    tp = torch.tensor(tp_np, device=dev, dtype=torch.long)
    fp = torch.tensor(fp_np, device=dev, dtype=torch.long)
    fn = torch.tensor(fn_np, device=dev, dtype=torch.long)
    support = torch.tensor(support_np, device=dev, dtype=torch.long)

    tp = accelerator.reduce(tp, reduction="sum")
    fp = accelerator.reduce(fp, reduction="sum")
    fn = accelerator.reduce(fn, reduction="sum")
    support = accelerator.reduce(support, reduction="sum")

    if not getattr(accelerator, "is_main_process", True):
        return {}

    tp_np = tp.detach().cpu().numpy().astype(np.int64)
    fp_np = fp.detach().cpu().numpy().astype(np.int64)
    fn_np = fn.detach().cpu().numpy().astype(np.int64)
    support_np = support.detach().cpu().numpy().astype(np.int64)

    tp_total = int(tp_np.sum())
    fp_total = int(fp_np.sum())
    fn_total = int(fn_np.sum())

    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else 0.0

    p_micro = _safe_div(tp_total, tp_total + fp_total)
    r_micro = _safe_div(tp_total, tp_total + fn_total)
    f_micro = _safe_div(2.0 * tp_total, 2.0 * tp_total + fp_total + fn_total)

    per_label_p = [_safe_div(tp_i, tp_i + fp_i) for tp_i, fp_i in zip(tp_np, fp_np)]
    per_label_r = [_safe_div(tp_i, tp_i + fn_i) for tp_i, fn_i in zip(tp_np, fn_np)]
    per_label_f = [
        _safe_div(2.0 * p_i * r_i, p_i + r_i) if (p_i + r_i) > 0 else 0.0
        for p_i, r_i in zip(per_label_p, per_label_r)
    ]

    p_macro = float(sum(per_label_p) / 18.0)
    r_macro = float(sum(per_label_r) / 18.0)
    f_macro = float(sum(per_label_f) / 18.0)

    label_names = (
        _load_ct_rate_label_names(str(label_csv))
        if label_csv is not None
        else tuple(f"label_{i}" for i in range(18))
    )

    metrics: Dict[str, float] = {
        "metrics/ct_rate_ce_precision_micro": float(p_micro),
        "metrics/ct_rate_ce_recall_micro": float(r_micro),
        "metrics/ct_rate_ce_f1_micro": float(f_micro),
        "metrics/ct_rate_ce_precision_macro": float(p_macro),
        "metrics/ct_rate_ce_recall_macro": float(r_macro),
        "metrics/ct_rate_ce_f1_macro": float(f_macro),
        "metrics/ce_precision": float(p_micro),
        "metrics/ce_recall": float(r_micro),
        "metrics/ce_f1": float(f_micro),
    }
    for name, p, r, f, sup in zip(label_names, per_label_p, per_label_r, per_label_f, support_np):
        key = _sanitize_metric_key(name)
        metrics[f"metrics/ct_rate_ce_f1/{key}"] = float(f)
        metrics[f"metrics/ct_rate_ce_precision/{key}"] = float(p)
        metrics[f"metrics/ct_rate_ce_recall/{key}"] = float(r)
        metrics[f"metrics/ct_rate_ce_support/{key}"] = float(sup)
    return metrics


def calculate_ce_metrics(
    gen_reports: Sequence[str],
    ref_reports: Sequence[str],
    checkpoint_path: Path = DEFAULT_CT_RATE_CE_CKPT,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_name: str = DEFAULT_CT_RATE_CE_MODEL_NAME,
    threshold: float = 0.5,
    max_length: int = 512,
    batch_size: int = 16,
) -> dict:
    """Minimal Reg2RG-style CE API (micro P/R/F1) for ad-hoc use."""
    metrics = compute_ct_rate_ce_metrics(
        gen_reports,
        ref_reports,
        label_csv=None,
        base_model_name=model_name,
        ckpt_path=checkpoint_path,
        device=device,
        threshold=threshold,
        max_length=max_length,
        batch_size=batch_size,
    )
    return {
        "CE_Precision": float(metrics.get("metrics/ct_rate_ce_precision_micro", 0.0)),
        "CE_Recall": float(metrics.get("metrics/ct_rate_ce_recall_micro", 0.0)),
        "CE_F1": float(metrics.get("metrics/ct_rate_ce_f1_micro", 0.0)),
    }


_GREEN_PROMPT_TEMPLATE = """GPT-4 Prompt Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.

Process Overview:
You will be presented with:
1) The criteria for making a judgment.
2) The reference radiology report.
3) The candidate radiology report.
4) The desired format for your assessment.

1. Criteria for Judgment:
For each candidate report, determine:
- The count of clinically significant errors.
- The count of clinically insignificant errors.
Errors can fall into one of these categories:
(a) False report of a finding in the candidate.
(b) Missing a finding present in the reference.
(c) Misidentification of a finding’s anatomic location/position.
(d) Misassessment of the severity of a finding.
(e) Mentioning a comparison that isn’t in the reference.
(f) Omitting a comparison detailing a change from a prior study.
Note: Concentrate on the clinical findings rather than the report’s writing style. Evaluate only the findings that appear in both reports.

2. Reference Report:
{reference_report}

3. Candidate Report:
{candidate_report}

4. Reporting Your Assessment:
Follow this specific format for your output, even if no errors are found:
'''
[Explanation]: <Explanation>
[Clinically Significant Errors]:
(a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
...
(f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
[Clinically Insignificant Errors]:
(a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
...
(f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
[Matched Findings]: <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>
'''
"""


@lru_cache(maxsize=1)  # type: ignore[misc]
def _load_green_components(*, model_id: str, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device="cuda")
    else:
        model = model.to(device="cpu")
    model.eval()
    return tok, model


_GREEN_SIG_RE = re.compile(r"\\((a|b|c|d|e|f)\\)\\s*[^:]{0,120}:\\s*(\\d+)", re.IGNORECASE)
_GREEN_MATCHED_RE = re.compile(r"\\[Matched Findings\\]\\s*:\\s*(\\d+)", re.IGNORECASE)


def _parse_green_output(text: str) -> Optional[Dict[str, float]]:
    raw = str(text or "")
    if not raw.strip():
        return None

    sig_section = raw
    m = re.search(
        r"\\[Clinically Significant Errors\\]\\s*:(.*?)(\\[Clinically Insignificant Errors\\]|\\[Matched Findings\\])",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        sig_section = m.group(1)

    sig_counts = {k: 0 for k in "abcdef"}
    for letter, count in _GREEN_SIG_RE.findall(sig_section):
        sig_counts[str(letter).lower()] = int(count)

    matched_m = _GREEN_MATCHED_RE.search(raw)
    matched = int(matched_m.group(1)) if matched_m else None
    if matched is None:
        matched_m = re.search(r"Matched Findings\\s*:\\s*(\\d+)", raw, flags=re.IGNORECASE)
        matched = int(matched_m.group(1)) if matched_m else None
    if matched is None:
        return None

    total_sig = sum(sig_counts.values())
    denom = matched + total_sig
    score = float(matched / denom) if denom > 0 else 0.0
    out: Dict[str, float] = {"green6": score, "matched_findings": float(matched), "sig_errors_total": float(total_sig)}
    for k in "abcdef":
        out[f"sig_errors_{k}"] = float(sig_counts[k])
    return out


def compute_green6_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_id: str = DEFAULT_GREEN_MODEL,
    device: str = "cpu",
    max_samples: int = 50,
    batch_size: int = 1,
    max_new_tokens: int = 256,
) -> dict:
    """Compute GREEN6 (mean GREEN score) using a GREEN evaluator LLM; slow."""
    if not predictions or not references:
        return {}
    n = min(len(predictions), len(references), int(max_samples))
    if n <= 0:
        return {}

    tok, model = _load_green_components(model_id=model_id, device=device)
    pairs = list(zip(list(references)[:n], list(predictions)[:n]))
    prompts = [
        _GREEN_PROMPT_TEMPLATE.format(reference_report=ref.strip(), candidate_report=pred.strip())
        for ref, pred in pairs
    ]

    device_t = next(model.parameters()).device
    parsed: List[Dict[str, float]] = []
    failed = 0
    with torch.inference_mode():
        for i in range(0, len(prompts), max(1, int(batch_size))):
            chunk = prompts[i : i + max(1, int(batch_size))]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            input_ids = enc["input_ids"].to(device_t)
            attn = enc["attention_mask"].to(device_t)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            input_lens = attn.sum(dim=1).tolist()
            for seq, in_len in zip(generated, input_lens):
                out_ids = seq[int(in_len) :]
                out_text = tok.decode(out_ids, skip_special_tokens=True)
                parsed_row = _parse_green_output(out_text)
                if parsed_row is None:
                    failed += 1
                else:
                    parsed.append(parsed_row)

    if not parsed:
        return {"metrics/green6_parse_success_rate": 0.0}

    mean_green = float(sum(r["green6"] for r in parsed) / len(parsed))
    success_rate = float(len(parsed) / float(n))

    def _mean(key: str) -> float:
        return float(sum(r.get(key, 0.0) for r in parsed) / len(parsed))

    metrics: Dict[str, float] = {
        "metrics/green6": mean_green,
        "metrics/green6_parse_success_rate": success_rate,
        "metrics/green6_num_scored": float(len(parsed)),
        "metrics/green6_num_failed": float(failed),
        "metrics/green6_matched_findings_mean": _mean("matched_findings"),
        "metrics/green6_sig_errors_total_mean": _mean("sig_errors_total"),
    }
    for k in "abcdef":
        metrics[f"metrics/green6_sig_errors_{k}_mean"] = _mean(f"sig_errors_{k}")
    return metrics
