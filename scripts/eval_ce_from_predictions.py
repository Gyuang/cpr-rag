#!/usr/bin/env python3
"""
Evaluate Clinical Efficacy (CE) from prediction CSV files.
Uses RadBERT-based CT-RATE CE evaluation.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.radiology_eval import compute_ct_rate_ce_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CE from predictions CSV")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions CSV")
    parser.add_argument("--output", type=Path, default=None, help="Output YAML file for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    # Load predictions
    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df)} predictions from {args.predictions}")

    predictions = df["prediction"].tolist()
    references = df["reference"].tolist()

    # Compute CE metrics
    print("Computing CT-RATE CE metrics...")
    ce_metrics = compute_ct_rate_ce_metrics(
        predictions=predictions,
        references=references,
        base_model_name="zzxslp/RadBERT-RoBERTa-4m",
        ckpt_path="/research/04-CT/00-RawData/CT-RATE/models/RadBertClassifier.pth",
        device=args.device,
        threshold=0.5,
        max_length=512,
        batch_size=args.batch_size,
    )

    print("\n" + "="*60)
    print("CT-RATE Clinical Efficacy Results")
    print("="*60)
    print(f"  CE Precision: {ce_metrics.get('metrics/ce_precision', ce_metrics.get('metrics/ct_rate_ce_precision_micro', 0)):.4f}")
    print(f"  CE Recall:    {ce_metrics.get('metrics/ce_recall', ce_metrics.get('metrics/ct_rate_ce_recall_micro', 0)):.4f}")
    print(f"  CE F1:        {ce_metrics.get('metrics/ce_f1', ce_metrics.get('metrics/ct_rate_ce_f1_micro', 0)):.4f}")
    print("="*60)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            yaml.dump(ce_metrics, f, default_flow_style=False)
        print(f"Results saved to {args.output}")

    return ce_metrics


if __name__ == "__main__":
    main()
