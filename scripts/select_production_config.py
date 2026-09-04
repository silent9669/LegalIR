#!/usr/bin/env python3
"""CLI script to aggregate OOF fold results, apply score promotion, and generate production_lock.json."""

import argparse
import json
import sys
from pathlib import Path

from src.validation.promotion import aggregate_oof_metrics, create_production_lock


def main():
    parser = argparse.ArgumentParser(description="Select and lock production configuration from OOF results.")
    parser.add_argument("--folds-dir", type=str, default="artifacts/factory/folds", help="Path to folds directory")
    parser.add_argument("--output-lock", type=str, default="artifacts/bundle/production_lock.json", help="Path to output lock")
    parser.add_argument("--runtime-commit", type=str, default="a0efb25", help="Approved runtime git commit SHA")
    args = parser.parse_args()

    folds_root = Path(args.folds_dir)
    print(f"[*] Aggregating OOF metrics from {folds_root} ...")

    metrics_list = []
    for f in range(5):
        m_file = folds_root / f"fold_{f}" / "fold_metrics.json"
        if m_file.is_file():
            with open(m_file, "r", encoding="utf-8") as fp:
                metrics_list.append(json.load(fp))

    if not metrics_list:
        print("[!] No fold metrics found to aggregate.")
        sys.exit(1)

    agg = aggregate_oof_metrics(metrics_list)
    print(f"[+] Aggregate OOF Metrics: {agg}")

    approved_config = {
        "fusion": {
            "method": "rrf",
            "weights": {"bm25_legal": 1.0, "bm25_pyvi": 1.0, "dense": 1.0, "exact": 0.5, "reranker": 2.5},
            "candidate_k": 150,
            "top_k": 5,
        },
        "reranker": {
            "model_name": "BAAI/bge-reranker-v2-m3",
            "max_length": 512,
            "effective_batch_size": 16,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }
    }

    create_production_lock(
        output_path=args.output_lock,
        metrics=agg,
        config=approved_config,
        runtime_commit=args.runtime_commit,
    )

    print(f"[+] Successfully locked production configuration to {args.output_lock}")


if __name__ == "__main__":
    main()
