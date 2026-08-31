"""Train final models and full-corpus question memory on all 7,000 training queries."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import pandas as pd

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.parameter_audit import audit_system_parameters, MAX_PARAMETER_BUDGET
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.question_memory import TrainQuestionMemory
from src.training.build_pairs import build_training_pairs
from src.training.train_reranker import train_reranker


def train_final_system(
    data_dir: str | Path = "artifacts/task1/data",
    index_dir: str | Path = "artifacts/task1/indexes",
    output_dir: str | Path = "artifacts/local/training/final_checkpoints",
    config_path: str | Path = "configs/experiments/reranker_lora.yaml",
    max_steps: int | None = None,
    device: str | None = None,
    limit_queries: int | None = None,
) -> dict[str, Any]:
    print("=" * 60)
    print("LegalIR Task 1: Training Final System on All 7,000 Queries")
    print("=" * 60)

    data_dir = Path(data_dir)
    index_dir = Path(index_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Preflight] Running Strict Parameter Budget Audit (<4B Rule)...")
    audit_report = audit_system_parameters(
        output_json=output_dir / "parameter_audit.json",
        raise_on_violation=True,
    )
    print(
        f"Parameter audit passed: {audit_report['total_learned_parameters']:,} params "
        f"({audit_report['total_parameters_billions']:.4f}B / 4.0B, "
        f"{audit_report['budget_utilization_pct']:.2f}% utilization). PASS"
    )

    queries_path = data_dir / "queries_train.parquet"
    qrels_path = data_dir / "qrels_train.parquet"

    # 1. Build full question memory on all queries
    if queries_path.exists() and qrels_path.exists():
        print("\n[1/3] Building Full Question Memory Index...")
        df_queries = pd.read_parquet(queries_path)
        df_qrels = pd.read_parquet(qrels_path)

        queries_dict = {
            str(r["query_id"]): str(r.get("question_norm") or r.get("question_raw") or r.get("question") or "")
            for r in df_queries.to_dict("records")
        }
        qrels_dict = {}
        for r in df_qrels.to_dict("records"):
            qid = str(r["query_id"])
            did = str(r["doc_id"])
            if qid not in qrels_dict:
                qrels_dict[qid] = []
            qrels_dict[qid].append(did)

        dense_dir = index_dir / "dense_dek21" if (index_dir / "dense_dek21").exists() else index_dir / "dense"
        dense = DenseMacroRetriever.load(dense_dir, device=device) if dense_dir.exists() else None

        memory = TrainQuestionMemory(min_similarity=0.82, dense_encoder=dense)
        memory.fit(queries_dict, qrels_dict)

        mem_final_dir = output_dir / "question_memory"
        memory.save(mem_final_dir)
        print(f"Full question memory saved to {mem_final_dir}")

    # 2. Build training pairs on all queries
    print("\n[2/3] Mining Training Pairs from All Training Queries...")
    pairs_dir = output_dir / "pairs"
    _, pairs_df = build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        fold=None,
        use_all_queries=True,
        limit=limit_queries,
    )
    pairs_file = pairs_dir / "reranker_pairs.parquet"

    # 3. Train final LoRA reranker checkpoint
    print("\n[3/3] Training Final LoRA Reranker Checkpoint...")
    reranker_dir = output_dir / "reranker"
    report = train_reranker(
        pairs_file=pairs_file,
        config_path=config_path,
        output_dir=reranker_dir,
        fold=None,
        max_steps=max_steps,
    )

    manifest = {
        "unique_training_queries": report.get("unique_training_queries", len(pairs_df["query_id"].unique())),
        "pair_count": len(pairs_df),
        "positive_count": int((pairs_df["label"] > 0.5).sum()) if "label" in pairs_df.columns else 0,
        "negative_count": int((pairs_df["label"] <= 0.5).sum()) if "label" in pairs_df.columns else 0,
        "optimizer_steps": report.get("global_steps", 0),
        "effective_examples_seen": report.get("effective_examples_seen", 0),
        "epochs_or_equivalent": report.get("epochs_or_equivalent", 1.0),
        "adapter_checksum": report.get("adapter_checksum"),
        "training_time_sec": report.get("training_time_sec", 0.0),
        "param_diff": report.get("param_diff", 0.0),
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "final_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Final training completed with status: {report.get('status')}. Manifest saved to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="LegalIR Final Model Trainer")
    parser.add_argument("--data-dir", type=str, default="artifacts/task1/data")
    parser.add_argument("--index-dir", type=str, default="artifacts/task1/indexes")
    parser.add_argument("--output-dir", type=str, default="artifacts/local/training/final_checkpoints")
    parser.add_argument("--config", type=str, default="configs/experiments/reranker_lora.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_final_system(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        max_steps=args.max_steps,
        limit_queries=args.limit_queries,
        device=args.device,
    )


if __name__ == "__main__":
    main()
