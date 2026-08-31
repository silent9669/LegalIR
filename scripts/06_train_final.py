"""Train final models and full-corpus question memory on all 7,000 training queries."""

import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.parameter_audit import audit_system_parameters, MAX_PARAMETER_BUDGET
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.question_memory import TrainQuestionMemory
from src.training.train_reranker import train_reranker


def train_final_system(
    data_dir: str = "artifacts/task1/data",
    index_dir: str = "artifacts/task1/indexes",
    output_dir: str = "artifacts/local/training/final_checkpoints",
    config_path: str = "configs/experiments/reranker_lora.yaml",
    max_steps: int | None = None,
    device: str | None = None,
):
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

    # 1. Build full question memory on all 7,000 queries
    if queries_path.exists() and qrels_path.exists():
        print("\n[1/2] Building Full Question Memory Index...")
        df_queries = pd.read_parquet(queries_path)
        df_qrels = pd.read_parquet(qrels_path)

        queries_dict = {str(r["query_id"]): str(r["question_raw"]) for r in df_queries.to_dict("records")}
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

    # 2. Train final reranker checkpoint
    print("\n[2/2] Training Final Reranker Checkpoint...")
    report = train_reranker(
        config_path=config_path,
        fold=0,
        output_dir=str(output_dir / "reranker"),
        max_steps=max_steps,
    )
    print(f"Final training completed with status: {report.get('status')}")


def main():
    parser = argparse.ArgumentParser(description="LegalIR Final Model Trainer")
    parser.add_argument("--data-dir", type=str, default="artifacts/task1/data")
    parser.add_argument("--index-dir", type=str, default="artifacts/task1/indexes")
    parser.add_argument("--output-dir", type=str, default="artifacts/local/training/final_checkpoints")
    parser.add_argument("--config", type=str, default="configs/experiments/reranker_lora.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_final_system(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        max_steps=args.max_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
