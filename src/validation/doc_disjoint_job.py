"""Document-disjoint validation job execution, candidate scoring, and robustness metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd

from src.core.hashing import sha256_directory, sha256_file
from src.core.manifests import JobManifest
from src.core.memory import (
    check_memory_guard,
    format_memory_report,
    release_memory,
    take_memory_snapshot,
)
from src.data.splits import load_doc_disjoint_split
from src.evaluation.evaluator import evaluate_predictions
from src.ranking.reranker import CrossEncoderReranker
from src.training.train_reranker import train_reranker


def verify_doc_disjoint_split(train_docs: Set[str], test_docs: Set[str]) -> bool:
    """Verify zero document overlap between train and test sets."""
    return len(train_docs.intersection(test_docs)) == 0


class DocDisjointRunner:
    """
    Executes the document-disjoint robustness validation job.
    Evaluates model generalization to completely unseen laws/statutes.
    """

    def __init__(
        self,
        work_dir: Union[str, Path],
        dataset_dir: Union[str, Path] = "data/task1_canonical_v2",
        runtime_commit: str = "unknown",
        dataset_sha256: str = "unknown",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.work_dir = Path(work_dir)
        self.dataset_dir = Path(dataset_dir)
        self.runtime_commit = runtime_commit
        self.dataset_sha256 = dataset_sha256
        self.config = config or {}

    def run(self, mock_run: bool = False) -> JobManifest:
        """Execute document-disjoint validation run."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        manifest_p = self.work_dir / "job_manifest.json"

        start_time = time.time()
        snap = take_memory_snapshot()
        print(format_memory_report(snap, stage="Doc-Disjoint Validation Start"))

        metrics_p = self.work_dir / "metrics.json"
        predictions_p = self.work_dir / "predictions.json"
        adapter_dir = self.work_dir / "adapter"

        if mock_run:
            metrics_data = {
                "recall@1": 0.55,
                "recall@3": 0.72,
                "recall@5": 0.81,
                "precision@5": 0.25,
                "candidate_recall@20": 0.88,
                "candidate_recall@50": 0.94,
                "candidate_recall@100": 0.96,
                "candidate_recall@150": 0.98,
            }
            with open(metrics_p, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)
            with open(predictions_p, "w", encoding="utf-8") as f:
                json.dump({}, f)
        else:
            # Check pair file
            pairs_p = self.work_dir / "train_pairs.parquet"
            if not pairs_p.is_file():
                pairs_p = self.work_dir / "reranker_pairs.parquet"
            if not pairs_p.is_file():
                raise FileNotFoundError(f"Missing doc-disjoint train pairs parquet in {self.work_dir}")

            # Train reranker
            cfg = dict(self.config)
            print(f"[*] Training doc-disjoint BGE LoRA reranker on {pairs_p} ...")
            train_reranker(
                pairs_file=pairs_p,
                output_dir=adapter_dir,
                fold=None,
                base_model_name=cfg.get("base_model_name", "mock"),
                max_steps=cfg.get("max_steps", 20),
                batch_size=cfg.get("batch_size", 2),
                learning_rate=cfg.get("learning_rate", 5e-5),
                device=cfg.get("device", "cpu"),
                enforce_full_coverage_steps=cfg.get("enforce_full_coverage_steps", False),
            )

            # Reload adapter
            reranker = CrossEncoderReranker(
                model_name=cfg.get("base_model_name", "BAAI/bge-reranker-v2-m3"),
                adapter_path=adapter_dir,
                device=cfg.get("device", "cpu"),
            )
            reranker.ensure_loaded()

            # Read validation candidates
            val_cands_p = self.work_dir / "validation_candidates.parquet"
            if not val_cands_p.is_file():
                raise FileNotFoundError(f"Missing doc-disjoint validation candidates in {self.work_dir}")

            val_df = pd.read_parquet(val_cands_p)
            pairs_to_score = list(zip(val_df["query_text"].astype(str), val_df["evidence_text"].astype(str)))
            reranker_scores = reranker.score_pairs(pairs_to_score, batch_size=16, max_length=512)
            val_df["reranker_score"] = [float(s) for s in reranker_scores]

            predictions: Dict[str, List[str]] = {}
            gold_dict: Dict[str, List[str]] = {}
            candidate_pools: Dict[str, List[str]] = {}

            for qid, group in val_df.groupby("query_id"):
                qid_str = str(qid)
                first_row = group.iloc[0]
                gold_ids = json.loads(first_row.get("gold_doc_ids", "[]"))
                gold_dict[qid_str] = [str(d) for d in gold_ids]

                group = group.copy()
                fused = group["rrf_score"].astype(float) + 2.5 * group["reranker_score"].astype(float)
                group["final_score"] = fused
                sorted_group = group.sort_values("final_score", ascending=False)
                candidate_pools[qid_str] = sorted_group["doc_id"].astype(str).tolist()
                predictions[qid_str] = candidate_pools[qid_str][:5]

            with open(predictions_p, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2)

            metrics_data = evaluate_predictions(
                y_pred=predictions,
                y_true=gold_dict,
                candidate_pools=candidate_pools,
                cutoffs=[20, 50, 100, 150],
            )
            with open(metrics_p, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)

        outputs = {
            "metrics.json": sha256_file(metrics_p),
            "predictions.json": sha256_file(predictions_p),
        }
        if adapter_dir.is_dir():
            outputs["adapter"] = sha256_directory(adapter_dir)

        duration = time.time() - start_time
        manifest = JobManifest(
            job_id="doc_disjoint",
            job_type="doc_disjoint",
            status="PASS",
            runtime_commit=self.runtime_commit,
            dataset_sha256=self.dataset_sha256,
            inputs={"split": "doc_disjoint"},
            outputs=outputs,
            metrics=metrics_data,
            duration_seconds=duration,
        )
        manifest.save(manifest_p)

        release_memory()
        return manifest
