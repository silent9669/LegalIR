"""
Isolated resumable 5-fold OOF job execution, model training, candidate reranking,
OOF feature extraction, and official Codabench metrics evaluation.
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.hashing import sha256_directory, sha256_file
from src.core.manifests import JobManifest
from src.core.memory import (
    check_memory_guard,
    format_memory_report,
    release_memory,
    take_memory_snapshot,
)
from src.evaluation.evaluator import evaluate_predictions
from src.ranking.oof_features import extract_candidate_features
from src.ranking.reranker import CrossEncoderReranker
from src.training.train_reranker import train_reranker


def should_resume_fold(fold_dir: Union[str, Path]) -> bool:
    """
    Verify if a fold job has already completed successfully and all output
    artifacts match their recorded SHA-256 digests.
    """
    fold_p = Path(fold_dir)
    manifest_p = fold_p / "job_manifest.json"
    if not manifest_p.is_file():
        return False

    try:
        manifest = JobManifest.load(manifest_p)
        if manifest.status != "PASS":
            return False

        if not manifest.outputs:
            return False

        for rel_path, expected_sha in manifest.outputs.items():
            artifact_p = fold_p / rel_path
            if not artifact_p.exists():
                return False
            if artifact_p.is_dir():
                actual_sha = sha256_directory(artifact_p)
            else:
                actual_sha = sha256_file(artifact_p)
            if actual_sha != expected_sha:
                return False

        return True
    except Exception:
        return False


class FoldJobRunner:
    """
    Manages the execution of a single OOF validation fold.
    Designed for process-isolated execution so OS reclaims all GPU/host memory on exit.
    """

    def __init__(
        self,
        fold_id: int,
        work_dir: Union[str, Path],
        runtime_commit: str = "unknown",
        dataset_sha256: str = "unknown",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.fold_id = int(fold_id)
        self.work_dir = Path(work_dir) / f"fold_{self.fold_id}"
        self.runtime_commit = runtime_commit
        self.dataset_sha256 = dataset_sha256
        self.config = config or {}

    def run(self, mock_run: bool = False) -> JobManifest:
        """Execute fold job or resume existing verified run."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if should_resume_fold(self.work_dir):
            print(f"[+] Fold {self.fold_id} is already completed and verified. Resuming (skipping).")
            return JobManifest.load(self.work_dir / "job_manifest.json")

        start_time = time.time()
        snap = take_memory_snapshot()
        print(format_memory_report(snap, stage=f"Fold {self.fold_id} Start"))

        # Output artifact paths
        metrics_p = self.work_dir / "fold_metrics.json"
        predictions_p = self.work_dir / "predictions.json"
        oof_features_p = self.work_dir / "oof_features.parquet"
        adapter_dir = self.work_dir / "adapter"

        if mock_run:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
            (adapter_dir / "adapter_model.bin").write_text("mock_weights", encoding="utf-8")

            metrics_data = {"recall@1": 0.60, "recall@3": 0.75, "recall@5": 0.85, "precision@5": 0.28}
            with open(metrics_p, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)
            with open(predictions_p, "w", encoding="utf-8") as f:
                json.dump({}, f)

            tbl = pa.Table.from_pydict({
                "query_id": ["q1"],
                "doc_id": ["d1"],
                "raw_bm25_rank": [1.0],
                "raw_bm25_score": [1.0],
                "reranker_score": [0.9],
                "fold": [self.fold_id],
            })
            pq.write_table(tbl, str(oof_features_p))

            training_report = {
                "optimizer_steps": 50,
                "final_loss": 0.25,
                "param_diff": 0.05,
            }
        else:
            # 1. Verify train pairs file
            pairs_p = self.work_dir / "train_pairs.parquet"
            if not pairs_p.is_file():
                pairs_p = self.work_dir / "reranker_pairs.parquet"
            if not pairs_p.is_file():
                raise FileNotFoundError(f"Missing train pairs parquet in {self.work_dir}")

            # 2. Train BGE+LoRA adapter
            print(f"[*] Training BGE LoRA reranker on {pairs_p} ...")
            cfg = dict(self.config)
            training_report = train_reranker(
                pairs_file=pairs_p,
                output_dir=adapter_dir,
                fold=self.fold_id,
                base_model_name=cfg.get("base_model_name", "mock"),
                max_steps=cfg.get("max_steps", 20),
                batch_size=cfg.get("batch_size", 2),
                learning_rate=cfg.get("learning_rate", 5e-5),
                device=cfg.get("device", "cpu"),
                enforce_full_coverage_steps=cfg.get("enforce_full_coverage_steps", False),
            )

            # 3. Fresh reload adapter
            print(f"[*] Fresh reloading adapter from {adapter_dir} ...")
            reranker = CrossEncoderReranker(
                model_name=cfg.get("base_model_name", "BAAI/bge-reranker-v2-m3"),
                adapter_path=adapter_dir,
                device=cfg.get("device", "cpu"),
            )
            reranker.ensure_loaded()

            # 4. Read validation candidates
            val_cands_p = self.work_dir / "validation_candidates.parquet"
            if not val_cands_p.is_file():
                raise FileNotFoundError(f"Missing validation candidates parquet in {self.work_dir}")

            val_df = pd.read_parquet(val_cands_p)
            if val_df.empty:
                raise ValueError(f"Validation candidates DataFrame in {self.work_dir} is empty")

            # 5. Score candidates with reloaded adapter
            print(f"[*] Scoring {len(val_df)} validation candidates ...")
            pairs_to_score = list(zip(val_df["query_text"].astype(str), val_df["evidence_text"].astype(str)))
            reranker_scores = reranker.score_pairs(pairs_to_score, batch_size=16, max_length=512)
            val_df["reranker_score"] = [float(s) for s in reranker_scores]

            # 6. Rank and select top 5 predictions
            predictions: Dict[str, List[str]] = {}
            gold_dict: Dict[str, List[str]] = {}
            feature_dfs: List[pd.DataFrame] = []

            for qid, group in val_df.groupby("query_id"):
                qid_str = str(qid)
                first_row = group.iloc[0]
                q_text = str(first_row["query_text"])
                gold_ids = json.loads(first_row.get("gold_doc_ids", "[]"))
                gold_dict[qid_str] = [str(d) for d in gold_ids]

                # RRF + reranker fusion scoring
                group = group.copy()
                fused_scores = group["rrf_score"].astype(float) + 2.5 * group["reranker_score"].astype(float)
                group["final_score"] = fused_scores
                sorted_group = group.sort_values("final_score", ascending=False)
                predictions[qid_str] = sorted_group["doc_id"].astype(str).head(5).tolist()

                # Extract candidate feature records
                cand_records = []
                for rank, (_, row) in enumerate(sorted_group.iterrows(), start=1):
                    cand_records.append({
                        "doc_id": str(row["doc_id"]),
                        "rrf_score": float(row["rrf_score"]),
                        "reranker_score": float(row["reranker_score"]),
                        "final_score": float(row["final_score"]),
                        "rank": rank,
                    })

                feat_df = extract_candidate_features(
                    query_id=qid_str,
                    candidate_records=cand_records,
                    query_text=q_text,
                    qrels=gold_dict,
                )
                feat_df["fold"] = self.fold_id
                feature_dfs.append(feat_df)

            # 7. Save OOF features and predictions
            combined_features = pd.concat(feature_dfs, ignore_index=True) if feature_dfs else pd.DataFrame()
            combined_features.to_parquet(oof_features_p, index=False)

            with open(predictions_p, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2)

            # 8. Calculate official metrics
            metrics_data = evaluate_predictions(y_pred=predictions, y_true=gold_dict)
            with open(metrics_p, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)

        # 9. Verify and write job manifest
        outputs = {
            "adapter": sha256_directory(adapter_dir),
            "fold_metrics.json": sha256_file(metrics_p),
            "predictions.json": sha256_file(predictions_p),
            "oof_features.parquet": sha256_file(oof_features_p),
        }

        duration = time.time() - start_time
        manifest = JobManifest(
            job_id=f"fold_{self.fold_id}",
            job_type="fold",
            status="PASS",
            runtime_commit=self.runtime_commit,
            dataset_sha256=self.dataset_sha256,
            inputs={"fold_id": str(self.fold_id)},
            outputs=outputs,
            metrics=metrics_data,
            duration_seconds=duration,
        )
        manifest.save(self.work_dir / "job_manifest.json")

        release_memory()
        return manifest
