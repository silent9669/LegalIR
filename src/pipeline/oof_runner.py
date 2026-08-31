"""Out-of-Fold (OOF) 5-fold cross-validation runner for LegalIR Task 1.

Provides leakage-safe cross-validation orchestration, feature generation,
official Codabench scorer parity, and full metrics reporting.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.paths import ProjectPaths
from src.evaluation.codabench_compat import assert_official_equivalence
from src.evaluation.evaluator import (
    DEFAULT_CANDIDATE_CUTOFFS,
    FINAL_RANKING_METRICS,
    compute_candidate_cutoffs,
    compute_candidate_recall,
    evaluate_predictions,
    normalize_candidate_cutoffs,
)
from src.models.parameter_audit import audit_system_parameters, MAX_PARAMETER_BUDGET
from src.evaluation.splits import (
    generate_document_disjoint_split,
    generate_random_5fold_split,
    verify_document_disjoint_isolation,
    verify_fold_isolation,
)
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.oof_features import extract_candidate_features
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.selector import TopKSelector
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import TrainQuestionMemory
from src.retrieval.types import CandidateRecord


class OOFRunner:
    """Orchestrates leakage-safe 5-fold OOF cross-validation and feature extraction."""

    def __init__(
        self,
        data_dir: str | Path = "artifacts/task1/data",
        index_dir: str | Path = "artifacts/task1/indexes",
        splits_path: str | Path = "artifacts/shared/canonical/v2/splits/random_5fold.json",
        output_dir: str | Path = "artifacts/local/cv",
        config_path: str | Path | None = "configs/pipeline.yaml",
        num_folds: int = 5,
        candidate_k: int = 150,
        rerank_k: int = 50,
        use_reranker: bool = False,
        reranker_model: str = "mock",
        train_reranker_per_fold: bool = False,
        device: str | None = None,
        smoke: bool = False,
        smoke_sample_size: int = 20,
        doc_disjoint: bool = False,
        doc_disjoint_splits_path: str | Path = "artifacts/shared/canonical/v2/splits/doc_disjoint_split.json",
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.splits_path = Path(splits_path)
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path) if config_path else None
        self.num_folds = int(num_folds)
        self.candidate_k = int(candidate_k)
        self.rerank_k = int(rerank_k)
        self.use_reranker = bool(use_reranker)
        self.reranker_model = str(reranker_model)
        self.train_reranker_per_fold = bool(train_reranker_per_fold)
        self.device = device
        self.smoke = bool(smoke)
        self.smoke_sample_size = int(smoke_sample_size)
        self.doc_disjoint = bool(doc_disjoint)
        self.doc_disjoint_splits_path = Path(doc_disjoint_splits_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded attributes
        self.docs_df: pd.DataFrame | None = None
        self.queries_df: pd.DataFrame | None = None
        self.qrels_df: pd.DataFrame | None = None
        self.doc_map: dict[str, dict[str, Any]] = {}
        self.queries_map: dict[str, str] = {}
        self.qrels_map: dict[str, list[str]] = defaultdict(list)
        self.evidence_builder: EvidencePackBuilder | None = None
        self.bm25: BM25MicroRetriever | None = None
        self.dense: DenseMacroRetriever | None = None
        self.exact: ExactMatcher | None = None
        self.selector = TopKSelector(max_k=5)

    def load_data(self) -> None:
        """Load canonical dataset tables and build lookups."""
        if self.docs_df is not None:
            return

        docs_path = self.data_dir / "documents.parquet"
        queries_path = self.data_dir / "queries_train.parquet"
        qrels_path = self.data_dir / "qrels_train.parquet"

        if not (docs_path.exists() and queries_path.exists() and qrels_path.exists()):
            raise FileNotFoundError(f"Canonical data files missing in {self.data_dir}")

        self.docs_df = pd.read_parquet(docs_path)
        self.queries_df = pd.read_parquet(queries_path)
        self.qrels_df = pd.read_parquet(qrels_path)

        for r in self.docs_df.to_dict("records"):
            self.doc_map[str(r["doc_id"])] = r

        for r in self.queries_df.to_dict("records"):
            # Prefer normalized question text, fallback to raw
            q_text = r.get("question_norm") or r.get("question_raw") or r.get("question") or ""
            self.queries_map[str(r["query_id"])] = str(q_text)

        self.qrels_map = defaultdict(list)
        for r in self.qrels_df.to_dict("records"):
            self.qrels_map[str(r["query_id"])].append(str(r["doc_id"]))

        # Build evidence pack builder if chunks exist
        chunks_path = self.data_dir / "chunks.parquet"
        if chunks_path.exists():
            self.evidence_builder = EvidencePackBuilder(
                chunks_path=chunks_path,
                doc_metadata=self.doc_map,
            )

    def load_retrievers(self) -> None:
        """Load or initialize retrieval components."""
        if self.exact is None:
            self.exact = ExactMatcher(documents=list(self.doc_map.values()))

        if self.bm25 is None:
            bm25_index_dir = self.index_dir / "bm25"
            bm25_file = bm25_index_dir / "bm25_micro_index.pkl"
            if bm25_file.exists():
                self.bm25 = BM25MicroRetriever.load(bm25_file)
            elif bm25_index_dir.exists() and list(bm25_index_dir.glob("*.pkl")):
                self.bm25 = BM25MicroRetriever.load(bm25_index_dir)
            else:
                # Fit on micro chunks
                chunks_path = self.data_dir / "chunks.parquet"
                if chunks_path.exists():
                    chunks_df = pd.read_parquet(chunks_path)
                    micro_chunks = chunks_df[chunks_df["granularity"] == "micro"].to_dict("records")
                    self.bm25 = BM25MicroRetriever()
                    self.bm25.fit(micro_chunks, show_progress=False)

        if self.dense is None:
            dense_dek21 = self.index_dir / "dense_dek21"
            dense_std = self.index_dir / "dense"
            dense_path = dense_dek21 if dense_dek21.exists() else dense_std
            if dense_path.exists() and (dense_path / "embeddings.npy").exists():
                try:
                    self.dense = DenseMacroRetriever.load(dense_path, device=self.device)
                except Exception as e:
                    print(f"Warning: Dense retriever could not be loaded from {dense_path}: {e}")
                    self.dense = None

    def get_splits(self) -> list[dict[str, Any]]:
        """Load or generate 5-fold cross-validation splits and verify isolation."""
        if self.splits_path.exists():
            with open(self.splits_path, "r", encoding="utf-8") as f:
                folds = json.load(f)
        else:
            print(f"Splits not found at {self.splits_path}; generating fresh 5-fold split...")
            queries_list = [{"query_id": qid} for qid in self.queries_map.keys()]
            folds = generate_random_5fold_split(queries_list, seed=42)
            self.splits_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.splits_path, "w", encoding="utf-8") as f:
                json.dump(folds, f, indent=2)

        # Strict fold isolation verification
        verify_fold_isolation(folds, self.qrels_map)
        return folds

    def run_fold(
        self,
        fold_idx: int,
        fold_info: dict[str, Any],
        reranker: CrossEncoderReranker | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], list[pd.DataFrame], dict[str, Any], dict[str, float]]:
        """
        Execute OOF evaluation on a single fold with strict isolation guarantees:
        1. Construct TrainQuestionMemory on training queries only.
        2. Retrieve top candidates for validation queries.
        3. Extract candidate features.
        4. Optional cross-encoder reranking.
        5. Top-5 selection and metric computation.
        6. Assert official scorer equivalence.
        """
        train_ids = set(str(x) for x in fold_info.get("train_query_ids", fold_info.get("train", [])))
        val_ids = [str(x) for x in fold_info.get("val_query_ids", fold_info.get("val", []))]

        if self.smoke:
            val_ids = val_ids[: self.smoke_sample_size]

        fold_train_queries = {qid: self.queries_map[qid] for qid in train_ids if qid in self.queries_map}
        fold_train_qrels = {qid: self.qrels_map[qid] for qid in train_ids if qid in self.qrels_map}

        # Build fold-isolated question memory
        memory = TrainQuestionMemory(min_similarity=0.82, dense_encoder=self.dense)
        memory.fit(fold_train_queries, fold_train_qrels)

        # Strict validation: memory must not contain any validation query
        leaked_val = set(val_ids) & memory.training_query_ids
        if leaked_val:
            raise AssertionError(f"Fold {fold_idx} Question Memory contains validation queries: {leaked_val}")

        hybrid_engine = HybridSearchEngine(
            bm25_retriever=self.bm25,
            dense_retriever=self.dense,
            question_memory=memory,
            exact_matcher=self.exact,
        )

        fold_preds: dict[str, list[str]] = {}
        fold_candidates: dict[str, list[str]] = {}
        fold_feature_dfs: list[pd.DataFrame] = []
        fold_runtimes: dict[str, float] = {}

        t0 = time.time()
        for qid in tqdm(val_ids, desc=f"Fold {fold_idx} OOF Inference", leave=False):
            q_text = self.queries_map.get(qid, "")
            t_q0 = time.time()

            candidates: list[CandidateRecord] = hybrid_engine.search_candidates(
                query=q_text,
                top_k=self.candidate_k,
                exclude_qid=str(qid),
            )
            cand_ids = [str(c["doc_id"]) for c in candidates]
            fold_candidates[qid] = cand_ids

            # Extract features for candidate union
            feat_df = extract_candidate_features(query_id=qid, candidate_records=candidates)
            if not feat_df.empty:
                gold_set = set(self.qrels_map.get(qid, []))
                feat_df["label"] = feat_df["doc_id"].apply(lambda did: 1 if did in gold_set else 0)
                feat_df["fold"] = fold_idx
                fold_feature_dfs.append(feat_df)

            # Rerank if configured
            if reranker is not None and self.evidence_builder is not None:
                candidates = reranker.rerank(
                    query=q_text,
                    candidates=candidates,
                    evidence_builder=self.evidence_builder,
                    top_k=self.rerank_k,
                )

            top5 = self.selector.select(candidates)
            fold_preds[qid] = top5
            fold_runtimes[qid] = time.time() - t_q0

        elapsed_total = time.time() - t0

        # Evaluate fold metrics
        fold_gold = {qid: self.qrels_map[qid] for qid in val_ids}
        fold_metrics = evaluate_predictions(
            y_pred=fold_preds,
            y_true=fold_gold,
            candidate_pools=fold_candidates,
            runtimes=fold_runtimes,
            cutoffs=DEFAULT_CANDIDATE_CUTOFFS,
        )
        fold_metrics["fold"] = fold_idx
        fold_metrics["val_queries"] = len(val_ids)
        fold_metrics["elapsed_seconds"] = elapsed_total

        # Verify exact Codabench official scorer equivalence
        assert_official_equivalence(fold_preds, fold_gold)

        return fold_preds, fold_candidates, fold_feature_dfs, fold_metrics, fold_runtimes

    def run(self) -> dict[str, Any]:
        """
        Execute full 5-fold out-of-fold cross-validation pipeline:
        - Orchestrates 5 folds with strict isolation
        - Gathers OOF predictions, OOF candidate pools, and OOF candidate features
        - Exports oof_predictions.parquet, oof_features.parquet, and cv_report.json
        """
        print("=" * 70)
        mode_str = "SMOKE / FAST" if self.smoke else "FULL"
        print(f"LegalIR Task 1: 5-Fold OOF Validation [{mode_str} MODE]")
        print("=" * 70)

        # Preflight parameter budget audit
        audit_path = self.output_dir / "parameter_audit.json"
        audit_report = audit_system_parameters(
            config_path=self.config_path,
            output_json=audit_path,
            raise_on_violation=True,
            offline_fallback=True,
        )
        print(
            f"[Preflight] Parameter Budget Audit: {audit_report['total_learned_parameters']:,} params "
            f"({audit_report['total_parameters_billions']:.4f}B / 4.0B, "
            f"{audit_report['budget_utilization_pct']:.2f}% utilization). PASS\n"
        )

        self.load_data()
        self.load_retrievers()
        folds = self.get_splits()

        reranker: CrossEncoderReranker | None = None
        if self.use_reranker:
            print(f"Initializing CrossEncoderReranker: {self.reranker_model}...")
            reranker = CrossEncoderReranker(model_name=self.reranker_model, device=self.device)

        all_oof_predictions: dict[str, list[str]] = {}
        all_candidate_pools: dict[str, list[str]] = {}
        all_feature_dfs: list[pd.DataFrame] = []
        all_runtimes: dict[str, float] = {}
        fold_records: list[dict[str, Any]] = []

        total_t0 = time.time()
        active_folds = folds[: self.num_folds]

        for f_idx, fold_info in enumerate(active_folds):
            print(f"\n>>> Running Fold {f_idx + 1}/{len(active_folds)} (Fold {f_idx})...")
            f_preds, f_cands, f_feat_dfs, f_metrics, f_runtimes = self.run_fold(
                fold_idx=f_idx,
                fold_info=fold_info,
                reranker=reranker,
            )

            all_oof_predictions.update(f_preds)
            all_candidate_pools.update(f_cands)
            all_feature_dfs.extend(f_feat_dfs)
            all_runtimes.update(f_runtimes)
            fold_records.append(f_metrics)

            rec5 = f_metrics.get("recall@5", 0.0)
            prec5 = f_metrics.get("precision@5", 0.0)
            cand50 = f_metrics.get("candidate_recall@50", 0.0)
            cand150 = f_metrics.get("candidate_recall@150", 0.0)
            mrr = f_metrics.get("mrr", 0.0)
            map_score = f_metrics.get("map", 0.0)
            ndcg5 = f_metrics.get("ndcg@5", 0.0)
            elapsed = f_metrics.get("elapsed_seconds", 0.0)

            print(
                f"Fold {f_idx} Results: "
                f"Recall@5 = {rec5 * 100:.2f}% | "
                f"Prec@5 = {prec5 * 100:.2f}% | "
                f"MRR = {mrr:.4f} | "
                f"MAP = {map_score:.4f} | "
                f"nDCG@5 = {ndcg5:.4f} | "
                f"Cand@50 = {cand50 * 100:.2f}% | "
                f"Cand@150 = {cand150 * 100:.2f}% "
                f"({elapsed:.1f}s)"
            )

        overall_elapsed = time.time() - total_t0

        # Global aggregate evaluation across all OOF queries
        all_gold = {qid: self.qrels_map[qid] for qid in all_oof_predictions.keys()}
        overall_metrics = evaluate_predictions(
            y_pred=all_oof_predictions,
            y_true=all_gold,
            candidate_pools=all_candidate_pools,
            runtimes=all_runtimes,
            cutoffs=DEFAULT_CANDIDATE_CUTOFFS,
        )

        # Cross-fold mean and std calculations
        rec5_scores = [f["recall@5"] for f in fold_records]
        prec5_scores = [f["precision@5"] for f in fold_records]
        rec1_scores = [f["recall@1"] for f in fold_records]
        rec3_scores = [f["recall@3"] for f in fold_records]
        mrr_scores = [f.get("mrr", 0.0) for f in fold_records]
        map_scores = [f.get("map", 0.0) for f in fold_records]
        ndcg5_scores = [f.get("ndcg@5", 0.0) for f in fold_records]

        cand_means = {
            f"mean_candidate@{k}": float(np.mean([f.get(f"candidate_recall@{k}", 0.0) for f in fold_records]))
            for k in DEFAULT_CANDIDATE_CUTOFFS
        }

        cv_report = {
            "mean_recall@5": float(np.mean(rec5_scores)),
            "std_recall@5": float(np.std(rec5_scores)),
            "mean_precision@5": float(np.mean(prec5_scores)),
            "std_precision@5": float(np.std(prec5_scores)),
            "mean_recall@1": float(np.mean(rec1_scores)),
            "mean_recall@3": float(np.mean(rec3_scores)),
            "mean_mrr": float(np.mean(mrr_scores)),
            "mean_map": float(np.mean(map_scores)),
            "mean_ndcg@5": float(np.mean(ndcg5_scores)),
            **cand_means,
            "overall_aggregate_metrics": overall_metrics,
            "total_evaluated_queries": len(all_oof_predictions),
            "total_runtime_seconds": overall_elapsed,
            "runtime_per_query_ms": float(np.mean(list(all_runtimes.values()))) * 1000.0 if all_runtimes else 0.0,
            "queries_per_second": (len(all_oof_predictions) / overall_elapsed) if overall_elapsed > 0 else 0.0,
            "official_scorer_parity_verified": True,
            "is_smoke_mode": self.smoke,
            "num_folds": len(active_folds),
            "folds": fold_records,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Save cv_report.json
        report_path = self.output_dir / "cv_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cv_report, f, indent=2)

        # 2. Save oof_predictions.parquet & oof_predictions.json
        pred_records = []
        for f_idx, fold_info in enumerate(active_folds):
            val_ids = [str(x) for x in fold_info.get("val_query_ids", fold_info.get("val", []))]
            if self.smoke:
                val_ids = val_ids[: self.smoke_sample_size]
            for qid in val_ids:
                pred_records.append({
                    "query_id": str(qid),
                    "answer": all_oof_predictions.get(qid, []),
                    "fold": f_idx,
                })

        pred_df = pd.DataFrame(pred_records)
        pred_parquet_path = self.output_dir / "oof_predictions.parquet"
        pred_df.to_parquet(pred_parquet_path, index=False)

        pred_json_path = self.output_dir / "oof_predictions.json"
        codabench_preds = {r["query_id"]: {"answer": r["answer"]} for r in pred_records}
        with open(pred_json_path, "w", encoding="utf-8") as f:
            json.dump(codabench_preds, f, indent=2)

        # 3. Save oof_features.parquet
        feat_parquet_path = self.output_dir / "oof_features.parquet"
        if all_feature_dfs:
            combined_features_df = pd.concat(all_feature_dfs, ignore_index=True)
            combined_features_df.to_parquet(feat_parquet_path, index=False)
            print(f"Saved OOF Features: {feat_parquet_path} ({len(combined_features_df)} candidate rows)")
        else:
            # Create empty placeholder DataFrame with schema
            empty_df = pd.DataFrame(columns=["query_id", "doc_id", "label", "fold"])
            empty_df.to_parquet(feat_parquet_path, index=False)

        # 4. Optional Document-Disjoint Robustness Split Evaluation
        if self.doc_disjoint:
            print("\n>>> Running Document-Disjoint Robustness Split Evaluation...")
            self.run_document_disjoint_evaluation(reranker=reranker)

        # Print final summary
        print("\n" + "=" * 70)
        print(">> 5-FOLD OUT-OF-FOLD (OOF) CV SUMMARY:")
        print(f"   Mean Recall@5       : {cv_report['mean_recall@5'] * 100:.4f}% (+/- {cv_report['std_recall@5'] * 100:.4f}%)")
        print(f"   Mean Precision@5    : {cv_report['mean_precision@5'] * 100:.4f}% (+/- {cv_report['std_precision@5'] * 100:.4f}%)")
        print(f"   Mean MRR            : {cv_report['mean_mrr']:.4f}")
        print(f"   Mean MAP            : {cv_report['mean_map']:.4f}")
        print(f"   Mean nDCG@5         : {cv_report['mean_ndcg@5']:.4f}")
        print(f"   Candidate Recall@50 : {cv_report.get('mean_candidate@50', 0.0) * 100:.4f}%")
        print(f"   Candidate Recall@150: {cv_report.get('mean_candidate@150', 0.0) * 100:.4f}%")
        print(f"   Candidate Recall@200: {cv_report.get('mean_candidate@200', 0.0) * 100:.4f}%")
        print(f"   Total Queries       : {cv_report['total_evaluated_queries']}")
        print(f"   Runtime / Query     : {cv_report['runtime_per_query_ms']:.2f} ms")
        print(f"   Artifacts Exported  : {self.output_dir}")
        print("=" * 70)

        return cv_report

    def run_document_disjoint_evaluation(
        self,
        reranker: CrossEncoderReranker | None = None,
    ) -> dict[str, Any]:
        """Evaluate document-disjoint split to test generalization to unseen documents."""
        if self.doc_disjoint_splits_path.exists():
            with open(self.doc_disjoint_splits_path, "r", encoding="utf-8") as f:
                disjoint_split = json.load(f)
        else:
            print("Generating fresh document-disjoint split...")
            queries_list = [{"query_id": qid} for qid in self.queries_map.keys()]
            qrels_list = self.qrels_df.to_dict("records") if self.qrels_df is not None else []
            disjoint_split = generate_document_disjoint_split(queries_list, qrels_list, val_ratio=0.2, seed=42)
            self.doc_disjoint_splits_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.doc_disjoint_splits_path, "w", encoding="utf-8") as f:
                json.dump(disjoint_split, f, indent=2)

        # Verify strict document disjoint isolation
        verify_document_disjoint_isolation(disjoint_split, self.qrels_map)

        train_ids = set(str(x) for x in disjoint_split.get("train_query_ids", disjoint_split.get("train", [])))
        val_ids = [str(x) for x in disjoint_split.get("val_query_ids", disjoint_split.get("val", []))]

        if self.smoke:
            val_ids = val_ids[: self.smoke_sample_size]

        fold_train_queries = {qid: self.queries_map[qid] for qid in train_ids if qid in self.queries_map}
        fold_train_qrels = {qid: self.qrels_map[qid] for qid in train_ids if qid in self.qrels_map}

        memory = TrainQuestionMemory(min_similarity=0.82, dense_encoder=self.dense)
        memory.fit(fold_train_queries, fold_train_qrels)

        hybrid_engine = HybridSearchEngine(
            bm25_retriever=self.bm25,
            dense_retriever=self.dense,
            question_memory=memory,
            exact_matcher=self.exact,
        )

        preds = {}
        candidates_map = {}
        runtimes = {}

        t0 = time.time()
        for qid in tqdm(val_ids, desc="Document-Disjoint Eval", leave=False):
            q_text = self.queries_map.get(qid, "")
            t_q0 = time.time()

            cands = hybrid_engine.search_candidates(
                query=q_text,
                top_k=self.candidate_k,
                exclude_qid=str(qid),
            )
            candidates_map[qid] = [str(c["doc_id"]) for c in cands]

            if reranker is not None and self.evidence_builder is not None:
                cands = reranker.rerank(
                    query=q_text,
                    candidates=cands,
                    evidence_builder=self.evidence_builder,
                    top_k=self.rerank_k,
                )

            top5 = self.selector.select(cands)
            preds[qid] = top5
            runtimes[qid] = time.time() - t_q0

        gold = {qid: self.qrels_map[qid] for qid in val_ids}
        metrics = evaluate_predictions(
            y_pred=preds,
            y_true=gold,
            candidate_pools=candidates_map,
            runtimes=runtimes,
            cutoffs=DEFAULT_CANDIDATE_CUTOFFS,
        )
        metrics["elapsed_seconds"] = time.time() - t0
        metrics["val_queries"] = len(val_ids)

        assert_official_equivalence(preds, gold)

        report_path = self.output_dir / "doc_disjoint_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Document-Disjoint Split Recall@5: {metrics['recall@5'] * 100:.2f}% | Precision@5: {metrics['precision@5'] * 100:.2f}%")
        return metrics
