import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from src.ranking.oof_features import FEATURE_COLUMNS, extract_candidate_features


class ReciprocalRankFusion:
    def __init__(
        self,
        k: int = 60,
        w_bm25: float = 1.0,
        w_exact: float = 2.5,
        w_memory: float = 2.0,
        w_dense: float = 1.2,
        w_rerank: float = 1.8,
    ):
        self.k = k
        self.w_bm25 = w_bm25
        self.w_exact = w_exact
        self.w_memory = w_memory
        self.w_dense = w_dense
        self.w_rerank = w_rerank

    def rank_candidates(self, candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Computes final fused score combining retrieval ranks + exact matches + reranker scores.
        """
        scored_records = []
        for c in candidate_records:
            score = 0.0

            # BM25 component
            if c.get("bm25_rank") is not None:
                score += self.w_bm25 / (self.k + c["bm25_rank"])

            # Exact Match component
            exact_sc = float(c.get("exact_match_score", c.get("exact_score", 0.0)))
            if exact_sc > 0.0:
                score += self.w_exact * 0.05 * exact_sc

            # Question Memory component
            if c.get("memory_rank") is not None:
                score += self.w_memory / (self.k + c["memory_rank"])

            # Dense component
            if c.get("dense_rank") is not None:
                score += self.w_dense / (self.k + c["dense_rank"])

            # Reranker component (sigmoid calibrated)
            if "reranker_score" in c and c["reranker_score"] is not None:
                r_score = float(c["reranker_score"])
                if r_score > -900:
                    sig_r = 1.0 / (1.0 + np.exp(-r_score)) if -50 < r_score < 50 else (1.0 if r_score >= 50 else 0.0)
                    score += self.w_rerank * 0.05 * sig_r

            item = dict(c)
            item["final_score"] = float(score)
            scored_records.append(item)

        # Stable sort: descending final_score, ascending doc_id
        scored_records.sort(key=lambda x: (-x["final_score"], x["doc_id"]))
        return scored_records


class LightGBMRanker:
    def __init__(self, model_file: str | Path | None = None):
        self.model = None
        self.feature_cols = list(FEATURE_COLUMNS)
        if model_file is not None:
            self.load(model_file)

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, num_boost_round: int = 100):
        import lightgbm as lgb
        # Ensure only available feature columns
        available_cols = [c for c in self.feature_cols if c in X.columns]
        self.feature_cols = available_cols

        train_data = lgb.Dataset(X[available_cols], label=y, group=groups)
        params = {
            "objective": "lambdarank",
            "learning_rate": 0.05,
            "num_leaves": min(31, max(7, len(X))),
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "verbose": -1,
        }
        self.model = lgb.train(params, train_data, num_boost_round=num_boost_round)

    def predict(self, candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None or not candidate_records:
            rrf = ReciprocalRankFusion()
            return rrf.rank_candidates(candidate_records)

        df = extract_candidate_features(candidate_records=candidate_records)
        available_cols = [c for c in self.feature_cols if c in df.columns]
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        scores = self.model.predict(df[self.feature_cols])

        scored_records = []
        for c, sc in zip(candidate_records, scores):
            item = dict(c)
            item["final_score"] = float(sc)
            scored_records.append(item)

        # Stable sort: descending final_score, ascending doc_id
        scored_records.sort(key=lambda x: (-x["final_score"], x["doc_id"]))
        return scored_records

    def save(self, file_path: str | Path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_model(str(file_path))

    def load(self, file_path: str | Path):
        import lightgbm as lgb
        file_path = Path(file_path)
        if file_path.exists():
            self.model = lgb.Booster(model_file=str(file_path))
            self.feature_cols = self.model.feature_name()
