import numpy as np
import pandas as pd
from src.ranking.oof_features import extract_candidate_features

class ReciprocalRankFusion:
    def __init__(
        self,
        k: int = 60,
        w_bm25: float = 1.0,
        w_exact: float = 2.5,
        w_memory: float = 2.0,
        w_dense: float = 1.2,
        w_rerank: float = 1.8
    ):
        self.k = k
        self.w_bm25 = w_bm25
        self.w_exact = w_exact
        self.w_memory = w_memory
        self.w_dense = w_dense
        self.w_rerank = w_rerank

    def rank_candidates(self, candidate_records: list) -> list:
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
            if c.get("exact_match_score", 0.0) > 0.0:
                score += self.w_exact * 0.05 * c["exact_match_score"]

            # Question Memory component
            if c.get("memory_rank") is not None:
                score += self.w_memory / (self.k + c["memory_rank"])

            # Dense component
            if c.get("dense_rank") is not None:
                score += self.w_dense / (self.k + c["dense_rank"])

            # Reranker component (sigmoid calibrated)
            if "reranker_score" in c and c["reranker_score"] is not None:
                r_score = float(c["reranker_score"])
                # Sigmoid scaling
                sig_r = 1.0 / (1.0 + np.exp(-r_score)) if -50 < r_score < 50 else (1.0 if r_score >= 50 else 0.0)
                score += self.w_rerank * 0.05 * sig_r

            item = dict(c)
            item["final_score"] = score
            scored_records.append(item)

        scored_records.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_records

class LightGBMRanker:
    def __init__(self):
        self.model = None
        self.feature_cols = [
            "bm25_rank", "bm25_score", "bm25_inv_rank",
            "exact_match_score",
            "memory_rank", "memory_score", "memory_inv_rank",
            "dense_rank", "dense_score", "dense_inv_rank",
            "reranker_score", "rrf_score"
        ]

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
        import lightgbm as lgb
        train_data = lgb.Dataset(X[self.feature_cols], label=y, group=groups)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1
        }
        self.model = lgb.train(params, train_data, num_boost_round=100)

    def predict(self, candidate_records: list) -> list:
        if self.model is None or not candidate_records:
            # Fallback to RRF
            rrf = ReciprocalRankFusion()
            return rrf.rank_candidates(candidate_records)

        df = extract_candidate_features(candidate_records)
        scores = self.model.predict(df[self.feature_cols])

        scored_records = []
        for c, sc in zip(candidate_records, scores):
            item = dict(c)
            item["final_score"] = float(sc)
            scored_records.append(item)

        scored_records.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_records
