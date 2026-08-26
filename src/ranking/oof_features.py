from typing import Any
import numpy as np
import pandas as pd
from src.retrieval.types import CandidateRecord

FEATURE_SCHEMA_VERSION = "v2"

FEATURE_COLUMNS: list[str] = [
    "bm25_score",
    "bm25_rank",
    "bm25_inv_rank",
    "bm25_best_score",
    "bm25_second_score",
    "bm25_mean_score",
    "exact_score",
    "exact_match_score",
    "exact_legal_number",
    "exact_title",
    "exact_year",
    "exact_doc_type",
    "memory_score",
    "memory_rank",
    "memory_inv_rank",
    "memory_lexical_similarity",
    "memory_dense_similarity",
    "memory_vote_count",
    "dense_score",
    "dense_rank",
    "dense_inv_rank",
    "dense_best_score",
    "dense_second_score",
    "reranker_score",
    "reranker_best_score",
    "reranker_second_score",
    "reranker_margin",
    "evidence_chunk_count",
    "rrf_score",
    "source_count",
]


def extract_candidate_features(
    query_id: str | None = None,
    candidate_records: list[CandidateRecord] | None = None,
) -> pd.DataFrame:
    """
    Converts list of candidate records into a tabular feature DataFrame for ranking/fusion.
    Flexible signature:
    extract_candidate_features(query_id, candidate_records)
    OR
    extract_candidate_features(candidate_records)
    """
    if isinstance(query_id, list):
        # Called as extract_candidate_features(candidate_records)
        candidate_records = query_id
        query_id = None

    if candidate_records is None:
        candidate_records = []

    rows = []
    for c in candidate_records:
        did = str(c.get("doc_id", ""))
        bm25_r = c.get("bm25_rank")
        dense_r = c.get("dense_rank")
        mem_r = c.get("memory_rank")

        row = {
            "query_id": str(query_id) if query_id is not None else None,
            "doc_id": did,
            "bm25_score": float(c.get("bm25_score", 0.0)),
            "bm25_rank": float(bm25_r) if bm25_r is not None else 100.0,
            "bm25_inv_rank": 1.0 / (60.0 + (bm25_r if bm25_r is not None else 100.0)),
            "bm25_best_score": float(c.get("bm25_best_score", c.get("bm25_score", 0.0))),
            "bm25_second_score": float(c.get("bm25_second_score", 0.0)),
            "bm25_mean_score": float(c.get("bm25_mean_score", 0.0)),
            "exact_score": float(c.get("exact_score", c.get("exact_match_score", 0.0))),
            "exact_match_score": float(c.get("exact_match_score", c.get("exact_score", 0.0))),
            "exact_legal_number": 1.0 if c.get("exact_legal_number") else 0.0,
            "exact_title": 1.0 if c.get("exact_title") else 0.0,
            "exact_year": 1.0 if c.get("exact_year") else 0.0,
            "exact_doc_type": 1.0 if c.get("exact_doc_type") else 0.0,
            "memory_score": float(c.get("memory_score", 0.0)),
            "memory_rank": float(mem_r) if mem_r is not None else 100.0,
            "memory_inv_rank": 1.0 / (60.0 + (mem_r if mem_r is not None else 100.0)),
            "memory_lexical_similarity": float(c.get("memory_lexical_similarity", c.get("memory_score", 0.0))),
            "memory_dense_similarity": float(c.get("memory_dense_similarity", 0.0)),
            "memory_vote_count": float(c.get("memory_vote_count", 0.0)),
            "dense_score": float(c.get("dense_score", 0.0)),
            "dense_rank": float(dense_r) if dense_r is not None else 100.0,
            "dense_inv_rank": 1.0 / (60.0 + (dense_r if dense_r is not None else 100.0)),
            "dense_best_score": float(c.get("dense_best_score", c.get("dense_score", 0.0))),
            "dense_second_score": float(c.get("dense_second_score", 0.0)),
            "reranker_score": float(c.get("reranker_score", -999.0)),
            "reranker_best_score": float(c.get("reranker_best_score", c.get("reranker_score", -999.0))),
            "reranker_second_score": float(c.get("reranker_second_score", -999.0)),
            "reranker_margin": float(c.get("reranker_margin", 0.0)),
            "evidence_chunk_count": float(c.get("evidence_chunk_count", 0.0)),
            "rrf_score": float(c.get("rrf_score", 0.0)),
            "source_count": float(c.get("source_count", 1.0)),
        }
        rows.append(row)

    if not rows:
        cols = ["query_id", "doc_id"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)
