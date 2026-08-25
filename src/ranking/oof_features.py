import pandas as pd
import numpy as np

def extract_candidate_features(candidate_records: list) -> pd.DataFrame:
    """
    Converts list of candidate records into a tabular feature DataFrame for ranking/fusion.
    """
    rows = []
    for c in candidate_records:
        did = str(c["doc_id"])
        bm25_r = c.get("bm25_rank")
        dense_r = c.get("dense_rank")
        mem_r = c.get("memory_rank")

        row = {
            "doc_id": did,
            "bm25_rank": float(bm25_r) if bm25_r is not None else 100.0,
            "bm25_score": float(c.get("bm25_score", 0.0)),
            "bm25_inv_rank": 1.0 / (60.0 + (bm25_r if bm25_r is not None else 100.0)),
            "exact_match_score": float(c.get("exact_match_score", 0.0)),
            "memory_rank": float(mem_r) if mem_r is not None else 100.0,
            "memory_score": float(c.get("memory_score", 0.0)),
            "memory_inv_rank": 1.0 / (60.0 + (mem_r if mem_r is not None else 100.0)),
            "dense_rank": float(dense_r) if dense_r is not None else 100.0,
            "dense_score": float(c.get("dense_score", 0.0)),
            "dense_inv_rank": 1.0 / (60.0 + (dense_r if dense_r is not None else 100.0)),
            "reranker_score": float(c.get("reranker_score", 0.0)),
            "rrf_score": float(c.get("rrf_score", 0.0))
        }
        rows.append(row)

    return pd.DataFrame(rows)
