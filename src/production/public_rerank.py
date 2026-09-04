"""Public candidate reranking and fusion top-5 selector."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import pyarrow.parquet as pq

from src.retrieval.static_cache import StaticCacheReader


def rerank_and_fuse_public_predictions(
    public_candidates_path: Union[str, Path],
    production_lock_path: Union[str, Path],
    top_k: int = 5,
    adapter_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, List[str]]:
    """
    Rerank public candidates combining static retrieval branch ranks and reranker scores,
    yielding top-k predicted document IDs per public query.
    """
    reader = StaticCacheReader(public_candidates_path)
    qids = reader.get_query_ids()

    predictions: Dict[str, List[str]] = {}

    for qid in qids:
        cands = reader.get_query_candidates(qid)
        # Group candidates by doc_id and compute reciprocal rank fusion
        doc_scores: collections.defaultdict[str, float] = collections.defaultdict(float)

        for c in cands:
            rrf_score = 1.0 / (60.0 + c.rank)
            doc_scores[c.doc_id] += rrf_score

        if adapter_scores and qid in adapter_scores:
            for doc_id, score in adapter_scores[qid].items():
                doc_scores[doc_id] += 2.5 * float(score)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        predictions[qid] = [doc for doc, _ in sorted_docs[:top_k]]

    return predictions
