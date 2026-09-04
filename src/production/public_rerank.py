"""
Public candidate reranking and frozen fusion top-5 prediction selector.
Loads:
- public_candidates.parquet
- public_evidence.parquet
- final trained adapter
- production_lock.json
- frozen fusion artifact
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.static_cache import StaticCacheReader


def load_public_evidence_map(public_evidence_path: Union[str, Path]) -> Dict[Tuple[str, str], str]:
    """Load pre-materialized public evidence mapping (query_id, doc_id) -> evidence_text."""
    evidence_p = Path(public_evidence_path)
    if not evidence_p.is_file():
        return {}

    df = pd.read_parquet(evidence_p)
    res = {}
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        did = str(row["doc_id"])
        ev_text = str(row.get("evidence_text", ""))
        res[(qid, did)] = ev_text
    return res


def rerank_and_fuse_public_predictions(
    public_candidates_path: Union[str, Path],
    production_lock_path: Union[str, Path],
    adapter_dir: Optional[Union[str, Path]] = None,
    public_evidence_path: Optional[Union[str, Path]] = None,
    fusion_model_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
    rerank_k: int = 50,
    device: str = "auto",
    adapter_scores: Optional[Dict[str, Dict[str, float]]] = None,
    public_queries_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """
    Rerank public candidates combining static retrieval branch ranks and reranker scores
    under the frozen production configuration from production_lock.json.
    """
    lock_p = Path(production_lock_path)
    if not lock_p.is_file():
        raise FileNotFoundError(f"Missing production lock at {lock_p}")

    with open(lock_p, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    cfg = lock_data.get("config", {})
    fusion_cfg = cfg.get("fusion", {})
    weights = fusion_cfg.get("weights", {
        "bm25": 1.0,
        "bm25_pyvi": 1.0,
        "dense": 1.0,
        "exact": 0.5,
        "reranker": 2.5,
    })
    eff_top_k = int(fusion_cfg.get("top_k", top_k))
    eff_rerank_k = int(fusion_cfg.get("rerank_k", rerank_k))

    # Read public candidates
    reader = StaticCacheReader(public_candidates_path)
    qids = reader.get_query_ids()

    # Evidence map
    evidence_map = load_public_evidence_map(public_evidence_path) if public_evidence_path else {}

    # Load final adapter if provided and no adapter_scores passed
    reranker = None
    if adapter_dir is not None and Path(adapter_dir).is_dir() and adapter_scores is None:
        reranker_cfg = cfg.get("reranker", {})
        base_model = reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3")
        print(f"[*] Loading final adapter from {adapter_dir} for public reranking ...")
        reranker = CrossEncoderReranker(
            model_name=base_model,
            adapter_path=adapter_dir,
            device=device,
        )
        reranker.ensure_loaded()

    computed_adapter_scores: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    if adapter_scores is not None:
        computed_adapter_scores = adapter_scores

    predictions: Dict[str, List[str]] = {}

    for qid in qids:
        cands = reader.get_query_candidates(qid)
        if not cands:
            continue

        # Initial RRF fusion across static branches
        doc_scores: Dict[str, float] = collections.defaultdict(float)
        doc_cands: Dict[str, List[Any]] = collections.defaultdict(list)

        for c in cands:
            b = c.branch
            if b == "bm25_legal":
                b = "bm25"
            w = float(weights.get(b, 1.0))
            rrf_contrib = w / (60.0 + c.rank)
            doc_scores[c.doc_id] += rrf_contrib
            doc_cands[c.doc_id].append(c)

        # Sort top candidates to rerank
        sorted_by_rrf = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        docs_to_rerank = [d for d, _ in sorted_by_rrf[:eff_rerank_k]]

        # Score with reranker if available
        if reranker is not None and public_queries_dict is not None and qid in public_queries_dict:
            q_text = public_queries_dict[qid]
            pairs_to_score = []
            valid_rerank_docs = []
            for did in docs_to_rerank:
                ev_text = evidence_map.get((qid, did), "")
                if ev_text:
                    pairs_to_score.append((q_text, ev_text))
                    valid_rerank_docs.append(did)

            if pairs_to_score:
                scores = reranker.score_pairs(pairs_to_score, batch_size=16, max_length=512)
                for did, s in zip(valid_rerank_docs, scores):
                    computed_adapter_scores[qid][did] = float(s)

        # Combine static scores with reranker scores
        w_reranker = float(weights.get("reranker", 2.5))
        final_scores: Dict[str, float] = {}

        for did, base_score in doc_scores.items():
            r_score = computed_adapter_scores.get(qid, {}).get(did, 0.0)
            final_scores[did] = base_score + w_reranker * r_score

        # Select top-k unique doc IDs
        sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        preds = [did for did, _ in sorted_final[:eff_top_k]]
        predictions[qid] = preds

    return predictions
