"""Candidate union evaluation and feature matrix construction."""

from collections.abc import Iterable, Mapping
from typing import Any
import numpy as np
import pandas as pd

from src.retrieval.types import CandidateRecord

DEFAULT_CANDIDATE_CUTOFFS = (20, 50, 100, 150, 200)


def _as_int(val: Any, default: int = 999) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _as_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def evaluate_candidate_recall(
    candidates: Mapping[str, Any],
    ground_truths: Mapping[str, Any],
    cutoffs: Iterable[int] = DEFAULT_CANDIDATE_CUTOFFS,
) -> dict[str, float]:
    """Compute candidate recall diagnostics at cutoffs (e.g., @20, @50, @100, @150, @200)."""
    normalized_truths: dict[str, set[str]] = {}
    for qid, gold in ground_truths.items():
        if isinstance(gold, (str, bytes)):
            normalized_truths[str(qid)] = {str(gold)}
        elif isinstance(gold, Mapping):
            for k in ("answer", "doc_ids", "doc_id", "document_id"):
                if k in gold:
                    val = gold[k]
                    if isinstance(val, (list, set, tuple)):
                        normalized_truths[str(qid)] = {str(x) for x in val if x is not None}
                    elif val is not None:
                        normalized_truths[str(qid)] = {str(val)}
                    break
        else:
            try:
                normalized_truths[str(qid)] = {str(x) for x in gold if x is not None}
            except TypeError:
                normalized_truths[str(qid)] = {str(gold)}

    cutoff_list = sorted([int(k) for k in cutoffs if int(k) > 0])
    recalls_by_k: dict[int, list[float]] = {k: [] for k in cutoff_list}

    for qid, gold_set in normalized_truths.items():
        if not gold_set:
            continue

        raw_cands = candidates.get(str(qid), candidates.get(qid, []))
        if isinstance(raw_cands, Mapping):
            cand_ids = [str(x) for x in raw_cands.keys()]
        else:
            cand_ids = []
            for item in raw_cands:
                if isinstance(item, Mapping):
                    did = item.get("doc_id", item.get("document_id"))
                    if did is not None:
                        cand_ids.append(str(did))
                elif isinstance(item, (tuple, list)) and item:
                    cand_ids.append(str(item[0]))
                elif item is not None:
                    cand_ids.append(str(item))

        for k in cutoff_list:
            top_k_ids = set(cand_ids[:k])
            recall_k = len(gold_set & top_k_ids) / len(gold_set)
            recalls_by_k[k].append(recall_k)

    return {
        f"candidate_recall@{k}": float(np.mean(vals)) if vals else 0.0
        for k, vals in recalls_by_k.items()
    }


def build_candidate_features(
    query_candidates: Mapping[str, list[CandidateRecord]],
    qrels: Mapping[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Build a tabular feature matrix from CandidateRecord objects for downstream fusion or LTR."""
    rows = []
    gold_map = {}
    if qrels is not None:
        for qid, docs in qrels.items():
            if isinstance(docs, (str, bytes)):
                gold_map[str(qid)] = {str(docs)}
            else:
                gold_map[str(qid)] = {str(d) for d in docs}

    for qid, cands in query_candidates.items():
        qid_str = str(qid)
        golds = gold_map.get(qid_str, set())

        for rank_idx, c in enumerate(cands):
            did = str(c["doc_id"])
            is_gold = int(did in golds) if qrels is not None else None

            row = {
                "query_id": qid_str,
                "doc_id": did,
                "union_rank": rank_idx + 1,
                "rrf_score": _as_float(c.get("rrf_score", 0.0)),
                "source_count": _as_int(c.get("source_count", 1), default=1),
                # BM25 raw/legal features
                "bm25_score": _as_float(c.get("bm25_score", 0.0)),
                "bm25_raw_score": _as_float(c.get("bm25_raw_score", 0.0)),
                "bm25_rank": _as_int(c.get("bm25_rank"), default=999),
                "bm25_best_score": _as_float(c.get("bm25_best_score", 0.0)),
                "bm25_second_score": _as_float(c.get("bm25_second_score", 0.0)),
                "bm25_mean_score": _as_float(c.get("bm25_mean_score", 0.0)),
                "bm25_legal_boost": _as_float(c.get("bm25_legal_boost", 0.0)),
                # PyVi BM25 features
                "bm25_pyvi_score": _as_float(c.get("bm25_pyvi_score", 0.0)),
                "bm25_pyvi_rank": _as_int(c.get("bm25_pyvi_rank"), default=999),
                "bm25_pyvi_best_score": _as_float(c.get("bm25_pyvi_best_score", 0.0)),
                "bm25_pyvi_second_score": _as_float(c.get("bm25_pyvi_second_score", 0.0)),
                # Dense macro features
                "dense_score": _as_float(c.get("dense_score", 0.0)),
                "dense_rank": _as_int(c.get("dense_rank"), default=999),
                "dense_best_score": _as_float(c.get("dense_best_score", 0.0)),
                "dense_second_score": _as_float(c.get("dense_second_score", 0.0)),
                # Question memory features
                "memory_score": _as_float(c.get("memory_score", 0.0)),
                "memory_rank": _as_int(c.get("memory_rank"), default=999),
                "memory_lexical_similarity": _as_float(c.get("memory_lexical_similarity", 0.0)),
                "memory_dense_similarity": _as_float(c.get("memory_dense_similarity", 0.0)),
                "memory_vote_count": _as_int(c.get("memory_vote_count", 0), default=0),
                # Exact match features
                "exact_score": _as_float(c.get("exact_score", 0.0)),
                "exact_legal_number": int(bool(c.get("exact_legal_number", False))),
                "exact_article": int(bool(c.get("exact_article", False))),
                "exact_clause": int(bool(c.get("exact_clause", False))),
                "exact_point": int(bool(c.get("exact_point", False))),
                "exact_year": int(bool(c.get("exact_year", False))),
                "exact_doc_type": int(bool(c.get("exact_doc_type", False))),
                "exact_title": int(bool(c.get("exact_title", False))),
                "exact_title_overlap": _as_float(c.get("exact_title_overlap", 0.0)),
            }
            if is_gold is not None:
                row["label"] = is_gold

            rows.append(row)

    return pd.DataFrame(rows)
