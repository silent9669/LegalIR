from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np


DEFAULT_CANDIDATE_CUTOFFS = (20, 50, 100)
FINAL_RANKING_METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "precision_at_5")


def _normalize_ids(value: Any) -> list[str]:
    """Normalize a prediction or qrel value to an ordered list of IDs."""
    if isinstance(value, Mapping):
        if "answer" in value:
            value = value["answer"]
        elif "doc_ids" in value:
            value = value["doc_ids"]
        elif "doc_id" in value:
            value = [value["doc_id"]]
        else:
            value = []
    elif isinstance(value, (str, bytes)):
        value = [value]
    elif value is None:
        value = []
    else:
        try:
            value = list(value)
        except TypeError:
            value = [value]

    return [str(item) for item in value if item is not None]


def _normalize_query_values(values: Mapping[Any, Any]) -> dict[str, list[str]]:
    return {str(query_id): _normalize_ids(value) for query_id, value in values.items()}


def _candidate_doc_id(candidate: Any) -> str | None:
    if isinstance(candidate, Mapping):
        candidate = candidate.get("doc_id", candidate.get("document_id"))
    elif isinstance(candidate, (tuple, list)):
        candidate = candidate[0] if candidate else None
    if candidate is None:
        return None
    return str(candidate)


def normalize_candidate_cutoffs(cutoffs: Iterable[int] | None = None) -> list[int]:
    """Return validated, de-duplicated candidate cutoffs in caller order."""
    values = DEFAULT_CANDIDATE_CUTOFFS if cutoffs is None else cutoffs
    normalized: list[int] = []
    for cutoff in values:
        if isinstance(cutoff, bool):
            raise ValueError("candidate cutoffs must be positive integers")
        try:
            cutoff_int = int(cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate cutoffs must be positive integers") from exc
        if cutoff_int <= 0:
            raise ValueError("candidate cutoffs must be positive integers")
        if cutoff_int not in normalized:
            normalized.append(cutoff_int)
    if not normalized:
        raise ValueError("at least one candidate cutoff is required")
    return normalized


def evaluate_predictions(y_pred: dict[str, Any], y_true: dict[str, Any]) -> dict[str, Any]:
    """Exact Codabench retrieval evaluation function."""
    normalized_pred = _normalize_query_values(y_pred)
    normalized_true = _normalize_query_values(y_true)

    recalls = []
    precisions = []
    r1_list = []
    r3_list = []
    r5_list = []

    for qid, gold_list in normalized_true.items():
        gold_set = set(gold_list)
        if not gold_set:
            continue

        preds = normalized_pred.get(qid, [])
        pred_set = set(preds)

        # Codabench rule: if 0 < len(preds) <= 5, compute set intersection ratio, else 0
        if 0 < len(preds) <= 5:
            rec = len(gold_set & pred_set) / len(gold_set)
            prec = len(gold_set & pred_set) / len(preds)
        else:
            rec = 0.0
            prec = 0.0

        recalls.append(rec)
        precisions.append(prec)

        # Standard Top-K Recall metrics
        top1 = set(preds[:1])
        top3 = set(preds[:3])
        top5 = set(preds[:5])
        r1_list.append(len(gold_set & top1) / len(gold_set))
        r3_list.append(len(gold_set & top3) / len(gold_set))
        r5_list.append(len(gold_set & top5) / len(gold_set))

    mean_rec = float(np.mean(recalls)) if recalls else 0.0
    mean_prec = float(np.mean(precisions)) if precisions else 0.0
    mean_r1 = float(np.mean(r1_list)) if r1_list else 0.0
    mean_r3 = float(np.mean(r3_list)) if r3_list else 0.0
    mean_r5 = float(np.mean(r5_list)) if r5_list else 0.0

    return {
        # Codabench-compatible aggregate names.
        "recall": mean_rec,
        "precision": mean_prec,
        # Machine-friendly final ranking metric names.
        "recall_at_1": mean_r1,
        "recall_at_3": mean_r3,
        "recall_at_5": mean_r5,
        "precision_at_5": mean_prec,
        # Public report aliases used by the benchmark acceptance contract.
        "recall@1": mean_r1,
        "recall@3": mean_r3,
        "recall@5": mean_r5,
        "precision@5": mean_prec,
        "Recall@1": mean_r1,
        "Recall@3": mean_r3,
        "Recall@5": mean_r5,
        "Precision@5": mean_prec,
        "total_evaluated_queries": len(recalls),
    }


def compute_candidate_recall(
    candidates: dict[str, Any],
    ground_truths: dict[str, Any],
    k: int = 50,
) -> float:
    """Compute candidate recall at ``k`` before final reranking/fusion."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("candidate recall cutoff must be a positive integer")

    normalized_candidates = {str(qid): value for qid, value in candidates.items()}
    normalized_truths = _normalize_query_values(ground_truths)
    recalls = []
    for qid, gold_list in normalized_truths.items():
        gold_set = set(gold_list)
        if not gold_set:
            continue

        raw_candidates = normalized_candidates.get(qid, [])
        try:
            raw_candidates = list(raw_candidates)
        except TypeError:
            raw_candidates = [raw_candidates]
        candidate_ids = [
            doc_id
            for doc_id in (_candidate_doc_id(candidate) for candidate in raw_candidates[:k])
            if doc_id is not None
        ]
        recalls.append(len(gold_set & set(candidate_ids)) / len(gold_set))

    return float(np.mean(recalls)) if recalls else 0.0


def compute_candidate_cutoffs(
    candidates: dict[str, Any],
    ground_truths: dict[str, Any],
    cutoffs: Iterable[int] | None = None,
) -> dict[str, float]:
    """Compute candidate recall for the standard diagnostic cutoffs."""
    normalized_cutoffs = normalize_candidate_cutoffs(cutoffs)
    return {
        f"candidate_recall@{cutoff}": compute_candidate_recall(
            candidates,
            ground_truths,
            k=cutoff,
        )
        for cutoff in normalized_cutoffs
    }
