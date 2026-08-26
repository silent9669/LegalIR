from typing import Any
import numpy as np


def evaluate_predictions(y_pred: dict[str, Any], y_true: dict[str, Any]) -> dict[str, Any]:
    """Exact Codabench retrieval evaluation function."""
    normalized_pred = {}
    for k, v in y_pred.items():
        if isinstance(v, dict) and "answer" in v:
            ans = v["answer"]
        elif isinstance(v, list):
            ans = v
        else:
            ans = []
        normalized_pred[str(k)] = [str(x) for x in ans]

    normalized_true = {str(k): [str(x) for x in v] for k, v in y_true.items()}

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
        "recall": mean_rec,
        "precision": mean_prec,
        "recall_at_1": mean_r1,
        "recall_at_3": mean_r3,
        "recall_at_5": mean_r5,
        "precision_at_5": mean_prec,
        "Recall@1": mean_r1,
        "Recall@3": mean_r3,
        "Recall@5": mean_r5,
        "total_evaluated_queries": len(recalls),
    }


def compute_candidate_recall(candidates: dict[str, Any], ground_truths: dict[str, Any], k: int = 50) -> float:
    """Compute Candidate Recall@K before final reranking/fusion."""
    recalls = []
    for qid, gold_list in ground_truths.items():
        gold_set = set(str(x) for x in gold_list)
        if not gold_set:
            continue
        cands = [str(x) for x in candidates.get(str(qid), [])[:k]]
        cand_set = set(cands)
        rec = len(gold_set & cand_set) / len(gold_set)
        recalls.append(rec)
    return float(np.mean(recalls)) if recalls else 0.0


def compute_candidate_cutoffs(candidates: dict[str, Any], ground_truths: dict[str, Any], cutoffs: list[int] = [20, 50, 100, 150]) -> dict[str, float]:
    """Compute candidate recall across standard diagnostic cutoffs."""
    return {
        f"candidate_recall@{k}": compute_candidate_recall(candidates, ground_truths, k=k)
        for k in cutoffs
    }
