import json
import numpy as np

def evaluate_predictions(y_pred: dict, y_true: dict) -> dict:
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

    return {
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@1": float(np.mean(r1_list)) if r1_list else 0.0,
        "Recall@3": float(np.mean(r3_list)) if r3_list else 0.0,
        "Recall@5": float(np.mean(r5_list)) if r5_list else 0.0,
        "total_evaluated_queries": len(recalls)
    }

def compute_candidate_recall(candidates: dict, ground_truths: dict, k: int = 50) -> float:
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
