from typing import Any
import sys
from pathlib import Path
import numpy as np

# Ensure scoring program is importable
repo_root = Path(__file__).resolve().parents[2]
scoring_prog_dir = repo_root / "Scoring-Program-Task-LegalIR"
if str(scoring_prog_dir) not in sys.path:
    sys.path.insert(0, str(scoring_prog_dir))

from scoring import eval_retrieval
from src.evaluation.evaluator import evaluate_predictions


def assert_official_equivalence(predictions: dict[str, Any], ground_truths: dict[str, Any]) -> dict[str, float]:
    # Normalize inputs for both evaluators
    # Official eval_retrieval expects y_pred as {qid: {"answer": [...]}} and y_true as {qid: [...]} or {qid: {"answer": [...]}}
    official_y_pred = {}
    eval_y_pred = {}

    for qid, val in predictions.items():
        qid_str = str(qid)
        if isinstance(val, dict) and "answer" in val:
            ans = [str(x) for x in val["answer"]]
            official_y_pred[qid_str] = {"answer": ans}
            eval_y_pred[qid_str] = ans
        elif isinstance(val, list):
            ans = [str(x) for x in val]
            official_y_pred[qid_str] = {"answer": ans}
            eval_y_pred[qid_str] = ans
        else:
            raise ValueError(f"Invalid prediction format for qid {qid}: {val}")

    official_y_true = {}
    eval_y_true = {}

    for qid, val in ground_truths.items():
        qid_str = str(qid)
        if isinstance(val, dict) and "answer" in val:
            ans = [str(x) for x in val["answer"]]
            official_y_true[qid_str] = ans
            eval_y_true[qid_str] = ans
        elif isinstance(val, list):
            ans = [str(x) for x in val]
            official_y_true[qid_str] = ans
            eval_y_true[qid_str] = ans
        else:
            raise ValueError(f"Invalid ground truth format for qid {qid}: {val}")

    # Official scoring
    official_metrics = eval_retrieval(official_y_pred, official_y_true)

    # Internal scoring
    internal_metrics = evaluate_predictions(eval_y_pred, eval_y_true)

    diff_rec = abs(official_metrics["recall"] - internal_metrics["recall_at_5"])
    diff_prec = abs(official_metrics["precision"] - internal_metrics["precision_at_5"])

    if diff_rec > 1e-9 or diff_prec > 1e-9:
        raise AssertionError(
            f"Official scorer discrepancy: official={official_metrics}, internal={internal_metrics}"
        )

    return {
        "recall": float(official_metrics["recall"]),
        "precision": float(official_metrics["precision"]),
    }
