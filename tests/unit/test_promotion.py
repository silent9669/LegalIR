import json
import pytest
from src.validation.promotion import compare_score_promotion, create_production_lock


def test_promotion_comparison_recall_improves():
    baseline = {"recall@5": 0.850, "precision@5": 0.300, "candidate_recall@150": 0.98}
    candidate = {"recall@5": 0.853, "precision@5": 0.295, "candidate_recall@150": 0.98}
    eligible, reason = compare_score_promotion(candidate, baseline)
    assert eligible is True
    assert "Recall@5 improved" in reason


def test_promotion_comparison_recall_ties_precision_improves():
    baseline = {"recall@5": 0.850, "precision@5": 0.300, "candidate_recall@150": 0.98}
    candidate = {"recall@5": 0.850, "precision@5": 0.305, "candidate_recall@150": 0.98}
    eligible, reason = compare_score_promotion(candidate, baseline)
    assert eligible is True
    assert "Recall@5 tied and Precision@5 improved" in reason


def test_promotion_comparison_rejects_candidate_recall_regression():
    baseline = {"recall@5": 0.850, "precision@5": 0.300, "candidate_recall@150": 0.98}
    candidate = {"recall@5": 0.855, "precision@5": 0.310, "candidate_recall@150": 0.92}
    eligible, reason = compare_score_promotion(candidate, baseline, max_candidate_recall_drop=0.03)
    assert eligible is False
    assert "Candidate recall regressed" in reason


def test_create_production_lock(tmp_path):
    out_file = tmp_path / "production_lock.json"
    metrics = {"recall@5": 0.853, "precision@5": 0.305}
    config = {"fusion": {"method": "rrf", "weights": {"bm25": 1.0, "dense": 1.0, "reranker": 2.0}}}
    create_production_lock(
        output_path=out_file,
        metrics=metrics,
        config=config,
        runtime_commit="abc1234",
    )

    assert out_file.is_file()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["runtime_commit"] == "abc1234"
    assert data["metrics"]["recall@5"] == 0.853
    assert data["status"] == "LOCKED"
