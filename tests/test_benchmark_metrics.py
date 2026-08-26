from src.evaluation.evaluator import compute_candidate_cutoffs, evaluate_predictions


def test_candidate_cutoffs_returns_all_expected_keys():
    candidates = {"q1": ["1", "2", "3", "4", "5", "6"]}
    ground_truth = {"q1": ["1", "6"]}
    cutoffs = compute_candidate_cutoffs(candidates, ground_truth, cutoffs=[20, 50, 100, 150])
    assert set(cutoffs.keys()) == {
        "candidate_recall@20",
        "candidate_recall@50",
        "candidate_recall@100",
        "candidate_recall@150",
    }
    assert cutoffs["candidate_recall@20"] == 1.0


def test_benchmark_metrics_structure():
    pred = {"q1": ["1", "2"], "q2": ["3"]}
    truth = {"q1": ["1"], "q2": ["3"]}
    metrics = evaluate_predictions(pred, truth)
    assert "recall_at_5" in metrics
    assert "precision_at_5" in metrics
    assert "Recall@1" in metrics
    assert "Recall@3" in metrics
    assert "Recall@5" in metrics
