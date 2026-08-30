from src.evaluation.benchmark import aggregate_fold_metrics, run_split_eval
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


def test_benchmark_dual_validation_metric_aliases():
    preds = {"q1": ["doc_1", "doc_2"]}
    golds = {"q1": ["doc_1"]}

    metrics = evaluate_predictions(preds, golds)

    assert metrics["recall@5"] == 1.0
    assert metrics["precision@5"] == 0.5
    assert metrics["Recall@1"] == 1.0
    assert metrics["Recall@3"] == 1.0
    assert metrics["Recall@5"] == 1.0
    assert metrics["Precision@5"] == 0.5


def test_default_candidate_cutoffs_cover_standard_diagnostics():
    candidates = {"q1": ["doc_1", "doc_2"]}
    ground_truth = {"q1": ["doc_1"]}

    metrics = compute_candidate_cutoffs(candidates, ground_truth)

    assert set(metrics) == {
        "candidate_recall@20",
        "candidate_recall@50",
        "candidate_recall@100",
    }


def test_candidate_recall_accepts_candidate_records():
    candidates = {
        "q1": [
            {"doc_id": "doc_x", "score": 1.0},
            {"doc_id": "doc_1", "score": 0.5},
        ]
    }
    ground_truth = {"q1": ["doc_1"]}

    metrics = compute_candidate_cutoffs(candidates, ground_truth, cutoffs=[1, 2])

    assert metrics == {
        "candidate_recall@1": 0.0,
        "candidate_recall@2": 1.0,
    }


def test_aggregate_fold_metrics_reports_all_final_metrics():
    fold_metrics = [
        {
            "recall_at_1": 0.5,
            "recall_at_3": 0.75,
            "recall_at_5": 1.0,
            "precision_at_5": 0.4,
            "candidate_recall@20": 0.8,
            "candidate_recall@50": 0.9,
            "candidate_recall@100": 1.0,
        },
        {
            "recall_at_1": 0.0,
            "recall_at_3": 0.5,
            "recall_at_5": 0.5,
            "precision_at_5": 0.2,
            "candidate_recall@20": 0.6,
            "candidate_recall@50": 0.8,
            "candidate_recall@100": 0.9,
        },
    ]

    summary = aggregate_fold_metrics(fold_metrics, [20, 50, 100])

    assert summary["mean_recall_at_1"] == 0.25
    assert summary["mean_recall_at_3"] == 0.625
    assert summary["mean_recall_at_5"] == 0.75
    assert summary["std_recall_at_5"] == 0.25
    assert summary["mean_precision_at_5"] == 0.30000000000000004
    assert summary["candidate_recalls"] == {
        "candidate_recall@20": 0.7,
        "candidate_recall@50": 0.8500000000000001,
        "candidate_recall@100": 0.95,
    }
    assert summary["final_ranking_metrics"] == {
        "Recall@1": 0.25,
        "Recall@3": 0.625,
        "Recall@5": 0.75,
        "Precision@5": 0.30000000000000004,
    }


def test_run_split_eval_reports_candidate_and_final_metrics():
    class StubEngine:
        def search_candidates(self, query, exclude_qid=None, top_k=100):
            return [
                {"doc_id": "doc_x", "score": 1.0},
                {"doc_id": "doc_1", "score": 0.5},
            ]

    class StubFuser:
        def rank_candidates(self, candidates):
            return candidates

    class StubSelector:
        def select(self, ranked):
            return [candidate["doc_id"] for candidate in ranked[:2]]

    metrics, predictions, candidates = run_split_eval(
        split_name="test",
        val_query_ids=["q1"],
        queries_dict={"q1": "question"},
        qrels_dict={"q1": ["doc_1"]},
        hybrid_engine=StubEngine(),
        fuser=StubFuser(),
        selector=StubSelector(),
    )

    assert set(key for key in metrics if key.startswith("candidate_recall@")) == {
        "candidate_recall@20",
        "candidate_recall@50",
        "candidate_recall@100",
    }
    assert metrics["Recall@1"] == 0.0
    assert metrics["Recall@3"] == 1.0
    assert metrics["Recall@5"] == 1.0
    assert metrics["Precision@5"] == 0.5
    assert predictions == {"q1": ["doc_x", "doc_1"]}
    assert candidates == {"q1": ["doc_x", "doc_1"]}
