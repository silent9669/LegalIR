import json
import pytest
import pandas as pd
from pathlib import Path
from src.core.manifests import JobManifest
from src.validation.doc_disjoint_job import DocDisjointRunner, verify_doc_disjoint_split


def test_verify_doc_disjoint_split():
    train_docs = {"d1", "d2", "d3"}
    test_docs = {"d4", "d5"}
    assert verify_doc_disjoint_split(train_docs, test_docs) is True

    # Overlapping docs
    overlap_docs = {"d3", "d6"}
    assert verify_doc_disjoint_split(train_docs, overlap_docs) is False


def test_doc_disjoint_runner_mock(tmp_path):
    runner = DocDisjointRunner(work_dir=tmp_path)
    manifest = runner.run(mock_run=True)
    assert manifest.job_type == "doc_disjoint"
    assert manifest.status == "PASS"
    assert "recall@5" in manifest.metrics
    assert (tmp_path / "metrics.json").is_file()


def test_doc_disjoint_nonmock_calls_real_pipeline(tmp_path):
    work_dir = tmp_path / "doc_disjoint_real"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Mock train pairs
    pairs_df = pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["hoi luat 1", "hoi luat 1"],
        "doc_id": ["docA", "docB"],
        "label": [1.0, 0.0],
        "evidence_text": ["van ban A", "van ban B"],
        "fold": [0, 0],
    })
    pairs_df.to_parquet(work_dir / "train_pairs.parquet", index=False)

    # Mock validation candidates
    val_df = pd.DataFrame({
        "query_id": ["q2"],
        "query_text": ["hoi luat 2"],
        "doc_id": ["docA"],
        "gold_doc_ids": [json.dumps(["docA"])],
        "rrf_score": [0.05],
        "evidence_text": ["van ban A"],
        "fold": [0],
    })
    val_df.to_parquet(work_dir / "validation_candidates.parquet", index=False)

    runner = DocDisjointRunner(
        work_dir=work_dir,
        config={
            "base_model_name": "mock",
            "max_steps": 1,
            "batch_size": 2,
            "device": "cpu",
            "enforce_full_coverage_steps": False,
        },
    )
    manifest = runner.run(mock_run=False)
    assert manifest.status == "PASS"
    assert (work_dir / "adapter" / "adapter_config.json").is_file()
    assert (work_dir / "metrics.json").is_file()
    assert (work_dir / "predictions.json").is_file()

    with open(work_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["recall@5"] == 1.0
    assert "candidate_recall@50" in metrics
