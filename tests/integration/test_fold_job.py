import json
import pytest
import pandas as pd
from pathlib import Path
from src.core.hashing import sha256_file
from src.core.manifests import JobManifest
from src.validation.fold_job import FoldJobRunner, should_resume_fold


def test_should_resume_fold_true(tmp_path):
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()

    metrics_file = fold_dir / "fold_metrics.json"
    metrics_file.write_text('{"recall@5": 0.85}', encoding="utf-8")
    metrics_hash = sha256_file(metrics_file)

    manifest = JobManifest(
        job_id="fold_0",
        job_type="fold",
        status="PASS",
        runtime_commit="test_commit",
        dataset_sha256="test_ds",
        outputs={"fold_metrics.json": metrics_hash},
    )
    manifest.save(fold_dir / "job_manifest.json")

    assert should_resume_fold(fold_dir) is True


def test_should_resume_fold_false_on_hash_mismatch(tmp_path):
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()

    metrics_file = fold_dir / "fold_metrics.json"
    metrics_file.write_text('{"recall@5": 0.85}', encoding="utf-8")

    manifest = JobManifest(
        job_id="fold_0",
        job_type="fold",
        status="PASS",
        runtime_commit="test_commit",
        dataset_sha256="test_ds",
        outputs={"fold_metrics.json": "wrong_hash"},
    )
    manifest.save(fold_dir / "job_manifest.json")

    assert should_resume_fold(fold_dir) is False


def test_should_resume_fold_false_on_failed_status(tmp_path):
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()

    metrics_file = fold_dir / "fold_metrics.json"
    metrics_file.write_text('{"recall@5": 0.0}', encoding="utf-8")

    manifest = JobManifest(
        job_id="fold_0",
        job_type="fold",
        status="FAIL",
        runtime_commit="test_commit",
        dataset_sha256="test_ds",
        outputs={"fold_metrics.json": sha256_file(metrics_file)},
    )
    manifest.save(fold_dir / "job_manifest.json")

    assert should_resume_fold(fold_dir) is False


def test_fold_nonmock_calls_real_training_interface_and_features(tmp_path):
    work_dir = tmp_path / "folds"
    runner = FoldJobRunner(
        fold_id=0,
        work_dir=work_dir,
        config={
            "base_model_name": "mock",
            "max_steps": 1,
            "batch_size": 2,
            "device": "cpu",
            "enforce_full_coverage_steps": False,
        },
    )
    fold_dir = work_dir / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Write small train pairs
    pairs_df = pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["hoi luat 1", "hoi luat 1"],
        "doc_id": ["docA", "docB"],
        "label": [1.0, 0.0],
        "evidence_text": ["van ban A", "van ban B"],
        "fold": [0, 0],
    })
    pairs_df.to_parquet(fold_dir / "train_pairs.parquet", index=False)

    # Write small validation candidates
    val_df = pd.DataFrame({
        "query_id": ["q2", "q2"],
        "query_text": ["hoi luat 2", "hoi luat 2"],
        "doc_id": ["docA", "docB"],
        "gold_doc_ids": [json.dumps(["docA"]), json.dumps(["docA"])],
        "rrf_score": [0.03, 0.01],
        "evidence_text": ["van ban A", "van ban B"],
        "fold": [0, 0],
    })
    val_df.to_parquet(fold_dir / "validation_candidates.parquet", index=False)

    manifest = runner.run(mock_run=False)
    assert manifest.status == "PASS"
    assert (fold_dir / "adapter" / "adapter_config.json").is_file()
    assert (fold_dir / "oof_features.parquet").is_file()
    assert (fold_dir / "fold_metrics.json").is_file()

    # Verify features schema
    feat_df = pd.read_parquet(fold_dir / "oof_features.parquet")
    assert "reranker_score" in feat_df.columns
    assert "fold" in feat_df.columns
    assert len(feat_df) == 2

    # Verify real metrics (not hardcoded constant)
    with open(fold_dir / "fold_metrics.json") as f:
        metrics = json.load(f)
    assert "recall@5" in metrics
    assert metrics["recall@5"] == 1.0  # docA was retrieved in top 5 for q2


def test_no_constant_fake_metrics_in_production(tmp_path):
    # Verify that runner calculates dynamic metrics based on predictions vs true labels
    work_dir = tmp_path / "folds_dyn"
    runner = FoldJobRunner(
        fold_id=1,
        work_dir=work_dir,
        config={
            "base_model_name": "mock",
            "max_steps": 1,
            "batch_size": 2,
            "device": "cpu",
            "enforce_full_coverage_steps": False,
        },
    )
    fold_dir = work_dir / "fold_1"
    fold_dir.mkdir(parents=True, exist_ok=True)

    pairs_df = pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["hoi luat 1", "hoi luat 1"],
        "doc_id": ["docA", "docB"],
        "label": [1.0, 0.0],
        "evidence_text": ["van ban A", "van ban B"],
        "fold": [1, 1],
    })
    pairs_df.to_parquet(fold_dir / "train_pairs.parquet", index=False)

    # Gold is docX (not in candidate list), so recall should be 0.0
    val_df = pd.DataFrame({
        "query_id": ["q2"],
        "query_text": ["hoi luat 2"],
        "doc_id": ["docA"],
        "gold_doc_ids": [json.dumps(["docX"])],
        "rrf_score": [0.03],
        "evidence_text": ["van ban A"],
        "fold": [1],
    })
    val_df.to_parquet(fold_dir / "validation_candidates.parquet", index=False)

    manifest = runner.run(mock_run=False)
    with open(fold_dir / "fold_metrics.json") as f:
        metrics = json.load(f)
    assert metrics["recall@5"] == 0.0  # Proves metrics are not constant fake 0.85
