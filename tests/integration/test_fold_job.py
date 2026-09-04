import json
import pytest
from pathlib import Path
from src.core.hashing import sha256_file
from src.core.manifests import JobManifest
from src.validation.fold_job import FoldJobRunner, should_resume_fold


def test_should_resume_fold_true(tmp_path):
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()

    # Create dummy output files
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

    # Resume check should be True
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
