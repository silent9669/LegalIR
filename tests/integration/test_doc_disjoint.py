import pytest
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
