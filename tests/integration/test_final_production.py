import json
import zipfile
import pytest
from pathlib import Path
from src.production.submission import validate_submission, package_submission


def test_validate_submission_valid():
    # Exactly 10 public queries for unit testing
    expected_qids = {f"pub_{i}" for i in range(10)}
    sub = {f"pub_{i}": [f"doc_{j}" for j in range(5)] for i in range(10)}
    is_valid, errors = validate_submission(sub, expected_qids=expected_qids, max_predictions=5)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_submission_missing_key():
    expected_qids = {"q1", "q2"}
    sub = {"q1": ["d1", "d2"]}
    is_valid, errors = validate_submission(sub, expected_qids=expected_qids)
    assert is_valid is False
    assert any("Missing 1 query IDs" in e for e in errors)


def test_validate_submission_duplicate_predictions():
    expected_qids = {"q1"}
    sub = {"q1": ["d1", "d1", "d2"]}
    is_valid, errors = validate_submission(sub, expected_qids=expected_qids)
    assert is_valid is False
    assert any("Duplicate document IDs" in e for e in errors)


def test_validate_submission_too_many_predictions():
    expected_qids = {"q1"}
    sub = {"q1": ["d1", "d2", "d3", "d4", "d5", "d6"]}
    is_valid, errors = validate_submission(sub, expected_qids=expected_qids, max_predictions=5)
    assert is_valid is False
    assert any("exceeds max 5" in e for e in errors)


def test_package_submission(tmp_path):
    sub = {f"q_{i}": ["d1", "d2", "d3"] for i in range(5)}
    out_dir = tmp_path / "submission_out"
    json_p, zip_p = package_submission(sub, out_dir=out_dir)

    assert json_p.is_file()
    assert zip_p.is_file()
    assert zipfile.is_zipfile(zip_p)

    with zipfile.ZipFile(zip_p, "r") as zf:
        assert "submission.json" in zf.namelist()
