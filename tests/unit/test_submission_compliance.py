import os
import json
import zipfile
import pytest
import pandas as pd
from src.evaluation.submission import validate_submission_zip
from src.ranking.selector import TopKSelector


def _write_submission_zip(tmp_path, sub: dict) -> object:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sub_file = tmp_path / "submission.json"
    sub_file.write_text(json.dumps(sub), encoding="utf-8")
    zip_file = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_file, arcname="submission.json")
    return zip_file


def test_private_submission_requires_exactly_five_per_query(tmp_path):
    """FULL private contract: 5 doc IDs per query, unique, string IDs."""
    sub = {f"q{i}": {"answer": [f"d{j}" for j in range(i * 5, i * 5 + 5)]} for i in range(5)}
    res = validate_submission_zip(_write_submission_zip(tmp_path, sub), exact_answer_count=5)
    assert bool(res.get("is_valid")) is True


def test_private_submission_rejects_four_or_six(tmp_path):
    base = {f"q{i}": {"answer": [f"d{j}" for j in range(i * 5, i * 5 + 5)]} for i in range(3)}
    short = dict(base)
    short["q0"] = {"answer": short["q0"]["answer"][:4]}
    assert bool(validate_submission_zip(
        _write_submission_zip(tmp_path / "s", short), exact_answer_count=5).get("is_valid")) is False
    long_ = dict(base)
    long_["q1"] = {"answer": long_["q1"]["answer"] + ["dx"]}
    assert bool(validate_submission_zip(
        _write_submission_zip(tmp_path / "l", long_), exact_answer_count=5).get("is_valid")) is False


def test_private_submission_rejects_duplicates(tmp_path):
    sub = {"q0": {"answer": ["d1", "d1", "d2", "d3", "d4"]}}
    res = validate_submission_zip(_write_submission_zip(tmp_path, sub), exact_answer_count=5)
    assert bool(res.get("is_valid")) is False

def test_submission_format_compliance(tmp_path):
    # Mock submission
    sub = {
        "1001": {"answer": ["740", "2113"]},
        "1002": {"answer": ["280282"]}
    }
    sub_file = tmp_path / "submission.json"
    with open(sub_file, "w", encoding="utf-8") as f:
        json.dump(sub, f)

    zip_file = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_file, arcname="submission.json")

    # Verify zip content
    with zipfile.ZipFile(zip_file, "r") as zf:
        namelist = zf.namelist()
        assert namelist == ["submission.json"]

        with zf.open("submission.json") as jf:
            loaded = json.load(jf)
            assert len(loaded) == 2
            for qid, v in loaded.items():
                assert "answer" in v
                assert isinstance(v["answer"], list)
                assert 1 <= len(v["answer"]) <= 5
                assert len(set(v["answer"])) == len(v["answer"])


def test_selector_accepts_scored_candidates_and_returns_unique_string_ids():
    selector = TopKSelector(max_k=5)

    result = selector.select([("doc_1", 10.0), ("doc_2", 8.0), ("doc_1", 7.0), ("doc_3", 6.0)])

    assert result == ["doc_1", "doc_2", "doc_3"]
    assert all(isinstance(doc_id, str) for doc_id in result)
    assert 1 <= len(result) <= 5
