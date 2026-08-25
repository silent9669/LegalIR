import os
import json
import zipfile
import pytest
import pandas as pd

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
