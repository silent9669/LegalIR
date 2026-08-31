"""Tests for Phase 9: Kaggle Notebook Orchestrator and Submission Packaging."""

import json
from pathlib import Path
import tempfile
import zipfile
import pytest

from src.evaluation.submission import (
    compute_sha256,
    create_submission_manifest,
    package_submission,
    validate_submission,
    validate_submission_zip,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_legalir_notebook_nbformat_v4_structure():
    """Verify that legalir_training.ipynb conforms to nbformat v4 schema."""
    nb_path = REPO_ROOT / "legalir_training.ipynb"
    assert nb_path.exists(), "legalir_training.ipynb must exist at repository root"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4, f"Expected nbformat 4, got {nb.get('nbformat')}"
    assert "cells" in nb, "Notebook must contain 'cells' list"
    assert len(nb["cells"]) >= 5, f"Expected at least 5 cells, found {len(nb['cells'])}"

    for idx, cell in enumerate(nb["cells"]):
        assert "cell_type" in cell, f"Cell {idx} missing 'cell_type'"
        assert cell["cell_type"] in ("markdown", "code"), f"Cell {idx} invalid cell_type: {cell['cell_type']}"
        assert "source" in cell, f"Cell {idx} missing 'source'"
        src_text = "".join(cell["source"]).strip()
        assert len(src_text) > 0, f"Cell {idx} source cannot be empty"


def test_notebook_is_thin_orchestrator_without_monolithic_classes():
    """Verify notebook is a thin orchestrator importing canonical src modules rather than embedding monolithic classes."""
    nb_path = REPO_ROOT / "legalir_training.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    all_code = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )

    # Core classes that must NOT be defined inside the notebook
    forbidden_class_defs = [
        "class BM25MicroRetriever",
        "class BM25PyViRetriever",
        "class DenseMacroRetriever",
        "class CrossEncoderReranker",
        "class LegalIRPipeline",
        "class HybridSearchEngine",
        "class TrainQuestionMemory",
        "class EvidencePackBuilder",
        "class ReciprocalRankFusion",
        "class TopKSelector",
        "class OOFRunner",
    ]

    for class_def in forbidden_class_defs:
        assert class_def not in all_code, f"Notebook must not define '{class_def}' monolithic copy. It must import from src.*."

    # Canonical src pipeline module that MUST be imported
    required_src_imports = [
        "src.pipeline",
    ]

    for imp in required_src_imports:
        assert imp in all_code, f"Notebook must import from '{imp}'"


def test_notebook_no_hardcoded_tokens_or_secret_leaks():
    """Verify that notebook contains no hardcoded tokens or secret leaks."""
    nb_path = REPO_ROOT / "legalir_training.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    all_text = json.dumps(nb)

    # Check for hardcoded HF tokens (hf_...)
    assert "hf_" not in all_text.lower() or "hf_token" in all_text.lower(), "Found potential hardcoded HF token"
    # Never print token
    assert "print(hf_token)" not in all_text, "Notebook must never print HF token"
    assert "print(f\"{hf_token}" not in all_text, "Notebook must never print HF token"


def test_notebook_supports_run_modes():
    """Verify notebook supports RUN_MODE = 'full' and 'smoke'."""
    nb_path = REPO_ROOT / "legalir_training.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    all_code = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )

    assert "RUN_MODE" in all_code, "Notebook must support RUN_MODE"
    assert "smoke" in all_code, "Notebook must support 'smoke' execution mode"


def test_submission_packaging_and_validation(tmp_path):
    """Verify strict submission formatting, validation, and zip packaging."""
    sample_preds = {
        "101": {"answer": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]},
        "102": {"answer": ["doc_6", "doc_7", "doc_8", "doc_9", "doc_10"]},
    }

    sub_json = tmp_path / "submission.json"
    sub_zip = tmp_path / "submission.zip"

    # Package submission
    package_submission(sample_preds, sub_json, sub_zip)

    assert sub_json.exists()
    assert sub_zip.exists()

    # Validate JSON
    val_res = validate_submission(sub_json, expected_qids={"101", "102"})
    assert val_res["is_valid"] is True
    assert val_res["total_queries"] == 2

    # Validate ZIP structure (strictly ['submission.json'] at root)
    zip_val = validate_submission_zip(sub_zip)
    assert zip_val["is_valid"] is True

    # Validate corrupt/nested zip failure
    nested_zip = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested_zip, "w") as zf:
        zf.writestr("subfolder/submission.json", "{}")
    assert validate_submission_zip(nested_zip)["is_valid"] is False

    extra_file_zip = tmp_path / "extra_file.zip"
    with zipfile.ZipFile(extra_file_zip, "w") as zf:
        zf.writestr("submission.json", '{"101": {"answer": ["doc_1"]}}')
        zf.writestr("extra.txt", "extra")
    assert validate_submission_zip(extra_file_zip)["is_valid"] is False


def test_submission_invariants_enforcement():
    """Verify validation catches invalid prediction structures."""
    # 1. >5 document IDs
    too_many = {"1": {"answer": ["1", "2", "3", "4", "5", "6"]}}
    res = validate_submission(too_many, raise_on_error=False)
    assert res["is_valid"] is False

    # 2. Empty answer list
    empty_ans = {"1": {"answer": []}}
    res = validate_submission(empty_ans, raise_on_error=False)
    assert res["is_valid"] is False

    # 3. Duplicate IDs in answer
    duplicates = {"1": {"answer": ["doc_1", "doc_1", "doc_2"]}}
    res = validate_submission(duplicates, raise_on_error=False)
    assert res["is_valid"] is False

    # 4. Non-string document IDs
    non_strings = {"1": {"answer": [123, 456]}}
    res = validate_submission(non_strings, raise_on_error=False)
    assert res["is_valid"] is False

    # 5. Missing / extra query IDs
    res = validate_submission({"1": {"answer": ["doc_1"]}}, expected_qids={"1", "2"}, raise_on_error=False)
    assert res["is_valid"] is False


def test_create_submission_manifest(tmp_path):
    """Verify submission manifest export with sha256 checksums and parameter counts."""
    sample_preds = {"q1": {"answer": ["d1", "d2", "d3", "d4", "d5"]}}
    sub_json = tmp_path / "submission.json"
    sub_zip = tmp_path / "submission.zip"
    manifest_path = tmp_path / "submission_manifest.json"

    package_submission(sample_preds, sub_json, sub_zip)

    manifest = create_submission_manifest(
        submission_json_path=sub_json,
        submission_zip_path=sub_zip,
        output_path=manifest_path,
        git_commit="test_commit_sha_12345",
        parameter_total=703000000,
        model_names_and_revisions=[
            {"name": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", "role": "dense_embedding"},
            {"name": "BAAI/bge-reranker-v2-m3", "role": "cross_encoder_reranker"},
        ],
    )

    assert manifest_path.exists()
    assert manifest["git_commit"] == "test_commit_sha_12345"
    assert manifest["query_count"] == 1
    assert manifest["parameter_total"] == 703000000
    assert len(manifest["submission_json_sha256"]) == 64
    assert len(manifest["submission_zip_sha256"]) == 64
    assert manifest["all_answers_valid"] is True
    assert manifest["created_utc"] is not None
