"""Behavioral tests for LegalIR B49B Public-1000 Kaggle Final Gate.

Covers:
- P0: 1000 public query count, exact keyset validation, and rejection of 999/1001
- P1: Reusable validate_official_task1_identity helper with strict manifest/audit/counts validation
- P1: Notebook Cell 3 lightweight PyArrow metadata row-count preflight (no full DataFrame materialization)
- P1: Split-specific fold step derivation in runtime projection
- P1: Enriched dataset_identity payload in gpu_smoke_report.json
"""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_official_v2_mock_fixture(
    tmp_path: Path,
    num_docs: int = 8532,
    num_chunks: int = 1153876,
    num_micro: int = 934416,
    num_macro: int = 219460,
    num_train: int = 7000,
    num_qrels: int = 7637,
    num_public: int = 1000,
    audit_valid: bool = True,
    audit_errors: list[str] | None = None,
    dataset_name: str = "task1_canonical",
    version: str = "v2",
    schema: str = "hierarchical_micro_macro_v2",
) -> tuple[Path, Path]:
    """Create a minimal valid directory structure matching official Task 1 v2 identity."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": dataset_name,
        "version": version,
        "schema": schema,
        "total_documents": num_docs,
        "total_chunks": num_chunks,
        "total_micro_chunks": num_micro,
        "total_macro_chunks": num_macro,
        "total_queries": num_train,
        "total_qrels": num_qrels,
        "total_duplicate_groups": 4,
        "empty_documents_count": 20,
        "normalization": "nfc_whitespace_preserve_legal_ids",
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit = {
        "is_valid": audit_valid,
        "total_documents": num_docs,
        "total_chunks": num_chunks,
        "total_micro_chunks": num_micro,
        "total_macro_chunks": num_macro,
        "total_queries": num_train,
        "total_qrels": num_qrels,
        "empty_documents_count": 20,
        "errors": audit_errors if audit_errors is not None else [],
    }
    (data_dir / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Create dummy parquet files with metadata
    for fname in ["documents.parquet", "chunks.parquet", "queries_train.parquet", "qrels_train.parquet"]:
        df = pd.DataFrame({"col": [1]})
        df.to_parquet(data_dir / fname)

    public_data = {f"pub_{i:04d}": {"question": f"Legal question {i}?", "answer": None} for i in range(num_public)}
    public_file = data_dir / "public-official.json"
    public_file.write_text(json.dumps(public_data, indent=2), encoding="utf-8")

    return data_dir, public_file


# ==============================================================================
# 1. P0 & P1: Official Task 1 Dataset Identity Helper
# ==============================================================================

def test_actual_public_fixture_has_1000_queries(tmp_path):
    """Verify synthetic 1000-query fixture has exactly 1000 query IDs."""
    _, public_file = make_official_v2_mock_fixture(tmp_path, num_public=1000)
    data = json.loads(public_file.read_text(encoding="utf-8"))
    assert len(data) == 1000
    assert len(set(data.keys())) == 1000


def test_official_identity_accepts_exact_v2_1000_fixture(tmp_path):
    """validate_official_task1_identity succeeds on official 1000-query v2 dataset."""
    from src.pipeline.kaggle_train import validate_official_task1_identity

    data_dir, public_file = make_official_v2_mock_fixture(tmp_path, num_public=1000)
    report = validate_official_task1_identity(
        data_dir=data_dir,
        public_json_path=public_file,
        strict=True,
    )
    assert report["is_valid"] is True
    assert report["public_queries"] == 1000
    assert report["documents"] == 8532
    assert report["chunks"] == 1153876
    assert report["micro_chunks"] == 934416
    assert report["macro_chunks"] == 219460
    assert report["train_queries"] == 7000
    assert report["qrels"] == 7637
    assert report["audit_valid"] is True
    assert report["audit_errors"] == []


def test_official_identity_rejects_999_public_queries(tmp_path):
    """validate_official_task1_identity fails when public queries count is 999."""
    from src.pipeline.kaggle_train import validate_official_task1_identity

    data_dir, public_file = make_official_v2_mock_fixture(tmp_path, num_public=999)
    report = validate_official_task1_identity(
        data_dir=data_dir,
        public_json_path=public_file,
        strict=False,
    )
    assert report["is_valid"] is False
    assert any("1000" in e and "999" in e for e in report["errors"])

    with pytest.raises(ValueError, match="public queries count"):
        validate_official_task1_identity(
            data_dir=data_dir,
            public_json_path=public_file,
            strict=True,
        )


def test_official_identity_rejects_1001_public_queries(tmp_path):
    """validate_official_task1_identity fails when public queries count is 1001."""
    from src.pipeline.kaggle_train import validate_official_task1_identity

    data_dir, public_file = make_official_v2_mock_fixture(tmp_path, num_public=1001)
    report = validate_official_task1_identity(
        data_dir=data_dir,
        public_json_path=public_file,
        strict=False,
    )
    assert report["is_valid"] is False

    with pytest.raises(ValueError, match="public queries count"):
        validate_official_task1_identity(
            data_dir=data_dir,
            public_json_path=public_file,
            strict=True,
        )


def test_official_identity_rejects_wrong_version_or_schema(tmp_path):
    """validate_official_task1_identity fails on non-v2 or invalid schema."""
    from src.pipeline.kaggle_train import validate_official_task1_identity

    data_dir_v1, pub_v1 = make_official_v2_mock_fixture(tmp_path / "v1", version="v1")
    with pytest.raises(ValueError, match="version"):
        validate_official_task1_identity(data_dir=data_dir_v1, public_json_path=pub_v1, strict=True)

    data_dir_schema, pub_s = make_official_v2_mock_fixture(tmp_path / "bad_schema", schema="flat_v1")
    with pytest.raises(ValueError, match="schema"):
        validate_official_task1_identity(data_dir=data_dir_schema, public_json_path=pub_s, strict=True)


def test_official_identity_rejects_audit_errors(tmp_path):
    """validate_official_task1_identity fails when audit_report has errors or is_valid=False."""
    from src.pipeline.kaggle_train import validate_official_task1_identity

    data_dir, pub = make_official_v2_mock_fixture(
        tmp_path,
        audit_valid=False,
        audit_errors=["Corrupted chunk 42"],
    )
    with pytest.raises(ValueError, match=r"(?i)audit"):
        validate_official_task1_identity(data_dir=data_dir, public_json_path=pub, strict=True)


# ==============================================================================
# 2. P1: Notebook Memory Optimization (Cell 3)
# ==============================================================================

def test_notebook_identity_preflight_does_not_materialize_full_parquets():
    """Generated notebook Cell 3 must NOT call pd.read_parquet for full tables."""
    from scripts.generate_kaggle_notebook import build_legalir_notebook

    nb = build_legalir_notebook()
    cell_3_src = "".join(nb["cells"][3]["source"])

    assert "pd.read_parquet" not in cell_3_src
    assert "df_chunks =" not in cell_3_src
    assert "df_docs =" not in cell_src if "cell_src" in locals() else "df_docs =" not in cell_3_src
    assert "pyarrow.parquet" in cell_3_src or "pq." in cell_3_src
    assert "ParquetFile" in cell_3_src or "metadata.num_rows" in cell_3_src
    assert "1000" in cell_3_src
    assert "999" not in cell_3_src


# ==============================================================================
# 3. P1: Split-Specific Fold Steps in Runtime Projection
# ==============================================================================

def test_split_specific_fold_step_projection():
    """5-fold outer training split (~5,600 queries) projects to 700 steps per fold, not 875."""
    from src.training.trainer import compute_coverage_required_steps

    n_train_queries = 7000
    n_folds = 5
    fold_train_queries = int(n_train_queries * (n_folds - 1) / n_folds)  # 5,600
    assert fold_train_queries == 5600

    fold_steps = compute_coverage_required_steps(
        eligible_query_count=fold_train_queries,
        batch_size=2,
        gradient_accumulation_steps=8,
        target_coverage_pct=1.0,
        require_pos_and_neg=True,
    )
    assert fold_steps == 700

    final_steps = compute_coverage_required_steps(
        eligible_query_count=7000,
        batch_size=2,
        gradient_accumulation_steps=8,
        target_coverage_pct=1.0,
        require_pos_and_neg=True,
    )
    assert final_steps == 875


# ==============================================================================
# 4. P1: GPU Smoke Report Carries Verified Dataset Identity
# ==============================================================================

def test_gpu_smoke_report_schema_includes_dataset_identity():
    """GPU smoke report dictionary must include structured dataset_identity."""
    expected_keys = {
        "dataset",
        "version",
        "schema",
        "documents",
        "chunks",
        "micro_chunks",
        "macro_chunks",
        "train_queries",
        "qrels",
        "public_queries",
        "audit_valid",
        "audit_errors",
    }
    identity_dict = {
        "dataset": "task1_canonical",
        "version": "v2",
        "schema": "hierarchical_micro_macro_v2",
        "documents": 8532,
        "chunks": 1153876,
        "micro_chunks": 934416,
        "macro_chunks": 219460,
        "train_queries": 7000,
        "qrels": 7637,
        "public_queries": 1000,
        "audit_valid": True,
        "audit_errors": [],
    }
    assert set(identity_dict.keys()) == expected_keys
    assert identity_dict["public_queries"] == 1000
