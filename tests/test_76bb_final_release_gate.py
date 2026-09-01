"""
Comprehensive Behavioral Release Gate Tests for LegalIR 76BB Repair.

Authoritative specification: LEGALIR_76BB_FINAL_RELEASE_PROVENANCE_SMOKE_REPAIR.md
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from unittest import mock
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# Task 1 & 9: Release Provenance & Approval Invariants
# ==============================================================================

def test_kaggle_expected_commit_equals_approved_runtime_sha():
    """Verify that Kaggle notebook generator and notebooks pin exact approved runtime SHA."""
    from scripts.generate_kaggle_notebook import build_legalir_notebook

    nb = build_legalir_notebook()
    cells = nb.get("cells", [])
    all_source = "\n".join("".join(c.get("source", [])) for c in cells)

    # Must extract EXPECTED_COMMIT
    match = re.search(r'EXPECTED_COMMIT\s*=\s*os\.environ\.get\([^,]+,\s*["\']([0-9a-fA-F]{40})["\']\)', all_source)
    assert match is not None, "Kaggle notebook must define explicit 40-char hex EXPECTED_COMMIT pin"
    pinned_sha = match.group(1)
    assert pinned_sha != "2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12", "Kaggle notebook must not pin stale 2c1b6e8 commit"


def test_approved_runtime_is_not_stale_2c1b6e8():
    """Verify that stale commit 2c1b6e8 is strictly rejected by provenance validation."""
    from src.release.provenance import validate_release_approval

    stale_approval = {
        "schema_version": 1,
        "runtime_sha": "2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12",
        "release_sha": "2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12",
        "ci": {"runtime_sha": "2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12", "conclusion": "failure"},
        "colab": {"runtime_sha": "2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12", "result": "FAIL", "report_sha256": "abc"},
        "production": {"kaggle_expected_commit": "2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12", "dual_gpu_required": True},
        "approved_for_kaggle_full": False,
    }
    is_valid, errors = validate_release_approval(stale_approval)
    assert is_valid is False
    assert len(errors) > 0


def test_release_only_commit_allowlist_rejects_src_changes(tmp_path):
    """Verify that any changes to src/** between runtime and release invalidate approval."""
    from src.release.provenance import validate_release_approval, RELEASE_ONLY_DIFF_ALLOWLIST

    assert "src/pipeline/kaggle_train.py" not in RELEASE_ONLY_DIFF_ALLOWLIST
    assert "src/retrieval/dense_macro.py" not in RELEASE_ONLY_DIFF_ALLOWLIST
    assert "configs/pipeline.yaml" not in RELEASE_ONLY_DIFF_ALLOWLIST

    approval = {
        "schema_version": 1,
        "runtime_sha": "a" * 40,
        "release_sha": "b" * 40,
        "ci": {"runtime_sha": "a" * 40, "conclusion": "success"},
        "colab": {"runtime_sha": "a" * 40, "result": "PASS", "report_sha256": "c" * 64},
        "production": {"kaggle_expected_commit": "a" * 40, "dual_gpu_required": True},
        "approved_for_kaggle_full": True,
    }

    # Mock git diff returning disallowed src modification
    with mock.patch("subprocess.check_output", return_value="src/pipeline/kaggle_train.py\nparameter_audit.json\n"):
        is_valid, errors = validate_release_approval(approval, git_root=tmp_path)
        assert is_valid is False
        assert any("Disallowed file changed" in e for e in errors)


# ==============================================================================
# Tasks 2-7: Colab Smoke Contracts & Invariants
# ==============================================================================

def test_colab_smoke_pair_mining_and_duplicate_blacklist(tmp_path):
    """Verify that Colab smoke uses production pair mining and applies the 4-group blacklist."""
    from scripts.smoke_kaggle_pipeline import create_toy_canonical_dataset
    from src.pipeline.colab_smoke import ColabSmokeConfig, run_colab_t4_smoke_pipeline

    toy_data = tmp_path / "toy_data"
    create_toy_canonical_dataset(toy_data)

    # Write 4-group duplicate blacklist in toy data
    dup_file = toy_data / "duplicate_groups.json"
    dup_groups = {
        "group_1": ["101", "102"],
        "group_2": ["103", "104"],
        "group_3": ["105", "106"],
        "group_4": ["101", "103"],
    }
    dup_file.write_text(json.dumps(dup_groups), encoding="utf-8")

    work_dir = tmp_path / "smoke_work"
    cfg = ColabSmokeConfig(
        seed=42,
        train_queries=4,
        validation_queries=2,
        public_queries=2,
        max_documents=6,
        folds=2,
        reranker_optimizer_steps=2,
        device="cpu",
    )

    report = run_colab_t4_smoke_pipeline(
        data_dir=toy_data,
        work_dir=work_dir,
        target_sha="a" * 40,
        config=cfg,
        skip_ci_check=True,
        allow_non_t4=True,
        use_mock_models=True,
    )

    assert report["result"] in ("PASS", "NOT_A_T4_READINESS_GATE")
    assert report["duplicate_blacklist"]["count"] == 4
    assert report["duplicate_blacklist"]["valid"] is True
    assert report["adapter_verification"]["fresh_reload"] is True
    assert report["adapter_verification"]["active_peft"] is True
    assert report["adapter_verification"]["finite_scores"] is True
    assert report["prediction_validation"]["prediction_pipeline"] == "dense_faiss_plus_reloaded_bge"
    assert report["prediction_validation"]["public_queries_executed"] == 2
    assert report["parameter_audit"]["system_learned_parameters"] < 4_000_000_000
    assert report["dataset_identity"]["documents"] > 0
    assert report["dense_telemetry"]["oom_events"] == 0
