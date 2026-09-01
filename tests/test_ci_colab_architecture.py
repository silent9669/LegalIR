"""
Tests for CI, Colab Single-T4 Contract Smoke, and Verification Architecture.

Authoritative specification: LEGALIR_CI_COLAB_KAGGLE_ARCHITECTURE_SPEC.md
Implementation plan: LEGALIR_CI_COLAB_KAGGLE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# Task 1: GitHub CI Correctness Gate & Notebook Parity
# ==============================================================================

def test_ci_workflow_structure_and_invariants():
    ci_file = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_file.exists(), f"Missing required CI workflow file: {ci_file}"

    content = ci_file.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in content, "CI workflow must not reference or require HF_TOKEN"

    data = yaml.safe_load(content)
    assert data.get("name") == "LegalIR CI", "Workflow name must be 'LegalIR CI'"

    # Triggers
    triggers = data.get("on") or data.get(True)  # YAML parser might parse 'on' as True
    if triggers is True:
        # Check raw text if YAML parsed 'on:' as boolean True
        assert "push:" in content
        assert "pull_request:" in content
        assert "workflow_dispatch:" in content
    else:
        assert "push" in triggers
        assert "pull_request" in triggers
        assert "workflow_dispatch" in triggers

    # Job specs
    jobs = data.get("jobs", {})
    assert "test" in jobs or "ci" in jobs, "Workflow must contain a test/ci job"
    test_job = jobs.get("test") or jobs.get("ci")

    assert test_job.get("runs-on") == "ubuntu-latest"
    assert test_job.get("timeout-minutes", 999) <= 35

    # Check env vars
    env = test_job.get("env", {})
    assert env.get("HF_HUB_OFFLINE") == "1"
    assert env.get("TRANSFORMERS_OFFLINE") == "1"

    # Inspect steps
    steps = test_job.get("steps", [])
    step_runs = [s.get("run", "") for s in steps if "run" in s]
    all_runs = "\n".join(step_runs)

    # Check python setup step
    setup_steps = [s for s in steps if "setup-python" in s.get("uses", "")]
    assert len(setup_steps) >= 1, "Must use actions/setup-python"
    setup_with = setup_steps[0].get("with", {})
    assert str(setup_with.get("python-version")) == "3.12"
    assert setup_with.get("cache") == "pip"

    # Required check commands
    assert "compileall" in all_runs, "Missing compileall check in CI"
    assert "run_kaggle_pipeline" in all_runs, "Missing pipeline import check in CI"
    assert "OOFRunner" in all_runs, "Missing OOFRunner import check in CI"
    assert "build_training_pairs" in all_runs, "Missing build_training_pairs import check in CI"
    assert "pytest" in all_runs, "Missing pytest run in CI"
    assert "audit_parameters.py" in all_runs, "Missing audit_parameters check in CI"
    assert "smoke_kaggle_pipeline.py" in all_runs, "Missing smoke_kaggle_pipeline run in CI"
    assert "check_notebook_parity.py" in all_runs, "Missing check_notebook_parity check in CI"


def test_notebook_parity_script():
    from scripts.check_notebook_parity import check_notebook_parity, main

    root_nb = REPO_ROOT / "legalir_training.ipynb"
    kaggle_nb = REPO_ROOT / "kaggle_kernel_task1" / "legalir_training.ipynb"

    assert root_nb.exists(), f"Missing root notebook: {root_nb}"
    assert kaggle_nb.exists(), f"Missing kaggle kernel notebook: {kaggle_nb}"

    is_identical, sha_root, sha_kaggle = check_notebook_parity(root_nb, kaggle_nb)
    assert sha_root == hashlib.sha256(root_nb.read_bytes()).hexdigest()
    assert sha_kaggle == hashlib.sha256(kaggle_nb.read_bytes()).hexdigest()
    assert is_identical == (sha_root == sha_kaggle)
