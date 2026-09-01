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


# ==============================================================================
# Task 2: CI-Status Verification for Colab (verify_github_ci)
# ==============================================================================

def test_verify_github_ci_green(monkeypatch):
    from scripts.verify_github_ci import check_ci_status

    target_sha = "a" * 40

    def mock_urlopen(req, *args, **kwargs):
        class MockResponse:
            def read(self):
                payload = {
                    "workflow_runs": [
                        {
                            "name": "LegalIR CI",
                            "head_sha": target_sha,
                            "status": "completed",
                            "conclusion": "success",
                            "html_url": "https://github.com/silent9669/LegalIR/actions/runs/123",
                        }
                    ]
                }
                return json.dumps(payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return MockResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    is_green, msg = check_ci_status(repo="silent9669/LegalIR", sha=target_sha)
    assert is_green is True
    assert "GREEN" in msg or "success" in msg


def test_verify_github_ci_failed(monkeypatch):
    from scripts.verify_github_ci import check_ci_status

    target_sha = "b" * 40

    def mock_urlopen(req, *args, **kwargs):
        class MockResponse:
            def read(self):
                payload = {
                    "workflow_runs": [
                        {
                            "name": "LegalIR CI",
                            "head_sha": target_sha,
                            "status": "completed",
                            "conclusion": "failure",
                            "html_url": "https://github.com/silent9669/LegalIR/actions/runs/124",
                        }
                    ]
                }
                return json.dumps(payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return MockResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    is_green, msg = check_ci_status(repo="silent9669/LegalIR", sha=target_sha)
    assert is_green is False
    assert "failure" in msg or "not successful" in msg


def test_verify_github_ci_in_progress(monkeypatch):
    from scripts.verify_github_ci import check_ci_status

    target_sha = "c" * 40

    def mock_urlopen(req, *args, **kwargs):
        class MockResponse:
            def read(self):
                payload = {
                    "workflow_runs": [
                        {
                            "name": "LegalIR CI",
                            "head_sha": target_sha,
                            "status": "in_progress",
                            "conclusion": None,
                            "html_url": "https://github.com/silent9669/LegalIR/actions/runs/125",
                        }
                    ]
                }
                return json.dumps(payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return MockResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    is_green, msg = check_ci_status(repo="silent9669/LegalIR", sha=target_sha)
    assert is_green is False
    assert "in_progress" in msg or "not completed" in msg


def test_verify_github_ci_no_workflow(monkeypatch):
    from scripts.verify_github_ci import check_ci_status

    target_sha = "d" * 40

    def mock_urlopen(req, *args, **kwargs):
        class MockResponse:
            def read(self):
                payload = {"workflow_runs": []}
                return json.dumps(payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return MockResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    is_green, msg = check_ci_status(repo="silent9669/LegalIR", sha=target_sha)
    assert is_green is False
    assert "No workflow runs found" in msg or "not found" in msg or "No matching" in msg


def test_verify_github_ci_rate_limit_fail_closed(monkeypatch):
    from scripts.verify_github_ci import check_ci_status
    import urllib.error

    target_sha = "e" * 40

    def mock_urlopen(req, *args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.github.com",
            code=403,
            msg="rate limit exceeded",
            hdrs={},
            fp=None,
        )

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    is_green, msg = check_ci_status(repo="silent9669/LegalIR", sha=target_sha)
    assert is_green is False
    assert "403" in msg or "rate limit" in msg.lower() or "GITHUB_TOKEN" in msg


# ==============================================================================
# Task 3: Protected Score Configuration & Smoke Overrides
# ==============================================================================

def test_protected_score_keys_rejected():
    from src.pipeline.colab_smoke import validate_smoke_overrides, PROTECTED_SCORE_KEYS

    assert "loss_type" in PROTECTED_SCORE_KEYS or any("loss" in k for k in PROTECTED_SCORE_KEYS)
    assert any("lora" in k for k in PROTECTED_SCORE_KEYS)
    assert any("weight" in k for k in PROTECTED_SCORE_KEYS)
    assert any("fusion" in k for k in PROTECTED_SCORE_KEYS)

    prod_config = {
        "retrieval": {"fusion": {"weights": {"bm25": 1.0, "dense": 1.5}}},
        "ranking": {"reranker": {"model_name": "BAAI/bge-reranker-v2-m3", "loss_type": "bce"}},
        "training": {"lora_r": 16, "learning_rate": 2e-5},
    }

    # Attempting to override RRF weights must raise ValueError
    bad_smoke_1 = {"weights": {"bm25": 2.0, "dense": 0.5}}
    with pytest.raises(ValueError, match="Protected score key"):
        validate_smoke_overrides(prod_config, bad_smoke_1)

    # Attempting to override loss_type must raise ValueError
    bad_smoke_2 = {"loss_type": "pairwise_logistic"}
    with pytest.raises(ValueError, match="Protected score key"):
        validate_smoke_overrides(prod_config, bad_smoke_2)

    # Attempting to override LoRA rank must raise ValueError
    bad_smoke_3 = {"lora_r": 8}
    with pytest.raises(ValueError, match="Protected score key"):
        validate_smoke_overrides(prod_config, bad_smoke_3)

    # Attempting to override learning rate must raise ValueError
    bad_smoke_4 = {"learning_rate": 1e-3}
    with pytest.raises(ValueError, match="Protected score key"):
        validate_smoke_overrides(prod_config, bad_smoke_4)


def test_allowed_smoke_overrides_accepted():
    from src.pipeline.colab_smoke import validate_smoke_overrides

    prod_config = {
        "retrieval": {"fusion": {"weights": {"bm25": 1.0, "dense": 1.5}}},
        "ranking": {"reranker": {"model_name": "BAAI/bge-reranker-v2-m3", "loss_type": "bce"}},
        "training": {"lora_r": 16, "learning_rate": 2e-5},
    }

    valid_smoke = {
        "train_queries": 64,
        "validation_queries": 32,
        "public_queries": 16,
        "max_documents": 2000,
        "folds": 2,
        "reranker_optimizer_steps": 10,
        "dense_batch_size": 16,
        "reranker_batch_size": 8,
        "seed": 42,
    }

    # Should succeed without raising exceptions
    validate_smoke_overrides(prod_config, valid_smoke)


def test_colab_smoke_yaml_file():
    yaml_path = REPO_ROOT / "configs" / "colab_smoke.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data.get("seed") == 42
    assert data.get("train_queries") == 64
    assert data.get("validation_queries") == 32
    assert data.get("public_queries") == 16
    assert data.get("max_documents") == 2000
    assert data.get("folds") == 2
    assert data.get("reranker_optimizer_steps") == 10
    assert data.get("dense_batch_size") == 16
    assert data.get("reranker_batch_size") == 8

    # Must NOT contain model names or ranking weights
    content = yaml_path.read_text(encoding="utf-8")
    assert "BAAI/bge-reranker-v2-m3" not in content
    assert "CODE4LIFEOFFICIAL" not in content
    assert "weights" not in data


