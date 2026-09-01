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


# ==============================================================================
# Task 4: Deterministic Official-Data Smoke Subset
# ==============================================================================

def test_build_colab_smoke_subset(tmp_path):
    from scripts.smoke_kaggle_pipeline import create_toy_canonical_dataset
    from src.pipeline.colab_smoke import ColabSmokeConfig, build_colab_subset

    toy_data_dir = tmp_path / "toy_data"
    create_toy_canonical_dataset(toy_data_dir)

    out_subset_1 = tmp_path / "subset_1"
    out_subset_2 = tmp_path / "subset_2"

    cfg = ColabSmokeConfig(
        seed=42,
        train_queries=4,
        validation_queries=2,
        public_queries=2,
        max_documents=5,
        folds=2,
    )

    manifest_1 = build_colab_subset(toy_data_dir, out_subset_1, cfg)
    manifest_2 = build_colab_subset(toy_data_dir, out_subset_2, cfg)

    # 1. Determinism: same seed -> exact same QIDs, doc IDs, and SHA
    assert manifest_1.selected_train_qids == manifest_2.selected_train_qids
    assert manifest_1.selected_val_qids == manifest_2.selected_val_qids
    assert manifest_1.selected_doc_ids == manifest_2.selected_doc_ids
    assert manifest_1.manifest_sha256 == manifest_2.manifest_sha256

    # 2. Output files exist
    assert (out_subset_1 / "documents.parquet").exists()
    assert (out_subset_1 / "chunks.parquet").exists()
    assert (out_subset_1 / "queries_train.parquet").exists()
    assert (out_subset_1 / "qrels_train.parquet").exists()
    assert (out_subset_1 / "public-official.json").exists()
    assert (out_subset_1 / "subset_manifest.json").exists()

    # 3. Selected train QIDs come from official train set
    import pandas as pd
    orig_queries = pd.read_parquet(toy_data_dir / "queries_train.parquet")
    orig_qids = set(orig_queries["query_id"].astype(str))
    for qid in manifest_1.selected_train_qids:
        assert qid in orig_qids

    # 4. All qrel-positive docs for selected queries are included
    orig_qrels = pd.read_parquet(toy_data_dir / "qrels_train.parquet")
    all_selected_qids = set(manifest_1.selected_train_qids) | set(manifest_1.selected_val_qids)
    pos_qrels = orig_qrels[orig_qrels["query_id"].astype(str).isin(all_selected_qids)]
    for doc_id in pos_qrels["doc_id"].astype(str):
        assert doc_id in manifest_1.selected_doc_ids

    # 5. Subset chunks belong ONLY to selected docs
    sub_chunks = pd.read_parquet(out_subset_1 / "chunks.parquet")
    assert set(sub_chunks["doc_id"].astype(str)).issubset(set(manifest_1.selected_doc_ids))

    # 6. No synthetic qrels (all subset qrels must be in original qrels)
    sub_qrels = pd.read_parquet(out_subset_1 / "qrels_train.parquet")
    orig_tuples = set(zip(orig_qrels["query_id"].astype(str), orig_qrels["doc_id"].astype(str)))
    sub_tuples = set(zip(sub_qrels["query_id"].astype(str), sub_qrels["doc_id"].astype(str)))
    assert sub_tuples.issubset(orig_tuples)

    # 7. Validation QIDs never enter smoke pair-training / train QIDs
    assert set(manifest_1.selected_train_qids).isdisjoint(set(manifest_1.selected_val_qids))


# ==============================================================================
# Task 5: Colab Single-T4 Smoke Runner & Hardware Contract
# ==============================================================================

def test_hardware_contract_validation():
    from src.pipeline.colab_smoke import check_gpu_readiness

    # 1. No CUDA available -> raises RuntimeError
    with pytest.raises(RuntimeError, match="CUDA not available"):
        check_gpu_readiness(cuda_available=False, gpu_name="Tesla T4")

    # 2. Tesla T4 -> Validated for readiness
    is_t4, verdict, msg = check_gpu_readiness(cuda_available=True, gpu_name="Tesla T4", allow_non_t4=False)
    assert is_t4 is True
    assert verdict == "READY_FOR_T4_SMOKE"

    # 3. L4 or A100 without override -> Raises RuntimeError / Fails readiness
    with pytest.raises(RuntimeError, match="not a Tesla T4"):
        check_gpu_readiness(cuda_available=True, gpu_name="NVIDIA A100-SXM4-40GB", allow_non_t4=False)

    # 4. Non-T4 with explicit override -> returns NOT_A_T4_READINESS_GATE verdict
    is_t4, verdict, msg = check_gpu_readiness(cuda_available=True, gpu_name="NVIDIA A100-SXM4-40GB", allow_non_t4=True)
    assert is_t4 is False
    assert verdict == "NOT_A_T4_READINESS_GATE"


def test_colab_smoke_report_schema_and_execution(tmp_path, monkeypatch):
    from scripts.smoke_kaggle_pipeline import create_toy_canonical_dataset
    from src.pipeline.colab_smoke import ColabSmokeConfig, run_colab_t4_smoke_pipeline

    toy_data = tmp_path / "toy_data"
    create_toy_canonical_dataset(toy_data)

    work_dir = tmp_path / "colab_work"
    cfg = ColabSmokeConfig(
        seed=42,
        train_queries=4,
        validation_queries=2,
        public_queries=2,
        max_documents=5,
        folds=2,
        reranker_optimizer_steps=3,
        device="cpu",  # CPU for local pytest execution
    )

    # Mock CI check to return True
    import scripts.verify_github_ci as vci
    monkeypatch.setattr(vci, "check_ci_status", lambda *args, **kwargs: (True, "GREEN"))

    report = run_colab_t4_smoke_pipeline(
        data_dir=toy_data,
        work_dir=work_dir,
        target_sha="a" * 40,
        config=cfg,
        skip_ci_check=False,
        allow_non_t4=True,
        use_mock_models=True,
    )

    # Assert required report fields
    required_fields = [
        "git_sha",
        "ci_green",
        "gpu_name",
        "cuda_version",
        "torch_version",
        "dataset_identity",
        "subset_manifest_hash",
        "split_provenance_sha",
        "duplicate_blacklist",
        "dense_device",
        "dense_backend",
        "dense_embeddings_finite",
        "dense_telemetry",
        "reranker_device",
        "optimizer_steps",
        "loss_finite",
        "param_diff",
        "adapter_verification",
        "parameter_audit",
        "prediction_validation",
        "stage_timings",
        "result",
    ]
    for field in required_fields:
        assert field in report, f"Report missing required field: {field}"

    assert report["loss_finite"] is True
    assert report["param_diff"] >= 0.0
    assert report["parameter_audit"]["system_learned_parameters"] < 4_000_000_000
    assert report["prediction_validation"]["valid"] is True
    assert (work_dir / "colab_smoke_report.json").exists()


# ==============================================================================
# Task 6: Colab Smoke Notebook Generator & Structure
# ==============================================================================

def test_colab_smoke_notebook_generator_and_content():
    from scripts.generate_colab_smoke_notebook import generate_colab_notebook

    nb_path = REPO_ROOT / "colab" / "legalir_t4_smoke.ipynb"
    generated_path = generate_colab_notebook(nb_path)
    assert generated_path.exists(), f"Failed to generate {generated_path}"

    nb_json = json.loads(generated_path.read_text(encoding="utf-8"))
    assert "cells" in nb_json
    cells = nb_json["cells"]

    all_source = "\n".join("".join(c.get("source", [])) for c in cells)

    # Required cells / keywords
    assert "nvidia-smi" in all_source, "Missing GPU inspection / nvidia-smi"
    assert "userdata.get('HF_TOKEN')" in all_source or 'userdata.get("HF_TOKEN")' in all_source, "Missing HF_TOKEN userdata read"
    assert "userdata.get('GITHUB_TOKEN')" in all_source or 'userdata.get("GITHUB_TOKEN")' in all_source, "Missing GITHUB_TOKEN userdata read"
    assert "drive.mount" in all_source, "Missing Google Drive mount"
    assert "TARGET_SHA" in all_source, "Missing TARGET_SHA configuration"
    assert "git clone" in all_source, "Missing git clone step"
    assert "git checkout" in all_source, "Missing detached checkout step"
    assert "verify_github_ci.py" in all_source, "Missing CI verification step"
    assert "run_colab_t4_smoke.py" in all_source, "Missing run_colab_t4_smoke execution step"
    assert "colab_smoke_report.json" in all_source, "Missing report summary inspection"

    # Reject any cell that prints secret values or token values
    assert "print(HF_TOKEN)" not in all_source, "Secret HF_TOKEN must never be printed"
    assert "print(GITHUB_TOKEN)" not in all_source, "Secret GITHUB_TOKEN must never be printed"
    assert "hf_" not in all_source, "No hardcoded Hugging Face token"
    assert "ghp_" not in all_source, "No hardcoded GitHub token"


# ==============================================================================
# Task 7: Preserve Kaggle FULL as Strict Dual-T4 Production
# ==============================================================================

def test_kaggle_full_strict_dual_gpu_isolation(tmp_path, monkeypatch):
    import torch
    from scripts.smoke_kaggle_pipeline import create_toy_canonical_dataset
    from src.pipeline.kaggle_train import run_kaggle_pipeline

    toy_data = tmp_path / "toy_data"
    create_toy_canonical_dataset(toy_data)
    work_dir = tmp_path / "kaggle_work"

    # 1. Colab smoke runner uses 1 GPU sequentially
    from src.pipeline.colab_smoke import ColabSmokeConfig
    smoke_cfg = ColabSmokeConfig()
    assert smoke_cfg.device == "cuda:0"
    assert smoke_cfg.folds == 2
    assert smoke_cfg.train_queries == 64

    # 2. FULL mode rejects 1 CUDA device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="requires Kaggle T4 x2 / >=2 CUDA devices"):
        run_kaggle_pipeline(
            data_dir=toy_data,
            working_dir=work_dir,
            run_mode="full",
            allow_nonstandard_production_devices=False,
        )

    # 3. FULL mode requires dense on cuda:0 and reranker on cuda:1
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="requires dense_device == 'cuda:0'"):
        run_kaggle_pipeline(
            data_dir=toy_data,
            working_dir=work_dir,
            run_mode="full",
            dense_device="cuda:1",
            reranker_device="cuda:0",
            allow_nonstandard_production_devices=False,
        )

    # 4. Production configs are not polluted by smoke limits
    prod_task1_cfg = yaml.safe_load((REPO_ROOT / "configs" / "task1.yaml").read_text(encoding="utf-8"))
    assert "max_documents" not in prod_task1_cfg.get("dataset", {})
    assert "train_queries" not in prod_task1_cfg


# ==============================================================================
# Task 8: Score Promotion Guardrails
# ==============================================================================

def test_score_promotion_rules():
    from scripts.check_score_promotion import evaluate_score_promotion

    baseline = {
        "baseline_metrics": {
            "oof_recall_at_5": 0.7396,
            "oof_precision_at_5": 0.1569,
            "candidate_recall_at_50": 0.9400,
            "candidate_recall_at_150": 0.9700,
            "doc_disjoint_recall_at_5": 0.6600,
        },
        "guardrails": {
            "require_leakage_checks_passed": True,
            "require_doc_disjoint_eval": True,
            "max_candidate_recall_regression": 0.005,
            "max_doc_disjoint_recall_regression": 0.02,
            "max_total_parameters": 4_000_000_000,
        },
    }

    # Case 1: Higher Recall@5 -> Eligible
    cand_higher_r5 = {
        "oof_recall_at_5": 0.7450,
        "oof_precision_at_5": 0.1570,
        "candidate_recall_at_50": 0.9420,
        "candidate_recall_at_150": 0.9710,
        "doc_disjoint_recall_at_5": 0.6650,
        "leakage_checks_passed": True,
        "total_learned_parameters": 702_754_049,
    }
    is_promoted, metrics, reasons = evaluate_score_promotion(cand_higher_r5, baseline)
    assert is_promoted is True
    assert metrics["recall_at_5_delta"] > 0

    # Case 2: Equal Recall@5 + higher Precision@5 -> Eligible
    cand_equal_r5_higher_p5 = {
        "oof_recall_at_5": 0.7396,
        "oof_precision_at_5": 0.1620,
        "candidate_recall_at_50": 0.9400,
        "candidate_recall_at_150": 0.9700,
        "doc_disjoint_recall_at_5": 0.6600,
        "leakage_checks_passed": True,
        "total_learned_parameters": 702_754_049,
    }
    is_promoted, metrics, reasons = evaluate_score_promotion(cand_equal_r5_higher_p5, baseline)
    assert is_promoted is True
    assert metrics["precision_at_5_delta"] > 0

    # Case 3: Lower Recall@5 -> Rejected
    cand_lower_r5 = {
        "oof_recall_at_5": 0.7350,
        "oof_precision_at_5": 0.1600,
        "candidate_recall_at_50": 0.9400,
        "candidate_recall_at_150": 0.9700,
        "doc_disjoint_recall_at_5": 0.6600,
        "leakage_checks_passed": True,
        "total_learned_parameters": 702_754_049,
    }
    is_promoted, metrics, reasons = evaluate_score_promotion(cand_lower_r5, baseline)
    assert is_promoted is False
    assert any("Recall@5 regressed" in r for r in reasons)

    # Case 4: Severe Candidate Recall regression -> Rejected
    cand_bad_candidate = {
        "oof_recall_at_5": 0.7450,
        "oof_precision_at_5": 0.1570,
        "candidate_recall_at_50": 0.9200,  # 2% drop > 0.5% tolerance
        "candidate_recall_at_150": 0.9700,
        "doc_disjoint_recall_at_5": 0.6600,
        "leakage_checks_passed": True,
        "total_learned_parameters": 702_754_049,
    }
    is_promoted, metrics, reasons = evaluate_score_promotion(cand_bad_candidate, baseline)
    assert is_promoted is False
    assert any("Candidate Recall@50 regressed" in r for r in reasons)

    # Case 5: Missing doc-disjoint evaluation -> Rejected
    cand_missing_disjoint = {
        "oof_recall_at_5": 0.7450,
        "oof_precision_at_5": 0.1570,
        "candidate_recall_at_50": 0.9420,
        "candidate_recall_at_150": 0.9710,
        "leakage_checks_passed": True,
        "total_learned_parameters": 702_754_049,
    }
    is_promoted, metrics, reasons = evaluate_score_promotion(cand_missing_disjoint, baseline)
    assert is_promoted is False
    assert any("doc-disjoint" in r.lower() for r in reasons)


def test_production_score_guard_config_file():
    guard_path = REPO_ROOT / "configs" / "production_score_guard.json"
    assert guard_path.exists(), f"Missing {guard_path}"
    data = json.loads(guard_path.read_text(encoding="utf-8"))
    assert "baseline_metrics" in data
    assert data["baseline_metrics"]["oof_recall_at_5"] >= 0.70
    assert "guardrails" in data


# ==============================================================================
# Task 9: Operating Workflow Documentation & Release Governance
# ==============================================================================

def test_workflow_documentation_and_invariants():
    doc_path = REPO_ROOT / "docs" / "CI_COLAB_KAGGLE_WORKFLOW.md"
    assert doc_path.exists(), f"Missing {doc_path}"

    content = doc_path.read_text(encoding="utf-8")
    assert "LegalIR CI" in content
    assert "Colab" in content
    assert "Kaggle" in content
    assert "Tesla T4" in content or "T4" in content
    assert "TARGET_SHA" in content
    assert "check_score_promotion.py" in content
    assert "Recall@5" in content

    # Check README updates
    readme_path = REPO_ROOT / "README.md"
    readme_content = readme_path.read_text(encoding="utf-8")
    assert "CI_COLAB_KAGGLE_WORKFLOW" in readme_content or "Colab" in readme_content
    assert "check_score_promotion.py" in readme_content or "Score Promotion" in readme_content









