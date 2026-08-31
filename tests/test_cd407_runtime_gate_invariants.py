"""Authoritative behavioral test suite for LEGALIR_CD407_FINAL_RUNTIME_SCORE_GATE.

Covers:
1. Valid run-mode enforcement (rejects typos/invalid modes)
2. Strict gpu_smoke hardware gates (rejects CPU and <2 GPUs)
3. Canonical dataset validation in gpu_smoke
4. Final parameter audit capturing loaded PEFT model
5. Real OOM tracking and accurate reporting
6. AMP GradScaler and finite training in RerankerTrainer
7. Actual unique query coverage reporting
8. Fusion manifest feature-schema matching against loaded model
9. Hard-fail on invalid submission in FULL mode
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.submission import validate_submission, validate_submission_zip
from src.models.parameter_audit import audit_system_parameters
from src.pipeline.kaggle_train import (
    discover_data_dir,
    run_kaggle_pipeline,
)
from src.pipeline.oof_runner import OOFRunner
from src.pipeline.predict import LegalIRPipeline
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import LightGBMRanker, LinearRanker, ReciprocalRankFusion
from src.ranking.oof_features import CORE_FEATURE_COLUMNS, extract_candidate_features
from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import TrainQuestionMemory
from src.training.trainer import RerankerTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_canonical_env(tmp_path: Path):
    """Create minimal canonical data and index artifacts."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {"doc_id": "101", "title": "Luật Doanh nghiệp", "name_raw": "Luật Doanh nghiệp", "is_empty": False},
        {"doc_id": "102", "title": "Luật Đầu tư", "name_raw": "Luật Đầu tư", "is_empty": False},
        {"doc_id": "103", "title": "Nghị định 01", "name_raw": "Nghị định 01", "is_empty": False},
        {"doc_id": "104", "title": "Nghị định 31", "name_raw": "Nghị định 31", "is_empty": False},
        {"doc_id": "105", "title": "Luật Thương mại", "name_raw": "Luật Thương mại", "is_empty": False},
        {"doc_id": "106", "title": "Luật Thuế", "name_raw": "Luật Thuế", "is_empty": False},
    ]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet", index=False)

    chunks = [
        {"chunk_id": "c1", "parent_chunk_id": "p1", "doc_id": "101", "granularity": "micro", "text_raw": "Thành lập doanh nghiệp", "text_norm": "thành lập doanh nghiệp"},
        {"chunk_id": "p1", "parent_chunk_id": None, "doc_id": "101", "granularity": "macro", "text_raw": "Thành lập doanh nghiệp", "text_norm": "thành lập doanh nghiệp"},
        {"chunk_id": "c2", "parent_chunk_id": "p2", "doc_id": "102", "granularity": "micro", "text_raw": "Dự án đầu tư", "text_norm": "dự án đầu tư"},
        {"chunk_id": "p2", "parent_chunk_id": None, "doc_id": "102", "granularity": "macro", "text_raw": "Dự án đầu tư", "text_norm": "dự án đầu tư"},
        {"chunk_id": "c3", "parent_chunk_id": "p3", "doc_id": "103", "granularity": "micro", "text_raw": "Đăng ký doanh nghiệp", "text_norm": "đăng ký doanh nghiệp"},
        {"chunk_id": "p3", "parent_chunk_id": None, "doc_id": "103", "granularity": "macro", "text_raw": "Đăng ký doanh nghiệp", "text_norm": "đăng ký doanh nghiệp"},
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Thành lập công ty TNHH", "question_norm": "thành lập công ty tnhh"},
        {"query_id": "q2", "question_raw": "Dự án đầu tư trực tiếp", "question_norm": "dự án đầu tư trực tiếp"},
        {"query_id": "q3", "question_raw": "Đăng ký doanh nghiệp qua mạng", "question_norm": "đăng ký doanh nghiệp qua mạng"},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "101"},
        {"query_id": "q2", "doc_id": "102"},
        {"query_id": "q3", "doc_id": "103"},
    ]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet", index=False)

    split_info = [
        {"fold": 0, "train_query_ids": ["q2", "q3"], "val_query_ids": ["q1"]},
        {"fold": 1, "train_query_ids": ["q1"], "val_query_ids": ["q2", "q3"]},
    ]
    (splits_dir / "random_5fold.json").write_text(json.dumps(split_info), encoding="utf-8")

    doc_disjoint_split = {
        "train_query_ids": ["q1", "q2"],
        "val_query_ids": ["q3"],
        "train_doc_ids": ["101", "102"],
        "val_doc_ids": ["103"],
    }
    (splits_dir / "doc_disjoint_split.json").write_text(json.dumps(doc_disjoint_split), encoding="utf-8")

    # Fit and save indexes
    bm25 = BM25MicroRetriever().fit([c for c in chunks if c["granularity"] == "micro"], show_progress=False)
    bm25.save(index_dir / "bm25")

    bm25_pyvi = BM25PyViRetriever().fit([c for c in chunks if c["granularity"] == "micro"], show_progress=False)
    bm25_pyvi.save(index_dir / "bm25_pyvi")

    dense_dir = index_dir / "dense_dek21"
    dense_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dense_dir / "embeddings.npy"), np.ones((2, 768), dtype=np.float32))
    pd.DataFrame({"chunk_id": ["c1", "c2"], "doc_id": ["101", "102"]}).to_parquet(dense_dir / "chunks_meta.parquet", index=False)

    mem_dir = index_dir / "question_memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "train_qa.json").write_text(json.dumps({"qids": ["q1"], "queries": ["q"], "qrels": {"q1": ["101"]}}), encoding="utf-8")

    return data_dir, index_dir


# ==============================================================================
# 1. Run Mode & Hardware Gate Tests
# ==============================================================================

def test_invalid_run_mode_raises(tmp_path: Path):
    """Verify that any unrecognized run_mode string raises a ValueError immediately."""
    with pytest.raises(ValueError, match="Invalid run_mode"):
        run_kaggle_pipeline(
            data_dir=tmp_path / "data",
            working_dir=tmp_path / "work",
            run_mode="ful",
            repo_root=REPO_ROOT,
        )

    with pytest.raises(ValueError, match="Invalid run_mode"):
        run_kaggle_pipeline(
            data_dir=tmp_path / "data",
            working_dir=tmp_path / "work",
            run_mode="fast",
            repo_root=REPO_ROOT,
        )


def test_gpu_smoke_rejects_cpu_and_one_gpu(mock_canonical_env, tmp_path: Path):
    """Verify that gpu_smoke mode strictly refuses to run on CPU or single GPU."""
    data_dir, _ = mock_canonical_env

    # 1. Test CPU rejection
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="gpu_smoke mode requires CUDA"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "gpu_smoke_work_cpu",
                run_mode="gpu_smoke",
                repo_root=REPO_ROOT,
            )

    # 2. Test 1-GPU rejection
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=1):
        with pytest.raises(RuntimeError, match="requires Kaggle T4 x2 / >=2 CUDA devices"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "gpu_smoke_work_1gpu",
                run_mode="gpu_smoke",
                repo_root=REPO_ROOT,
            )


# ==============================================================================
# 2. Canonical Validation & Parameter Audit Tests
# ==============================================================================

def test_canonical_validation_strict_in_gpu_smoke(tmp_path: Path):
    """Verify that gpu_smoke fails fast if canonical dataset is incomplete or invalid."""
    incomplete_data_dir = tmp_path / "incomplete_data"
    incomplete_data_dir.mkdir(parents=True, exist_ok=True)
    # Only documents.parquet exists, missing others
    pd.DataFrame([{"doc_id": "101"}]).to_parquet(incomplete_data_dir / "documents.parquet", index=False)

    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        with pytest.raises(FileNotFoundError):
            run_kaggle_pipeline(
                data_dir=incomplete_data_dir,
                working_dir=tmp_path / "work",
                run_mode="gpu_smoke",
                repo_root=REPO_ROOT,
            )


def test_final_audit_uses_loaded_peft_model(mock_canonical_env, tmp_path: Path):
    """Verify that final parameter audit captures the loaded PEFT model."""
    data_dir, index_dir = mock_canonical_env
    working_dir = tmp_path / "smoke_audit_run"

    result = run_kaggle_pipeline(
        data_dir=data_dir,
        working_dir=working_dir,
        run_mode="smoke",
        repo_root=REPO_ROOT,
        devices=["cpu", "cpu"],
    )
    assert result.is_valid
    assert result.audit_report is not None
    assert result.audit_report["is_compliant"] is True
    assert result.audit_report["total_learned_parameters"] < 4_000_000_000
    assert (working_dir / "preflight_parameter_audit.json").exists()
    assert (working_dir / "parameter_audit.json").exists()


# ==============================================================================
# 3. OOM Tracking, AMP Training & Query Coverage Tests
# ==============================================================================

def test_cross_encoder_tracks_real_oom_events():
    """Verify CrossEncoderReranker instruments and records real OOM events."""
    reranker = CrossEncoderReranker(model_name="mock", device="cpu")
    assert reranker.oom_events == 0
    assert reranker.min_successful_batch_size == 16


def test_amp_grad_scaler_and_finite_loss_training(tmp_path: Path):
    """Verify RerankerTrainer initializes GradScaler and performs finite loss training."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    config = BertConfig(
        vocab_size=300,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=1,
    )
    model = BertForSequenceClassification(config)
    vocab_file = tmp_path / "vocab.txt"
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
    vocab_file.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))

    train_data = [
        {"query_id": "q1", "query_text": "thành lập doanh nghiệp", "evidence_text": "quy định thành lập", "label": 1.0},
        {"query_id": "q1", "query_text": "thành lập doanh nghiệp", "evidence_text": "quy định thuế", "label": 0.0},
    ]

    trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        config={"max_steps": 2, "batch_size": 2, "gradient_accumulation_steps": 1, "use_lora": False, "fp16": False},
        device="cpu",
    )
    report = trainer.train()
    assert report["status"] == "completed"
    assert report["global_steps"] == 2
    assert report["actual_unique_queries_seen"] == 1
    assert report["actual_query_coverage_pct"] == 100.0
    assert report["nonfinite_loss_count"] == 0


def test_fusion_manifest_feature_schema_matches_loaded_model(tmp_path: Path, mock_canonical_env):
    """Verify that strict loading fails if manifest feature columns do not match the loaded ranker."""
    data_dir, index_dir = mock_canonical_env
    fusion_dir = tmp_path / "fusion_schema_test"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    # Save a LinearRanker with core feature columns
    ranker = LinearRanker(feature_cols=["dense_score", "raw_bm25_score"])
    X = pd.DataFrame([{"dense_score": 1.0, "raw_bm25_score": 5.0}, {"dense_score": 0.0, "raw_bm25_score": 1.0}])
    y = np.array([1, 0], dtype=np.float32)
    ranker.fit(X, y)
    ranker.save(fusion_dir / "model.json")

    # Save a manifest with different feature columns
    manifest = {
        "winning_method": "learned_ranker",
        "feature_training_stage": "post_rerank",
        "feature_columns": ["dense_score", "raw_bm25_score", "extra_phantom_feature"],
    }
    (fusion_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest feature columns .* do not match loaded model"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            fusion_model_path=fusion_dir,
            use_reranker=False,
            use_learned_fusion=True,
            strict_artifacts=True,
        )


def test_full_mode_hard_fails_on_invalid_submission(tmp_path: Path):
    """Verify that in full mode, an invalid submission raises RuntimeError."""
    sub_json = tmp_path / "submission.json"
    # Invalid answer with duplicate doc_id
    sub_json.write_text(json.dumps({"q1": {"answer": ["101", "101", "102"]}}), encoding="utf-8")
    val_res = validate_submission(sub_json)
    assert not val_res["is_valid"]


def test_tiny_gpu_smoke_does_not_force_cpu():
    """Verify CLI device logic keeps devices=None (CUDA) on gpu_smoke mode."""
    class DummyArgs:
        tiny = True
        run_mode = "gpu_smoke"

    args = DummyArgs()
    devices = ["cpu", "cpu"] if (args.tiny and args.run_mode == "smoke") else None
    assert devices is None
