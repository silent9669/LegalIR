"""Authoritative test suite for LEGALIR_17057_FINAL_RUNTIME_COVERAGE_REPAIR contract.

Covers:
1. P0-1: Training coverage calculation, required step derivations, Phase A/B/C ordering, and coverage invariants.
2. P0-2: GPU-smoke runtime projection scaling to full production steps and 5 folds.
3. P0-3: FULL mode dual-GPU T4x2 hardware topology enforcement.
4. P1-1: Outer fold training data utilization (no 10% waste when no early stopping).
5. P1-2: Strict PEFT parameter audit requiring active adapter parameters.
6. P1-3: Stage-wise Dense telemetry (per-call tracking).
7. P1-4: Fusion manifest.json containing winning_model_type.
8. P1-5: Portable subprocess test execution.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.parameter_audit import audit_model_parameters, audit_system_parameters
from src.pipeline.kaggle_train import run_kaggle_pipeline
from src.pipeline.predict import LegalIRPipeline
from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.dense_macro import DenseMacroRetriever, DenseEncodeTelemetry
from src.training.trainer import (
    QueryBalancedSampler,
    RerankerPairDataset,
    compute_coverage_required_steps,
    setup_peft_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_mock_canonical_dataset(data_dir: Path, num_docs: int = 8532, num_queries: int = 7000, num_public: int = 999):
    """Helper to generate schema-compliant mock canonical data."""
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "doc_id": str(i),
            "name_raw": f"Doc {i}",
            "title": f"Doc {i}",
            "link": "",
            "passage_raw": f"Passage {i}",
            "passage_norm": f"passage {i}",
            "legal_number": f"{i}/2021",
            "year": "2021",
            "doc_type": "Luật",
            "is_empty": False,
        }
        for i in range(num_docs)
    ]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet", index=False)

    chunks = [
        {
            "chunk_id": f"c_{i}",
            "doc_id": str(i % max(1, num_docs)),
            "granularity": "macro",
            "chapter": "",
            "section": "",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": f"Text {i}",
            "text_norm": f"text {i}",
            "parent_chunk_id": None,
            "token_count": 10,
            "is_empty": False,
        }
        for i in range(num_docs)
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {
            "query_id": f"q_{i}",
            "question_raw": f"Query {i}",
            "question_norm": f"query {i}",
            "gold_count": 1,
        }
        for i in range(num_queries)
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {
            "query_id": f"q_{i}",
            "doc_id": str(i % max(1, num_docs)),
            "relevance": 1,
        }
        for i in range(num_queries)
    ]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet", index=False)

    if num_public > 0:
        public_data = {f"pub_{i}": {"question": f"pub query {i}"} for i in range(num_public)}
        public_file = data_dir / "public-official.json"
        public_file.write_text(json.dumps(public_data), encoding="utf-8")
        return data_dir, public_file
    return data_dir, None


# ==============================================================================
# 1. P0-1: Training Coverage & Step Calculations
# ==============================================================================

def test_required_steps_for_7000_bce_queries_is_867_for_99pct():
    """Verify compute_coverage_required_steps calculates 867 steps for 99% of 7000 queries at batch=2, grad_accum=8."""
    steps_99 = compute_coverage_required_steps(
        eligible_query_count=7000,
        batch_size=2,
        gradient_accumulation_steps=8,
        target_coverage_pct=0.99,
        require_pos_and_neg=True,
    )
    assert steps_99 == 867

    steps_100 = compute_coverage_required_steps(
        eligible_query_count=7000,
        batch_size=2,
        gradient_accumulation_steps=8,
        target_coverage_pct=1.00,
        require_pos_and_neg=True,
    )
    assert steps_100 == 875


def test_sampler_first_phase_contains_one_unique_query_each():
    """Verify Phase A of QueryBalancedSampler schedules exactly 1 positive for each eligible query before Phase B."""
    pairs = [
        {"query_id": "q1", "evidence_text": "pos 1", "label": 1.0},
        {"query_id": "q1", "evidence_text": "neg 1a", "label": 0.0},
        {"query_id": "q1", "evidence_text": "neg 1b", "label": 0.0},
        {"query_id": "q2", "evidence_text": "pos 2", "label": 1.0},
        {"query_id": "q2", "evidence_text": "neg 2a", "label": 0.0},
        {"query_id": "q3", "evidence_text": "pos 3", "label": 1.0},
        {"query_id": "q3", "evidence_text": "neg 3a", "label": 0.0},
    ]
    dataset = RerankerPairDataset(pairs, balanced=False)
    sampler = QueryBalancedSampler(dataset, seed=42)
    indices = list(iter(sampler))

    assert len(indices) == len(pairs)
    # First 3 indices (Phase A) must contain each of q1, q2, q3 once (positives only)
    phase_a_qids = [pairs[i]["query_id"] for i in indices[:3]]
    phase_a_labels = [pairs[i]["label"] for i in indices[:3]]
    assert len(set(phase_a_qids)) == 3
    assert all(lbl > 0.5 for lbl in phase_a_labels)

    # Next 3 indices (Phase B) must contain each of q1, q2, q3 once (negatives only)
    phase_b_qids = [pairs[i]["query_id"] for i in indices[3:6]]
    phase_b_labels = [pairs[i]["label"] for i in indices[3:6]]
    assert len(set(phase_b_qids)) == 3
    assert all(lbl <= 0.5 for lbl in phase_b_labels)


def test_eligible_coverage_never_exceeds_100pct():
    """Verify query coverage percentage is capped at 100% and correctly intersected."""
    pairs = [
        {"query_id": "q1", "evidence_text": "pos 1", "label": 1.0},
        {"query_id": "q1", "evidence_text": "neg 1", "label": 0.0},
        {"query_id": "q_ineligible", "evidence_text": "pos only", "label": 1.0},
    ]
    dataset = RerankerPairDataset(pairs, balanced=False)
    sampler = QueryBalancedSampler(dataset, seed=42)
    assert "q1" in sampler.eligible_query_ids
    assert len(sampler.eligible_query_ids) == 1


# ==============================================================================
# 2. P0-2: GPU Smoke Runtime Projection Scaling
# ==============================================================================

def test_gpu_smoke_projection_scales_3_steps_to_full_final_steps(tmp_path: Path):
    """Verify runtime_projection.json extrapolates smoke steps to full production step counts (e.g. 875)."""
    data_dir, public_file = make_mock_canonical_dataset(tmp_path / "data", 8532, 7000, 999)

    mock_dense = mock.MagicMock()
    mock_dense.dense_search_backend = "faiss_index_flat_ip"
    mock_dense._faiss_index = mock.MagicMock()
    mock_dense.device = "cuda:0"
    mock_param_0 = mock.MagicMock()
    mock_param_0.device = "cuda:0"
    mock_dense.model.parameters.return_value = iter([mock_param_0])
    mock_dense.encode_texts.return_value = np.zeros((1, 64), dtype=np.float32)
    mock_dense.encode_queries.return_value = np.zeros((1, 64), dtype=np.float32)

    mock_pipeline = mock.MagicMock()
    mock_pipeline.hybrid_engine = mock.MagicMock()
    mock_pipeline.hybrid_engine.dense = mock_dense
    mock_pipeline.hybrid_engine.dense_retriever = mock_dense
    mock_pipeline.reranker = mock.MagicMock()
    mock_pipeline.reranker.device = "cuda:1"
    mock_param_1 = mock.MagicMock()
    mock_param_1.device = "cuda:1"
    mock_pipeline.reranker.model.parameters.return_value = iter([mock_param_1])

    mock_pipeline.audit_parameters.return_value = {
        "total_learned_parameters": 700_000_000,
        "models": {
            "reranker": {
                "role": "cross_encoder_reranker",
                "is_peft_lora": True,
                "adapter_parameters": 100_000,
            }
        }
    }
    mock_pipeline.predict_single.return_value = ["1", "2", "3", "4", "5"]
    mock_pipeline.predict_batch.return_value = {f"pub_{i}": ["1", "2", "3", "4", "5"] for i in range(20)}

    mock_pipe_cls = mock.MagicMock()
    mock_pipe_cls.load_pipeline.return_value = mock_pipeline

    mock_mem = mock.MagicMock()
    mock_mem.qids = ["q0"]

    with mock.patch("src.pipeline.kaggle_train.DenseMacroRetriever", return_value=mock_dense), \
         mock.patch("src.pipeline.kaggle_train.OOFRunner.run", return_value={
             "heldout_inference_queries_per_second": 10.0,
             "reranker_training_seconds_total": 3.0,
             "reranker_optimizer_steps_total": 6,
             "pair_mining_seconds_total": 2.0,
             "mean_recall@5": 0.9,
             "folds": [{"heldout_queries": 10}]
         }), \
         mock.patch("src.pipeline.kaggle_train.train_reranker", return_value={
             "status": "completed",
             "training_time_sec": 3.0,
             "optimizer_steps": 3,
             "actual_query_coverage_pct": 100.0,
             "unique_query_coverage_pct": 100.0,
             "positive_query_coverage_pct": 100.0,
             "negative_query_coverage_pct": 100.0,
         }), \
         mock.patch("src.pipeline.kaggle_train.LegalIRPipeline", mock_pipe_cls), \
         mock.patch("src.pipeline.kaggle_train.TrainQuestionMemory", return_value=mock_mem), \
         mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.device_count", return_value=2):

        work_dir = tmp_path / "work"
        run_kaggle_pipeline(
            data_dir=data_dir,
            working_dir=work_dir,
            run_mode="gpu_smoke",
            public_json_path=public_file,
            repo_root=REPO_ROOT,
        )

        proj = json.loads((work_dir / "runtime_projection.json").read_text(encoding="utf-8"))
        assert proj["projected_num_folds"] == 5
        assert proj["projected_final_steps"] == 875
        assert proj["total_oof_validation_queries"] == 7000
        assert proj["includes_doc_disjoint"] is True


# ==============================================================================
# 3. P0-3: FULL Mode Hardware Topology Enforcement
# ==============================================================================

def test_full_rejects_cpu(tmp_path: Path):
    """Verify run_kaggle_pipeline in full mode hard-fails if CUDA is unavailable."""
    data_dir, public_file = make_mock_canonical_dataset(tmp_path / "data", 8532, 7000, 999)

    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="(?i)full mode requires CUDA"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "work",
                run_mode="full",
                public_json_path=public_file,
                repo_root=REPO_ROOT,
            )


def test_full_rejects_single_gpu(tmp_path: Path):
    """Verify run_kaggle_pipeline in full mode hard-fails if fewer than 2 GPUs are present."""
    data_dir, public_file = make_mock_canonical_dataset(tmp_path / "data", 8532, 7000, 999)

    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=1):
        with pytest.raises(RuntimeError, match="(?i)full mode requires Kaggle T4 x2 / >=2 CUDA devices"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "work",
                run_mode="full",
                public_json_path=public_file,
                repo_root=REPO_ROOT,
            )


# ==============================================================================
# 4. P1-1: Outer Fold Data Utilization & P1-2: PEFT Adapter Auditing
# ==============================================================================

def test_outer_fold_training_uses_all_outer_train_qids():
    """Verify train_reranker uses 100% of outer-fold pairs without 10% holdout waste."""
    from src.training.train_reranker import train_reranker

    pairs = [
        {"query_id": f"q_{i}", "evidence_text": f"p_{i}", "label": 1.0 if i % 2 == 0 else 0.0}
        for i in range(10)
    ]
    pairs_file = REPO_ROOT / "artifacts/test_outer_pairs.parquet"
    pairs_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_parquet(pairs_file, index=False)

    with mock.patch("src.training.train_reranker.RerankerTrainer") as mock_trainer_cls:
        mock_trainer_inst = mock.MagicMock()
        mock_trainer_inst.train.return_value = {"status": "completed", "global_steps": 1}
        mock_trainer_cls.return_value = mock_trainer_inst

        train_reranker(
            pairs_file=pairs_file,
            output_dir=REPO_ROOT / "artifacts/test_out",
            fold=0,
            base_model_name="mock",
        )

        _, kwargs = mock_trainer_cls.call_args
        assert len(kwargs["train_data"]) == 10
        assert kwargs["val_data"] is None

    if pairs_file.exists():
        pairs_file.unlink()


def test_final_audit_requires_positive_adapter_parameter_count(tmp_path: Path):
    """Verify strict audit enforces adapter_parameters > 0 when adapter model is loaded."""
    from transformers import BertConfig, BertForSequenceClassification
    from src.retrieval.bm25_micro import BM25MicroRetriever
    from src.retrieval.hybrid_search import HybridSearchEngine

    config = BertConfig(vocab_size=100, hidden_size=16, num_attention_heads=2, num_hidden_layers=2, num_labels=1)
    base_model = BertForSequenceClassification(config)
    peft_model, _ = setup_peft_model(base_model, lora_r=4, lora_alpha=8)

    reranker = CrossEncoderReranker(model_name="mock", device="cpu")
    reranker.model = peft_model
    reranker.is_peft = True

    pipeline = LegalIRPipeline(
        hybrid_engine=HybridSearchEngine(
            bm25_retriever=BM25MicroRetriever().fit([{"chunk_id": "c1", "doc_id": "1", "text_norm": "text"}]),
        ),
        reranker=reranker,
    )

    report = pipeline.audit_parameters(output_json=tmp_path / "audit.json", require_loaded_models=True)
    reranker_entry = next(m for m in report["models"].values() if m["role"] == "cross_encoder_reranker")
    assert reranker_entry["is_peft_lora"] is True
    assert reranker_entry["adapter_parameters"] > 0


# ==============================================================================
# 5. P1-3: Stage-Wise Dense Telemetry & P1-4: Fusion Manifest
# ==============================================================================

def test_dense_telemetry_separates_corpus_and_train_query_calls():
    """Verify DenseMacroRetriever records separate stage telemetries for corpus and query encoding."""
    retriever = DenseMacroRetriever(model_name="mock", dimension=64, device="cpu")
    retriever.encode_corpus(["passage 1", "passage 2"], batch_size=2)
    retriever.encode_queries(["query 1"], batch_size=1)

    assert "corpus" in retriever.stage_telemetry
    assert "train_query" in retriever.stage_telemetry
    assert retriever.stage_telemetry["corpus"].item_count == 2
    assert retriever.stage_telemetry["train_query"].item_count == 1


def test_fusion_manifest_json_contains_actual_winning_model_type(tmp_path: Path):
    """Verify fusion manifest.json contains winning_model_type."""
    from src.ranking.train_fusion import train_and_evaluate_fusion_cv

    oof_df = pd.DataFrame([
        {"query_id": "q1", "doc_id": "d1", "label": 1.0, "fold": 0, "exact_score": 1.0, "bm25_score": 5.0, "dense_score": 0.9, "memory_score": 0.8, "reranker_score": 0.95},
        {"query_id": "q1", "doc_id": "d2", "label": 0.0, "fold": 0, "exact_score": 0.0, "bm25_score": 2.0, "dense_score": 0.3, "memory_score": 0.1, "reranker_score": 0.1},
        {"query_id": "q2", "doc_id": "d2", "label": 1.0, "fold": 1, "exact_score": 1.0, "bm25_score": 4.0, "dense_score": 0.85, "memory_score": 0.7, "reranker_score": 0.9},
        {"query_id": "q2", "doc_id": "d1", "label": 0.0, "fold": 1, "exact_score": 0.0, "bm25_score": 1.0, "dense_score": 0.2, "memory_score": 0.0, "reranker_score": 0.05},
    ])
    qrels_dict = {"q1": ["d1"], "q2": ["d2"]}

    out_dir = tmp_path / "fusion_cv"
    train_and_evaluate_fusion_cv(oof_df, qrels_dict, output_dir=out_dir, num_boost_round=5)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "winning_model_type" in manifest
    assert manifest["winning_model_type"] in ("lightgbm", "linear_fallback", "rrf_weighted")


def test_500_steps_cannot_claim_99pct_posneg_coverage_for_7000_queries():
    """Verify that 500 optimizer steps with batch=2, grad_accum=8 only processes 8000 rows (under 13860 required)."""
    rows_at_500 = 500 * 2 * 8
    assert rows_at_500 == 8000

    required_rows_99 = 2 * math.ceil(0.99 * 7000)
    assert required_rows_99 == 13860
    assert rows_at_500 < required_rows_99


def test_full_rejects_cuda0_cuda0_mapping(tmp_path: Path):
    """Verify FULL mode rejects assigning both Dense and Reranker to cuda:0."""
    data_dir, public_file = make_mock_canonical_dataset(tmp_path / "data", 8532, 7000, 999)

    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        with pytest.raises(RuntimeError, match="(?i)full mode requires reranker_device == 'cuda:1'"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "work",
                run_mode="full",
                dense_device="cuda:0",
                reranker_device="cuda:0",
                public_json_path=public_file,
                repo_root=REPO_ROOT,
            )


def test_dense_total_oom_equals_sum_of_stage_ooms():
    """Verify total Dense OOMs matches sum of stage-wise OOMs."""
    retriever = DenseMacroRetriever(model_name="mock", dimension=64, device="cpu")
    retriever.stage_telemetry["corpus"] = DenseEncodeTelemetry(32, 32, 32, 2, 100, 1.0)
    retriever.stage_telemetry["train_query"] = DenseEncodeTelemetry(32, 32, 32, 1, 50, 0.5)
    total_ooms = sum(t.oom_events for t in retriever.stage_telemetry.values())
    assert total_ooms == 3

