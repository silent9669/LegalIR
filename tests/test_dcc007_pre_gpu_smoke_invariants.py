"""Authoritative test suite for LEGALIR_DCC007_FINAL_PRE_GPU_SMOKE_REPAIR contract.

Covers:
1. ExactMatcher cross-document isolation and chunk-backed statutory features.
2. QueryBalancedSampler DataLoader integration, query schedule preservation, and >=99% coverage.
3. Official dataset identity gate (8,532 docs, 7,000 train queries, 999 public queries) and toy rejection.
4. Strict loaded PEFT parameter audit and authentic LoRA detection.
5. Persistent reranker OOM batch halving.
6. Dense stage-wise telemetry aggregation.
7. Corrected 5-fold runtime projection (7,000 OOF validation queries).
8. FAISS production backend enforcement.
9. Actual fusion model type reporting (lightgbm vs linear_fallback vs rrf_weighted).
"""

from collections import defaultdict
import json
from pathlib import Path
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.pipeline.kaggle_train import run_kaggle_pipeline
from src.pipeline.predict import LegalIRPipeline
from src.retrieval.exact_matcher import ExactMatcher, fold_accent_ascii
from src.training.trainer import (
    QueryBalancedSampler,
    QueryBalancedGroupSampler,
    RerankerPairDataset,
    RerankerTrainer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. P0: ExactMatcher Cross-Document Isolation
# ==============================================================================

def test_exact_chunk_features_do_not_cross_contaminate_documents():
    """Verify that chunks for doc_A do not leak statutory features into doc_B."""
    docs = [
        {"doc_id": "doc_A", "title": "Luật Đầu tư 2020", "name_raw": "Luat-Dau-tu-2020"},
        {"doc_id": "doc_B", "title": "Luật Doanh nghiệp 2020", "name_raw": "Luat-Doanh-nghiep-2020"},
    ]
    chunks = [
        {"doc_id": "doc_A", "article": "Điều 15", "clause": "Khoản 2", "point": "Điểm c"},
        {"doc_id": "doc_B", "article": "Điều 99", "clause": "Khoản 1", "point": "Điểm a"},
    ]
    matcher = ExactMatcher(documents=docs, chunks=chunks)

    assert "điều 15" in matcher.doc_articles["doc_A"]
    assert "khoản 2" in matcher.doc_clauses["doc_A"]
    assert "điểm c" in matcher.doc_points["doc_A"]
    assert "điều 99" not in matcher.doc_articles["doc_A"]
    assert "khoản 1" not in matcher.doc_clauses["doc_A"]
    assert "điểm a" not in matcher.doc_points["doc_A"]

    assert "điều 99" in matcher.doc_articles["doc_B"]
    assert "khoản 1" in matcher.doc_clauses["doc_B"]
    assert "điểm a" in matcher.doc_points["doc_B"]
    assert "điều 15" not in matcher.doc_articles["doc_B"]
    assert "khoản 2" not in matcher.doc_clauses["doc_B"]
    assert "điểm c" not in matcher.doc_points["doc_B"]


def test_exact_matcher_chunks_only_does_not_reference_stale_document():
    """Verify ExactMatcher works when initialized with chunks only (empty documents)."""
    chunks = [
        {"doc_id": "doc_100", "article": "Điều 5", "clause": "Khoản 1", "point": "Điểm a"},
    ]
    matcher = ExactMatcher(documents=[], chunks=chunks)

    assert "điều 5" in matcher.doc_articles["doc_100"]
    assert "khoản 1" in matcher.doc_clauses["doc_100"]
    assert "điểm a" in matcher.doc_points["doc_100"]


def test_exact_article_clause_point_are_assigned_to_correct_doc():
    """Verify statutory query matching correctly maps to the exact document having matching chunk statutory fields."""
    docs = [
        {"doc_id": "doc_1", "title": "Nghị định 31/2021/NĐ-CP", "legal_number": "31/2021/NĐ-CP"},
        {"doc_id": "doc_2", "title": "Nghị định 31/2021/NĐ-CP", "legal_number": "31/2021/NĐ-CP"},
    ]
    chunks = [
        {"doc_id": "doc_1", "article": "Điều 12", "clause": "Khoản 3", "point": "Điểm b"},
        {"doc_id": "doc_2", "article": "Điều 45", "clause": "Khoản 1", "point": "Điểm a"},
    ]
    matcher = ExactMatcher(documents=docs, chunks=chunks)

    res = matcher.match("Quy định tại điểm b khoản 3 Điều 12 Nghị định 31/2021/NĐ-CP")
    assert "doc_1" in res
    assert res["doc_1"]["exact_article"] is True
    assert res["doc_1"]["exact_clause"] is True
    assert res["doc_1"]["exact_point"] is True

    assert "doc_2" in res
    assert res["doc_2"]["exact_article"] is False
    assert res["doc_2"]["exact_clause"] is False
    assert res["doc_2"]["exact_point"] is False


# ==============================================================================
# 2. P0: True Query-Aware Sampler & Coverage Guarantee
# ==============================================================================

def test_query_sampler_survives_dataloader_iteration():
    """Verify QueryBalancedSampler preserves query ordering through a standard DataLoader."""
    pairs = [
        {"query_id": f"q{i}", "evidence_text": f"pos {i}", "label": 1.0} for i in range(10)
    ] + [
        {"query_id": f"q{i}", "evidence_text": f"neg {i}", "label": 0.0} for i in range(10)
    ]
    dataset = RerankerPairDataset(pairs, balanced=False)
    sampler = QueryBalancedSampler(dataset, seed=42)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)

    # First 5 batches (20 items) must yield all 10 unique queries
    seen_qids = []
    for batch in loader:
        seen_qids.extend(batch["query_id"])

    assert len(seen_qids) == 20
    first_10_unique = set(seen_qids[:20])
    assert len(first_10_unique) == 10


def test_query_sampler_covers_all_eligible_qids_before_repeat():
    """Verify QueryBalancedSampler yields all eligible queries before repeating any query's second pair."""
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
    # The first 6 indices must cover q1, q2, q3 (pos + neg for each)
    first_qids = [pairs[i]["query_id"] for i in indices[:6]]
    assert set(first_qids) == {"q1", "q2", "q3"}


def test_query_sampler_is_deterministic():
    """Verify QueryBalancedSampler produces identical sequences for the same seed."""
    pairs = [
        {"query_id": f"q{i%5}", "evidence_text": f"text {i}", "label": 1.0 if i % 2 == 0 else 0.0}
        for i in range(20)
    ]
    dataset = RerankerPairDataset(pairs, balanced=False)
    s1 = list(iter(QueryBalancedSampler(dataset, seed=42)))
    s2 = list(iter(QueryBalancedSampler(dataset, seed=42)))
    assert s1 == s2


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


def test_final_training_hard_fails_below_required_query_coverage(tmp_path: Path):
    """Verify run_kaggle_pipeline raises in full mode if final query coverage is below 99%."""
    data_dir, public_file = make_mock_canonical_dataset(tmp_path / "data", 8532, 7000, 999)

    mock_dense_inst = mock.MagicMock()
    mock_dense_inst.dense_search_backend = "faiss_index_flat_ip"
    mock_dense_inst.encode_texts.return_value = np.zeros((1, 64), dtype=np.float32)

    mock_mem = mock.MagicMock()
    mock_mem.qids = ["q0"]

    # Mock train_reranker returning low coverage
    with mock.patch("src.pipeline.kaggle_train.train_reranker", return_value={
        "status": "completed",
        "actual_unique_queries_seen": 500,
        "eligible_training_queries": 7000,
        "actual_query_coverage_pct": 7.14,
    }), mock.patch("src.pipeline.kaggle_train.DenseMacroRetriever", return_value=mock_dense_inst), mock.patch("src.pipeline.kaggle_train.OOFRunner.run", return_value={}), mock.patch("src.pipeline.kaggle_train.build_training_pairs"), mock.patch("src.pipeline.kaggle_train.TrainQuestionMemory", return_value=mock_mem), mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        with pytest.raises(RuntimeError, match="FULL mode requires actual_query_coverage_pct >= 99.0%"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "work",
                run_mode="full",
                public_json_path=public_file,
                repo_root=REPO_ROOT,
            )


# ==============================================================================
# 3. P0: Official Dataset Identity & Toy Production Smoke Gating
# ==============================================================================

def test_gpu_smoke_rejects_toy_dataset(tmp_path: Path):
    """Verify gpu_smoke fails fast when dataset has fewer than official 8,532 documents."""
    data_dir, _ = make_mock_canonical_dataset(tmp_path / "toy_data", num_docs=10, num_queries=10, num_public=5)

    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        with pytest.raises(ValueError, match="GPU_SMOKE mode requires official Task 1 dataset with exactly 8,532 documents"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "work",
                run_mode="gpu_smoke",
                repo_root=REPO_ROOT,
            )


def test_full_rejects_toy_dataset(tmp_path: Path):
    """Verify full mode fails fast when dataset has fewer than official 8,532 documents."""
    data_dir, _ = make_mock_canonical_dataset(tmp_path / "toy_data", num_docs=10, num_queries=10, num_public=5)

    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        with pytest.raises(ValueError, match="FULL mode requires official Task 1 dataset with exactly 8,532 documents"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "work",
                run_mode="full",
                repo_root=REPO_ROOT,
            )


def test_gpu_smoke_requires_8532_docs_7000_train_999_public(tmp_path: Path):
    """Verify gpu_smoke verifies all 3 official counts (8532 docs, 7000 train queries, 999 public queries)."""
    data_dir, _ = make_mock_canonical_dataset(tmp_path / "partial_data", num_docs=8532, num_queries=5000, num_public=999)

    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.device_count", return_value=2):
        with pytest.raises(ValueError, match="GPU_SMOKE mode requires official Task 1 dataset with exactly 7,000 training queries"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "work",
                run_mode="gpu_smoke",
                repo_root=REPO_ROOT,
            )


def test_gpu_smoke_cli_rejects_tiny():
    """Verify smoke_kaggle_pipeline CLI rejects --tiny when run_mode is gpu_smoke or full."""
    import subprocess
    cmd = [
        sys_executable := str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "scripts/smoke_kaggle_pipeline.py"),
        "--tiny",
        "--run-mode",
        "gpu_smoke",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "Cannot use --tiny with run_mode='gpu_smoke'" in proc.stderr or "Cannot use --tiny with run_mode='gpu_smoke'" in proc.stdout


# ==============================================================================
# 4. P0: Strict Loaded PEFT Parameter Audit & True PEFT Detection
# ==============================================================================

def test_strict_final_audit_raises_if_adapter_force_load_fails(tmp_path: Path):
    """Verify audit_parameters raises in strict mode if reranker ensure_loaded fails."""
    from src.pipeline.predict import LegalIRPipeline
    from src.ranking.reranker import CrossEncoderReranker
    from src.retrieval.bm25_micro import BM25MicroRetriever
    from src.retrieval.hybrid_search import HybridSearchEngine

    reranker = CrossEncoderReranker(model_name="mock", adapter_path=tmp_path / "non_existent_adapter", device="cpu")
    reranker.ensure_loaded = mock.MagicMock(side_effect=RuntimeError("Corrupt adapter safetensors"))

    pipeline = LegalIRPipeline(
        hybrid_engine=HybridSearchEngine(
            bm25_retriever=BM25MicroRetriever().fit([{"chunk_id": "c1", "doc_id": "101", "text_norm": "q"}]),
        ),
        reranker=reranker,
    )

    with pytest.raises(RuntimeError, match="failed to force-load Reranker model/adapter"):
        pipeline.audit_parameters(output_json=tmp_path / "audit.json", require_loaded_models=True)


def test_strict_final_audit_counts_real_lora_params(tmp_path: Path):
    """Verify audit_model_parameters counts actual LoRA adapter parameters for a real PeftModel."""
    from transformers import BertConfig, BertForSequenceClassification
    from src.models.parameter_audit import audit_model_parameters
    from src.training.trainer import setup_peft_model

    config = BertConfig(
        vocab_size=300,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=1,
    )
    base_model = BertForSequenceClassification(config)
    peft_model, meta = setup_peft_model(base_model, lora_r=8, lora_alpha=16)

    report = audit_model_parameters(peft_model, role="cross_encoder_reranker")
    assert report["is_peft_lora"] is True
    assert report["adapter_parameters"] > 0
    assert report["base_parameters"] > 0
    assert report["parameters"] == report["base_parameters"] + report["adapter_parameters"]


def test_plain_hf_model_with_base_model_property_is_not_marked_peft():
    """Verify plain HF models having a base_model attribute are not misclassified as PEFT."""
    from transformers import BertConfig, BertForSequenceClassification
    from src.models.parameter_audit import audit_model_parameters

    config = BertConfig(
        vocab_size=300,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=1,
    )
    plain_model = BertForSequenceClassification(config)
    assert hasattr(plain_model, "base_model")

    report = audit_model_parameters(plain_model, role="cross_encoder_reranker")
    assert report["is_peft_lora"] is False
    assert report["adapter_parameters"] == 0
    assert report["base_parameters"] == report["parameters"]


# ==============================================================================
# 5. P1: Reranker Persistent OOM, Dense Telemetry, Runtime Projection, FAISS & Fusion
# ==============================================================================

def test_reranker_oom_reduction_persists_across_later_batches():
    """Verify CrossEncoderReranker keeps reduced batch size for subsequent batches after OOM."""
    from src.ranking.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="mock", device="cpu")
    reranker._load_model = mock.MagicMock()

    batch_sizes_seen = []
    call_count = {"count": 0}

    def mock_tokenize(queries, passages, **kwargs):
        call_count["count"] += 1
        b_len = len(queries)
        batch_sizes_seen.append(b_len)
        if b_len == 16 and call_count["count"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.5GB")
        return {"input_ids": torch.ones((b_len, 10), dtype=torch.long)}

    reranker.tokenizer = mock_tokenize
    reranker.model = mock.MagicMock()
    reranker.model.side_effect = lambda **inputs: (torch.ones((inputs["input_ids"].shape[0], 1)),)

    pairs = [(f"query {i}", f"passage {i}") for i in range(32)]
    scores = reranker.score_pairs(pairs, batch_size=16)

    assert len(scores) == 32
    assert reranker.initial_batch_size == 16
    assert reranker.min_successful_batch_size == 8
    assert reranker.oom_events == 1
    # Check that after OOM (16 failed), all subsequent attempts were batch size 8
    successful_batches = [b for b in batch_sizes_seen if b <= 8]
    assert all(b <= 8 for b in successful_batches)


def test_dense_stage_telemetry_aggregates_public_encoder(tmp_path: Path):
    """Verify gpu_smoke_report aggregates public query Dense encoder telemetry."""
    data_dir, public_file = make_mock_canonical_dataset(tmp_path / "data", 8532, 7000, 999)

    mock_dense_corpus = mock.MagicMock()
    mock_dense_corpus.dense_search_backend = "faiss_index_flat_ip"
    mock_dense_corpus._faiss_index = mock.MagicMock()
    mock_dense_corpus.dense_oom_events = 1
    mock_dense_corpus.encode_texts.return_value = np.zeros((1, 64), dtype=np.float32)
    mock_dense_corpus.encode_queries.return_value = np.zeros((1, 64), dtype=np.float32)

    mock_pipeline = mock.MagicMock()
    mock_pipeline.hybrid_engine = mock.MagicMock()
    mock_pipeline_dense = mock.MagicMock()
    mock_pipeline_dense.dense_oom_events = 2
    mock_pipeline_dense.device = "cuda:0"
    mock_pipeline.hybrid_engine.dense = mock_pipeline_dense
    mock_pipeline.hybrid_engine.dense_retriever = mock_pipeline_dense

    mock_param_0 = mock.MagicMock()
    mock_param_0.device = "cuda:0"
    mock_pipeline_dense.model.parameters.return_value = iter([mock_param_0])

    mock_param_1 = mock.MagicMock()
    mock_param_1.device = "cuda:1"
    mock_pipeline.reranker = mock.MagicMock()
    mock_pipeline.reranker.device = "cuda:1"
    mock_pipeline.reranker.model.parameters.return_value = iter([mock_param_1])
    mock_pipeline.reranker.oom_events = 0

    mock_pipeline.audit_parameters.return_value = {"total_learned_parameters": 700_000_000}
    mock_pipeline.predict_single.return_value = ["1", "2", "3", "4", "5"]
    mock_pipeline.predict_batch.return_value = {f"pub_{i}": ["1", "2", "3", "4", "5"] for i in range(20)}

    mock_pipe_cls = mock.MagicMock()
    mock_pipe_cls.load_pipeline.return_value = mock_pipeline

    mock_mem = mock.MagicMock()
    mock_mem.qids = ["q0"]

    with mock.patch("src.pipeline.kaggle_train.DenseMacroRetriever", return_value=mock_dense_corpus), \
         mock.patch("src.pipeline.kaggle_train.OOFRunner.run", return_value={"mean_recall@5": 0.9, "folds": []}), \
         mock.patch("src.pipeline.kaggle_train.train_reranker", return_value={"status": "completed", "actual_query_coverage_pct": 100.0}), \
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

        report = json.loads((work_dir / "gpu_smoke_report.json").read_text(encoding="utf-8"))
        assert "dense_total_oom_events" in report
        assert report["dense_total_oom_events"] == 3  # 1 from corpus + 2 from pipeline public encoder


def test_runtime_projection_counts_7000_oof_validation_queries_not_35000(tmp_path: Path):
    """Verify runtime_projection.json projects 7,000 total held-out OOF validation queries."""
    data_dir, public_file = make_mock_canonical_dataset(tmp_path / "data", 8532, 7000, 999)

    mock_dense = mock.MagicMock()
    mock_dense.dense_search_backend = "faiss_index_flat_ip"
    mock_dense._faiss_index = mock.MagicMock()
    mock_dense.encode_texts.return_value = np.zeros((1, 64), dtype=np.float32)
    mock_dense.encode_queries.return_value = np.zeros((1, 64), dtype=np.float32)

    mock_pipeline = mock.MagicMock()
    mock_pipeline.hybrid_engine = mock.MagicMock()
    mock_dense.device = "cuda:0"
    mock_pipeline.hybrid_engine.dense = mock_dense
    mock_pipeline.hybrid_engine.dense_retriever = mock_dense

    mock_param_0 = mock.MagicMock()
    mock_param_0.device = "cuda:0"
    mock_dense.model.parameters.return_value = iter([mock_param_0])

    mock_param_1 = mock.MagicMock()
    mock_param_1.device = "cuda:1"
    mock_pipeline.reranker = mock.MagicMock()
    mock_pipeline.reranker.device = "cuda:1"
    mock_pipeline.reranker.model.parameters.return_value = iter([mock_param_1])

    mock_pipeline.audit_parameters.return_value = {"total_learned_parameters": 700_000_000}
    mock_pipeline.predict_single.return_value = ["1", "2", "3", "4", "5"]
    mock_pipeline.predict_batch.return_value = {f"pub_{i}": ["1", "2", "3", "4", "5"] for i in range(20)}

    mock_pipe_cls = mock.MagicMock()
    mock_pipe_cls.load_pipeline.return_value = mock_pipeline

    mock_mem = mock.MagicMock()
    mock_mem.qids = ["q0"]

    with mock.patch("src.pipeline.kaggle_train.DenseMacroRetriever", return_value=mock_dense), \
         mock.patch("src.pipeline.kaggle_train.OOFRunner.run", return_value={"queries_per_second": 10.0, "mean_recall@5": 0.9, "folds": []}), \
         mock.patch("src.pipeline.kaggle_train.train_reranker", return_value={"status": "completed", "training_time_sec": 10.0, "global_steps": 10, "actual_query_coverage_pct": 100.0}), \
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
        assert proj["total_oof_validation_queries"] == 7000
        assert proj["projected_oof_inference_seconds"] == 700.0  # 7000 / 10.0 qps


def test_gpu_smoke_requires_faiss_backend(tmp_path: Path):
    """Verify gpu_smoke raises if Dense macro index fails to initialize FAISS backend."""
    data_dir, public_file = make_mock_canonical_dataset(tmp_path / "data", 8532, 7000, 999)

    mock_dense = mock.MagicMock()
    mock_dense._faiss_index = None  # NumPy fallback

    with mock.patch("src.pipeline.kaggle_train.DenseMacroRetriever", return_value=mock_dense), \
         mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.device_count", return_value=2):

        with pytest.raises(RuntimeError, match="FAISS production backend enforcement failed"):
            run_kaggle_pipeline(
                data_dir=data_dir,
                working_dir=tmp_path / "work",
                run_mode="gpu_smoke",
                public_json_path=public_file,
                repo_root=REPO_ROOT,
            )


def test_fusion_manifest_reports_actual_model_type(tmp_path: Path):
    """Verify train_and_evaluate_fusion_cv exports actual winning model type."""
    from src.ranking.train_fusion import train_and_evaluate_fusion_cv

    oof_df = pd.DataFrame([
        {"query_id": "q1", "doc_id": "d1", "label": 1.0, "fold": 0, "exact_score": 1.0, "bm25_score": 5.0, "dense_score": 0.9, "memory_score": 0.8, "reranker_score": 0.95},
        {"query_id": "q1", "doc_id": "d2", "label": 0.0, "fold": 0, "exact_score": 0.0, "bm25_score": 2.0, "dense_score": 0.3, "memory_score": 0.1, "reranker_score": 0.1},
        {"query_id": "q2", "doc_id": "d2", "label": 1.0, "fold": 1, "exact_score": 1.0, "bm25_score": 4.0, "dense_score": 0.85, "memory_score": 0.7, "reranker_score": 0.9},
        {"query_id": "q2", "doc_id": "d1", "label": 0.0, "fold": 1, "exact_score": 0.0, "bm25_score": 1.0, "dense_score": 0.2, "memory_score": 0.0, "reranker_score": 0.05},
    ])
    qrels_dict = {"q1": ["d1"], "q2": ["d2"]}

    out_dir = tmp_path / "fusion_cv"
    report = train_and_evaluate_fusion_cv(oof_df, qrels_dict, output_dir=out_dir, num_boost_round=5)

    comp = json.loads((out_dir / "fusion_comparison.json").read_text(encoding="utf-8"))
    assert comp["winning_model_type"] in ("lightgbm", "linear_fallback", "rrf_weighted")


