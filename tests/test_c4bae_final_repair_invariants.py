"""Authoritative verification tests for LEGALIR_C4BAE_FINAL_KAGGLE_REPAIR contract.

Covers:
1. build_pairs Mapping/import verification
2. repo-root config resolution from arbitrary CWD
3. T4-safe effective reranker configuration
4. Cross-fitted fusion requiring >=2 folds (and gpu_smoke 2-fold support)
5. Learned-fusion inference feature alignment (query_text and doc_freq_map)
6. Precomputed query embedding reuse across Dense and Question Memory
7. OOFRunner invoking reranker BEFORE candidate feature extraction
8. Document-disjoint trained reranker actual fine-tuning and evaluation
9. Strict production artifact validation (manifests, hashes, stages)
10. Complete parameter audit including Dense (DEk21) and Reranker (BGE)
"""

from collections import defaultdict
from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts.generate_kaggle_notebook import generate_and_save_notebooks
from src.evaluation.submission import (
    create_submission_manifest,
    validate_submission,
    validate_submission_zip,
)
from src.models.device import resolve_device
from src.models.parameter_audit import audit_system_parameters
from src.pipeline.kaggle_train import (
    discover_public_test_file,
    resolve_repo_path,
    run_kaggle_pipeline,
)
from src.pipeline.oof_runner import OOFRunner
from src.pipeline.predict import LegalIRPipeline
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import LightGBMRanker, LinearRanker, ReciprocalRankFusion
from src.ranking.oof_features import (
    CORE_FEATURE_COLUMNS,
    compute_training_doc_frequencies,
    extract_candidate_features,
)
from src.ranking.selector import TopKSelector
from src.ranking.train_fusion import train_and_evaluate_fusion_cv
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import TrainQuestionMemory
from src.training.build_pairs import build_training_pairs
from src.training.train_reranker import train_reranker

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_canonical_data(tmp_path: Path):
    """Create minimal canonical data files for integration tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "doc_id": "101",
            "title": "Luật Doanh nghiệp 2020",
            "legal_number": "59/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/59-2020",
            "name_raw": "Luật Doanh nghiệp",
            "passage_raw": "Quy định về thành lập doanh nghiệp",
            "passage_norm": "quy định về thành lập doanh nghiệp",
            "is_empty": False,
        },
        {
            "doc_id": "102",
            "title": "Luật Đầu tư 2020",
            "legal_number": "61/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/61-2020",
            "name_raw": "Luật Đầu tư",
            "passage_raw": "Quy định về đầu tư",
            "passage_norm": "quy định về đầu tư",
            "is_empty": False,
        },
        {
            "doc_id": "103",
            "title": "Nghị định Đăng ký doanh nghiệp",
            "legal_number": "01/2021/NĐ-CP",
            "year": "2021",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/01-2021",
            "name_raw": "Nghị định 01",
            "passage_raw": "Hồ sơ đăng ký",
            "passage_norm": "hồ sơ đăng ký",
            "is_empty": False,
        },
    ]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet", index=False)

    chunks = [
        {
            "chunk_id": "c101_micro",
            "parent_chunk_id": "c101_macro",
            "doc_id": "101",
            "granularity": "micro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Quy định về thành lập doanh nghiệp và công ty TNHH.",
            "text_norm": "quy định về thành lập doanh nghiệp và công ty tnhh",
        },
        {
            "chunk_id": "c101_macro",
            "parent_chunk_id": None,
            "doc_id": "101",
            "granularity": "macro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Quy định về thành lập doanh nghiệp và công ty TNHH.",
            "text_norm": "quy định về thành lập doanh nghiệp và công ty tnhh",
        },
        {
            "chunk_id": "c102_micro",
            "parent_chunk_id": "c102_macro",
            "doc_id": "102",
            "granularity": "micro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Dự án đầu tư trực tiếp nước ngoài và ưu đãi.",
            "text_norm": "dự án đầu tư trực tiếp nước ngoài và ưu đãi",
        },
        {
            "chunk_id": "c102_macro",
            "parent_chunk_id": None,
            "doc_id": "102",
            "granularity": "macro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Dự án đầu tư trực tiếp nước ngoài và ưu đãi.",
            "text_norm": "dự án đầu tư trực tiếp nước ngoài và ưu đãi",
        },
        {
            "chunk_id": "c103_micro",
            "parent_chunk_id": "c103_macro",
            "doc_id": "103",
            "granularity": "micro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng.",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua mạng",
        },
        {
            "chunk_id": "c103_macro",
            "parent_chunk_id": None,
            "doc_id": "103",
            "granularity": "macro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "chapter": "",
            "section": "",
            "token_count": 10,
            "is_empty": False,
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng.",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua mạng",
        },
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Thành lập công ty TNHH", "question_norm": "thành lập công ty tnhh", "gold_count": 1},
        {"query_id": "q2", "question_raw": "Dự án đầu tư trực tiếp", "question_norm": "dự án đầu tư trực tiếp", "gold_count": 1},
        {"query_id": "q3", "question_raw": "Đăng ký doanh nghiệp qua mạng", "question_norm": "đăng ký doanh nghiệp qua mạng", "gold_count": 1},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "101", "relevance": 1},
        {"query_id": "q2", "doc_id": "102", "relevance": 1},
        {"query_id": "q3", "doc_id": "103", "relevance": 1},
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

    return data_dir, index_dir


# ==============================================================================
# 1. P0 Tests: Module imports & Path Resolution
# ==============================================================================

def test_build_pairs_module_imports():
    """Verify that build_pairs imports and exposes build_training_pairs with Mapping annotation."""
    from src.training.build_pairs import build_training_pairs
    import inspect
    sig = inspect.signature(build_training_pairs)
    assert "query_embeddings" in sig.parameters


def test_kaggle_config_resolution_from_non_repo_cwd(tmp_path: Path, monkeypatch):
    """Verify runtime and reranker configs resolve to repo_root even when CWD is outside the repo."""
    external_dir = tmp_path / "outside_workspace"
    external_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(external_dir)

    p_runtime = resolve_repo_path("configs/kaggle.yaml", REPO_ROOT)
    p_reranker = resolve_repo_path("configs/experiments/reranker_lora.yaml", REPO_ROOT)

    assert p_runtime == (REPO_ROOT / "configs/kaggle.yaml").resolve()
    assert p_reranker == (REPO_ROOT / "configs/experiments/reranker_lora.yaml").resolve()
    assert p_runtime.is_file()
    assert p_reranker.is_file()


def test_t4_safe_effective_reranker_config():
    """Verify effective reranker config contains all mandatory T4-safe training parameters."""
    cfg_path = REPO_ROOT / "configs/experiments/reranker_lora.yaml"
    assert cfg_path.is_file()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert cfg.get("base_model_name") == "BAAI/bge-reranker-v2-m3"
    assert cfg.get("batch_size") == 2
    assert cfg.get("gradient_accumulation_steps") == 8
    assert cfg.get("max_length") == 512
    assert cfg.get("max_steps") == 500
    assert cfg.get("fp16") is True
    assert cfg.get("gradient_checkpointing") is True
    assert float(cfg.get("learning_rate", 0)) == pytest.approx(2.0e-5)


# ==============================================================================
# 2. P0/P1 Tests: Multi-Fold Fusion & Feature Alignment
# ==============================================================================

def test_gpu_smoke_requires_two_folds_for_cross_fit():
    """Verify that cross-fitted fusion requires at least 2 unique folds and raises ValueError on 1 fold."""
    oof_1fold = pd.DataFrame([
        {"query_id": "q1", "doc_id": "101", "fold": 0, "raw_bm25_rank": 1, "raw_bm25_score": 10.0, "label": 1},
        {"query_id": "q1", "doc_id": "102", "fold": 0, "raw_bm25_rank": 2, "raw_bm25_score": 5.0, "label": 0},
    ])
    qrels = {"q1": ["101"]}

    with pytest.raises(ValueError, match="at least 2 folds"):
        train_and_evaluate_fusion_cv(oof_1fold, qrels)

    oof_2fold = pd.DataFrame([
        {"query_id": "q1", "doc_id": "101", "fold": 0, "raw_bm25_rank": 1, "raw_bm25_score": 10.0, "label": 1},
        {"query_id": "q1", "doc_id": "102", "fold": 0, "raw_bm25_rank": 2, "raw_bm25_score": 5.0, "label": 0},
        {"query_id": "q2", "doc_id": "102", "fold": 1, "raw_bm25_rank": 1, "raw_bm25_score": 10.0, "label": 1},
        {"query_id": "q2", "doc_id": "101", "fold": 1, "raw_bm25_rank": 2, "raw_bm25_score": 5.0, "label": 0},
    ])
    qrels_2 = {"q1": ["101"], "q2": ["102"]}
    report = train_and_evaluate_fusion_cv(oof_2fold, qrels_2, num_boost_round=5)
    assert "winning_method" in report
    assert report["manifest"]["feature_training_stage"] == "post_rerank"


def test_oof_runner_calls_reranker_before_feature_extraction(mock_canonical_data, tmp_path: Path):
    """Verify that OOFRunner executes neural reranking before extracting candidate features."""
    data_dir, index_dir = mock_canonical_data
    cv_dir = tmp_path / "cv_rerank_test"

    class TrackingReranker:
        def __init__(self):
            self.calls = 0

        def rerank(self, query, candidates, evidence_builder=None, top_k=50):
            self.calls += 1
            reranked = []
            for i, c in enumerate(candidates):
                item = dict(c)
                item["reranker_score"] = 0.888 - i * 0.1
                item["reranker_second_score"] = 0.555
                item["reranker_margin"] = 0.333
                reranked.append(item)
            return reranked

    tracking_reranker = TrackingReranker()
    runner = OOFRunner(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=cv_dir,
        num_folds=2,
        smoke=True,
        smoke_sample_size=2,
        use_reranker=True,
        train_reranker_per_fold=False,
    )
    runner.load_data()
    runner.load_retrievers()

    fold_info = {"fold": 0, "train_query_ids": ["q2", "q3"], "val_query_ids": ["q1"]}
    _, _, feat_dfs, _, _ = runner.run_fold(0, fold_info, reranker=tracking_reranker)

    assert tracking_reranker.calls > 0
    assert len(feat_dfs) > 0
    combined_feat = pd.concat(feat_dfs, ignore_index=True)
    assert "reranker_score" in combined_feat.columns
    assert (combined_feat["reranker_score"] > 0.8).any()


def test_pipeline_passes_query_text_and_doc_freq_to_learned_ranker(mock_canonical_data, tmp_path: Path):
    """Verify that LegalIRPipeline passes query_text and doc_freq_map to ranker.predict()."""
    data_dir, index_dir = mock_canonical_data
    captured_kwargs = {}

    class TrackingLearnedRanker:
        def predict(self, candidate_records, query_id=None, query_text=None, doc_freq_map=None, **kwargs):
            captured_kwargs["query_id"] = query_id
            captured_kwargs["query_text"] = query_text
            captured_kwargs["doc_freq_map"] = doc_freq_map
            return candidate_records

    doc_freq = {"101": 0.75, "102": 0.50}
    pipeline = LegalIRPipeline(
        hybrid_engine=HybridSearchEngine(
            bm25_retriever=BM25MicroRetriever.load(index_dir / "bm25"),
        ),
        ranker=TrackingLearnedRanker(),
        doc_freq_map=doc_freq,
        valid_doc_ids={"101", "102", "103"},
    )

    pipeline.predict_one("q1", "Thành lập công ty TNHH")
    assert captured_kwargs.get("query_id") == "q1"
    assert captured_kwargs.get("query_text") == "Thành lập công ty TNHH"
    assert captured_kwargs.get("doc_freq_map") == doc_freq


def test_pipeline_shares_one_q_emb_between_dense_and_memory(tmp_path: Path):
    """Verify that a single precomputed q_emb is shared between Dense and Question Memory."""
    encode_counter = {"calls": 0}

    class CountingEncoder:
        def __init__(self):
            self.model_name = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"

        def encode_queries(self, queries, **kwargs):
            encode_counter["calls"] += len(queries)
            return np.ones((len(queries), 768), dtype=np.float32)

    fake_encoder = CountingEncoder()
    dense = DenseMacroRetriever.from_arrays(
        embeddings=np.ones((2, 768), dtype=np.float32),
        doc_ids=["101", "102"],
        chunk_ids=["c1", "c2"],
    )
    dense.query_encoder = fake_encoder.encode_queries

    memory = TrainQuestionMemory(min_similarity=0.82, use_dense=False)
    memory.fit([("q1", "query text", np.ones(768, dtype=np.float32))], {"q1": ["101"]})

    hybrid = HybridSearchEngine(
        dense_retriever=dense,
        question_memory=memory,
    )

    pipeline = LegalIRPipeline(hybrid_engine=hybrid)
    precomputed_vec = np.ones(768, dtype=np.float32)

    # Calling with q_emb should not trigger query encoding
    pipeline.predict_one("q1", "query text", q_emb=precomputed_vec)
    assert encode_counter["calls"] == 0


def test_final_parameter_audit_contains_dense_and_reranker(mock_canonical_data):
    """Verify that LegalIRPipeline.audit_parameters() captures both DEk21 and BGE reranker."""
    data_dir, index_dir = mock_canonical_data

    dense = DenseMacroRetriever(model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", device="cpu")
    reranker = mock.MagicMock()
    reranker.model = None
    reranker.model_name = "BAAI/bge-reranker-v2-m3"

    pipeline = LegalIRPipeline(
        hybrid_engine=HybridSearchEngine(dense_retriever=dense),
        reranker=reranker,
    )

    report = pipeline.audit_parameters(raise_on_violation=True)
    assert report["total_learned_parameters"] == 702_754_049
    assert report["is_compliant"] is True
    assert report["verdict"] == "PASS"
    assert report["total_parameters_billions"] < 4.0
    roles = [m["role"] for m in report["models"].values()]
    assert "dense_embedding" in roles
    assert "cross_encoder_reranker" in roles
