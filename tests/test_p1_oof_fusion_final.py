import json
import os
import importlib
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from src.pipeline.oof_runner import OOFRunner
from src.pipeline.predict import LegalIRPipeline
from src.ranking.fusion import LightGBMRanker, ReciprocalRankFusion
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.train_fusion import train_and_evaluate_fusion_cv
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.build_indexes import build_bm25_index, build_bm25_pyvi_index
from src.training.build_pairs import build_training_pairs

train_final_module = importlib.import_module("scripts.06_train_final")
train_final_system = train_final_module.train_final_system


@pytest.fixture
def tiny_system_data(tmp_path: Path):
    """Create a self-contained multi-query canonical dataset with BM25 & PyVi indexes."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {"doc_id": "101", "title": "Luật Doanh nghiệp 2020", "legal_number": "59/2020/QH14", "year": "2020", "doc_type": "Luật", "link": "https://tvpl.vn/59-2020", "name_raw": "Luật Doanh nghiệp"},
        {"doc_id": "102", "title": "Luật Đầu tư 2020", "legal_number": "61/2020/QH14", "year": "2020", "doc_type": "Luật", "link": "https://tvpl.vn/61-2020", "name_raw": "Luật Đầu tư"},
        {"doc_id": "103", "title": "Nghị định Đăng ký kinh doanh", "legal_number": "01/2021/NĐ-CP", "year": "2021", "doc_type": "Nghị định", "link": "https://tvpl.vn/01-2021", "name_raw": "Nghị định 01"},
        {"doc_id": "104", "title": "Nghị định Đầu tư nước ngoài", "legal_number": "31/2021/NĐ-CP", "year": "2021", "doc_type": "Nghị định", "link": "https://tvpl.vn/31-2021", "name_raw": "Nghị định 31"},
        {"doc_id": "105", "title": "Luật Thương mại", "legal_number": "36/2005/QH11", "year": "2005", "doc_type": "Luật", "link": "https://tvpl.vn/36-2005", "name_raw": "Luật Thương mại"},
    ]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet", index=False)

    chunks = [
        {"chunk_id": "c101_micro", "doc_id": "101", "granularity": "micro", "article": "Điều 1", "clause": "Khoản 1", "point": "", "text_raw": "Quy định về thành lập doanh nghiệp", "text_norm": "quy định về thành lập doanh nghiệp"},
        {"chunk_id": "c101_macro", "doc_id": "101", "granularity": "macro", "article": "Điều 1", "clause": "Khoản 1", "point": "", "text_raw": "Quy định về thành lập doanh nghiệp và quản lý công ty", "text_norm": "quy định về thành lập doanh nghiệp và quản lý công ty"},
        {"chunk_id": "c102_micro", "doc_id": "102", "granularity": "micro", "article": "Điều 2", "clause": "Khoản 1", "point": "", "text_raw": "Quy định về dự án đầu tư trực tiếp", "text_norm": "quy định về dự án đầu tư trực tiếp"},
        {"chunk_id": "c102_macro", "doc_id": "102", "granularity": "macro", "article": "Điều 2", "clause": "Khoản 1", "point": "", "text_raw": "Quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư", "text_norm": "quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư"},
        {"chunk_id": "c103_micro", "doc_id": "103", "granularity": "micro", "article": "Điều 3", "clause": "Khoản 1", "point": "", "text_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng", "text_norm": "hồ sơ đăng ký doanh nghiệp qua mạng"},
        {"chunk_id": "c103_macro", "doc_id": "103", "granularity": "macro", "article": "Điều 3", "clause": "Khoản 1", "point": "", "text_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng điện tử", "text_norm": "hồ sơ đăng ký doanh nghiệp qua mạng điện tử"},
        {"chunk_id": "c104_micro", "doc_id": "104", "granularity": "micro", "article": "Điều 4", "clause": "Khoản 1", "point": "", "text_raw": "Thủ tục đầu tư cho nhà đầu tư nước ngoài", "text_norm": "thủ tục đầu tư cho nhà đầu tư nước ngoài"},
        {"chunk_id": "c104_macro", "doc_id": "104", "granularity": "macro", "article": "Điều 4", "clause": "Khoản 1", "point": "", "text_raw": "Thủ tục đầu tư cho nhà đầu tư nước ngoài tại Việt Nam", "text_norm": "thủ tục đầu tư cho nhà đầu tư nước ngoài tại việt nam"},
        {"chunk_id": "c105_micro", "doc_id": "105", "granularity": "micro", "article": "Điều 5", "clause": "Khoản 1", "point": "", "text_raw": "Hoạt động mua bán hàng hóa quốc tế", "text_norm": "hoạt động mua bán hàng hóa quốc tế"},
        {"chunk_id": "c105_macro", "doc_id": "105", "granularity": "macro", "article": "Điều 5", "clause": "Khoản 1", "point": "", "text_raw": "Hoạt động mua bán hàng hóa quốc tế và cung ứng dịch vụ", "text_norm": "hoạt động mua bán hàng hóa quốc tế và cung ứng dịch vụ"},
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Thành lập doanh nghiệp như thế nào?", "question_norm": "thành lập doanh nghiệp như thế nào"},
        {"query_id": "q2", "question_raw": "Dự án đầu tư trực tiếp", "question_norm": "dự án đầu tư trực tiếp"},
        {"query_id": "q3", "question_raw": "Đăng ký kinh doanh qua mạng", "question_norm": "đăng ký kinh doanh qua mạng"},
        {"query_id": "q4", "question_raw": "Đầu tư nước ngoài tại Việt Nam", "question_norm": "đầu tư nước ngoài tại việt nam"},
        {"query_id": "q5", "question_raw": "Mua bán hàng hóa quốc tế", "question_norm": "mua bán hàng hóa quốc tế"},
        {"query_id": "q6", "question_raw": "Hồ sơ thành lập công ty", "question_norm": "hồ sơ thành lập công ty"},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "101"},
        {"query_id": "q2", "doc_id": "102"},
        {"query_id": "q3", "doc_id": "103"},
        {"query_id": "q4", "doc_id": "104"},
        {"query_id": "q5", "doc_id": "105"},
        {"query_id": "q6", "doc_id": "101"},
    ]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet", index=False)

    split_info = [
        {"fold": 0, "train_query_ids": ["q4", "q5", "q6"], "val_query_ids": ["q1", "q2", "q3"]},
        {"fold": 1, "train_query_ids": ["q1", "q2", "q3"], "val_query_ids": ["q4", "q5", "q6"]},
    ]
    (splits_dir / "random_5fold.json").write_text(json.dumps(split_info), encoding="utf-8")

    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    build_bm25_index(data_dir=data_dir, output_dir=index_dir / "bm25")
    build_bm25_pyvi_index(data_dir=data_dir, output_dir=index_dir / "bm25_pyvi")

    return data_dir, index_dir


@pytest.fixture
def tiny_bert_model(tmp_path: Path):
    """Create a minimal BERT model and tokenizer for fast LoRA training tests."""
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
    model_dir = tmp_path / "tiny_bert"
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    return str(model_dir)


# ==============================================================================
# 1. P1.1: Fold-Specific Adapter Training & Evaluation During OOF
# ==============================================================================

def test_oof_train_reranker_per_fold(tiny_system_data, tiny_bert_model, tmp_path: Path):
    """Test that train_reranker_per_fold=True trains fold-isolated adapters and evaluates with them."""
    data_dir, index_dir = tiny_system_data
    cv_out = tmp_path / "cv_output"

    runner = OOFRunner(
        data_dir=data_dir,
        index_dir=index_dir,
        splits_path=data_dir / "splits" / "random_5fold.json",
        output_dir=cv_out,
        num_folds=2,
        candidate_k=10,
        rerank_k=5,
        use_reranker=True,
        reranker_model=tiny_bert_model,
        train_reranker_per_fold=True,
        smoke=False,
    )

    report = runner.run()

    assert report["total_evaluated_queries"] == 6
    assert len(report["folds"]) == 2

    # Verify fold-specific artifacts exist on disk
    for f_idx in range(2):
        fold_dir = cv_out / f"fold_{f_idx}"
        assert fold_dir.exists()

        # Pair files must exist
        pairs_file = fold_dir / "pairs" / "reranker_pairs.parquet"
        assert pairs_file.exists()
        pairs_df = pd.read_parquet(pairs_file)
        assert len(pairs_df) > 0

        # Assert no validation query IDs leaked into fold training pairs
        val_qids = {"q1", "q2", "q3"} if f_idx == 0 else {"q4", "q5", "q6"}
        assert set(pairs_df["query_id"]).isdisjoint(val_qids)

        # Adapter directory must exist with adapter_config.json
        adapter_dir = fold_dir / "reranker_adapter"
        assert adapter_dir.exists()
        assert (adapter_dir / "adapter_config.json").exists()

        # Fold metrics must have recorded adapter metadata
        f_metric = report["folds"][f_idx]
        assert f_metric["training_queries"] == 3
        assert f_metric["training_pairs"] == len(pairs_df)
        assert f_metric["adapter_path"] == str(adapter_dir)
        assert "adapter_checksum" in f_metric


# ==============================================================================
# 2. P1.2: PyVi BM25 Retriever Participation in OOF Search
# ==============================================================================

def test_oof_pyvi_retriever_participation(tiny_system_data, tmp_path: Path):
    """Test that OOFRunner loads PyVi BM25 and includes PyVi scores in candidate features."""
    data_dir, index_dir = tiny_system_data
    cv_out = tmp_path / "cv_pyvi_test"

    runner = OOFRunner(
        data_dir=data_dir,
        index_dir=index_dir,
        splits_path=data_dir / "splits" / "random_5fold.json",
        output_dir=cv_out,
        num_folds=2,
        candidate_k=10,
        use_reranker=False,
        smoke=False,
    )

    runner.load_data()
    runner.load_retrievers()

    # Verify PyVi retriever is loaded and non-None
    assert runner.bm25_pyvi is not None
    assert isinstance(runner.bm25_pyvi, BM25PyViRetriever)
    assert len(runner.bm25_pyvi.chunk_ids) > 0

    cv_report = runner.run()
    assert cv_report["total_evaluated_queries"] == 6

    # Verify OOF features contains pyvi_bm25 / bm25_pyvi feature columns
    feat_parquet = cv_out / "oof_features.parquet"
    assert feat_parquet.exists()
    feat_df = pd.read_parquet(feat_parquet)
    assert not feat_df.empty

    pyvi_cols = [c for c in feat_df.columns if "pyvi" in c]
    assert len(pyvi_cols) > 0, f"Expected PyVi feature columns, got: {feat_df.columns}"


# ==============================================================================
# 3. P1.3: All-Query Final Pair Mining & Final Training Manifest
# ==============================================================================

def test_all_queries_pair_mining_and_final_training(tiny_system_data, tiny_bert_model, tmp_path: Path):
    """Test mining pairs from all queries without fold exclusion and producing final manifest."""
    data_dir, index_dir = tiny_system_data
    final_out = tmp_path / "final_checkpoints"

    # Test build_training_pairs with fold=None, use_all_queries=True
    pairs_dir = final_out / "pairs"
    retriever_df, pairs_df = build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        fold=None,
        use_all_queries=True,
    )

    all_qids = {"q1", "q2", "q3", "q4", "q5", "q6"}
    mined_qids = set(pairs_df["query_id"].astype(str))
    assert mined_qids == all_qids, f"Expected all 6 queries in mined pairs, got {mined_qids}"

    # Test train_final_system
    manifest = train_final_system(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=final_out,
        config_path={"base_model_name": tiny_bert_model, "epochs": 1, "batch_size": 2},
        max_steps=2,
    )

    assert manifest["unique_training_queries"] == 6
    assert manifest["pair_count"] == len(pairs_df)
    assert manifest["positive_count"] > 0
    assert manifest["negative_count"] > 0
    assert "optimizer_steps" in manifest
    assert "effective_examples_seen" in manifest
    assert "epochs_or_equivalent" in manifest
    assert "adapter_checksum" in manifest

    # Manifest file must exist on disk
    manifest_file = final_out / "final_manifest.json"
    assert manifest_file.exists()
    saved_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert saved_manifest["unique_training_queries"] == 6


# ==============================================================================
# 4. P1.4: Cross-Fitted Fusion Evaluation and Model Selection Gate
# ==============================================================================

def test_cross_fitted_fusion_evaluation(tmp_path: Path):
    """Test 5-fold cross-fitted fusion evaluation, selection gate, and final model artifact generation."""
    fusion_out = tmp_path / "fusion_output"

    # Construct synthetic OOF candidate features for 2 folds
    oof_rows = [
        {"query_id": "q1", "doc_id": "101", "fold": 0, "label": 1, "bm25_score": 5.0, "bm25_rank": 1, "dense_score": 0.8, "dense_rank": 1, "reranker_score": 3.0},
        {"query_id": "q1", "doc_id": "102", "fold": 0, "label": 0, "bm25_score": 2.0, "bm25_rank": 2, "dense_score": 0.3, "dense_rank": 2, "reranker_score": -1.0},
        {"query_id": "q2", "doc_id": "102", "fold": 0, "label": 1, "bm25_score": 6.0, "bm25_rank": 1, "dense_score": 0.9, "dense_rank": 1, "reranker_score": 4.0},
        {"query_id": "q2", "doc_id": "101", "fold": 0, "label": 0, "bm25_score": 1.0, "bm25_rank": 2, "dense_score": 0.2, "dense_rank": 2, "reranker_score": -2.0},
        {"query_id": "q3", "doc_id": "103", "fold": 1, "label": 1, "bm25_score": 4.5, "bm25_rank": 1, "dense_score": 0.85, "dense_rank": 1, "reranker_score": 2.5},
        {"query_id": "q3", "doc_id": "104", "fold": 1, "label": 0, "bm25_score": 1.5, "bm25_rank": 2, "dense_score": 0.1, "dense_rank": 2, "reranker_score": -1.5},
        {"query_id": "q4", "doc_id": "104", "fold": 1, "label": 1, "bm25_score": 5.5, "bm25_rank": 1, "dense_score": 0.92, "dense_rank": 1, "reranker_score": 3.8},
        {"query_id": "q4", "doc_id": "103", "fold": 1, "label": 0, "bm25_score": 2.2, "bm25_rank": 2, "dense_score": 0.4, "dense_rank": 2, "reranker_score": -0.5},
    ]
    oof_df = pd.DataFrame(oof_rows)
    qrels_dict = {
        "q1": ["101"],
        "q2": ["102"],
        "q3": ["103"],
        "q4": ["104"],
    }

    result = train_and_evaluate_fusion_cv(
        oof_df=oof_df,
        qrels_dict=qrels_dict,
        output_dir=fusion_out,
        num_boost_round=10,
    )

    assert "winning_method" in result
    assert "winner_mean_recall@5" in result
    assert result["winner_mean_recall@5"] > 0.0

    # Verify comparison and manifest files
    assert (fusion_out / "fusion_comparison.json").exists()
    assert (fusion_out / "manifest.json").exists()
    assert (fusion_out / "winning_method.json").exists()
    assert (fusion_out / "model_full.txt").exists()
    assert (fusion_out / "fusion_final" / "model.txt").exists()


# ==============================================================================
# 5. P1.8: Final Public Pipeline Loads All Selected Artifacts
# ==============================================================================

def test_final_pipeline_loads_all_artifacts(tiny_system_data, tiny_bert_model, tmp_path: Path):
    """Test LegalIRPipeline.load_pipeline with BM25, PyVi, QuestionMemory, LoRA adapter, and Final Fusion."""
    data_dir, index_dir = tiny_system_data

    # 1. Train a tiny LoRA adapter
    pairs_dir = tmp_path / "pairs"
    build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        fold=None,
        use_all_queries=True,
    )
    from src.training.train_reranker import train_reranker
    adapter_dir = tmp_path / "reranker_final"
    train_reranker(
        pairs_file=pairs_dir / "reranker_pairs.parquet",
        config_path={"base_model_name": tiny_bert_model, "epochs": 1, "batch_size": 2},
        output_dir=adapter_dir,
        max_steps=2,
    )

    # 2. Train a final fusion model
    fusion_dir = tmp_path / "fusion_final"
    oof_rows = [
        {"query_id": "q1", "doc_id": "101", "fold": 0, "label": 1, "bm25_score": 5.0, "bm25_rank": 1, "dense_score": 0.8, "dense_rank": 1, "reranker_score": 3.0},
        {"query_id": "q1", "doc_id": "102", "fold": 0, "label": 0, "bm25_score": 2.0, "bm25_rank": 2, "dense_score": 0.3, "dense_rank": 2, "reranker_score": -1.0},
        {"query_id": "q2", "doc_id": "102", "fold": 1, "label": 1, "bm25_score": 6.0, "bm25_rank": 1, "dense_score": 0.9, "dense_rank": 1, "reranker_score": 4.0},
    ]
    oof_df = pd.DataFrame(oof_rows)
    qrels = {"q1": ["101"], "q2": ["102"]}
    train_and_evaluate_fusion_cv(oof_df=oof_df, qrels_dict=qrels, output_dir=fusion_dir)

    # 3. Load full pipeline
    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        reranker_adapter_path=adapter_dir,
        fusion_model_path=fusion_dir,
        use_reranker=True,
        use_learned_fusion=True,
        device="cpu",
    )

    # Verify pipeline components
    assert pipeline.hybrid_engine.bm25 is not None
    assert pipeline.hybrid_engine.bm25_pyvi is not None
    assert pipeline.hybrid_engine.exact is not None
    assert pipeline.hybrid_engine.memory is not None
    assert pipeline.reranker is not None
    assert isinstance(pipeline.ranker, LightGBMRanker)

    # 4. Predict on a test query
    preds = pipeline.predict_single(query="Thành lập doanh nghiệp")
    assert isinstance(preds, list)
    assert 1 <= len(preds) <= 5
    assert all(isinstance(x, str) for x in preds)
