import json
from pathlib import Path
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from src.pipeline.predict import LegalIRPipeline
from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.training.build_pairs import build_training_pairs
from src.training.train_reranker import train_reranker


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    """Create a minimal self-contained dataset in tmp_path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {"doc_id": "1", "title": "Luật Đầu tư 2020", "legal_number": "61/2020/QH14", "year": "2020", "doc_type": "Luật", "name_raw": "Luật Đầu tư"},
        {"doc_id": "2", "title": "Luật Doanh nghiệp 2020", "legal_number": "59/2020/QH14", "year": "2020", "doc_type": "Luật", "name_raw": "Luật Doanh nghiệp"},
        {"doc_id": "3", "title": "Nghị định Đầu tư", "legal_number": "31/2021/NĐ-CP", "year": "2021", "doc_type": "Nghị định", "name_raw": "Nghị định 31"},
    ]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet", index=False)

    chunks = [
        {"chunk_id": "c1_macro", "doc_id": "1", "granularity": "macro", "article": "Điều 1", "clause": "Khoản 1", "point": "Điểm a", "text_norm": "quy định về dự án đầu tư nước ngoài", "text_raw": "quy định về dự án đầu tư nước ngoài"},
        {"chunk_id": "c1_micro", "doc_id": "1", "granularity": "micro", "article": "Điều 1", "clause": "Khoản 1", "point": "Điểm a", "text_norm": "quy định về dự án đầu tư", "text_raw": "quy định về dự án đầu tư"},
        {"chunk_id": "c2_macro", "doc_id": "2", "granularity": "macro", "article": "Điều 2", "clause": "Khoản 1", "point": "Điểm a", "text_norm": "thành lập doanh nghiệp tư nhân", "text_raw": "thành lập doanh nghiệp tư nhân"},
        {"chunk_id": "c2_micro", "doc_id": "2", "granularity": "micro", "article": "Điều 2", "clause": "Khoản 1", "point": "Điểm a", "text_norm": "thành lập doanh nghiệp cổ phần", "text_raw": "thành lập doanh nghiệp cổ phần"},
        {"chunk_id": "c3_macro", "doc_id": "3", "granularity": "macro", "article": "Điều 3", "clause": "Khoản 1", "point": "Điểm a", "text_norm": "hướng dẫn luật đầu tư", "text_raw": "hướng dẫn luật đầu tư"},
        {"chunk_id": "c3_micro", "doc_id": "3", "granularity": "micro", "article": "Điều 3", "clause": "Khoản 1", "point": "Điểm a", "text_norm": "hướng dẫn thi hành luật đầu tư", "text_raw": "hướng dẫn thi hành luật đầu tư"},
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Dự án đầu tư nước ngoài", "question_norm": "dự án đầu tư nước ngoài"},
        {"query_id": "q2", "question_raw": "Thành lập doanh nghiệp", "question_norm": "thành lập doanh nghiệp"},
        {"query_id": "q3", "question_raw": "Hướng dẫn đầu tư", "question_norm": "hướng dẫn đầu tư"},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "1"},
        {"query_id": "q2", "doc_id": "2"},
        {"query_id": "q3", "doc_id": "3"},
    ]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet", index=False)

    split_info = [
        {"fold": 0, "train_query_ids": ["q1", "q2"], "val_query_ids": ["q3"]},
        {"fold": 1, "train_query_ids": ["q2", "q3"], "val_query_ids": ["q1"]},
    ]
    (splits_dir / "random_5fold.json").write_text(json.dumps(split_info), encoding="utf-8")

    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir = index_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    micro_chunks = [c for c in chunks if c["granularity"] == "micro"]
    bm25 = BM25MicroRetriever().fit(micro_chunks)
    bm25.save(bm25_dir)

    return data_dir, index_dir


@pytest.fixture
def tiny_bert_fixture(tmp_path: Path):
    """Creates and saves a tiny BERT model and tokenizer for fast unit testing."""
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

    return str(model_dir), model, tokenizer


# ==============================================================================
# P0.1 Tests: build_training_pairs explicit path signature & all_queries support
# ==============================================================================

def test_build_training_pairs_explicit_paths(tmp_path: Path, tiny_dataset):
    data_dir, index_dir = tiny_dataset
    out_dir = tmp_path / "pairs_fold_0"

    retriever_df, reranker_df = build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=out_dir,
        fold=0,
        use_all_queries=False,
        limit=2,
        negatives_per_positive=2,
        max_evidence_chunks=3,
        include_dense_negatives=False,
        include_pyvi_negatives=False,
    )

    assert (out_dir / "retriever_pairs.parquet").is_file()
    assert (out_dir / "reranker_pairs.parquet").is_file()
    assert (out_dir / "manifest.json").is_file()

    assert len(reranker_df) > 0
    assert set(reranker_df["query_id"].unique()).issubset({"q1", "q2"})
    assert "q3" not in reranker_df["query_id"].unique()  # q3 was val in fold 0


def test_build_training_pairs_use_all_queries(tmp_path: Path, tiny_dataset):
    data_dir, index_dir = tiny_dataset
    out_dir = tmp_path / "pairs_all"

    retriever_df, reranker_df = build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=out_dir,
        fold=None,
        use_all_queries=True,
        negatives_per_positive=2,
        max_evidence_chunks=3,
        include_dense_negatives=False,
        include_pyvi_negatives=False,
    )

    assert set(reranker_df["query_id"].unique()) == {"q1", "q2", "q3"}
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_queries"] == 3
    assert manifest["use_all_queries"] is True


# ==============================================================================
# P0.2 Tests: train_reranker requires pairs_file, raises FileNotFoundError
# ==============================================================================

def test_train_reranker_missing_pairs_file_raises_filenotfounderror(tmp_path: Path):
    missing_file = tmp_path / "non_existent_pairs.parquet"
    out_dir = tmp_path / "out_ckpt"

    with pytest.raises(FileNotFoundError, match="Training pairs file not found"):
        train_reranker(
            pairs_file=missing_file,
            output_dir=out_dir,
        )


def test_train_reranker_explicit_pairs_file_success(tmp_path: Path, tiny_bert_fixture):
    model_dir, _, _ = tiny_bert_fixture
    pairs_file = tmp_path / "custom_pairs.parquet"
    out_dir = tmp_path / "custom_ckpt"

    pairs = [
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d1", "evidence_text": "tok_1 tok_2", "label": 1.0},
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d2", "evidence_text": "tok_90 tok_91", "label": 0.0},
        {"query_id": "q2", "query_text": "tok_3 tok_4", "doc_id": "d3", "evidence_text": "tok_3 tok_4", "label": 1.0},
        {"query_id": "q2", "query_text": "tok_3 tok_4", "doc_id": "d4", "evidence_text": "tok_92 tok_93", "label": 0.0},
    ] * 3
    pd.DataFrame(pairs).to_parquet(pairs_file, index=False)

    report = train_reranker(
        pairs_file=pairs_file,
        output_dir=out_dir,
        base_model_name=model_dir,
        max_steps=4,
        batch_size=2,
        learning_rate=1e-3,
    )

    assert report["status"] == "completed"
    assert report["input_pair_count"] == len(pairs)
    assert (out_dir / "adapter_config.json").is_file()
    assert (out_dir / "training_manifest.json").is_file()


# ==============================================================================
# P0.3 Tests: LegalIRPipeline.load_pipeline EvidencePackBuilder construction
# ==============================================================================

def test_load_pipeline_evidence_pack_builder_no_conflict(tiny_dataset):
    data_dir, index_dir = tiny_dataset

    # Must load cleanly without raising ValueError from EvidencePackBuilder
    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        use_reranker=False,
    )

    assert pipeline is not None
    assert pipeline.evidence_builder is not None
    assert pipeline.evidence_builder.max_chunks == 3
    assert pipeline.evidence_builder.max_tokens == 430


# ==============================================================================
# P0.4 Tests: LegalIRPipeline.load_pipeline loads reranker_adapter_path & fusion
# ==============================================================================

def test_load_pipeline_missing_adapter_raises_filenotfounderror(tiny_dataset):
    data_dir, index_dir = tiny_dataset

    with pytest.raises(FileNotFoundError, match="Reranker adapter path not found"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            reranker_adapter_path="non_existent_adapter_dir",
            use_reranker=True,
        )


def test_load_pipeline_missing_fusion_raises_filenotfounderror(tiny_dataset):
    data_dir, index_dir = tiny_dataset

    with pytest.raises(FileNotFoundError, match="Fusion model path not found"):
        LegalIRPipeline.load_pipeline(
            data_dir=data_dir,
            index_dir=index_dir,
            fusion_model_path="non_existent_fusion_file",
            use_reranker=False,
        )


def test_load_pipeline_with_reranker_adapter(tmp_path: Path, tiny_dataset, tiny_bert_fixture):
    data_dir, index_dir = tiny_dataset
    model_dir, _, _ = tiny_bert_fixture

    # Train a tiny adapter
    pairs_file = tmp_path / "pairs.parquet"
    adapter_dir = tmp_path / "reranker_adapter"
    pairs = [
        {"query_id": "q1", "query_text": "tok_1", "doc_id": "1", "evidence_text": "tok_1 tok_2", "label": 1.0},
        {"query_id": "q1", "query_text": "tok_1", "doc_id": "2", "evidence_text": "tok_99", "label": 0.0},
    ] * 4
    pd.DataFrame(pairs).to_parquet(pairs_file, index=False)

    train_reranker(
        pairs_file=pairs_file,
        output_dir=adapter_dir,
        base_model_name=model_dir,
        max_steps=2,
    )

    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        reranker_adapter_path=adapter_dir,
        use_reranker=True,
    )

    assert pipeline.reranker is not None
    assert pipeline.reranker.adapter_path == adapter_dir


def test_load_pipeline_with_fusion_model(tmp_path: Path, tiny_dataset):
    data_dir, index_dir = tiny_dataset
    from src.ranking.fusion import LinearRanker, LightGBMRanker

    fusion_file = tmp_path / "fusion_model.json"
    linear = LinearRanker()
    linear.save(fusion_file)

    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        fusion_model_path=fusion_file,
        use_reranker=False,
    )

    assert pipeline.ranker is not None
    assert isinstance(pipeline.ranker, LightGBMRanker)
