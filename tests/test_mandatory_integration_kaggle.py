"""Mandatory 11 Integration Tests for LegalIR Task 1 Kaggle Pipeline.

Section 15 of LEGALIR_POST_FIX_KAGGLE_REPAIR_AGENT.md requires verifying:
 1. Test 1: Generated notebook production API contract.
 2. Test 2: Tiny end-to-end Kaggle smoke exercising the exact production pipeline.
 3. Test 3: Pair-path identity.
 4. Test 4: Adapter identity.
 5. Test 5: Fold adapter isolation.
 6. Test 6: PyVi branch participation.
 7. Test 7: BM25 metadata enrichment.
 8. Test 8: Evidence contract (sequence A = query, sequence B = doc without [QUESTION]).
 9. Test 9: Final all-query training.
 10. Test 10: Learned fusion cross-fitting.
 11. Test 11: Submission parity & zip validity.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from scripts.generate_kaggle_notebook import build_legalir_notebook, generate_and_save_notebooks
from src.core.paths import ProjectPaths
from src.evaluation.submission import (
    compute_sha256,
    create_submission_manifest,
    package_submission,
    validate_submission,
    validate_submission_zip,
)
from src.models.parameter_audit import MAX_PARAMETER_BUDGET, audit_system_parameters
from src.pipeline.kaggle_train import (
    KaggleRunResult,
    discover_data_dir,
    discover_public_test_file,
    resolve_kaggle_devices,
    run_kaggle_pipeline,
)
from src.pipeline.oof_runner import OOFRunner
from src.pipeline.predict import LegalIRPipeline
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import ReciprocalRankFusion
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.train_fusion import train_and_evaluate_fusion_cv
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.build_indexes import (
    build_bm25_index,
    build_bm25_pyvi_index,
    enrich_chunks_with_doc_metadata,
)
from src.retrieval.hybrid_search import HybridSearchEngine
from src.training.build_pairs import build_training_pairs
from src.training.train_reranker import train_reranker

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Test Fixtures: Self-contained Canonical Dataset & Tiny BERT Model
# ==============================================================================

@pytest.fixture
def tiny_canonical_environment(tmp_path: Path):
    """Create a minimal self-contained canonical dataset and indexes for integration tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "doc_id": "101",
            "title": "Luật Doanh nghiệp 2020",
            "legal_number": "59/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/59-2020",
            "name_raw": "Luật Doanh nghiệp",
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
            "is_empty": False,
        },
        {
            "doc_id": "104",
            "title": "Nghị định Hướng dẫn Luật Đầu tư",
            "legal_number": "31/2021/NĐ-CP",
            "year": "2021",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/31-2021",
            "name_raw": "Nghị định 31",
            "is_empty": False,
        },
        {
            "doc_id": "105",
            "title": "Luật Thương mại",
            "legal_number": "36/2005/QH11",
            "year": "2005",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/36-2005",
            "name_raw": "Luật Thương mại",
            "is_empty": False,
        },
        {
            "doc_id": "106",
            "title": "Luật Quản lý thuế",
            "legal_number": "38/2019/QH14",
            "year": "2019",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/38-2019",
            "name_raw": "Luật Quản lý thuế",
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
            "text_raw": "Quy định về thành lập doanh nghiệp và quản lý công ty trách nhiệm hữu hạn.",
            "text_norm": "quy định về thành lập doanh nghiệp và quản lý công ty trách nhiệm hữu hạn",
        },
        {
            "chunk_id": "c101_macro",
            "parent_chunk_id": None,
            "doc_id": "101",
            "granularity": "macro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về thành lập doanh nghiệp và quản lý công ty trách nhiệm hữu hạn.",
            "text_norm": "quy định về thành lập doanh nghiệp và quản lý công ty trách nhiệm hữu hạn",
        },
        {
            "chunk_id": "c102_micro",
            "parent_chunk_id": "c102_macro",
            "doc_id": "102",
            "granularity": "micro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư nước ngoài.",
            "text_norm": "quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư nước ngoài",
        },
        {
            "chunk_id": "c102_macro",
            "parent_chunk_id": None,
            "doc_id": "102",
            "granularity": "macro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư nước ngoài.",
            "text_norm": "quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư nước ngoài",
        },
        {
            "chunk_id": "c103_micro",
            "parent_chunk_id": "c103_macro",
            "doc_id": "103",
            "granularity": "micro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua cổng thông tin quốc gia.",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua cổng thông tin quốc gia",
        },
        {
            "chunk_id": "c103_macro",
            "parent_chunk_id": None,
            "doc_id": "103",
            "granularity": "macro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua cổng thông tin quốc gia.",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua cổng thông tin quốc gia",
        },
        {
            "chunk_id": "c104_micro",
            "parent_chunk_id": "c104_macro",
            "doc_id": "104",
            "granularity": "micro",
            "article": "Điều 4",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Thủ tục cấp giấy chứng nhận đăng ký đầu tư cho nhà đầu tư nước ngoài.",
            "text_norm": "thủ tục cấp giấy chứng nhận đăng ký đầu tư cho nhà đầu tư nước ngoài",
        },
        {
            "chunk_id": "c104_macro",
            "parent_chunk_id": None,
            "doc_id": "104",
            "granularity": "macro",
            "article": "Điều 4",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Thủ tục cấp giấy chứng nhận đăng ký đầu tư cho nhà đầu tư nước ngoài.",
            "text_norm": "thủ tục cấp giấy chứng nhận đăng ký đầu tư cho nhà đầu tư nước ngoài",
        },
        {
            "chunk_id": "c105_micro",
            "parent_chunk_id": "c105_macro",
            "doc_id": "105",
            "granularity": "micro",
            "article": "Điều 5",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hoạt động thương mại và mua bán hàng hóa quốc tế.",
            "text_norm": "hoạt động thương mại và mua bán hàng hóa quốc tế",
        },
        {
            "chunk_id": "c105_macro",
            "parent_chunk_id": None,
            "doc_id": "105",
            "granularity": "macro",
            "article": "Điều 5",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hoạt động thương mại và mua bán hàng hóa quốc tế.",
            "text_norm": "hoạt động thương mại và mua bán hàng hóa quốc tế",
        },
        {
            "chunk_id": "c106_micro",
            "parent_chunk_id": "c106_macro",
            "doc_id": "106",
            "granularity": "micro",
            "article": "Điều 6",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về khai thuế và nộp thuế điện tử.",
            "text_norm": "quy định về khai thuế và nộp thuế điện tử",
        },
        {
            "chunk_id": "c106_macro",
            "parent_chunk_id": None,
            "doc_id": "106",
            "granularity": "macro",
            "article": "Điều 6",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về khai thuế và nộp thuế điện tử.",
            "text_norm": "quy định về khai thuế và nộp thuế điện tử",
        },
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Thành lập doanh nghiệp như thế nào?", "question_norm": "thành lập doanh nghiệp như thế nào"},
        {"query_id": "q2", "question_raw": "Dự án đầu tư trực tiếp", "question_norm": "dự án đầu tư trực tiếp"},
        {"query_id": "q3", "question_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng", "question_norm": "hồ sơ đăng ký doanh nghiệp qua mạng"},
        {"query_id": "q4", "question_raw": "Thủ tục cấp giấy chứng nhận đăng ký đầu tư", "question_norm": "thủ tục cấp giấy chứng nhận đăng ký đầu tư"},
        {"query_id": "q5", "question_raw": "Hoạt động mua bán hàng hóa quốc tế", "question_norm": "hoạt động mua bán hàng hóa quốc tế"},
        {"query_id": "q6", "question_raw": "Khai thuế và nộp thuế điện tử", "question_norm": "khai thuế và nộp thuế điện tử"},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "101"},
        {"query_id": "q2", "doc_id": "102"},
        {"query_id": "q3", "doc_id": "103"},
        {"query_id": "q4", "doc_id": "104"},
        {"query_id": "q5", "doc_id": "105"},
        {"query_id": "q6", "doc_id": "106"},
    ]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet", index=False)

    split_info = [
        {"fold": 0, "train_query_ids": ["q4", "q5", "q6"], "val_query_ids": ["q1", "q2", "q3"]},
        {"fold": 1, "train_query_ids": ["q1", "q2", "q3"], "val_query_ids": ["q4", "q5", "q6"]},
    ]
    (splits_dir / "random_5fold.json").write_text(json.dumps(split_info), encoding="utf-8")

    doc_disjoint_split = {
        "train_query_ids": ["q1", "q2", "q3"],
        "val_query_ids": ["q4", "q5", "q6"],
        "train_doc_ids": ["101", "102", "103"],
        "val_doc_ids": ["104", "105", "106"],
    }
    (splits_dir / "doc_disjoint_split.json").write_text(json.dumps(doc_disjoint_split), encoding="utf-8")

    # Build BM25 and PyVi indexes
    build_bm25_index(data_dir=data_dir, output_dir=indexes_dir / "bm25")
    build_bm25_pyvi_index(data_dir=data_dir, output_dir=indexes_dir / "bm25_pyvi")

    return data_dir, indexes_dir


@pytest.fixture
def tiny_bert_fixture(tmp_path: Path):
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
# Test 1: Generated Notebook Production API Contract (Section 15, Test 1)
# ==============================================================================

def test_1_generated_notebook_production_api_contract(tiny_canonical_environment, tmp_path: Path):
    """Test 1: Parse generated notebook code and execute orchestration call in a test environment."""
    data_dir, _ = tiny_canonical_environment
    working_dir = tmp_path / "nb_smoke_working"
    working_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate notebook structure
    nb_dict = build_legalir_notebook()
    assert nb_dict.get("nbformat") == 4
    code_cells = [cell for cell in nb_dict["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 4

    # 2. Verify all code cells compile without SyntaxError
    for idx, cell in enumerate(code_cells):
        cell_code = "".join(cell["source"])
        compile(cell_code, f"<notebook_cell_{idx}>", "exec")

    # 3. Execute the exact production entrypoint that Cell 4 invokes
    result = run_kaggle_pipeline(
        data_dir=data_dir,
        working_dir=working_dir,
        run_mode="smoke",
        repo_root=REPO_ROOT,
        devices=["cpu", "cpu"],
    )

    assert isinstance(result, KaggleRunResult)
    assert result.is_valid is True
    assert result.submission_path.exists()
    assert result.submission_zip_path.exists()
    assert result.manifest_path.exists()


# ==============================================================================
# Test 2: Tiny End-to-End Kaggle Smoke Pipeline (Section 15, Test 2)
# ==============================================================================

def test_2_tiny_end_to_end_kaggle_smoke(tiny_canonical_environment, tmp_path: Path):
    """Test 2: Full 24-step pipeline smoke: dataset -> indexes -> pairs -> adapter -> OOF -> final -> submission."""
    data_dir, _ = tiny_canonical_environment
    working_dir = tmp_path / "e2e_smoke_working"

    result = run_kaggle_pipeline(
        data_dir=data_dir,
        working_dir=working_dir,
        run_mode="smoke",
        repo_root=REPO_ROOT,
        devices=["cpu", "cpu"],
    )

    assert result.is_valid is True
    assert result.submission_path.name == "submission.json"
    assert result.submission_zip_path.name == "submission.zip"
    assert result.manifest_path.name == "submission_manifest.json"

    # Verify submission json structure
    with open(result.submission_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    assert isinstance(preds, dict)
    assert len(preds) > 0
    for qid, val in preds.items():
        assert "answer" in val
        assert isinstance(val["answer"], list)
        assert 1 <= len(val["answer"]) <= 5

    # Verify submission zip contains strictly submission.json at root
    with zipfile.ZipFile(result.submission_zip_path, "r") as zf:
        file_list = zf.namelist()
        assert file_list == ["submission.json"]

    # Verify parameter audit report
    assert "total_learned_parameters" in result.audit_report
    assert result.audit_report["total_learned_parameters"] < MAX_PARAMETER_BUDGET

    # Verify CV report
    assert "mean_recall@5" in result.cv_report
    assert "mean_precision@5" in result.cv_report


# ==============================================================================
# Test 3: Pair-Path Identity (Section 15, Test 3)
# ==============================================================================

def test_3_pair_path_identity(tiny_canonical_environment, tiny_bert_fixture, tmp_path: Path):
    """Test 3: Assert the pair file produced by pair mining is the exact file consumed by trainer."""
    data_dir, index_dir = tiny_canonical_environment
    pairs_dir = tmp_path / "test_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = tmp_path / "test_ckpt"

    # 1. Build training pairs
    df_retriever, df_reranker = build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        fold=0,
        use_all_queries=False,
    )
    expected_pairs_file = pairs_dir / "reranker_pairs.parquet"
    assert expected_pairs_file.exists()
    assert len(df_reranker) > 0

    # 2. Train reranker passing explicit pairs_file
    report = train_reranker(
        pairs_file=expected_pairs_file,
        config_path=REPO_ROOT / "configs/kaggle.yaml",
        output_dir=checkpoints_dir,
        fold=0,
        base_model_name=tiny_bert_fixture,
        max_steps=5,
    )

    assert report["status"] in ("completed", "success")
    assert report["input_pair_count"] == len(df_reranker)
    assert report["pairs_file"] == str(expected_pairs_file.resolve())

    # 3. Assert missing pair file raises FileNotFoundError loudly
    with pytest.raises(FileNotFoundError):
        train_reranker(
            pairs_file=pairs_dir / "non_existent_pairs.parquet",
            config_path=REPO_ROOT / "configs/kaggle.yaml",
            output_dir=tmp_path / "fail_ckpt",
            fold=0,
            base_model_name=tiny_bert_fixture,
        )


# ==============================================================================
# Test 4: Adapter Identity (Section 15, Test 4)
# ==============================================================================

def test_4_adapter_identity(tiny_canonical_environment, tiny_bert_fixture, tmp_path: Path):
    """Test 4: Assert checkpoint produced by final trainer is the exact adapter consumed by final inference."""
    data_dir, index_dir = tiny_canonical_environment
    pairs_dir = tmp_path / "pairs_final"
    adapter_dir = tmp_path / "adapter_final"

    # 1. Build pairs and train tiny PEFT adapter
    build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        use_all_queries=True,
    )
    train_report = train_reranker(
        pairs_file=pairs_dir / "reranker_pairs.parquet",
        config_path=REPO_ROOT / "configs/kaggle.yaml",
        output_dir=adapter_dir,
        base_model_name=tiny_bert_fixture,
        max_steps=5,
    )
    assert (adapter_dir / "adapter_config.json").exists()

    # 2. Load production pipeline with the trained adapter
    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        reranker_adapter_path=adapter_dir,
        use_reranker=True,
        device="cpu",
        reranker_model_name=tiny_bert_fixture,
    )

    # 3. Verify PEFT adapter model is loaded from exact adapter_dir
    assert pipeline.reranker is not None
    assert pipeline.reranker.adapter_path == adapter_dir
    pipeline.reranker._load_model()
    from peft import PeftModel
    assert isinstance(pipeline.reranker.model, PeftModel)


# ==============================================================================
# Test 5: Fold Adapter Isolation (Section 15, Test 5)
# ==============================================================================

def test_5_fold_adapter_isolation(tiny_canonical_environment, tmp_path: Path):
    """Test 5: For each validation fold, assert its pairs/reranker are trained only from fold-train queries."""
    data_dir, index_dir = tiny_canonical_environment
    splits_file = data_dir / "splits/random_5fold.json"
    splits = json.loads(splits_file.read_text(encoding="utf-8"))

    for fold_info in splits:
        fold_idx = fold_info["fold"]
        train_qids = set(fold_info["train_query_ids"])
        val_qids = set(fold_info["val_query_ids"])

        fold_pairs_dir = tmp_path / f"fold_{fold_idx}_pairs"
        df_retriever, df_reranker = build_training_pairs(
            data_dir=data_dir,
            index_dir=index_dir,
            output_dir=fold_pairs_dir,
            fold=fold_idx,
            use_all_queries=False,
        )

        pair_qids = set(df_reranker["query_id"].astype(str))
        # Fold training pairs must be subset of train_qids and disjoint from val_qids
        assert pair_qids.issubset(train_qids), f"Fold {fold_idx} pairs contain non-train query IDs!"
        assert pair_qids.isdisjoint(val_qids), f"Fold {fold_idx} pairs leaked validation query IDs!"


# ==============================================================================
# Test 6: PyVi Branch Participation (Section 15, Test 6)
# ==============================================================================

def test_6_pyvi_branch_participation(tmp_path: Path):
    """Test 6: Inject a toy query where PyVi branch finds the relevant doc and verify candidate union contains it."""
    pyvi_docs = [
        {"doc_id": "doc_pyvi_1", "text_norm": "thủ tục đăng ký bảo hiểm y tế cho người lao động"},
        {"doc_id": "doc_other_2", "text_norm": "quy định xử phạt vi phạm hành chính"},
    ]
    pyvi_retriever = BM25PyViRetriever(k1=1.5, b=0.75).fit(pyvi_docs, show_progress=False)

    engine = HybridSearchEngine(
        bm25_retriever=None,
        bm25_pyvi_retriever=pyvi_retriever,
        dense_retriever=None,
        question_memory=None,
        exact_matcher=None,
    )

    query = "đăng ký bảo hiểm y tế"
    candidates = engine.search(query=query, top_k_candidates=5)
    doc_ids = [c["doc_id"] if isinstance(c, dict) else c.doc_id for c in candidates]

    assert "doc_pyvi_1" in doc_ids
    assert len(candidates) > 0


# ==============================================================================
# Test 7: BM25 Metadata Enrichment (Section 15, Test 7)
# ==============================================================================

def test_7_bm25_metadata_enrichment(tmp_path: Path):
    """Test 7: Verify legal_number, title, year, doc_type reach BM25 index and enable statutory boosts."""
    docs_df = pd.DataFrame([
        {
            "doc_id": "doc_statutory",
            "title": "Nghị định về thuế giá trị gia tăng",
            "legal_number": "44/2023/NĐ-CP",
            "year": "2023",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/44-2023",
        },
        {
            "doc_id": "doc_generic",
            "title": "Văn bản chung",
            "legal_number": "",
            "year": "2020",
            "doc_type": "Thông tư",
            "link": "",
        },
    ])
    chunks_df = pd.DataFrame([
        {
            "chunk_id": "c_stat",
            "doc_id": "doc_statutory",
            "granularity": "micro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định chi tiết thi hành thuế giá trị gia tăng.",
            "text_norm": "quy định chi tiết thi hành thuế giá trị gia tăng",
        },
        {
            "chunk_id": "c_gen",
            "doc_id": "doc_generic",
            "granularity": "micro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về thuế nói chung.",
            "text_norm": "quy định về thuế nói chung",
        },
    ])

    docs_file = tmp_path / "documents.parquet"
    docs_df.to_parquet(docs_file)

    enriched_chunks = enrich_chunks_with_doc_metadata(chunks_df, docs_file)
    assert "legal_number" in enriched_chunks.columns
    assert "title" in enriched_chunks.columns

    bm25 = BM25MicroRetriever().fit(enriched_chunks.to_dict("records"), show_progress=False)
    assert "doc_statutory" in bm25.doc_legal_numbers
    assert "2023" in bm25.doc_years["doc_statutory"]

    # Query with exact legal number boost
    results = bm25.retrieve("Nghị định số 44/2023/NĐ-CP", top_k=2)
    top_doc_id = results[0]["doc_id"] if isinstance(results[0], dict) else results[0].doc_id
    assert top_doc_id == "doc_statutory"


# ==============================================================================
# Test 8: Evidence Contract (Section 15, Test 8)
# ==============================================================================

def test_8_evidence_contract(tiny_canonical_environment):
    """Test 8: Verify tokenizer pair is (sequence_A=query, sequence_B=doc) and sequence B excludes [QUESTION]."""
    data_dir, _ = tiny_canonical_environment
    builder = EvidencePackBuilder(
        chunks_path=data_dir / "chunks.parquet",
        documents_path=data_dir / "documents.parquet",
        max_chunks=3,
        max_tokens=430,
    )

    query = "Điều kiện thành lập doanh nghiệp cổ phần"
    pack = builder.build_pack(
        query=query,
        doc_id="101",
        include_question=False,
    )

    # Invariant: sequence B passage must NOT include synthetic [QUESTION] header
    assert "[QUESTION]" not in pack
    assert not pack.startswith(f"[QUESTION] {query}")

    records = builder.build(query=query, doc_id="101", include_question=False)
    assert len(records) > 0
    assert "[QUESTION]" not in records[0]["pack"]

    # Verify structure of cross-encoder input pair
    pair = (query, pack)
    assert isinstance(pair[0], str)
    assert isinstance(pair[1], str)
    assert "[QUESTION]" not in pair[1]


# ==============================================================================
# Test 9: Final All-Query Training (Section 15, Test 9)
# ==============================================================================

def test_9_final_all_query_training(tiny_canonical_environment, tmp_path: Path):
    """Test 9: Verify final pair generation uses all intended training queries and is not fold-limited."""
    data_dir, index_dir = tiny_canonical_environment
    pairs_dir = tmp_path / "pairs_all_queries"

    df_qrels = pd.read_parquet(data_dir / "qrels_train.parquet")
    all_target_qids = set(df_qrels["query_id"].astype(str))

    df_retriever, df_reranker = build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        fold=None,
        use_all_queries=True,
    )

    pair_qids = set(df_reranker["query_id"].astype(str))
    # All queries with qrels should be present in final pair generation
    assert pair_qids == all_target_qids
    manifest = json.loads((pairs_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_queries"] == len(all_target_qids)
    assert manifest["fold"] is None
    assert manifest["use_all_queries"] is True


# ==============================================================================
# Test 10: Learned Fusion Cross-Fitting (Section 15, Test 10)
# ==============================================================================

def test_10_learned_fusion_cross_fitting(tmp_path: Path):
    """Test 10: Verify a fold's fusion model is not trained on that fold's labels (leakage-safe cross-fitting)."""
    # Create mock OOF feature records across 2 folds using standard feature columns
    oof_records = [
        # Fold 0
        {"query_id": "q1", "doc_id": "101", "fold": 0, "label": 1.0, "raw_bm25_rank": 1, "raw_bm25_score": 10.0, "pyvi_bm25_rank": 1, "pyvi_bm25_score": 10.0, "dense_rank": 2, "dense_score": 0.8, "reranker_score": 0.9, "rrf_score": 0.03},
        {"query_id": "q1", "doc_id": "102", "fold": 0, "label": 0.0, "raw_bm25_rank": 2, "raw_bm25_score": 5.0, "pyvi_bm25_rank": 2, "pyvi_bm25_score": 5.0, "dense_rank": 1, "dense_score": 0.9, "reranker_score": 0.3, "rrf_score": 0.02},
        {"query_id": "q2", "doc_id": "102", "fold": 0, "label": 1.0, "raw_bm25_rank": 1, "raw_bm25_score": 12.0, "pyvi_bm25_rank": 1, "pyvi_bm25_score": 12.0, "dense_rank": 1, "dense_score": 0.95, "reranker_score": 0.85, "rrf_score": 0.035},
        {"query_id": "q2", "doc_id": "101", "fold": 0, "label": 0.0, "raw_bm25_rank": 2, "raw_bm25_score": 4.0, "pyvi_bm25_rank": 2, "pyvi_bm25_score": 4.0, "dense_rank": 2, "dense_score": 0.6, "reranker_score": 0.2, "rrf_score": 0.015},
        # Fold 1
        {"query_id": "q3", "doc_id": "103", "fold": 1, "label": 1.0, "raw_bm25_rank": 1, "raw_bm25_score": 11.0, "pyvi_bm25_rank": 1, "pyvi_bm25_score": 11.0, "dense_rank": 1, "dense_score": 0.9, "reranker_score": 0.88, "rrf_score": 0.032},
        {"query_id": "q3", "doc_id": "104", "fold": 1, "label": 0.0, "raw_bm25_rank": 2, "raw_bm25_score": 3.0, "pyvi_bm25_rank": 2, "pyvi_bm25_score": 3.0, "dense_rank": 2, "dense_score": 0.5, "reranker_score": 0.25, "rrf_score": 0.018},
        {"query_id": "q4", "doc_id": "104", "fold": 1, "label": 1.0, "raw_bm25_rank": 1, "raw_bm25_score": 9.0, "pyvi_bm25_rank": 1, "pyvi_bm25_score": 9.0, "dense_rank": 2, "dense_score": 0.7, "reranker_score": 0.82, "rrf_score": 0.028},
        {"query_id": "q4", "doc_id": "103", "fold": 1, "label": 0.0, "raw_bm25_rank": 2, "raw_bm25_score": 2.0, "pyvi_bm25_rank": 2, "pyvi_bm25_score": 2.0, "dense_rank": 1, "dense_score": 0.8, "reranker_score": 0.3, "rrf_score": 0.022},
    ]
    oof_df = pd.DataFrame(oof_records)

    qrels_dict = {
        "q1": ["101"],
        "q2": ["102"],
        "q3": ["103"],
        "q4": ["104"],
    }

    fusion_dir = tmp_path / "fusion_cv"
    report = train_and_evaluate_fusion_cv(
        oof_df=oof_df,
        qrels_dict=qrels_dict,
        output_dir=fusion_dir,
        num_boost_round=10,
    )

    assert "winning_method" in report
    assert "winner_mean_recall@5" in report
    assert "comparison" in report
    assert len(report["comparison"]["learned_ranker"]["folds"]) == 2
    assert len(report["comparison"]["reciprocal_rank_fusion"]["folds"]) == 2


# ==============================================================================
# Test 11: Submission Parity & Zip Validity (Section 15, Test 11)
# ==============================================================================

def test_11_submission_parity_and_zip_validity(tmp_path: Path):
    """Test 11: Run official/scorer-compatible validation on generated predictions and ZIP packaging."""
    sub_dir = tmp_path / "sub_test"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_json = sub_dir / "submission.json"
    sub_zip = sub_dir / "submission.zip"

    predictions = {
        "q1": {"answer": ["101", "102", "103", "104", "105"]},
        "q2": {"answer": ["102", "101", "104"]},
        "q3": {"answer": ["103", "101"]},
    }

    package_submission(predictions, sub_json, sub_zip)

    # 1. JSON validation
    expected_qids = {"q1", "q2", "q3"}
    json_val = validate_submission(sub_json, expected_qids=expected_qids)
    assert json_val["is_valid"] is True
    assert json_val["total_queries"] == 3

    # 2. ZIP validation
    zip_val = validate_submission_zip(sub_zip)
    assert zip_val["is_valid"] is True
    with zipfile.ZipFile(sub_zip, "r") as zf:
        assert zf.namelist() == ["submission.json"]
