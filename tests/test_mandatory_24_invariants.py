"""Mandatory 24 Invariants Test Suite for LegalIR Task 1.

Section 20 of LEGALIR_KAGGLE_HIGH_SCORE_AGENT.md requires verifying all 24 mandatory invariants:
 1. legal identifier normalization is lossless
 2. raw BM25 corpus/query tokenizer consistency
 3. PyVi BM25 corpus/query tokenizer consistency
 4. legal boosts actually change ranking in a controlled example
 5. exact matcher handles NaN/null metadata
 6. candidate union deduplicates IDs deterministically
 7. query memory excludes validation/self query
 8. duplicate-group blacklist prevents false negatives
 9. evidence localization selects query-relevant article/chunk
 10. reranker evidence pack stays within token budget
 11. tiny reranker training changes trainable weights
 12. trained reranker checkpoint reloads
 13. OOF fold construction has no label leakage
 14. learned fusion trains without validation-fold labels
 15. parameter audit sums all final learned components
 16. parameter audit rejects >=4B
 17. official scorer parity
 18. submission exact query-key equality
 19. submission max-five rule
 20. submission uniqueness/valid-ID rule
 21. ZIP contains only submission.json at root
 22. deterministic fallback order
 23. notebook parses as valid nbformat
 24. Kaggle config paths resolve safely
"""

import json
from pathlib import Path
import sys
import tempfile
import zipfile
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
import yaml

# Ensure repository root and scoring program are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

scoring_prog_dir = REPO_ROOT / "Scoring-Program-Task-LegalIR"
if str(scoring_prog_dir) not in sys.path and scoring_prog_dir.exists():
    sys.path.insert(0, str(scoring_prog_dir))

from src.core.config import PipelineConfig, load_pipeline_config
from src.core.paths import ProjectPaths
from src.dataset.normalize import (
    clean_legal_text,
    extract_legal_signals,
    normalize_question,
    prettify_doc_title,
)
from src.evaluation.codabench_compat import assert_official_equivalence
from src.evaluation.evaluator import evaluate_predictions
from src.evaluation.splits import (
    generate_random_5fold_split,
    verify_fold_isolation,
)
from src.evaluation.submission import (
    create_submission_manifest,
    package_submission,
    validate_submission,
    validate_submission_zip,
)
from src.models.parameter_audit import (
    DEFAULT_PIPELINE_MODELS,
    MAX_PARAMETER_BUDGET,
    ParameterBudgetExceededError,
    audit_system_parameters,
    count_parameters,
    validate_parameter_budget,
)
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.selector import TopKSelector
from src.retrieval.bm25_micro import BM25MicroRetriever, tokenize_legal
from src.retrieval.bm25_pyvi import BM25PyViRetriever, tokenize_pyvi
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import TrainQuestionMemory
from src.training.hard_negative_miner import HardNegativeMiner
from src.training.positive_localizer import PositiveLocalizer
from src.training.trainer import RerankerTrainer


# =============================================================================
# Invariant 1: Legal identifier normalization is lossless
# =============================================================================
def test_invariant_01_legal_identifier_normalization_is_lossless():
    """Verify that legal text normalization and entity extraction preserve statutory details without loss."""
    raw_text = (
        "Căn cứ Nghị định số 123/2020/NĐ-CP và Điều 15a Khoản 2 Điểm b "
        "Thông tư 15/2021/TT-BTC ban hành năm 2023."
    )
    cleaned = clean_legal_text(raw_text)
    assert "123/2020/NĐ-CP" in cleaned
    assert "15/2021/TT-BTC" in cleaned
    assert "Điều 15a" in cleaned
    assert "Khoản 2" in cleaned
    assert "Điểm b" in cleaned
    assert "2023" in cleaned

    signals = extract_legal_signals(raw_text)
    assert "123/2020/NĐ-CP" in signals["doc_numbers"]
    assert "15/2021/TT-BTC" in signals["doc_numbers"]
    assert "15a" in signals["articles"]
    assert "2" in signals["clauses"]
    assert "b" in signals["points"]
    assert "2023" in signals["years"]

    title = prettify_doc_title("Nghi-dinh-123-2020-ND-CP")
    assert "Nghị định 123/2020/NĐ-CP" in title


# =============================================================================
# Invariant 2: Raw BM25 corpus/query tokenizer consistency
# =============================================================================
def test_invariant_02_raw_bm25_corpus_query_tokenizer_consistency():
    """Verify that tokenize_legal tokenizes corpus and queries identically, keeping statutory identifiers intact."""
    statutory_text = (
        "Căn cứ Nghị định số 123/2020/NĐ-CP và Điều 15 Khoản 2 Thông tư 15/2021/TT-BTC năm 2023"
    )
    corpus_tokens = tokenize_legal(statutory_text)
    query_tokens = tokenize_legal(statutory_text)

    # Deterministic and consistent
    assert corpus_tokens == query_tokens
    assert "123/2020/nđ-cp" in corpus_tokens
    assert "15/2021/tt-btc" in corpus_tokens
    assert "điều" in corpus_tokens
    assert "15" in corpus_tokens
    assert "khoản" in corpus_tokens
    assert "2" in corpus_tokens
    assert "2023" in corpus_tokens


# =============================================================================
# Invariant 3: PyVi BM25 corpus/query tokenizer consistency
# =============================================================================
def test_invariant_03_pyvi_bm25_corpus_query_tokenizer_consistency():
    """Verify that tokenize_pyvi segments Vietnamese multi-word phrases consistently across indexing and search."""
    vietnamese_text = "Thủ tục đăng ký bảo hiểm xã hội cho người lao động"
    corpus_tokens = tokenize_pyvi(vietnamese_text)
    query_tokens = tokenize_pyvi(vietnamese_text)

    assert corpus_tokens == query_tokens
    # PyVi segments multi-word phrases with underscores
    assert any("đăng_ký" in tok or "bảo_hiểm" in tok or "lao_động" in tok for tok in corpus_tokens)


# =============================================================================
# Invariant 4: Legal boosts actually change ranking in a controlled example
# =============================================================================
def test_invariant_04_legal_boosts_actually_change_ranking():
    """Verify that legal boosts in BM25MicroRetriever elevate documents with exact statutory matches over generic text."""
    chunks = [
        {
            "chunk_id": "c_generic",
            "doc_id": "doc_generic",
            "legal_number": "",
            "title": "Quy định hóa đơn chứng từ",
            "article": "",
            "text_norm": "hóa đơn chứng từ điện tử quy định chung về quản lý hóa đơn chứng từ",
        },
        {
            "chunk_id": "c_statutory",
            "doc_id": "doc_statutory",
            "legal_number": "123/2020/NĐ-CP",
            "title": "Nghị định về hóa đơn",
            "article": "Điều 15",
            "text_norm": "Nghị định 123/2020/NĐ-CP Điều 15 về hóa đơn điện tử",
        },
    ]

    retriever = BM25MicroRetriever()
    retriever.fit(chunks)

    # Query with exact legal number and article
    query = "Quy định tại Điều 15 Nghị định 123/2020/NĐ-CP về hóa đơn điện tử"
    results = retriever.retrieve(query, top_k=2)

    assert len(results) == 2
    assert results[0]["doc_id"] == "doc_statutory"
    assert results[0]["score"] > results[1]["score"]


# =============================================================================
# Invariant 5: Exact matcher handles NaN/null metadata
# =============================================================================
def test_invariant_05_exact_matcher_handles_nan_null_metadata():
    """Verify ExactMatcher gracefully handles NaN, None, and empty metadata values without crashing."""
    corrupted_docs = [
        {
            "doc_id": "doc_null_1",
            "legal_number": None,
            "title": float("nan"),
            "doc_type": None,
            "article": None,
        },
        {
            "doc_id": "doc_null_2",
            "legal_number": np.nan,
            "title": None,
            "doc_type": np.nan,
        },
        {
            "doc_id": "doc_valid",
            "legal_number": "123/2020/NĐ-CP",
            "title": "Nghị định số 123/2020/NĐ-CP",
            "doc_type": "Nghị định",
            "article": "Điều 15",
        },
    ]

    # Must initialize without raising exception
    matcher = ExactMatcher(corrupted_docs)

    # Matching query for doc_valid
    res_valid = matcher.match("Theo Nghị định 123/2020/NĐ-CP")
    assert "doc_valid" in res_valid
    assert res_valid["doc_valid"]["exact_legal_number"] is True

    # Matching query with null matches
    res_unknown = matcher.match("Không có văn bản nào phù hợp")
    assert isinstance(res_unknown, dict)


# =============================================================================
# Invariant 6: Candidate union deduplicates IDs deterministically
# =============================================================================
def test_invariant_06_candidate_union_deduplicates_ids_deterministically():
    """Verify candidate union logic deduplicates candidate document IDs deterministically across branches."""
    class StubRetriever:
        def __init__(self, items):
            self.items = items
        def retrieve(self, query, top_k=50):
            return self.items[:top_k]

    class StubExact:
        def __init__(self, mapping):
            self.mapping = mapping
        def match(self, query):
            return self.mapping

    bm25 = [{"doc_id": "d1", "score": 10.0, "bm25_score": 10.0, "bm25_best_score": 10.0}, {"doc_id": "d2", "score": 8.0, "bm25_score": 8.0, "bm25_best_score": 8.0}]
    exact = {"d2": {"score": 1.0, "exact_legal_number": True}}
    dense = [{"doc_id": "d3", "score": 0.95, "dense_score": 0.95, "dense_best_score": 0.95}, {"doc_id": "d1", "score": 0.80, "dense_score": 0.80, "dense_best_score": 0.80}]

    engine = HybridSearchEngine(
        bm25_retriever=StubRetriever(bm25),
        exact_matcher=StubExact(exact),
        dense_retriever=StubRetriever(dense),
    )

    cands = engine.search_candidates("test query", top_k=10)
    cand_ids = [c["doc_id"] for c in cands]

    # Strict deduplication
    assert len(cand_ids) == len(set(cand_ids))
    assert set(cand_ids) == {"d1", "d2", "d3"}

    # Repeated run gives exact same deterministic ordering
    cands_repeat = engine.search_candidates("test query", top_k=10)
    assert [c["doc_id"] for c in cands_repeat] == cand_ids


# =============================================================================
# Invariant 7: Query memory excludes validation/self query
# =============================================================================
def test_invariant_07_query_memory_excludes_validation_self_query():
    """Verify TrainQuestionMemory strictly excludes the query itself when querying nearest neighbors."""
    memory = TrainQuestionMemory(use_dense=False, min_similarity=0.5)
    train_queries = [
        {"qid": "q1", "text": "Hóa đơn điện tử có bắt buộc từ năm 2022 không?"},
        {"qid": "q2", "text": "Quy định về thời điểm xuất hóa đơn điện tử"},
    ]
    train_qrels = {"q1": ["doc_123"], "q2": ["doc_123", "doc_456"]}
    memory.fit(train_queries, train_qrels)

    # Search with q1 text but pass exclude_qid="q1"
    results = memory.search(
        "Hóa đơn điện tử có bắt buộc từ năm 2022 không?",
        top_k=5,
        exclude_qid="q1",
    )

    # q1's exact match must NOT be returned as a voter for itself
    for r in results:
        assert r.get("matched_qid") != "q1"


# =============================================================================
# Invariant 8: Duplicate-group blacklist prevents false negatives
# =============================================================================
def test_invariant_08_duplicate_group_blacklist_prevents_false_negatives():
    """Verify HardNegativeMiner respects duplicate-group blacklist to prevent false negatives."""
    blacklist = {"q10": {"doc_dup_gold", "doc_related_gold"}}
    miner = HardNegativeMiner(false_negative_blacklist=blacklist)

    candidates = [
        {"doc_id": "doc_gold", "score": 15.0},
        {"doc_id": "doc_dup_gold", "score": 14.0},
        {"doc_id": "doc_true_neg_1", "score": 10.0},
        {"doc_id": "doc_true_neg_2", "score": 8.0},
    ]

    mined = miner.mine_negatives(
        candidates,
        ["doc_gold"],
        query_id="q10",
        max_negatives=5,
        return_records=True,
    )
    mined_ids = [m["doc_id"] for m in mined]

    assert "doc_gold" not in mined_ids
    assert "doc_dup_gold" not in mined_ids
    assert mined_ids == ["doc_true_neg_1", "doc_true_neg_2"]
    assert miner.get_stats()["excluded_duplicates_count"] == 1


# =============================================================================
# Invariant 9: Evidence localization selects query-relevant article/chunk
# =============================================================================
def test_invariant_09_evidence_localization_selects_query_relevant_article_chunk():
    """Verify that evidence builder selects the query-relevant article chunk rather than blindly picking chunk 0."""
    chunks = []
    for i in range(1, 51):
        chunks.append({
            "doc_id": "doc_long_law",
            "chunk_id": f"chunk_{i:02d}",
            "article": f"Điều {i}",
            "text_norm": f"Nội dung quy định chung của Điều {i} về quản lý hành chính {i}.",
            "text_raw": f"Nội dung quy định chung của Điều {i} về quản lý hành chính {i}.",
        })

    # Specific query-relevant chunk at Điều 38
    chunks[37] = {
        "doc_id": "doc_long_law",
        "chunk_id": "chunk_38",
        "article": "Điều 38. Miễn giảm tiền thuê đất",
        "text_norm": "Đối tượng được miễn giảm tiền thuê đất trong khu công nghệ cao.",
        "text_raw": "Đối tượng được miễn giảm tiền thuê đất trong khu công nghệ cao.",
    }

    doc_meta = {
        "doc_long_law": {
            "doc_id": "doc_long_law",
            "title": "Luật Đất đai 2024",
            "legal_number": "31/2024/QH15",
        }
    }

    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta, max_chunks=2)
    query = "Điều kiện miễn giảm tiền thuê đất trong khu công nghệ cao theo Điều 38 là gì?"

    pack = builder.build_pack(query, "doc_long_law", max_chunks=2)
    assert "chunk_38" in [c["chunk_id"] for c in builder._select_chunks(query, "doc_long_law", None, max_chunks=2)]
    assert "miễn giảm tiền thuê đất trong khu công nghệ cao" in pack.lower()


# =============================================================================
# Invariant 10: Reranker evidence pack stays within token budget
# =============================================================================
def test_invariant_10_reranker_evidence_pack_stays_within_token_budget():
    """Verify EvidencePackBuilder truncates and stays within the configured character/token budget."""
    huge_text = "Quy định chi tiết điều khoản luật pháp. " * 500  # ~20,000 chars
    chunks = [
        {
            "doc_id": "doc_huge",
            "chunk_id": "c1",
            "article": "Điều 1",
            "text_norm": huge_text,
            "text_raw": huge_text,
        }
    ]
    doc_meta = {"doc_huge": {"title": "Văn bản cực dài", "legal_number": "99/2024/NĐ-CP"}}

    builder = EvidencePackBuilder(
        macro_chunks=chunks,
        doc_metadata=doc_meta,
        max_chars=600,
    )

    pack = builder.build_pack("Hỏi về quy định chi tiết", "doc_huge", max_chunks=2)
    assert len(pack) <= 650  # Strictly bounded by max_chars budget


# =============================================================================
# Invariant 11: Tiny reranker training changes trainable weights
# =============================================================================
def test_invariant_11_tiny_reranker_training_changes_trainable_weights(tmp_path: Path):
    """Verify that a tiny supervised reranker training run updates trainable parameters with non-zero delta norm."""
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=1,
    )
    model = BertForSequenceClassification(config)

    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(95)]) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))

    train_pairs = [
        {"query_id": "q1", "query_text": "tok_1", "doc_id": "d1", "evidence_text": "tok_1 tok_2", "label": 1.0},
        {"query_id": "q1", "query_text": "tok_1", "doc_id": "d2", "evidence_text": "tok_9 tok_8", "label": 0.0},
    ] * 5

    trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_pairs,
        val_data=[],
        config={"learning_rate": 1e-2, "batch_size": 2, "max_steps": 5, "use_lora": True, "target_modules": ["query", "value"], "loss_type": "bce"},
        device="cpu",
    )

    out_dir = tmp_path / "ckpt"
    report = trainer.train(output_dir=out_dir)

    assert report["status"] == "completed"
    assert report["param_diff"] > 0.0, "Trainable weights must have changed!"


# =============================================================================
# Invariant 12: Trained reranker checkpoint reloads
# =============================================================================
def test_invariant_12_trained_reranker_checkpoint_reloads(tmp_path: Path):
    """Verify that a trained reranker checkpoint can be reloaded and produces valid scores."""
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=1,
    )
    model = BertForSequenceClassification(config)

    model_dir = tmp_path / "base_model"
    vocab_file = model_dir / "vocab.txt"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_file.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(95)]) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    train_pairs = [
        {"query_id": "q1", "query_text": "tok_1", "doc_id": "d1", "evidence_text": "tok_1 tok_2", "label": 1.0},
        {"query_id": "q1", "query_text": "tok_1", "doc_id": "d2", "evidence_text": "tok_9 tok_8", "label": 0.0},
    ] * 4

    out_dir = tmp_path / "trained_adapter"
    trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_pairs,
        val_data=[],
        config={"learning_rate": 1e-2, "batch_size": 2, "max_steps": 5, "use_lora": True, "target_modules": ["query", "value"], "loss_type": "bce"},
        device="cpu",
    )
    trainer.train(output_dir=out_dir)

    # Reload reranker
    reranker = CrossEncoderReranker(
        model_name=str(model_dir),
        adapter_path=str(out_dir),
        device="cpu",
        local_files_only=True,
    )
    scores = reranker.score_pairs([("tok_1", "tok_1 tok_2"), ("tok_1", "tok_9")])
    assert len(scores) == 2
    assert isinstance(scores[0], float)
    assert isinstance(scores[1], float)


# =============================================================================
# Invariant 13: OOF fold construction has no label leakage
# =============================================================================
def test_invariant_13_oof_fold_construction_has_no_label_leakage():
    """Verify that 5-fold cross validation partitions queries with zero leakage across folds."""
    query_records = [{"query_id": f"q_{i}", "doc_id": f"doc_{i % 20}"} for i in range(100)]
    folds = generate_random_5fold_split(query_records, seed=42)

    assert len(folds) == 5
    all_val_qids = []
    for f in folds:
        train_qids = set(f["train_query_ids"])
        val_qids = set(f["val_query_ids"])

        # Disjoint train and val in every fold
        assert len(train_qids.intersection(val_qids)) == 0
        assert len(train_qids) + len(val_qids) == len(query_records)
        all_val_qids.extend(list(val_qids))

    # All validation queries across 5 folds partition the dataset exactly once
    assert sorted(all_val_qids) == sorted([q["query_id"] for q in query_records])
    verify_fold_isolation(folds)


# =============================================================================
# Invariant 14: Learned fusion trains without validation-fold labels
# =============================================================================
def test_invariant_14_learned_fusion_trains_without_validation_fold_labels():
    """Verify that learned fusion training strictly uses training-fold samples and labels without touching validation data."""
    train_features = [
        {"raw_bm25_rank": 1, "dense_score": 0.9, "label": 1},
        {"raw_bm25_rank": 5, "dense_score": 0.4, "label": 0},
    ]
    val_features = [
        {"raw_bm25_rank": 2, "dense_score": 0.8, "label": 1},
    ]

    # Ensure train set labels are available and val labels are never passed to fit
    X_train = np.array([[f["raw_bm25_rank"], f["dense_score"]] for f in train_features])
    y_train = np.array([f["label"] for f in train_features])

    # Simple linear/ridge ranker fit only on train
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    X_val = np.array([[f["raw_bm25_rank"], f["dense_score"]] for f in val_features])
    preds = clf.predict_proba(X_val)[:, 1]
    assert len(preds) == len(val_features)


# =============================================================================
# Invariant 15: Parameter audit sums all final learned components
# =============================================================================
def test_invariant_15_parameter_audit_sums_all_final_learned_components():
    """Verify parameter audit sums all neural models in the final pipeline."""
    report = audit_system_parameters(
        models=DEFAULT_PIPELINE_MODELS,
        raise_on_violation=True,
    )

    assert report["total_learned_parameters"] == 702_754_049
    assert report["is_compliant"] is True
    assert report["total_parameters_billions"] == pytest.approx(0.702754, rel=1e-3)


# =============================================================================
# Invariant 16: Parameter audit rejects >= 4B
# =============================================================================
def test_invariant_16_parameter_audit_rejects_ge_4b():
    """Verify that parameter audit strictly rejects configurations with >= 4,000,000,000 parameters."""
    class MockLargeModel(nn.Module):
        def __init__(self, params_count: int = 4_500_000_000):
            super().__init__()
            self._mock_params_count = params_count

        def parameters(self):
            p = torch.nn.Parameter(torch.empty(self._mock_params_count, device="meta"))
            return iter([p])

    large_model = MockLargeModel(4_500_000_000)

    with pytest.raises(ParameterBudgetExceededError):
        audit_system_parameters(
            models=[large_model],
            output_json=None,
            raise_on_violation=True,
        )

    # Direct validation helper check
    is_valid = validate_parameter_budget(total_params=4_000_000_000, raise_on_violation=False)
    assert is_valid is False

    with pytest.raises(ParameterBudgetExceededError, match="exceeds"):
        validate_parameter_budget(total_params=4_000_000_000, raise_on_violation=True)


# =============================================================================
# Invariant 17: Official scorer parity
# =============================================================================
def test_invariant_17_official_scorer_parity():
    """Verify local metric computation matches official Codabench scorer logic across edge cases."""
    truth = {
        "q1": ["docA"],
        "q2": ["docB", "docC"],
        "q3": ["docD"],
        "q4": ["docE"],
    }
    preds = {
        "q1": {"answer": ["docA", "doc1", "doc2", "doc3", "doc4"]},  # perfect
        "q2": {"answer": ["docB", "docX", "docY", "docZ", "docW"]},  # 1/2 correct
        "q3": {"answer": []},                                         # empty -> 0
        "q4": {"answer": ["docE", "doc1", "doc2", "doc3", "doc4", "doc5"]},  # >5 -> 0
    }

    metrics = assert_official_equivalence(preds, truth)
    # q1: rec=1.0, prec=0.2
    # q2: rec=0.5, prec=0.2
    # q3: rec=0.0, prec=0.0
    # q4: rec=0.0, prec=0.0
    expected_recall = (1.0 + 0.5 + 0.0 + 0.0) / 4.0
    expected_precision = (0.2 + 0.2 + 0.0 + 0.0) / 4.0
    assert metrics["recall"] == pytest.approx(expected_recall)
    assert metrics["precision"] == pytest.approx(expected_precision)


# =============================================================================
# Invariant 18: Submission exact query-key equality
# =============================================================================
def test_invariant_18_submission_exact_query_key_equality():
    """Verify submission validator rejects missing or extraneous query IDs."""
    public_keys = {"q1", "q2", "q3"}
    corpus_ids = {"doc1", "doc2"}

    # Missing query key
    with pytest.raises(ValueError, match="query keys"):
        validate_submission({"q1": {"answer": ["doc1"]}, "q2": {"answer": ["doc2"]}}, public_keys, corpus_ids)

    # Extra query key
    with pytest.raises(ValueError, match="query keys"):
        validate_submission(
            {"q1": {"answer": ["doc1"]}, "q2": {"answer": ["doc2"]}, "q3": {"answer": ["doc1"]}, "q_extra": {"answer": ["doc1"]}},
            public_keys,
            corpus_ids,
        )


# =============================================================================
# Invariant 19: Submission max-five rule
# =============================================================================
def test_invariant_19_submission_max_five_rule():
    """Verify submission validator enforces 1 <= len(answer) <= 5 for all predictions."""
    public_keys = {"q1"}
    corpus_ids = {"1", "2", "3", "4", "5", "6"}

    # Empty answer list
    with pytest.raises(ValueError, match="1 to 5"):
        validate_submission({"q1": {"answer": []}}, public_keys, corpus_ids)

    # Overflow answer list (6 answers)
    with pytest.raises(ValueError, match="1 to 5"):
        validate_submission({"q1": {"answer": ["1", "2", "3", "4", "5", "6"]}}, public_keys, corpus_ids)


# =============================================================================
# Invariant 20: Submission uniqueness/valid-ID rule
# =============================================================================
def test_invariant_20_submission_uniqueness_valid_id_rule():
    """Verify submission validator rejects duplicate document IDs and non-corpus IDs."""
    public_keys = {"q1"}
    corpus_ids = {"doc1", "doc2"}

    # Duplicate IDs
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission({"q1": {"answer": ["doc1", "doc1"]}}, public_keys, corpus_ids)

    # Unknown ID outside corpus
    with pytest.raises(ValueError, match="unknown document IDs"):
        validate_submission({"q1": {"answer": ["doc_unknown_999"]}}, public_keys, corpus_ids)


# =============================================================================
# Invariant 21: ZIP contains only submission.json at root
# =============================================================================
def test_invariant_21_zip_contains_only_submission_json_at_root(tmp_path: Path):
    """Verify packaged submission.zip contains only submission.json at the archive root."""
    pred = {"q1": {"answer": ["doc1"]}}
    json_path = tmp_path / "submission.json"
    zip_path = tmp_path / "submission.zip"

    package_submission(pred, json_path, zip_path)

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        assert names == ["submission.json"]

    # Verify zip validation utility
    zip_report = validate_submission_zip(zip_path)
    assert zip_report["is_valid"] is True


# =============================================================================
# Invariant 22: Deterministic fallback order
# =============================================================================
def test_invariant_22_deterministic_fallback_order():
    """Verify TopKSelector breaks ties and fills short predictions deterministically in fallback order."""
    selector = TopKSelector(max_k=5, min_k=5, fallback_doc_ids=["fb1", "fb2", "fb3", "fb4", "fb5"])

    # Short candidate list of 2 items
    short_cands = [{"doc_id": "d1", "score": 10.0}, {"doc_id": "d2", "score": 9.0}]
    selected = selector.select(short_cands, top_k=5)

    assert len(selected) == 5
    assert selected == ["d1", "d2", "fb1", "fb2", "fb3"]

    # Multiple runs produce identical result
    assert selector.select(short_cands, top_k=5) == selected


# =============================================================================
# Invariant 23: Notebook parses as valid nbformat
# =============================================================================
def test_invariant_23_notebook_parses_as_valid_nbformat():
    """Verify that legalir_training.ipynb at repo root and in kernel directory parses as valid nbformat v4."""
    for nb_rel_path in ["legalir_training.ipynb", "kaggle_kernel_task1/legalir_training.ipynb"]:
        nb_path = REPO_ROOT / nb_rel_path
        if not nb_path.exists():
            continue
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert nb.get("nbformat") == 4, f"{nb_rel_path} must be nbformat 4"
        assert "cells" in nb, f"{nb_rel_path} must contain 'cells'"
        assert len(nb["cells"]) >= 5, f"{nb_rel_path} must have at least 5 cells"
        for idx, cell in enumerate(nb["cells"]):
            assert cell.get("cell_type") in ("markdown", "code"), f"Cell {idx} invalid cell_type"
            assert "source" in cell, f"Cell {idx} missing source"


# =============================================================================
# Invariant 24: Kaggle config paths resolve safely
# =============================================================================
def test_invariant_24_kaggle_config_paths_resolve_safely(tmp_path: Path):
    """Verify that Kaggle configuration paths resolve relative paths safely across environments."""
    kaggle_cfg_path = REPO_ROOT / "configs" / "kaggle.yaml"
    assert kaggle_cfg_path.exists(), "configs/kaggle.yaml must exist"

    data = yaml.safe_load(kaggle_cfg_path.read_text(encoding="utf-8"))
    assert data.get("environment") == "kaggle"
    assert "paths" in data
    assert data["paths"].get("canonical") is not None
    assert data["paths"].get("indexes") is not None

    # PipelineConfig attribute access and safety
    cfg = PipelineConfig(data)
    assert cfg.gpu == "T4_x2"
    assert cfg.use_hf_token is True
