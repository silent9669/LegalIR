import json
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from src.evidence.macro_store import MacroEvidenceStore
from src.evidence.pair_materializer import (
    PairMaterializer,
    build_duplicate_closure,
)
from src.retrieval.question_memory import QuestionMemory
from src.retrieval.static_cache import StaticCacheWriter, StaticCandidateRecord, StaticCacheReader


def test_build_duplicate_closure():
    dup_groups = [
        {"doc_ids": ["docA", "docB"]},
        {"doc_ids": ["docC", "docD", "docE"]},
    ]
    closure = build_duplicate_closure(dup_groups)
    assert closure["docA"] == {"docA", "docB"}
    assert closure["docB"] == {"docA", "docB"}
    assert closure["docC"] == {"docC", "docD", "docE"}
    assert "docZ" not in closure


def test_pair_materializer_leakage_assertion():
    train_qids = {"q1", "q2"}
    val_qids = {"q3", "q4"}

    pm = PairMaterializer(
        train_qids=train_qids,
        val_qids=val_qids,
        qrels={"q1": ["docA"], "q2": ["docB"]},
        duplicate_groups=[{"doc_ids": ["docA", "docA_dup"]}],
    )

    # Validating query IDs: train_qid is allowed
    pm.assert_fold_isolation("q1")

    # Validation query ID in train pairs must raise ValueError
    with pytest.raises(ValueError, match="Validation leakage detected"):
        pm.assert_fold_isolation("q3")


def test_duplicate_equivalent_gold_not_negative():
    train_qids = {"q1"}
    val_qids = {"q2"}
    pm = PairMaterializer(
        train_qids=train_qids,
        val_qids=val_qids,
        qrels={"q1": ["docA"]},
        duplicate_groups=[{"doc_ids": ["docA", "docA_dup"]}],
    )
    # Negative checking: docA_dup must be blacklisted for q1 because docA is gold
    assert pm.is_negative_allowed(qid="q1", neg_doc_id="docA_dup") is False
    assert pm.is_negative_allowed(qid="q1", neg_doc_id="docA") is False
    assert pm.is_negative_allowed(qid="q1", neg_doc_id="docX") is True


def test_pair_builder_uses_fold_local_question_memory():
    train_qids = {"q1", "q2"}
    val_qids = {"q3"}

    # Memory built only with train queries
    valid_memory_rows = [
        {"query_id": "q1", "question_norm": "cau hoi 1", "doc_ids": ["docA"]},
        {"query_id": "q2", "question_norm": "cau hoi 2", "doc_ids": ["docB"]},
    ]
    valid_memory = QuestionMemory(valid_memory_rows)

    pm = PairMaterializer(
        train_qids=train_qids,
        val_qids=val_qids,
        qrels={"q1": ["docA"]},
        duplicate_groups=[],
        question_memory=valid_memory,
    )
    assert pm.question_memory is not None

    # Leaking memory containing validation query q3
    leaking_memory_rows = [
        {"query_id": "q3", "question_norm": "cau hoi 3", "doc_ids": ["docC"]},
    ]
    leaking_memory = QuestionMemory(leaking_memory_rows)

    with pytest.raises(ValueError, match="Memory leakage detected"):
        PairMaterializer(
            train_qids=train_qids,
            val_qids=val_qids,
            qrels={"q1": ["docA"]},
            duplicate_groups=[],
            question_memory=leaking_memory,
        )


def test_pair_qids_subset_train_and_disjoint_val(tmp_path):
    train_qids = {"q1", "q2"}
    val_qids = {"q3", "q4"}

    # Mock chunks
    chunks_p = tmp_path / "chunks.parquet"
    tbl = pa.Table.from_pydict({
        "doc_id": ["docA", "docB", "docC"],
        "chunk_id": ["c1", "c2", "c3"],
        "granularity": ["macro", "macro", "macro"],
        "text_raw": ["Điều 1. Nội dung A", "Điều 2. Nội dung B", "Điều 3. Nội dung C"],
        "text_norm": ["điều 1 nội dung a", "điều 2 nội dung b", "điều 3 nội dung c"],
        "article": ["Điều 1", "Điều 2", "Điều 3"],
    })
    pq.write_table(tbl, str(chunks_p))
    store = MacroEvidenceStore(chunks_p)

    # Mock static cache
    cache_p = tmp_path / "cache.parquet"
    writer = StaticCacheWriter(str(cache_p))
    writer.write_record(StaticCandidateRecord("q1", "bm25", 1, "docC", 10.0))
    writer.write_record(StaticCandidateRecord("q2", "bm25", 1, "docC", 10.0))
    writer.write_record(StaticCandidateRecord("q3", "bm25", 1, "docA", 10.0))
    writer.close()
    reader = StaticCacheReader(str(cache_p))

    pm = PairMaterializer(
        train_qids=train_qids,
        val_qids=val_qids,
        qrels={"q1": ["docA"], "q2": ["docB"]},
        duplicate_groups=[],
        evidence_store=store,
    )

    out_pairs = tmp_path / "train_pairs.parquet"
    queries_dict = {"q1": "cau hoi 1", "q2": "cau hoi 2", "q3": "cau hoi 3"}
    rows = pm.materialize_train_pairs(
        output_parquet=out_pairs,
        queries_dict=queries_dict,
        static_cache_reader=reader,
    )
    assert rows > 0

    pairs_df = pd.read_parquet(out_pairs)
    pair_qids = set(pairs_df["query_id"].unique())

    # Hard assertions from P0-6
    assert pair_qids.issubset(train_qids)
    assert pair_qids.isdisjoint(val_qids)


def test_pair_negative_source_policy_matches_legacy(tmp_path):
    train_qids = {"q1"}
    val_qids = {"q2"}

    chunks_p = tmp_path / "chunks.parquet"
    tbl = pa.Table.from_pydict({
        "doc_id": ["docA", "docB", "docC"],
        "chunk_id": ["c1", "c2", "c3"],
        "granularity": ["macro", "macro", "macro"],
        "text_raw": ["A", "B", "C"],
        "text_norm": ["a", "b", "c"],
        "article": ["Điều 1", "Điều 2", "Điều 3"],
    })
    pq.write_table(tbl, str(chunks_p))
    store = MacroEvidenceStore(chunks_p)

    # Static cache with distinct branches
    cache_p = tmp_path / "cache.parquet"
    writer = StaticCacheWriter(str(cache_p))
    writer.write_record(StaticCandidateRecord("q1", "exact", 1, "docB", 10.0))
    writer.write_record(StaticCandidateRecord("q1", "bm25", 1, "docC", 5.0))
    writer.close()
    reader = StaticCacheReader(str(cache_p))

    pm = PairMaterializer(
        train_qids=train_qids,
        val_qids=val_qids,
        qrels={"q1": ["docA"]},
        duplicate_groups=[],
        evidence_store=store,
    )

    out_pairs = tmp_path / "train_pairs.parquet"
    pm.materialize_train_pairs(
        output_parquet=out_pairs,
        queries_dict={"q1": "hoi ve a"},
        static_cache_reader=reader,
    )

    df = pd.read_parquet(out_pairs)
    sources = set(df["negative_source"].unique())
    # Must have "gold" for positive and source branch for negatives
    assert "gold" in sources
    assert any(s in ("exact", "bm25", "hybrid") for s in sources)
