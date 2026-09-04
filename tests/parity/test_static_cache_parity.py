import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.static_cache import (
    StaticCacheWriter,
    StaticCacheReader,
    StaticCandidateRecord,
)


@pytest.fixture
def sample_corpus():
    docs = [
        {"doc_id": "doc_1", "title": "Luật Đất đai 2024", "legal_number": "31/2024/QH15"},
        {"doc_id": "doc_2", "title": "Nghị định quy định bồi thường đất", "legal_number": "43/2014/NĐ-CP"},
        {"doc_id": "doc_3", "title": "Thông tư thuế đất", "legal_number": "10/2020/TT-BTC"},
    ]
    chunks = [
        {"chunk_id": "c1", "doc_id": "doc_1", "granularity": "micro", "text_norm": "điều 15 bồi thường tái định cư khi nhà nước thu hồi đất", "article": "Điều 15"},
        {"chunk_id": "c2", "doc_id": "doc_2", "granularity": "micro", "text_norm": "điều 20 trình tự thủ tục bồi thường hỗ trợ đất", "article": "Điều 20"},
        {"chunk_id": "c3", "doc_id": "doc_3", "granularity": "micro", "text_norm": "quy định về mức thuế đất hàng năm", "article": "Điều 1"},
    ]
    return docs, chunks


def test_static_cache_live_branch_parity(tmp_path, sample_corpus):
    docs, chunks = sample_corpus

    bm25 = BM25MicroRetriever()
    bm25.fit(chunks)

    exact = ExactMatcher(documents=docs, chunks=chunks)

    queries = [
        ("q_1", "bồi thường thu hồi đất theo Điều 15"),
        ("q_2", "31/2024/QH15"),
    ]

    cache_file = tmp_path / "cache.parquet"
    writer = StaticCacheWriter(str(cache_file))

    for qid, q_text in queries:
        bm25_res = bm25.retrieve(q_text, top_k=5)
        for rank, r in enumerate(bm25_res, start=1):
            writer.write_record(
                StaticCandidateRecord(
                    query_id=qid,
                    branch="bm25",
                    rank=rank,
                    doc_id=r["doc_id"],
                    score=float(r["score"]),
                    best_chunk_id=r.get("bm25_best_chunk_id"),
                )
            )

        exact_res = exact.search(q_text, top_k=5)
        for rank, r in enumerate(exact_res, start=1):
            writer.write_record(
                StaticCandidateRecord(
                    query_id=qid,
                    branch="exact",
                    rank=rank,
                    doc_id=r["doc_id"],
                    score=float(r.get("score", 1.0)),
                )
            )
    writer.close()

    # Verify parity via reader
    reader = StaticCacheReader(str(cache_file))
    for qid, q_text in queries:
        # Live bm25
        live_bm25 = bm25.retrieve(q_text, top_k=5)
        cached_bm25 = reader.get_query_candidates(qid, branch="bm25")

        assert len(cached_bm25) == len(live_bm25)
        for live_item, cached_item in zip(live_bm25, cached_bm25):
            assert live_item["doc_id"] == cached_item.doc_id
            assert abs(live_item["score"] - cached_item.score) <= 1e-4


def test_static_cache_fusion_parity(tmp_path, sample_corpus):
    docs, chunks = sample_corpus

    bm25 = BM25MicroRetriever()
    bm25.fit(chunks)
    exact = ExactMatcher(documents=docs, chunks=chunks)

    engine = HybridSearchEngine(bm25=bm25, exact=exact)

    query = "Luật đất đai thu hồi bồi thường"
    live_fused = engine.search(query, top_k_candidates=3, rrf_k=60)

    # Cache branches
    cache_file = tmp_path / "fusion_cache.parquet"
    writer = StaticCacheWriter(str(cache_file))

    bm25_res = bm25.retrieve(query, top_k=5)
    for rank, r in enumerate(bm25_res, start=1):
        writer.write_record(
            StaticCandidateRecord(
                query_id="q1",
                branch="bm25",
                rank=rank,
                doc_id=r["doc_id"],
                score=float(r["score"]),
            )
        )

    exact_res = exact.search(query, top_k=5)
    for rank, r in enumerate(exact_res, start=1):
        writer.write_record(
            StaticCandidateRecord(
                query_id="q1",
                branch="exact",
                rank=rank,
                doc_id=r["doc_id"],
                score=float(r.get("score", 1.0)),
            )
        )
    writer.close()

    # Reconstruct fusion from cache
    reader = StaticCacheReader(str(cache_file))
    cands = reader.get_query_candidates("q1")

    # Compute RRF
    rrf_scores = {}
    weights = {"bm25": engine.branch_weights.get("bm25", 1.0), "exact": engine.branch_weights.get("exact", 1.0)}
    for c in cands:
        w = weights.get(c.branch, 1.0)
        rrf_scores[c.doc_id] = rrf_scores.get(c.doc_id, 0.0) + w / (60 + c.rank)

    sorted_cached = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    cached_doc_ids = [d for d, _ in sorted_cached]
    live_doc_ids = [r["doc_id"] if isinstance(r, dict) else r.doc_id for r in live_fused]

    assert cached_doc_ids == live_doc_ids
