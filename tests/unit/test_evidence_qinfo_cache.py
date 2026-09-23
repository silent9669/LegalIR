"""Output-equivalence tests for the label-free query-info cache.

The cache keys on raw query text only (no qrels, no fold state), so reuse
across a query's ~200 candidates — and across folds — cannot leak labels.
Every test asserts identical packs with and without the cache.
"""
from __future__ import annotations

from src.ranking.evidence_pack import EvidencePackBuilder


def _chunks():
    return [
        {"chunk_id": "D1_C0", "doc_id": "D1", "granularity": "macro",
         "article": "Điều 61", "clause": "Khoản 2", "point": "",
         "text_raw": "Điều 61 về hợp đồng lao động và bồi thường thiệt hại tài sản. " * 3,
         "text_norm": ""},
        {"chunk_id": "D1_C1", "doc_id": "D1", "granularity": "macro",
         "article": "Điều 62", "clause": "", "point": "",
         "text_raw": "Điều 62 về quyền lợi bảo hiểm và thời hạn giải quyết. " * 3,
         "text_norm": ""},
        {"chunk_id": "D2_C0", "doc_id": "D2", "granularity": "macro",
         "article": "Điều 61", "clause": "", "point": "",
         "text_raw": "Văn bản khác về hợp đồng và bồi thường theo Điều 61. " * 3,
         "text_norm": ""},
    ]


def test_cached_and_uncached_packs_identical():
    chunks = _chunks()
    cached = EvidencePackBuilder(macro_chunks=chunks)
    nocache = EvidencePackBuilder(macro_chunks=chunks, qinfo_cache_size=0)
    queries = ["Bồi thường theo Điều 61 Khoản 2?", "Quyền lợi bảo hiểm Điều 62?"]
    for q in queries:
        for did in ("D1", "D2"):
            assert cached.build_pack(q, did) == nocache.build_pack(q, did)
            assert cached.build(q, did) == nocache.build(q, did)


def test_repeated_query_hits_cache_without_changing_output():
    b = EvidencePackBuilder(macro_chunks=_chunks())
    q = "Bồi thường theo Điều 61 Khoản 2?"
    first = b.build_pack(q, "D1")
    s1 = b.get_qinfo_cache_stats()
    assert s1["misses"] == 1 and s1["size"] == 1
    for _ in range(5):
        assert b.build_pack(q, "D1") == first
        assert b.build_pack(q, "D2") == b.build_pack(q, "D2")
    s2 = b.get_qinfo_cache_stats()
    assert s2["hits"] > s1["hits"]
    assert s2["misses"] == s1["misses"]  # same query never recomputes


def test_distinct_queries_tracked_separately():
    b = EvidencePackBuilder(macro_chunks=_chunks())
    b.build_pack("query one Điều 61?", "D1")
    b.build_pack("query two Điều 62?", "D1")
    stats = b.get_qinfo_cache_stats()
    assert stats["misses"] == 2 and stats["size"] == 2


def test_cache_bounded_fifo_eviction():
    b = EvidencePackBuilder(macro_chunks=_chunks(), qinfo_cache_size=2)
    for i in range(4):
        b.build_pack(f"unique query {i} Điều 61?", "D1")
    stats = b.get_qinfo_cache_stats()
    assert stats["size"] <= 2
    assert stats["max_size"] == 2


def test_cache_disabled_records_nothing():
    b = EvidencePackBuilder(macro_chunks=_chunks(), qinfo_cache_size=0)
    p1 = b.build_pack("Bồi thường Điều 61?", "D1")
    p2 = b.build_pack("Bồi thường Điều 61?", "D1")
    assert p1 == p2
    assert b.get_qinfo_cache_stats() == {"hits": 0, "misses": 0, "size": 0, "max_size": 0}


def test_clear_resets_counters():
    b = EvidencePackBuilder(macro_chunks=_chunks())
    b.build_pack("q Điều 61?", "D1")
    b.clear_qinfo_cache()
    assert b.get_qinfo_cache_stats()["size"] == 0
    # Output unchanged after clear.
    ref = EvidencePackBuilder(macro_chunks=_chunks(), qinfo_cache_size=0)
    assert b.build_pack("q Điều 61?", "D1") == ref.build_pack("q Điều 61?", "D1")
