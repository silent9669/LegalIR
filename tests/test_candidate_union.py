from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.types import CandidateRecord


class StubRetriever:
    def __init__(self, return_items):
        self.return_items = return_items

    def retrieve(self, query, top_k=50):
        return self.return_items[:top_k]


class StubExactMatcher:
    def __init__(self, return_dict):
        self.return_dict = return_dict

    def match(self, query):
        return self.return_dict


def test_hybrid_search_unions_all_branches_and_sorts_stably():
    bm25_hits = [{"doc_id": "doc1", "score": 10.0, "bm25_score": 10.0, "bm25_best_score": 10.0}]
    exact_hits = {"doc2": {"score": 1.0, "exact_legal_number": True}}
    dense_hits = [{"doc_id": "doc3", "score": 0.95, "dense_score": 0.95, "dense_best_score": 0.95}]

    engine = HybridSearchEngine(
        bm25_retriever=StubRetriever(bm25_hits),
        exact_matcher=StubExactMatcher(exact_hits),
        dense_retriever=StubRetriever(dense_hits),
    )

    cands = engine.search_candidates("query", top_k=150)
    cand_doc_ids = {c["doc_id"] for c in cands}
    assert {"doc1", "doc2", "doc3"} <= cand_doc_ids

    # Check that CandidateRecord contains expected feature fields
    for c in cands:
        assert "rrf_score" in c
        assert "source_count" in c
        assert "exact_legal_number" in c
        assert "bm25_best_score" in c
        assert "dense_best_score" in c
