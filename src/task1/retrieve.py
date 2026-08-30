import re
from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.rrf import reciprocal_rank_fusion
from src.task1.memory import QuestionMemory
from src.common.normalize import extract_legal_signals

class LegalMatcher:
    def __init__(self, doc_index: dict = None):
        self.doc_index = doc_index or {}

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        signals = extract_legal_signals(query)
        hits = []
        for d in signals.get("doc_numbers", []):
            for doc_id, doc in self.doc_index.items():
                if d in doc.get("legal_number", "") or d in doc.get("name_raw", ""):
                    hits.append({
                        "doc_id": str(doc_id),
                        "score": 100.0,
                        "branch": "exact"
                    })
        return hits[:top_k]

class CandidateRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DEk21Retriever, memory: QuestionMemory, exact: LegalMatcher = None):
        self.bm25 = bm25
        self.dense = dense
        self.memory = memory
        self.exact = exact or LegalMatcher()

    def retrieve_candidates(self, query: str, top_k: int = 60, rrf_k: int = 60, weights: dict = None) -> list[dict]:
        weights = weights or {"bm25": 1.0, "dense": 1.2, "memory": 2.0, "exact": 2.5}

        runs = []
        w_list = []

        if self.bm25 is not None:
            bm25_hits = self.bm25.search(query, top_k=top_k)
            if bm25_hits:
                runs.append(bm25_hits)
                w_list.append(weights.get("bm25", 1.0))

        if self.dense is not None:
            dense_hits = self.dense.search(query, top_k=top_k)
            if dense_hits:
                runs.append(dense_hits)
                w_list.append(weights.get("dense", 1.2))

        if self.memory is not None:
            mem_hits = self.memory.search(query, top_k=10)
            if mem_hits:
                runs.append(mem_hits)
                w_list.append(weights.get("memory", 2.0))

        if self.exact is not None:
            exact_hits = self.exact.search(query, top_k=10)
            if exact_hits:
                runs.append(exact_hits)
                w_list.append(weights.get("exact", 2.5))

        if not runs:
            return []

        fused = reciprocal_rank_fusion(runs, k=rrf_k, weights=w_list, key="doc_id")
        return fused[:top_k]
