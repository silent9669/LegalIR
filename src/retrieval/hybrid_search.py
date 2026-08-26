from collections import defaultdict
from typing import Any
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.question_memory import QuestionMemory
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.types import CandidateRecord


class HybridSearchEngine:
    def __init__(
        self,
        bm25_retriever: BM25MicroRetriever | None = None,
        exact_matcher: ExactMatcher | None = None,
        question_memory: QuestionMemory | None = None,
        dense_retriever: DenseMacroRetriever | None = None,
    ):
        self.bm25 = bm25_retriever
        self.exact = exact_matcher
        self.memory = question_memory
        self.dense = dense_retriever

    def search_candidates(
        self,
        query: str,
        exclude_qid: str | None = None,
        top_k: int = 50,
        rrf_k: int = 60,
    ) -> list[CandidateRecord]:
        all_candidates: set[str] = set()
        branch_ranks: dict[str, dict[str, int]] = {}
        branch_details: dict[str, dict[str, Any]] = defaultdict(dict)

        # 1. BM25 on micro chunks
        if self.bm25:
            bm25_res = self.bm25.retrieve(query, top_k=top_k * 2)
            # Handle both list of dicts and list of tuples
            bm25_dict = {}
            for r, item in enumerate(bm25_res, 1):
                if isinstance(item, dict):
                    did = str(item["doc_id"])
                    bm25_dict[did] = item
                else:
                    did = str(item[0])
                    bm25_dict[did] = {"doc_id": did, "score": float(item[1]), "bm25_score": float(item[1])}

            branch_ranks["bm25"] = {did: r for r, did in enumerate(bm25_dict.keys(), 1)}
            branch_details["bm25"] = bm25_dict
            all_candidates.update(list(bm25_dict.keys())[:top_k])

        # 2. Exact Matcher
        if self.exact:
            exact_res = self.exact.match(query)
            # Sort by match confidence
            sorted_exact = sorted(
                exact_res.items(),
                key=lambda x: x[1]["score"] if isinstance(x[1], dict) else float(x[1]),
                reverse=True,
            )
            exact_dict = {}
            for r, (did, val) in enumerate(sorted_exact, 1):
                if isinstance(val, dict):
                    exact_dict[did] = val
                else:
                    exact_dict[did] = {"score": float(val), "exact_score": float(val)}

            branch_ranks["exact"] = {did: r for r, (did, _) in enumerate(sorted_exact, 1)}
            branch_details["exact"] = exact_dict
            all_candidates.update(list(exact_dict.keys())[:15])

        # 3. Question Memory
        if self.memory:
            mem_res = self.memory.retrieve(query, top_k=5, exclude_qid=exclude_qid)
            sorted_mem = sorted(mem_res.items(), key=lambda x: x[1]["score"], reverse=True)
            branch_ranks["memory"] = {did: r for r, (did, _) in enumerate(sorted_mem, 1)}
            branch_details["memory"] = mem_res
            all_candidates.update(list(mem_res.keys())[:10])

        # 4. Dense Macro Retriever (if enabled)
        if self.dense is not None:
            dense_res = self.dense.retrieve(query, top_k=top_k * 2)
            dense_dict = {}
            for r, item in enumerate(dense_res, 1):
                if isinstance(item, dict):
                    did = str(item["doc_id"])
                    dense_dict[did] = item
                else:
                    did = str(item[0])
                    dense_dict[did] = {"doc_id": did, "score": float(item[1]), "dense_score": float(item[1])}

            branch_ranks["dense"] = {did: r for r, did in enumerate(dense_dict.keys(), 1)}
            branch_details["dense"] = dense_dict
            all_candidates.update(list(dense_dict.keys())[:top_k])

        # Reciprocal Rank Fusion (RRF)
        weights = {
            "bm25": 1.0,
            "exact": 2.5,
            "memory": 2.0,
            "dense": 1.2,
        }

        candidate_records: list[CandidateRecord] = []
        for did in all_candidates:
            rrf_score = 0.0
            source_count = 0
            for branch, w in weights.items():
                if branch in branch_ranks and did in branch_ranks[branch]:
                    rank = branch_ranks[branch][did]
                    rrf_score += w / (rrf_k + rank)
                    source_count += 1

            # Exact matching boost
            exact_info = branch_details.get("exact", {}).get(did, {})
            exact_sc = float(exact_info.get("score", 0.0))
            if exact_sc > 0:
                rrf_score += 0.05 * exact_sc

            bm25_info = branch_details.get("bm25", {}).get(did, {})
            mem_info = branch_details.get("memory", {}).get(did, {})
            dense_info = branch_details.get("dense", {}).get(did, {})

            record: CandidateRecord = {
                "doc_id": did,
                "rrf_score": rrf_score,
                "source_count": source_count,
                "bm25_rank": branch_ranks.get("bm25", {}).get(did),
                "bm25_score": float(bm25_info.get("bm25_score", bm25_info.get("score", 0.0))),
                "bm25_best_score": float(bm25_info.get("bm25_best_score", 0.0)),
                "bm25_second_score": float(bm25_info.get("bm25_second_score", 0.0)),
                "bm25_mean_score": float(bm25_info.get("bm25_mean_score", 0.0)),
                "bm25_best_chunk_id": bm25_info.get("bm25_best_chunk_id"),
                "exact_score": exact_sc,
                "exact_match_score": exact_sc,
                "exact_legal_number": bool(exact_info.get("exact_legal_number", False)),
                "exact_title": bool(exact_info.get("exact_title", False)),
                "exact_year": bool(exact_info.get("exact_year", False)),
                "exact_doc_type": bool(exact_info.get("exact_doc_type", False)),
                "memory_rank": branch_ranks.get("memory", {}).get(did),
                "memory_score": float(mem_info.get("score", 0.0)),
                "memory_lexical_similarity": float(mem_info.get("lexical_similarity", mem_info.get("score", 0.0))),
                "memory_dense_similarity": float(mem_info.get("dense_similarity", 0.0)),
                "memory_vote_count": int(mem_info.get("vote_count", 0)),
                "dense_rank": branch_ranks.get("dense", {}).get(did),
                "dense_score": float(dense_info.get("dense_score", dense_info.get("score", 0.0))),
                "dense_best_score": float(dense_info.get("dense_best_score", 0.0)),
                "dense_second_score": float(dense_info.get("dense_second_score", 0.0)),
                "dense_best_chunk_id": dense_info.get("dense_best_chunk_id"),
            }
            candidate_records.append(record)

        # Stable sort by descending rrf_score, ascending doc_id
        candidate_records.sort(key=lambda x: (-x["rrf_score"], x["doc_id"]))
        return candidate_records[:top_k]
