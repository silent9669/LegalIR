from collections import defaultdict
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.question_memory import QuestionMemory
from src.retrieval.dense_macro import DenseMacroRetriever

class HybridSearchEngine:
    def __init__(
        self,
        bm25_retriever: BM25MicroRetriever = None,
        exact_matcher: ExactMatcher = None,
        question_memory: QuestionMemory = None,
        dense_retriever: DenseMacroRetriever = None
    ):
        self.bm25 = bm25_retriever
        self.exact = exact_matcher
        self.memory = question_memory
        self.dense = dense_retriever

    def search_candidates(
        self,
        query: str,
        exclude_qid: str = None,
        top_k: int = 50,
        rrf_k: int = 60
    ) -> list:
        """
        Runs multi-branch retrieval and fuses into a Candidate Pool.
        Returns list of dicts:
        [
            {
                "doc_id": did,
                "rrf_score": score,
                "bm25_rank": int or None,
                "bm25_score": float,
                "exact_match_score": float,
                "memory_score": float,
                "memory_rank": int or None,
                "dense_rank": int or None,
                "dense_score": float
            },
            ...
        ]
        """
        all_candidates = set()
        branch_ranks = {}
        branch_scores = {}

        # 1. BM25 on micro chunks
        if self.bm25:
            bm25_res = self.bm25.retrieve(query, top_k=top_k * 2)
            branch_scores["bm25"] = {did: sc for did, sc in bm25_res}
            branch_ranks["bm25"] = {did: r for r, (did, _) in enumerate(bm25_res, 1)}
            all_candidates.update(list(branch_scores["bm25"].keys())[:top_k])

        # 2. Exact Matcher
        if self.exact:
            exact_res = self.exact.match(query)
            branch_scores["exact"] = exact_res
            sorted_exact = sorted(exact_res.items(), key=lambda x: x[1], reverse=True)
            branch_ranks["exact"] = {did: r for r, (did, _) in enumerate(sorted_exact, 1)}
            all_candidates.update(list(branch_scores["exact"].keys())[:10])

        # 3. Question Memory
        if self.memory:
            mem_res = self.memory.retrieve(query, top_k=5, exclude_qid=exclude_qid)
            mem_scores = {did: v["score"] for did, v in mem_res.items()}
            branch_scores["memory"] = mem_scores
            sorted_mem = sorted(mem_scores.items(), key=lambda x: x[1], reverse=True)
            branch_ranks["memory"] = {did: r for r, (did, _) in enumerate(sorted_mem, 1)}
            all_candidates.update(list(branch_scores["memory"].keys())[:10])

        # 4. Dense Macro Retriever (if enabled)
        if self.dense and self.dense.embeddings is not None:
            dense_res = self.dense.retrieve(query, top_k=top_k * 2)
            branch_scores["dense"] = {did: sc for did, sc in dense_res}
            branch_ranks["dense"] = {did: r for r, (did, _) in enumerate(dense_res, 1)}
            all_candidates.update(list(branch_scores["dense"].keys())[:top_k])

        # Reciprocal Rank Fusion (RRF)
        weights = {
            "bm25": 1.0,
            "exact": 2.5,
            "memory": 2.0,
            "dense": 1.2
        }

        candidate_records = []
        for did in all_candidates:
            rrf_score = 0.0
            for branch, w in weights.items():
                if branch in branch_ranks and did in branch_ranks[branch]:
                    rank = branch_ranks[branch][did]
                    rrf_score += w / (rrf_k + rank)

            # Bonus for exact match
            if "exact" in branch_scores and did in branch_scores["exact"]:
                rrf_score += 0.05 * branch_scores["exact"][did]

            candidate_records.append({
                "doc_id": did,
                "rrf_score": rrf_score,
                "bm25_rank": branch_ranks.get("bm25", {}).get(did),
                "bm25_score": branch_scores.get("bm25", {}).get(did, 0.0),
                "exact_match_score": branch_scores.get("exact", {}).get(did, 0.0),
                "memory_rank": branch_ranks.get("memory", {}).get(did),
                "memory_score": branch_scores.get("memory", {}).get(did, 0.0),
                "dense_rank": branch_ranks.get("dense", {}).get(did),
                "dense_score": branch_scores.get("dense", {}).get(did, 0.0),
            })

        candidate_records.sort(key=lambda x: x["rrf_score"], reverse=True)
        return candidate_records[:top_k]
