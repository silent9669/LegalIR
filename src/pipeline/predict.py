from typing import Any
from tqdm import tqdm
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import ReciprocalRankFusion, LightGBMRanker
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.selector import TopKSelector
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.types import CandidateRecord


class LegalIRPipeline:
    def __init__(
        self,
        hybrid_engine: HybridSearchEngine,
        evidence_builder: EvidencePackBuilder,
        reranker: CrossEncoderReranker | None = None,
        ranker: ReciprocalRankFusion | LightGBMRanker | None = None,
        selector: TopKSelector | None = None,
        candidate_k: int = 150,
        rerank_k: int = 50,
        fallback_doc_ids: list[str] | None = None,
    ):
        self.hybrid_engine = hybrid_engine
        self.evidence_builder = evidence_builder
        self.reranker = reranker
        self.ranker = ranker or ReciprocalRankFusion()
        self.selector = selector or TopKSelector(max_k=5)
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.fallback_doc_ids = fallback_doc_ids or ["2113", "58389", "84570"]

    def predict_one(self, query_id: str, question: str) -> list[str]:
        """Runs full multi-stage retrieval & ranking for one query."""
        if not question:
            return self.fallback_doc_ids[:5]

        cands = self.hybrid_engine.search_candidates(
            query=question,
            exclude_qid=str(query_id) if query_id else None,
            top_k=self.candidate_k,
        )

        if not cands:
            return self.fallback_doc_ids[:5]

        # Stage 2: Reranking (if enabled)
        if self.reranker is not None:
            cands = self.reranker.rerank(
                query=question,
                candidates=cands,
                evidence_builder=self.evidence_builder,
                top_k=self.rerank_k,
            )

        # Stage 3: Fusion ranking
        if hasattr(self.ranker, "predict"):
            ranked = self.ranker.predict(cands)
        elif hasattr(self.ranker, "rank_candidates"):
            ranked = self.ranker.rank_candidates(cands)
        else:
            ranked = cands

        # Stage 4: Top-5 selection
        top5 = self.selector.select(ranked)
        if not top5:
            top5 = self.fallback_doc_ids[:5]

        return [str(x) for x in top5]

    def predict_batch(self, queries: dict[str, str], show_progress: bool = False) -> dict[str, dict[str, list[str]]]:
        """
        Takes {query_id: question_text}
        Returns {query_id: {"answer": [doc_id_1, ...]}}
        """
        results = {}
        iterator = sorted(queries.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        if show_progress:
            iterator = tqdm(iterator, desc="Generating predictions")

        for qid, q_text in iterator:
            ans = self.predict_one(str(qid), q_text)
            results[str(qid)] = {"answer": ans}

        return results
