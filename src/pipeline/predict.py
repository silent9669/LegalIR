from collections.abc import Iterable
from typing import Any

from tqdm import tqdm

from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import LightGBMRanker, ReciprocalRankFusion
from src.ranking.reranker import CrossEncoderReranker
from src.ranking.selector import TopKSelector
from src.retrieval.hybrid_search import HybridSearchEngine


class LegalIRPipeline:
    """Run the four-branch retrieval, ranking, and selection pipeline."""

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
        valid_doc_ids: Iterable[str] | None = None,
    ):
        self.hybrid_engine = hybrid_engine
        self.evidence_builder = evidence_builder
        self.reranker = reranker
        self.ranker = ranker or ReciprocalRankFusion()
        self.selector = selector or TopKSelector(max_k=5)
        self.candidate_k = int(candidate_k)
        self.rerank_k = int(rerank_k)
        self.valid_doc_ids = (
            {str(doc_id) for doc_id in valid_doc_ids}
            if valid_doc_ids is not None
            else None
        )

        configured_fallbacks = fallback_doc_ids or ["2113", "58389", "84570"]
        self.fallback_doc_ids = self._unique_ids(configured_fallbacks)
        if self.valid_doc_ids is not None:
            self.fallback_doc_ids = [
                doc_id
                for doc_id in self.fallback_doc_ids
                if doc_id in self.valid_doc_ids
            ]

    @staticmethod
    def _unique_ids(doc_ids: Iterable[Any]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for doc_id in doc_ids:
            if doc_id is None:
                continue
            normalized = str(doc_id).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _fallback_answer(self) -> list[str]:
        """Return a deterministic, duplicate-free answer for empty retrieval."""
        return self.fallback_doc_ids[: self.selector.max_k]

    def _sanitize_answer(self, candidates: Iterable[Any]) -> list[str]:
        answer = self._unique_ids(candidates)
        if self.valid_doc_ids is not None:
            answer = [doc_id for doc_id in answer if doc_id in self.valid_doc_ids]
        answer = answer[: self.selector.max_k]

        if len(answer) < self.selector.min_k:
            for doc_id in self._fallback_answer():
                if doc_id not in answer:
                    answer.append(doc_id)
                if len(answer) == self.selector.max_k:
                    break
        return answer

    def predict_one(self, query_id: str, question: str | None) -> list[str]:
        """Run all configured retrieval/ranking stages for one query."""
        question = "" if question is None else str(question)
        if not question.strip():
            return self._fallback_answer()

        candidates = self.hybrid_engine.search_candidates(
            query=question,
            exclude_qid=str(query_id) if query_id else None,
            top_k=self.candidate_k,
        )
        if not candidates:
            return self._fallback_answer()

        if self.reranker is not None:
            candidates = self.reranker.rerank(
                query=question,
                candidates=candidates,
                evidence_builder=self.evidence_builder,
                top_k=self.rerank_k,
            )

        if hasattr(self.ranker, "predict"):
            ranked = self.ranker.predict(candidates)
        elif hasattr(self.ranker, "rank_candidates"):
            ranked = self.ranker.rank_candidates(candidates)
        else:
            ranked = candidates

        selected = self.selector.select(ranked)
        return self._sanitize_answer(selected)

    def predict_batch(
        self,
        queries: dict[str, str],
        show_progress: bool = False,
    ) -> dict[str, dict[str, list[str]]]:
        """Return ``{query_id: {"answer": [document_id, ...]}}``."""
        results: dict[str, dict[str, list[str]]] = {}
        iterator = sorted(
            queries.items(),
            key=lambda item: (
                0,
                int(item[0]),
            )
            if str(item[0]).isdigit()
            else (1, str(item[0])),
        )
        if show_progress:
            iterator = tqdm(iterator, desc="Generating predictions")

        for query_id, question in iterator:
            answer = self.predict_one(str(query_id), question)
            results[str(query_id)] = {"answer": answer}

        return results
