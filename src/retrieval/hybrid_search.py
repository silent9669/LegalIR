"""Candidate retrieval and reciprocal-rank fusion across LegalIR branches."""

from collections.abc import Mapping
import inspect
from typing import Any

from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.question_memory import QuestionMemory, TrainQuestionMemory
from src.retrieval.types import CandidateRecord


class HybridSearchEngine:
    """Unify lexical, dense, memory, and exact candidate retrieval.

    Each branch returns a ranked list of document records.  The engine removes
    duplicate document IDs within a branch, unions the four branch outputs, and
    ranks the union with weighted reciprocal-rank fusion.
    """

    DEFAULT_BRANCH_WEIGHTS = {
        "bm25": 1.0,
        "dense": 1.2,
        "memory": 2.0,
        "exact": 2.5,
    }

    def __init__(
        self,
        bm25_retriever: BM25MicroRetriever | None = None,
        exact_matcher: ExactMatcher | None = None,
        question_memory: QuestionMemory | TrainQuestionMemory | None = None,
        dense_retriever: DenseMacroRetriever | None = None,
        *,
        bm25_engine: Any | None = None,
        dense_macro: Any | None = None,
        memory: Any | None = None,
        branch_weights: Mapping[str, float] | None = None,
    ):
        """Create a hybrid engine.

        The explicit ``*_retriever`` names are the public API used by the
        pipeline.  The aliases keep the branch names from the design spec
        usable by callers that construct the engine directly.
        """
        self.bm25 = bm25_retriever if bm25_retriever is not None else bm25_engine
        self.exact = exact_matcher
        self.memory = question_memory if question_memory is not None else memory
        self.dense = dense_retriever if dense_retriever is not None else dense_macro

        self.branch_weights = dict(self.DEFAULT_BRANCH_WEIGHTS)
        if branch_weights is not None:
            unknown = set(branch_weights) - set(self.branch_weights)
            if unknown:
                raise ValueError(f"unknown branch weights: {sorted(unknown)}")
            for branch, weight in branch_weights.items():
                if weight < 0:
                    raise ValueError("branch weights must be non-negative")
                self.branch_weights[branch] = float(weight)

    def search(
        self,
        query: str,
        top_k_candidates: int = 100,
        rrf_k: int = 60,
        exclude_qid: str | None = None,
    ) -> list[CandidateRecord]:
        """Retrieve and fuse up to ``top_k_candidates`` unique documents.

        ``rrf_k`` is the standard RRF smoothing constant.  A branch's rank is
        one-based and contributes ``weight / (rrf_k + rank)`` to its document.
        Branch-specific details and each contribution are retained in the
        returned candidate record for downstream ranking and inspection.
        """
        if top_k_candidates <= 0 or not query:
            return []
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")

        branch_results: dict[str, tuple[dict[str, int], dict[str, dict[str, Any]]]] = {}

        if self.bm25 is not None:
            branch_results["bm25"] = self._rank_branch(
                self._retrieve(self.bm25, query, top_k_candidates), "bm25"
            )
        if self.dense is not None:
            branch_results["dense"] = self._rank_branch(
                self._retrieve(self.dense, query, top_k_candidates), "dense"
            )
        if self.memory is not None:
            branch_results["memory"] = self._rank_branch(
                self._retrieve_memory(
                    self.memory,
                    query,
                    top_k_candidates,
                    exclude_qid=exclude_qid,
                ),
                "memory",
            )
        if self.exact is not None:
            branch_results["exact"] = self._rank_branch(
                self._match_exact(self.exact, query), "exact"
            )

        return self._fuse(
            branch_results,
            top_k_candidates,
            rrf_k,
            branch_weights=self.branch_weights,
        )

    def search_candidates(
        self,
        query: str,
        exclude_qid: str | None = None,
        top_k: int = 50,
        rrf_k: int = 60,
        *,
        top_k_candidates: int | None = None,
    ) -> list[CandidateRecord]:
        """Backward-compatible wrapper around :meth:`search`."""
        if top_k_candidates is not None:
            top_k = top_k_candidates
        return self.search(
            query=query,
            top_k_candidates=top_k,
            rrf_k=rrf_k,
            exclude_qid=exclude_qid,
        )

    @staticmethod
    def _method(obj: Any, names: tuple[str, ...]) -> Any:
        for name in names:
            method = getattr(obj, name, None)
            if callable(method):
                return method
        if callable(obj):
            return obj
        raise TypeError(f"retriever must expose one of {', '.join(names)}")

    @staticmethod
    def _invoke(method: Any, query: str, top_k: int, exclude_qid: str | None = None) -> Any:
        """Call branch methods while supporting legacy and new signatures."""
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {}
        if "top_k" in parameters or accepts_kwargs:
            kwargs["top_k"] = top_k
        elif "k" in parameters:
            kwargs["k"] = top_k
        if exclude_qid is not None and (
            "exclude_qid" in parameters or accepts_kwargs
        ):
            kwargs["exclude_qid"] = exclude_qid

        if parameters:
            return method(query, **kwargs)
        try:
            return method(query, top_k=top_k)
        except TypeError:
            return method(query)

    @classmethod
    def _retrieve(cls, retriever: Any, query: str, top_k: int) -> Any:
        method = cls._method(retriever, ("retrieve", "search", "query"))
        return cls._invoke(method, query, top_k)

    @classmethod
    def _retrieve_memory(
        cls,
        memory: Any,
        query: str,
        top_k: int,
        exclude_qid: str | None,
    ) -> Any:
        method = cls._method(memory, ("query", "search", "retrieve"))
        return cls._invoke(method, query, top_k, exclude_qid=exclude_qid)

    @classmethod
    def _match_exact(cls, matcher: Any, query: str) -> Any:
        method = cls._method(matcher, ("match", "search", "retrieve"))
        try:
            return method(query)
        except TypeError:
            return cls._invoke(method, query, 0)

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _score(cls, info: Mapping[str, Any], branch: str) -> float:
        for key in (f"{branch}_score", "exact_score" if branch == "exact" else "", "score"):
            if key and key in info:
                return cls._as_float(info[key])
        return 0.0

    @classmethod
    def _normalise_item(cls, item: Any, branch: str) -> tuple[str, dict[str, Any]] | None:
        if isinstance(item, Mapping):
            doc_id = item.get("doc_id", item.get("document_id"))
            if doc_id is None:
                return None
            info = dict(item)
            info["doc_id"] = str(doc_id)
            return str(doc_id), info

        if isinstance(item, (tuple, list)) and item:
            doc_id = item[0]
            if doc_id is None:
                return None
            score = cls._as_float(item[1]) if len(item) > 1 else 0.0
            return str(doc_id), {
                "doc_id": str(doc_id),
                "score": score,
                f"{branch}_score": score,
            }

        if item is None:
            return None
        return str(item), {"doc_id": str(item), "score": 0.0}

    @classmethod
    def _rank_branch(
        cls,
        raw_results: Any,
        branch: str,
    ) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
        """Normalize a branch result and assign unique one-based ranks."""
        if raw_results is None:
            return {}, {}

        if isinstance(raw_results, Mapping):
            if "doc_id" in raw_results or "document_id" in raw_results:
                items: list[Any] = [raw_results]
            else:
                items = [
                    {"doc_id": doc_id, **(dict(value) if isinstance(value, Mapping) else {"score": value})}
                    for doc_id, value in raw_results.items()
                ]
                items.sort(
                    key=lambda item: (-cls._score(item, branch), str(item["doc_id"]))
                )
        else:
            try:
                items = list(raw_results)
            except TypeError:
                items = [raw_results]

        ranks: dict[str, int] = {}
        details: dict[str, dict[str, Any]] = {}
        for item in items:
            normalized = cls._normalise_item(item, branch)
            if normalized is None:
                continue
            doc_id, info = normalized
            if doc_id in ranks:
                continue
            ranks[doc_id] = len(ranks) + 1
            details[doc_id] = info
        return ranks, details

    @classmethod
    def _fuse(
        cls,
        branch_results: dict[str, tuple[dict[str, int], dict[str, dict[str, Any]]]],
        top_k_candidates: int,
        rrf_k: int,
        branch_weights: Mapping[str, float] | None = None,
    ) -> list[CandidateRecord]:
        weights = branch_weights or cls.DEFAULT_BRANCH_WEIGHTS
        doc_ids = {
            doc_id
            for ranks, _ in branch_results.values()
            for doc_id in ranks
        }
        records: list[CandidateRecord] = []

        for doc_id in doc_ids:
            branch_contributions: dict[str, float] = {}
            branch_ranks: dict[str, int] = {}
            branch_metadata: dict[str, dict[str, Any]] = {}
            rrf_score = 0.0

            for branch in ("bm25", "dense", "memory", "exact"):
                ranks, details = branch_results.get(branch, ({}, {}))
                rank = ranks.get(doc_id)
                if rank is None:
                    continue
                contribution = float(weights[branch]) / (rrf_k + rank)
                branch_ranks[branch] = rank
                branch_contributions[branch] = contribution
                branch_metadata[branch] = details[doc_id]
                rrf_score += contribution

            bm25_info = branch_metadata.get("bm25", {})
            dense_info = branch_metadata.get("dense", {})
            memory_info = branch_metadata.get("memory", {})
            exact_info = branch_metadata.get("exact", {})
            exact_score = cls._score(exact_info, "exact")

            record: CandidateRecord = {
                "doc_id": doc_id,
                "rrf_score": rrf_score,
                "source_count": len(branch_ranks),
                "branch_ranks": branch_ranks,
                "branch_contributions": branch_contributions,
                "branch_metadata": branch_metadata,
                "bm25_rank": branch_ranks.get("bm25"),
                "bm25_score": cls._score(bm25_info, "bm25"),
                "bm25_best_score": cls._as_float(bm25_info.get("bm25_best_score", 0.0)),
                "bm25_second_score": cls._as_float(bm25_info.get("bm25_second_score", 0.0)),
                "bm25_mean_score": cls._as_float(bm25_info.get("bm25_mean_score", 0.0)),
                "bm25_best_chunk_id": bm25_info.get("bm25_best_chunk_id"),
                "exact_score": exact_score,
                "exact_match_score": exact_score,
                "exact_legal_number": bool(exact_info.get("exact_legal_number", False)),
                "exact_title": bool(exact_info.get("exact_title", False)),
                "exact_year": bool(exact_info.get("exact_year", False)),
                "exact_doc_type": bool(exact_info.get("exact_doc_type", False)),
                "memory_rank": branch_ranks.get("memory"),
                "memory_score": cls._score(memory_info, "memory"),
                "memory_lexical_similarity": cls._as_float(
                    memory_info.get("lexical_similarity", memory_info.get("score", 0.0))
                ),
                "memory_dense_similarity": cls._as_float(memory_info.get("dense_similarity", 0.0)),
                "memory_vote_count": int(cls._as_float(memory_info.get("vote_count", 0))),
                "dense_rank": branch_ranks.get("dense"),
                "dense_score": cls._score(dense_info, "dense"),
                "dense_best_score": cls._as_float(dense_info.get("dense_best_score", 0.0)),
                "dense_second_score": cls._as_float(dense_info.get("dense_second_score", 0.0)),
                "dense_best_chunk_id": dense_info.get("dense_best_chunk_id"),
            }
            records.append(record)

        records.sort(key=lambda candidate: (-candidate["rrf_score"], candidate["doc_id"]))
        return records[:top_k_candidates]
