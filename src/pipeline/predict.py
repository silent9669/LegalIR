from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import LightGBMRanker, ReciprocalRankFusion
from src.ranking.reranker import CrossEncoderReranker, DocumentReranker
from src.ranking.selector import TopKSelector
from src.retrieval.hybrid_search import HybridSearchEngine


class LegalIRPipeline:
    """Run the four-branch retrieval, ranking, and selection pipeline."""

    def __init__(
        self,
        hybrid_engine: HybridSearchEngine | None = None,
        evidence_builder: EvidencePackBuilder | None = None,
        reranker: CrossEncoderReranker | DocumentReranker | None = None,
        ranker: ReciprocalRankFusion | LightGBMRanker | None = None,
        selector: TopKSelector | None = None,
        candidate_k: int = 150,
        rerank_k: int = 50,
        fallback_doc_ids: list[str] | None = None,
        valid_doc_ids: Iterable[str] | None = None,
        *,
        retriever: Any | None = None,
    ):
        engine = hybrid_engine if hybrid_engine is not None else retriever
        if engine is None:
            raise ValueError("hybrid_engine or retriever must be provided")
        self.hybrid_engine = engine
        self.evidence_builder = evidence_builder or EvidencePackBuilder()
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
            valid_fallbacks = [
                doc_id
                for doc_id in self.fallback_doc_ids
                if doc_id in self.valid_doc_ids
            ]
            valid_fallbacks.extend(sorted(self.valid_doc_ids))
            self.fallback_doc_ids = self._unique_ids(valid_fallbacks)

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

        if hasattr(self.hybrid_engine, "search_candidates"):
            candidates = self.hybrid_engine.search_candidates(
                query=question,
                exclude_qid=str(query_id) if query_id else None,
                top_k=self.candidate_k,
            )
        elif hasattr(self.hybrid_engine, "retrieve_candidates"):
            candidates = self.hybrid_engine.retrieve_candidates(
                query=question,
                top_k=self.candidate_k,
            )
        elif hasattr(self.hybrid_engine, "search"):
            candidates = self.hybrid_engine.search(
                query=question,
                top_k_candidates=self.candidate_k,
                exclude_qid=str(query_id) if query_id else None,
            )
        else:
            candidates = []

        if not candidates:
            return self._fallback_answer()

        if self.reranker is not None:
            if hasattr(self.reranker, "rerank"):
                candidates = self.reranker.rerank(
                    query=question,
                    candidates=candidates,
                    evidence_builder=self.evidence_builder,
                    top_k=self.rerank_k,
                )
            elif hasattr(self.reranker, "rerank_documents"):
                candidates = self.reranker.rerank_documents(
                    query=question,
                    candidates=candidates,
                    top_k=self.rerank_k,
                )

        if hasattr(self.ranker, "predict"):
            ranked = self.ranker.predict(candidates)
        elif hasattr(self.ranker, "rank_candidates"):
            ranked = self.ranker.rank_candidates(candidates)
        else:
            ranked = candidates

        selected = self.selector.select(ranked, valid_doc_ids=self.valid_doc_ids)
        return self._sanitize_answer(selected)

    def audit_parameters(
        self,
        output_json: str | Path | None = None,
        raise_on_violation: bool = True,
    ) -> dict[str, Any]:
        """Perform a strict parameter audit of all models in this pipeline against the 4B limit."""
        from src.models.parameter_audit import audit_system_parameters

        models_to_audit = []
        if hasattr(self.hybrid_engine, "dense_retriever") and self.hybrid_engine.dense_retriever is not None:
            dense_ret = self.hybrid_engine.dense_retriever
            if hasattr(dense_ret, "model") and dense_ret.model is not None:
                models_to_audit.append((dense_ret.model, getattr(dense_ret, "model_name", "dense_embedding"), "dense_embedding"))
            elif getattr(dense_ret, "model_name", None):
                models_to_audit.append({"name": str(dense_ret.model_name), "role": "dense_embedding"})

        if self.reranker is not None:
            if hasattr(self.reranker, "model") and self.reranker.model is not None:
                models_to_audit.append((self.reranker.model, getattr(self.reranker, "model_name", "cross_encoder_reranker"), "cross_encoder_reranker"))
            elif getattr(self.reranker, "model_name", None):
                models_to_audit.append({"name": str(self.reranker.model_name), "role": "cross_encoder_reranker"})

        if not models_to_audit:
            models_to_audit = None

        return audit_system_parameters(
            models=models_to_audit,
            output_json=output_json,
            raise_on_violation=raise_on_violation,
            offline_fallback=True,
        )

    def predict_single(
        self,
        query: str,
        query_id: str | None = None,
        top_k_candidates: int | None = None,
        top_k_rerank: int | None = None,
    ) -> list[str]:
        """Predict top document IDs for a single query."""
        old_cand_k = self.candidate_k
        old_rerank_k = self.rerank_k
        try:
            if top_k_candidates is not None:
                self.candidate_k = int(top_k_candidates)
            if top_k_rerank is not None:
                self.rerank_k = int(top_k_rerank)
            return self.predict_one(query_id=query_id or "0", question=query)
        finally:
            self.candidate_k = old_cand_k
            self.rerank_k = old_rerank_k

    def predict_batch(
        self,
        queries: dict[str, str] | list[dict[str, Any]],
        show_progress: bool = False,
    ) -> dict[str, dict[str, list[str]]]:
        """Return ``{query_id: {"answer": [document_id, ...]}}``."""
        if isinstance(queries, list):
            query_map = {}
            for item in queries:
                qid = str(item.get("id") or item.get("query_id") or "")
                qtext = item.get("question") or item.get("question_raw") or item.get("question_norm") or ""
                query_map[qid] = qtext
            query_dict = query_map
        else:
            query_dict = queries

        results: dict[str, dict[str, list[str]]] = {}
        iterator = sorted(
            query_dict.items(),
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

    @classmethod
    def load_pipeline(
        cls,
        data_dir: str | Path = "artifacts/task1/data",
        index_dir: str | Path = "artifacts/task1/indexes",
        reranker_adapter_path: str | Path | None = None,
        fusion_model_path: str | Path | None = None,
        use_reranker: bool = True,
        use_learned_fusion: bool | None = None,
        dense_device: str | None = None,
        reranker_device: str | None = None,
        device: str | None = None,
        audit_preflight: bool = False,
        audit_output_json: str | Path | None = None,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        strict_artifacts: bool = False,
    ) -> "LegalIRPipeline":
        """Load fully instantiated pipeline from index and data artifacts."""
        import json
        import pandas as pd
        from src.ranking.evidence_pack import EvidencePackBuilder
        from src.ranking.fusion import LightGBMRanker, ReciprocalRankFusion
        from src.ranking.reranker import CrossEncoderReranker
        from src.retrieval.bm25_micro import BM25MicroRetriever
        from src.retrieval.bm25_pyvi import BM25PyViRetriever
        from src.retrieval.dense_macro import DenseMacroRetriever
        from src.retrieval.exact_matcher import ExactMatcher
        from src.retrieval.question_memory import QuestionMemory, TrainQuestionMemory

        data_dir = Path(data_dir)
        index_dir = Path(index_dir)

        resolved_dense_device = dense_device if dense_device is not None else device
        resolved_reranker_device = reranker_device if reranker_device is not None else device

        docs_path = data_dir / "documents.parquet"
        chunks_path = data_dir / "chunks.parquet"

        doc_map = {}
        valid_doc_ids = set()
        if docs_path.exists():
            target_cols = ["doc_id", "title", "name_raw", "legal_number", "year", "doc_type", "link"]
            try:
                sample_df = pd.read_parquet(docs_path)
                load_cols = [c for c in target_cols if c in sample_df.columns]
                docs_df = sample_df[load_cols]
            except Exception:
                docs_df = pd.read_parquet(docs_path)
            for r in docs_df.to_dict("records"):
                did = str(r["doc_id"])
                doc_map[did] = r
                valid_doc_ids.add(did)

        # 1. BM25 Micro (Legal / Raw)
        bm25_path = index_dir / "bm25"
        if bm25_path.exists():
            bm25 = BM25MicroRetriever.load(bm25_path)
        else:
            bm25 = BM25MicroRetriever()

        # 1b. BM25 PyVi
        bm25_pyvi_path = index_dir / "bm25_pyvi"
        if bm25_pyvi_path.exists():
            bm25_pyvi = BM25PyViRetriever.load(bm25_pyvi_path)
        else:
            bm25_pyvi = None

        # 2. DEk21 Dense Macro
        dense_path = index_dir / "dense_dek21"
        if not dense_path.exists():
            dense_path = index_dir / "dense"
        if dense_path.exists():
            try:
                dense = DenseMacroRetriever.load(dense_path, device=resolved_dense_device)
            except Exception:
                dense = None
        else:
            dense = None

        # 3. Question Memory
        mem_path = index_dir / "question_memory"
        if not mem_path.exists():
            mem_path = data_dir / "question_memory"
        if mem_path.exists():
            memory = TrainQuestionMemory.load(mem_path, dense_retriever=dense)
        else:
            memory = TrainQuestionMemory(dense_device=resolved_dense_device)

        # Strict artifact validation in production mode
        if strict_artifacts:
            if not bm25_path.exists() or len(getattr(bm25, "corpus", [])) == 0:
                raise FileNotFoundError(f"Strict artifact check failed: Legal BM25 index at {bm25_path} is missing or empty")
            if bm25_pyvi is None or not bm25_pyvi_path.exists() or len(getattr(bm25_pyvi, "corpus", [])) == 0:
                raise FileNotFoundError(f"Strict artifact check failed: PyVi BM25 index at {bm25_pyvi_path} is missing or empty")
            if dense is None or not (dense_path / "embeddings.npy").exists():
                raise FileNotFoundError(f"Strict artifact check failed: Dense index at {dense_path} is missing or empty")
            if len(memory.qids) == 0:
                raise FileNotFoundError(f"Strict artifact check failed: Question Memory at {mem_path} has 0 indexed queries")
            if use_reranker and reranker_model_name != "mock":
                if reranker_adapter_path is None:
                    raise FileNotFoundError("Strict artifact check failed: reranker_adapter_path is None")
                ad_path = Path(reranker_adapter_path)
                if not (ad_path / "adapter_config.json").exists():
                    raise FileNotFoundError(f"Strict artifact check failed: adapter_config.json missing in {ad_path}")
                has_weights = (ad_path / "adapter_model.safetensors").exists() or (ad_path / "adapter_model.bin").exists()
                if not has_weights:
                    raise FileNotFoundError(f"Strict artifact check failed: adapter weights (safetensors/bin) missing in {ad_path}")
                ad_manifest = ad_path / "training_manifest.json"
                if ad_manifest.exists():
                    m_data = json.loads(ad_manifest.read_text(encoding="utf-8"))
                    if m_data.get("status") != "completed":
                        raise ValueError(f"Strict artifact check failed: adapter training status is {m_data.get('status')}")
                    if m_data.get("param_diff", 1) is not None and m_data.get("param_diff", 1) <= 0:
                        raise ValueError("Strict artifact check failed: adapter param_diff <= 0")
                    if not m_data.get("adapter_checksum"):
                        raise ValueError("Strict artifact check failed: adapter checksum is missing")
            if use_learned_fusion:
                if fusion_model_path is None:
                    raise FileNotFoundError("Strict artifact check failed: learned fusion requested but fusion_model_path is None")
                f_p = Path(fusion_model_path)
                if not f_p.exists():
                    raise FileNotFoundError(f"Strict artifact check failed: fusion model path {f_p} does not exist")

        # 4. Exact Matcher
        exact = ExactMatcher(documents=list(doc_map.values()))

        hybrid_engine = HybridSearchEngine(
            bm25_retriever=bm25,
            bm25_pyvi_retriever=bm25_pyvi,
            dense_retriever=dense,
            question_memory=memory,
            exact_matcher=exact,
        )

        evidence_builder = EvidencePackBuilder(
            chunks_path=chunks_path if chunks_path.exists() else None,
            doc_metadata=doc_map,
            max_chunks=3,
            max_tokens=430,
        )

        # 5. Reranker
        if use_reranker:
            if reranker_adapter_path is not None:
                adapter_path = Path(reranker_adapter_path)
                if not adapter_path.exists():
                    raise FileNotFoundError(f"Reranker adapter path not found: {reranker_adapter_path}")
            reranker = CrossEncoderReranker(
                model_name=reranker_model_name,
                adapter_path=reranker_adapter_path,
                device=resolved_reranker_device,
            )
        else:
            reranker = None

        # 6. Ranker / Fusion
        if fusion_model_path is not None:
            f_path = Path(fusion_model_path)
            if not f_path.exists():
                raise FileNotFoundError(f"Fusion model path not found: {fusion_model_path}")
            if f_path.is_dir():
                candidates = [
                    f_path / "model.txt",
                    f_path / "model_full.txt",
                    f_path / "model.json",
                    f_path / "model_full.json",
                    f_path / "fusion_ranker.json",
                    f_path / "fusion_model.json",
                    f_path / "fusion_ranker.txt",
                    f_path / "fusion_model.txt",
                ]
                actual_file = next((p for p in candidates if p.is_file()), f_path)
            else:
                actual_file = f_path
            ranker = LightGBMRanker(model_file=actual_file)
        elif use_learned_fusion:
            ranker = LightGBMRanker()
        else:
            ranker = ReciprocalRankFusion()

        selector = TopKSelector(
            max_k=5,
            min_k=1,
            fallback_doc_ids=sorted(list(valid_doc_ids))[:5] if valid_doc_ids else None,
        )

        pipeline = cls(
            hybrid_engine=hybrid_engine,
            evidence_builder=evidence_builder,
            reranker=reranker,
            ranker=ranker,
            selector=selector,
            valid_doc_ids=valid_doc_ids,
        )

        if audit_preflight or audit_output_json is not None:
            pipeline.audit_parameters(output_json=audit_output_json, raise_on_violation=True)

        return pipeline

