import os
import json
import pandas as pd
from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.reranker import BGEReranker
from src.common.evidence import EvidencePackBuilder
from src.task1.memory import QuestionMemory
from src.task1.retrieve import CandidateRetriever, LegalMatcher
from src.task1.rerank import DocumentReranker
from src.task1.selector import TopKSelector

class LegalIRPipeline:
    def __init__(
        self,
        retriever: CandidateRetriever,
        reranker: DocumentReranker,
        selector: TopKSelector,
        valid_doc_ids: set[str] = None
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.selector = selector
        self.valid_doc_ids = valid_doc_ids

    @classmethod
    def load_pipeline(
        cls,
        data_dir: str = "artifacts/task1/data",
        index_dir: str = "artifacts/task1/indexes",
        use_reranker: bool = True,
        device: str = None
    ):
        docs_path = os.path.join(data_dir, "documents.parquet")
        chunks_path = os.path.join(data_dir, "chunks.parquet")
        precomputed_chunk_map = os.path.join(index_dir, "evidence", "chunk_map.json")

        doc_map = {}
        valid_doc_ids = set()
        if os.path.exists(docs_path):
            docs_df = pd.read_parquet(docs_path, columns=["doc_id", "title", "name_raw", "legal_number"])
            for r in docs_df.to_dict("records"):
                did = str(r["doc_id"])
                doc_map[did] = r
                valid_doc_ids.add(did)

        chunk_map = {}
        if os.path.exists(precomputed_chunk_map):
            with open(precomputed_chunk_map, "r", encoding="utf-8") as f:
                chunk_map = json.load(f)
        elif os.path.exists(chunks_path):
            chunks_df = pd.read_parquet(chunks_path, columns=["doc_id", "granularity", "article", "text_raw"])
            macro_df = chunks_df[chunks_df["granularity"] == "macro"] if "granularity" in chunks_df.columns else chunks_df
            grouped = macro_df.groupby("doc_id").head(2)
            for r in grouped.to_dict("records"):
                did = str(r["doc_id"])
                if did not in chunk_map:
                    chunk_map[did] = []
                chunk_map[did].append(r)

        # 1. BM25 Micro
        bm25_dir = os.path.join(index_dir, "bm25")
        if os.path.exists(bm25_dir):
            bm25 = BM25Retriever.load(bm25_dir)
        else:
            bm25 = BM25Retriever()

        # 2. DEk21 Dense Macro
        dense_dir = os.path.join(index_dir, "dense_dek21")
        if os.path.exists(dense_dir):
            dense = DEk21Retriever.load(dense_dir, device=device)
        else:
            dense = None

        # 3. Question Memory
        mem_dir = os.path.join(index_dir, "question_memory")
        if os.path.exists(mem_dir):
            memory = QuestionMemory.load(mem_dir, dense_retriever=dense)
        else:
            memory = QuestionMemory()

        # 4. Exact Matcher
        exact = LegalMatcher(doc_index=doc_map)

        candidate_retriever = CandidateRetriever(bm25=bm25, dense=dense, memory=memory, exact=exact)

        # 5. Reranker
        if use_reranker:
            bge_reranker = BGEReranker(model_name="BAAI/bge-reranker-v2-m3", device=device, batch_size=32)
        else:
            bge_reranker = None

        doc_reranker = DocumentReranker(
            reranker=bge_reranker,
            evidence_builder=EvidencePackBuilder(max_chunks_per_doc=2),
            doc_map=doc_map,
            chunk_map=chunk_map
        )

        selector = TopKSelector(max_k=5, min_k=1, fallback_doc_ids=list(valid_doc_ids)[:5] if valid_doc_ids else None)

        return cls(
            retriever=candidate_retriever,
            reranker=doc_reranker,
            selector=selector,
            valid_doc_ids=valid_doc_ids
        )

    def predict_single(self, query: str, top_k_candidates: int = 50, top_k_rerank: int = 5) -> list[str]:
        # 1. Candidate Retrieval (4 Branches + RRF)
        candidates = self.retriever.retrieve_candidates(query, top_k=top_k_candidates)

        # 2. Cross-Encoder Reranking
        if self.reranker and self.reranker.reranker is not None and candidates:
            ranked = self.reranker.rerank_documents(query, candidates, top_k=min(len(candidates), top_k_rerank * 4))
        else:
            ranked = candidates

        # 3. Top-K Selection
        selected = self.selector.select(ranked, valid_doc_ids=self.valid_doc_ids)
        return selected

    def predict_batch(self, items: dict[str, dict] | list[dict], top_k_candidates: int = 50, top_k_rerank: int = 5) -> dict[str, dict]:
        results = {}
        if isinstance(items, dict):
            for qid, data in items.items():
                q_text = data.get("question", "") if isinstance(data, dict) else str(data)
                pred_ids = self.predict_single(q_text, top_k_candidates=top_k_candidates, top_k_rerank=top_k_rerank)
                results[str(qid)] = {"answer": pred_ids}
        else:
            for item in items:
                qid = str(item.get("id") or item.get("qa_id") or "")
                q_text = str(item.get("question", ""))
                pred_ids = self.predict_single(q_text, top_k_candidates=top_k_candidates, top_k_rerank=top_k_rerank)
                results[qid] = {"answer": pred_ids}
        return results
