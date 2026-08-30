from src.common.evidence import EvidencePackBuilder
from src.common.reranker import BGEReranker

class DocumentReranker:
    def __init__(self, reranker: BGEReranker, evidence_builder: EvidencePackBuilder = None, doc_map: dict = None, chunk_map: dict = None):
        self.reranker = reranker
        self.evidence_builder = evidence_builder or EvidencePackBuilder()
        self.doc_map = doc_map or {}
        self.chunk_map = chunk_map or {}

    def rerank_documents(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates or self.reranker is None:
            return candidates[:top_k]

        evidence_texts = []
        valid_candidates = []

        for c in candidates:
            doc_id = str(c.get("doc_id", ""))
            doc_info = self.doc_map.get(doc_id, {"doc_id": doc_id})
            chunks = self.chunk_map.get(doc_id, [])

            if not chunks and "best_chunk" in c:
                chunks = [c["best_chunk"]]

            ev_text = self.evidence_builder.build_evidence_text(query, doc_info, chunks)
            evidence_texts.append(ev_text)
            valid_candidates.append(c)

        reranked = self.reranker.rerank_candidates(query, valid_candidates, evidence_texts=evidence_texts, top_k=top_k)
        return reranked
