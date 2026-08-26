from collections import defaultdict, Counter
from typing import Any
import math
import re
import unicodedata

TOKEN_PATTERN = re.compile(r'\b\w+\b', re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize('NFC', text).lower()
    return TOKEN_PATTERN.findall(text)


class EvidencePackBuilder:
    def __init__(
        self,
        macro_chunks: list[dict[str, Any]],
        doc_metadata: dict[str, dict[str, Any]] | None = None,
    ):
        self.chunks_by_doc = defaultdict(list)
        for c in macro_chunks:
            did = str(c["doc_id"])
            self.chunks_by_doc[did].append(c)

        self.doc_metadata = doc_metadata or {}

    def format_evidence_text(self, chunk: dict[str, Any], doc_meta: dict[str, Any] | None = None) -> str:
        did = str(chunk.get("doc_id", ""))
        meta = doc_meta or self.doc_metadata.get(did, {})

        title = meta.get("title") or f"Văn bản {did}"
        legal_num = meta.get("legal_number") or ""
        article = chunk.get("article") or "Thông tin văn bản"
        body = chunk.get("text_raw") or chunk.get("text_norm") or ""

        header = f"[VĂN BẢN]: {title}"
        if legal_num:
            header += f" (Số: {legal_num})"

        evidence = f"{header}\n[ĐIỀU KHOẢN]: {article}\n[NỘI DUNG]:\n{body.strip()}"
        return evidence

    def build(
        self,
        query: str,
        doc_id: str,
        candidate_record: dict[str, Any] | None = None,
        max_chunks: int = 2,
        max_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        """
        Builds top 1-2 evidence packs for candidate document.
        Returns list of dicts: [{"chunk_id": cid, "text": evidence_text, "article": article}]
        """
        doc_id = str(doc_id)
        doc_chunks = self.chunks_by_doc.get(doc_id, [])

        if not doc_chunks:
            meta = self.doc_metadata.get(doc_id, {})
            title = meta.get("title") or f"Văn bản pháp luật {doc_id}"
            fallback = f"[VĂN BẢN]: {title}\n[ĐIỀU KHOẢN]: Toàn văn\n[NỘI DUNG]:\nVăn bản pháp luật {doc_id}"
            return [{
                "chunk_id": f"{doc_id}_fallback",
                "text": fallback,
                "evidence_text": fallback,
                "article": "Văn bản",
            }]

        if len(doc_chunks) <= max_chunks:
            packs = []
            for c in doc_chunks:
                f_text = self.format_evidence_text(c)[:max_chars]
                packs.append({
                    "chunk_id": str(c["chunk_id"]),
                    "text": f_text,
                    "evidence_text": f_text,
                    "article": c.get("article", ""),
                })
            return packs

        # Score chunks with lexical query overlap
        q_tokens = tokenize(query)
        q_tf = Counter(q_tokens)
        scored_chunks = []

        # If candidate record has dense/bm25 best chunk ID, give it a prior boost
        prior_best_cid = None
        if candidate_record:
            prior_best_cid = candidate_record.get("dense_best_chunk_id") or candidate_record.get("bm25_best_chunk_id")

        for chunk in doc_chunks:
            cid = str(chunk["chunk_id"])
            text = chunk.get("text_norm", "")
            c_tokens = tokenize(text)
            if not c_tokens:
                scored_chunks.append((0.0, chunk))
                continue

            c_tf = Counter(c_tokens)
            overlap = 0.0
            for term, qf in q_tf.items():
                if term in c_tf:
                    overlap += qf * (1.0 + math.log(1.0 + c_tf[term]))

            score = overlap / (math.sqrt(len(c_tokens)) + 1.0)
            if prior_best_cid and cid == str(prior_best_cid):
                score += 5.0

            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored_chunks[:max_chunks]]

        packs = []
        for c in top_chunks:
            f_text = self.format_evidence_text(c)[:max_chars]
            packs.append({
                "chunk_id": str(c["chunk_id"]),
                "text": f_text,
                "evidence_text": f_text,
                "article": c.get("article", ""),
            })
        return packs

    def build_evidence(self, query: str, doc_id: str, max_chars: int = 1200) -> dict[str, Any]:
        """Backwards compatible helper returning single top evidence dict."""
        packs = self.build(query, doc_id, max_chunks=1, max_chars=max_chars)
        return packs[0] if packs else {
            "doc_id": str(doc_id),
            "chunk_id": f"{doc_id}_fallback",
            "evidence_text": f"Văn bản pháp luật {doc_id}",
            "article": "Văn bản",
        }
