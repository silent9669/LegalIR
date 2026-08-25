import re
import unicodedata
from collections import defaultdict, Counter
import math

TOKEN_PATTERN = re.compile(r'\b\w+\b', re.UNICODE)

def tokenize(text: str) -> list:
    if not text:
        return []
    text = unicodedata.normalize('NFC', text).lower()
    return TOKEN_PATTERN.findall(text)

class EvidencePackBuilder:
    def __init__(self, macro_chunks: list):
        self.chunks_by_doc = defaultdict(list)
        for c in macro_chunks:
            did = str(c["doc_id"])
            self.chunks_by_doc[did].append(c)

    def build_evidence(self, query: str, doc_id: str, max_chars: int = 1200) -> dict:
        """
        Selects the best macro chunk in doc_id matching the query and returns:
        {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "evidence_text": formatted_text,
            "article": article_header
        }
        """
        doc_id = str(doc_id)
        doc_chunks = self.chunks_by_doc.get(doc_id, [])

        if not doc_chunks:
            return {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_fallback",
                "evidence_text": f"Văn bản pháp luật {doc_id}",
                "article": "Văn bản"
            }

        if len(doc_chunks) == 1:
            chunk = doc_chunks[0]
            return {
                "doc_id": doc_id,
                "chunk_id": chunk["chunk_id"],
                "evidence_text": chunk.get("text_norm", "")[:max_chars],
                "article": chunk.get("article", "")
            }

        q_tokens = tokenize(query)
        q_tf = Counter(q_tokens)
        best_chunk = doc_chunks[0]
        best_score = -1.0

        for chunk in doc_chunks:
            text = chunk.get("text_norm", "")
            c_tokens = tokenize(text)
            if not c_tokens:
                continue

            c_tf = Counter(c_tokens)
            overlap = 0.0
            for term, qf in q_tf.items():
                if term in c_tf:
                    overlap += qf * (1.0 + math.log(1.0 + c_tf[term]))

            score = overlap / (math.sqrt(len(c_tokens)) + 1.0)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        return {
            "doc_id": doc_id,
            "chunk_id": best_chunk["chunk_id"],
            "evidence_text": best_chunk.get("text_norm", "")[:max_chars],
            "article": best_chunk.get("article", "")
        }
