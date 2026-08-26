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


class PositiveLocalizer:
    def __init__(self, macro_chunks: list[dict[str, Any]]):
        self.chunks_by_doc = defaultdict(list)
        for c in macro_chunks:
            did = str(c["doc_id"])
            self.chunks_by_doc[did].append(c)

    def localize(
        self,
        query: str,
        gold_doc_id: str,
        top_k: int | None = None,
    ) -> Any:
        """Finds the most relevant macro chunk(s) inside gold_doc_id for the given query."""
        gold_doc_id = str(gold_doc_id)
        doc_chunks = self.chunks_by_doc.get(gold_doc_id, [])

        if not doc_chunks:
            return [] if top_k is not None else None

        if len(doc_chunks) == 1:
            return doc_chunks if top_k is not None else doc_chunks[0]

        q_tokens = tokenize(query)
        if not q_tokens:
            return doc_chunks[:top_k] if top_k is not None else doc_chunks[0]

        q_tf = Counter(q_tokens)
        scored_chunks = []

        for chunk in doc_chunks:
            text = chunk.get("text_norm") or chunk.get("text_raw", "")
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
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored_chunks]

        if top_k is not None:
            return top_chunks[:top_k]
        return top_chunks[0]
