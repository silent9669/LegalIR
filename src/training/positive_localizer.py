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

class PositiveLocalizer:
    def __init__(self, macro_chunks: list):
        self.chunks_by_doc = defaultdict(list)
        for c in macro_chunks:
            did = str(c["doc_id"])
            self.chunks_by_doc[did].append(c)

    def localize(self, query: str, gold_doc_id: str) -> dict:
        """Finds the most relevant macro chunk inside gold_doc_id for the given query."""
        gold_doc_id = str(gold_doc_id)
        doc_chunks = self.chunks_by_doc.get(gold_doc_id, [])

        if not doc_chunks:
            return None

        if len(doc_chunks) == 1:
            return doc_chunks[0]

        q_tokens = tokenize(query)
        if not q_tokens:
            return doc_chunks[0]

        q_tf = Counter(q_tokens)
        best_chunk = None
        best_score = -1.0

        for chunk in doc_chunks:
            text = chunk.get("text_norm", "")
            c_tokens = tokenize(text)
            if not c_tokens:
                continue

            c_tf = Counter(c_tokens)
            # Compute term overlap score weighted by query frequency and length normalization
            overlap = 0.0
            for term, qf in q_tf.items():
                if term in c_tf:
                    overlap += qf * (1.0 + math.log(1.0 + c_tf[term]))

            score = overlap / (math.sqrt(len(c_tokens)) + 1.0)

            if score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk or doc_chunks[0]
