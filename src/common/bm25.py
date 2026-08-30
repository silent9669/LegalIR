import json
import os
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from src.common.normalize import clean_legal_text, tokenize_vietnamese, extract_legal_signals

try:
    import bm25s
except ImportError:
    bm25s = None

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_ids = []
        self.chunk_to_doc = []
        self.corpus_size = 0
        self.bm25s_index = None

    def fit(self, corpus: list[dict]):
        self.corpus = corpus
        self.doc_ids = [str(c.get("doc_id", c.get("chunk_id", ""))) for c in corpus]
        self.chunk_to_doc = self.doc_ids
        self.corpus_size = len(corpus)

        tokenized_corpus = []
        for c in corpus:
            tokens = c.get("text_norm", "").split()
            if not tokens:
                tokens = tokenize_vietnamese(c.get("text_raw", "")).split()
            tokenized_corpus.append(tokens)

        if bm25s is not None and self.corpus_size > 0:
            try:
                self.bm25s_index = bm25s.BM25(k1=self.k1, b=self.b)
                self.bm25s_index.index(tokenized_corpus)
            except Exception:
                self.bm25s_index = None

    def search(self, query: str, top_k: int = 60) -> list[dict]:
        if self.corpus_size == 0 or self.bm25s_index is None:
            return []

        signals = extract_legal_signals(query)
        seg_query = tokenize_vietnamese(query.lower())
        if not seg_query:
            return []

        try:
            tokens = bm25s.tokenize(seg_query, stopwords=None, show_progress=False)
            retrieve_k = min(max(top_k * 5, 250), self.corpus_size)
            bm25_res = self.bm25s_index.retrieve(tokens, k=retrieve_k, show_progress=False)
            doc_indices = bm25_res.documents[0]
            bm25_scores = bm25_res.scores[0]
        except Exception:
            return []

        # Document-level score accumulation + legal entity boosting
        doc_scores = defaultdict(float)
        best_chunks = {}

        for idx, sc in zip(doc_indices, bm25_scores):
            if not isinstance(idx, (int, np.integer)) or idx < 0 or idx >= self.corpus_size:
                continue

            doc_id = str(self.chunk_to_doc[idx])
            score = float(sc)

            if self.corpus and idx < len(self.corpus):
                raw = self.corpus[idx].get("text_raw", "")
                for d in signals.get("doc_numbers", []):
                    if d in raw:
                        score += 30.0
                for a in signals.get("articles", []):
                    if f"Điều {a}." in raw or f"Điều {a} " in raw:
                        score += 15.0
                for cl in signals.get("clauses", []):
                    if f"Khoản {cl}." in raw or f"\n{cl}. " in raw:
                        score += 8.0

            if score > doc_scores[doc_id]:
                doc_scores[doc_id] = score
                if self.corpus and idx < len(self.corpus):
                    best_chunks[doc_id] = self.corpus[idx]

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            results.append({
                "doc_id": doc_id,
                "score": score,
                "rank": rank,
                "branch": "bm25",
                "best_chunk": best_chunks.get(doc_id)
            })
        return results

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, "chunk_to_doc.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunk_to_doc, f)
        if self.bm25s_index is not None:
            self.bm25s_index.save(os.path.join(index_dir, "bm25s_index"))

    @classmethod
    def load(cls, index_dir: str):
        retriever = cls()
        chunk_to_doc_path = os.path.join(index_dir, "chunk_to_doc.json")
        bm25s_dir = os.path.join(index_dir, "bm25s_index")

        if os.path.exists(chunk_to_doc_path) and bm25s is not None and os.path.exists(os.path.join(bm25s_dir, "params.index.json")):
            with open(chunk_to_doc_path, "r", encoding="utf-8") as f:
                retriever.chunk_to_doc = json.load(f)
            retriever.corpus_size = len(retriever.chunk_to_doc)
            retriever.doc_ids = retriever.chunk_to_doc
            retriever.bm25s_index = bm25s.BM25.load(bm25s_dir, mmap=True)
        return retriever
