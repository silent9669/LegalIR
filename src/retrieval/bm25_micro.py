from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import math
import os
import pickle
import re
import unicodedata
import numpy as np
from tqdm import tqdm

TOKEN_PATTERN = re.compile(r'\b\w+\b', re.UNICODE)


def tokenize_vietnamese(text: str) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize('NFC', text).lower()
    return TOKEN_PATTERN.findall(text)


class BM25MicroRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.chunk_lens: np.ndarray | None = None
        self.avg_len: float = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, chunks: list[dict[str, Any]], show_progress: bool = False) -> "BM25MicroRetriever":
        """Fit BM25 inverted index on a list of chunk dicts (chunk_id, doc_id, text_norm)."""
        self.chunk_ids = []
        self.doc_ids = []
        lens = []
        term_df = Counter()
        term_postings = defaultdict(list)

        iterator = enumerate(chunks)
        if show_progress:
            iterator = tqdm(iterator, total=len(chunks), desc="Indexing BM25 micro chunks")

        for idx, c in iterator:
            cid = str(c["chunk_id"])
            did = str(c["doc_id"])
            text = c.get("text_norm", "")

            tokens = tokenize_vietnamese(text)
            doc_len = len(tokens)
            lens.append(doc_len)
            self.chunk_ids.append(cid)
            self.doc_ids.append(did)

            tf = Counter(tokens)
            for term, freq in tf.items():
                term_df[term] += 1
                term_postings[term].append((idx, freq))

        N = len(chunks)
        self.chunk_lens = np.array(lens, dtype=np.float32)
        self.avg_len = float(np.mean(self.chunk_lens)) if N > 0 else 1.0

        # Calculate BM25 IDF
        self.idf = {}
        for term, df in term_df.items():
            self.idf[term] = float(math.log((N - df + 0.5) / (df + 0.5) + 1.0))

        # Convert postings to numpy arrays for fast vectorized scoring
        self.postings = {}
        for term, post_list in term_postings.items():
            c_indices = np.array([p[0] for p in post_list], dtype=np.int32)
            t_freqs = np.array([p[1] for p in post_list], dtype=np.float32)
            self.postings[term] = (c_indices, t_freqs)

        return self

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Returns list of (doc_id, score) sorted by descending score.
        Aggregates micro-chunk scores to document level using max + 0.1 * mean.
        """
        if not query or self.chunk_lens is None:
            return []

        tokens = tokenize_vietnamese(query)
        if not tokens:
            return []

        q_tf = Counter(tokens)
        scores_arr = np.zeros(len(self.chunk_ids), dtype=np.float32)
        has_matches = False

        for term, qf in q_tf.items():
            if term not in self.postings:
                continue

            c_indices, t_freqs = self.postings[term]
            idf_val = self.idf[term]
            lens = self.chunk_lens[c_indices]

            # Vectorized BM25 formula
            num = t_freqs * (self.k1 + 1.0)
            den = t_freqs + self.k1 * (1.0 - self.b + self.b * (lens / self.avg_len))
            term_scores = (idf_val * qf) * (num / den)

            np.add.at(scores_arr, c_indices, term_scores)
            has_matches = True

        if not has_matches:
            return []

        # Find top chunks quickly with argpartition
        candidate_count = min(top_k * 10, len(scores_arr))
        top_chunk_indices = np.argpartition(scores_arr, -candidate_count)[-candidate_count:]
        top_chunk_indices = top_chunk_indices[scores_arr[top_chunk_indices] > 0]

        # Aggregate to document level
        doc_chunk_scores = defaultdict(list)
        for c_idx in top_chunk_indices:
            did = self.doc_ids[c_idx]
            doc_chunk_scores[did].append(float(scores_arr[c_idx]))

        doc_scores = {}
        for did, s_list in doc_chunk_scores.items():
            max_s = max(s_list)
            mean_s = sum(s_list) / len(s_list)
            doc_scores[did] = max_s + 0.1 * mean_s

        sorted_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return sorted_docs

    def save(self, file_path: str | Path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25MicroRetriever":
        with open(file_path, "rb") as f:
            return pickle.load(f)
