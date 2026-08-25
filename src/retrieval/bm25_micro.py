import os
import re
import math
import pickle
import unicodedata
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

TOKEN_PATTERN = re.compile(r'\b\w+\b', re.UNICODE)

def tokenize_vietnamese(text: str) -> list:
    if not text:
        return []
    text = unicodedata.normalize('NFC', text).lower()
    return TOKEN_PATTERN.findall(text)

class BM25MicroRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids = []
        self.doc_ids = []
        self.chunk_lens = None
        self.avg_len = 0.0
        self.idf = {}
        self.postings = {}
        self.doc_to_chunk_indices = defaultdict(list)

    def fit(self, chunks: list, show_progress: bool = False):
        """Fit BM25 inverted index on a list of chunk dicts (chunk_id, doc_id, text_norm)."""
        self.chunk_ids = []
        self.doc_ids = []
        self.doc_to_chunk_indices = defaultdict(list)
        lens = []
        term_df = Counter()
        term_postings = defaultdict(list)

        iterator = enumerate(chunks)
        if show_progress:
            iterator = tqdm(iterator, total=len(chunks), desc="Indexing BM25 micro chunks")

        for idx, c in iterator:
            cid = c["chunk_id"]
            did = str(c["doc_id"])
            text = c.get("text_norm", "")

            tokens = tokenize_vietnamese(text)
            doc_len = len(tokens)
            lens.append(doc_len)
            self.chunk_ids.append(cid)
            self.doc_ids.append(did)
            self.doc_to_chunk_indices[did].append(idx)

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
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        # Convert postings to numpy arrays for fast vectorized scoring
        self.postings = {}
        for term, post_list in term_postings.items():
            c_indices = np.array([p[0] for p in post_list], dtype=np.int32)
            t_freqs = np.array([p[1] for p in post_list], dtype=np.float32)
            self.postings[term] = (c_indices, t_freqs)

    def retrieve(self, query: str, top_k: int = 100) -> list:
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
        chunk_scores = defaultdict(float)

        for term, qf in q_tf.items():
            if term not in self.postings:
                continue

            c_indices, t_freqs = self.postings[term]
            idf_val = self.idf[term]
            lens = self.chunk_lens[c_indices]

            # Vectorized BM25 formula
            num = t_freqs * (self.k1 + 1.0)
            den = t_freqs + self.k1 * (1.0 - self.b + self.b * (lens / self.avg_len))
            scores = idf_val * (num / den) * qf

            for idx, sc in zip(c_indices, scores):
                chunk_scores[idx] += float(sc)

        if not chunk_scores:
            return []

        # Aggregate to document level
        doc_chunk_scores = defaultdict(list)
        for c_idx, score in chunk_scores.items():
            did = self.doc_ids[c_idx]
            doc_chunk_scores[did].append(score)

        doc_scores = {}
        for did, s_list in doc_chunk_scores.items():
            max_s = max(s_list)
            mean_s = sum(s_list) / len(s_list)
            doc_scores[did] = max_s + 0.1 * mean_s

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "rb") as f:
            return pickle.load(f)
