from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import math
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
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        field_weights: dict[str, float] | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.field_weights = field_weights or {
            "legal_number": 5.0,
            "title": 3.0,
            "article": 2.0,
            "body": 1.0,
            "url_slug": 1.5,
        }
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.chunk_lens: np.ndarray | None = None
        self.avg_len: float = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, chunks: list[dict[str, Any]], show_progress: bool = False) -> "BM25MicroRetriever":
        """Fit BM25 inverted index on a list of chunk dicts."""
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

            # Field-weighted tokens
            weighted_tokens = []

            body_text = c.get("text_norm") or c.get("text_raw", "")
            body_toks = tokenize_vietnamese(body_text)
            weighted_tokens.extend(body_toks * int(self.field_weights.get("body", 1.0)))

            legal_num = c.get("legal_number") or ""
            if legal_num:
                num_toks = tokenize_vietnamese(legal_num)
                weighted_tokens.extend(num_toks * int(self.field_weights.get("legal_number", 5.0)))

            title = c.get("title") or ""
            if title:
                title_toks = tokenize_vietnamese(title)
                weighted_tokens.extend(title_toks * int(self.field_weights.get("title", 3.0)))

            article = c.get("article") or ""
            if article:
                article_toks = tokenize_vietnamese(article)
                weighted_tokens.extend(article_toks * int(self.field_weights.get("article", 2.0)))

            link = c.get("link") or ""
            if link:
                slug = link.rstrip("/").split("/")[-1].replace("-", " ")
                slug_toks = tokenize_vietnamese(slug)
                weighted_tokens.extend(slug_toks * int(self.field_weights.get("url_slug", 1.5)))

            doc_len = len(weighted_tokens)
            lens.append(doc_len)
            self.chunk_ids.append(cid)
            self.doc_ids.append(did)

            tf = Counter(weighted_tokens)
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

    def retrieve(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """
        Returns list of candidate dicts with:
        doc_id, score, bm25_score, bm25_best_score, bm25_second_score, bm25_mean_score, bm25_best_chunk_id
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
            cid = self.chunk_ids[c_idx]
            doc_chunk_scores[did].append((cid, float(scores_arr[c_idx])))

        doc_records = []
        for did, items in doc_chunk_scores.items():
            sorted_items = sorted(items, key=lambda x: -x[1])
            best_cid, best_s = sorted_items[0]
            second_s = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
            mean_s = sum(x[1] for x in items) / len(items)
            agg_score = best_s + 0.1 * second_s

            doc_records.append({
                "doc_id": did,
                "score": agg_score,
                "bm25_score": agg_score,
                "bm25_best_score": best_s,
                "bm25_second_score": second_s,
                "bm25_mean_score": mean_s,
                "bm25_best_chunk_id": best_cid,
            })

        # Stable sort: descending score, ascending doc_id
        doc_records.sort(key=lambda x: (-x["score"], x["doc_id"]))
        return doc_records[:top_k]

    def save(self, file_path: str | Path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25MicroRetriever":
        with open(file_path, "rb") as f:
            return pickle.load(f)
