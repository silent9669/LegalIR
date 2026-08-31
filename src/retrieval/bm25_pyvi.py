"""PyVi Segmented BM25 Retriever for natural-language semantic lexical retrieval."""

from collections import Counter, defaultdict
import math
from pathlib import Path
import pickle
import re
from typing import Any, Mapping
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.normalize import clean_legal_text

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

PYVI_TOKEN_PATTERN = re.compile(r'\b[a-zà-ỹ0-9_]+\b', re.IGNORECASE | re.UNICODE)


def tokenize_pyvi(text: str) -> list[str]:
    """Tokenize Vietnamese text with PyVi word segmentation consistently."""
    if not text or not isinstance(text, str):
        return []
    cleaned = clean_legal_text(text)
    if ViTokenizer is not None:
        segmented = ViTokenizer.tokenize(cleaned)
    else:
        segmented = cleaned
    segmented = unicodedata.normalize("NFC", segmented).lower()
    return PYVI_TOKEN_PATTERN.findall(segmented)


def _normalize_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return unicodedata.normalize("NFC", str(val)).strip()


class BM25PyViRetriever:
    """Branch B: Lexical BM25 retriever indexed with PyVi word segmentation."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        field_weights: dict[str, float] | None = None,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.field_weights = field_weights or {
            "legal_number": 4.0,
            "title": 3.0,
            "article": 2.0,
            "clause": 1.0,
            "body": 1.0,
            "url_slug": 1.0,
        }
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.chunk_lens: np.ndarray | None = None
        self.avg_len: float = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def corpus(self) -> list[str]:
        """Backward-compatibility property returning indexed chunk IDs."""
        return self.chunk_ids

    def fit(self, chunks: Any, show_progress: bool = False) -> "BM25PyViRetriever":
        """Fit BM25 index on micro chunks using PyVi tokenization."""
        if isinstance(chunks, pd.DataFrame):
            records = chunks.to_dict(orient="records")
        elif isinstance(chunks, Mapping):
            records = [dict(chunks)]
        else:
            records = list(chunks)

        self.chunk_ids = []
        self.doc_ids = []
        lens = []
        term_df = Counter()
        term_postings = defaultdict(list)

        iterator = enumerate(records)
        if show_progress:
            iterator = tqdm(iterator, total=len(records), desc="Indexing PyVi BM25 chunks")

        for idx, c in iterator:
            cid = str(c.get("chunk_id", idx))
            did = str(c.get("doc_id", c.get("document_id", cid)))

            weighted_tokens: list[str] = []

            body_text = _normalize_str(c.get("text_norm") or c.get("text_raw", ""))
            body_toks = tokenize_pyvi(body_text)
            weighted_tokens.extend(body_toks * int(self.field_weights.get("body", 1.0)))

            legal_num = _normalize_str(c.get("legal_number", ""))
            if legal_num:
                num_toks = tokenize_pyvi(legal_num)
                weighted_tokens.extend(num_toks * int(self.field_weights.get("legal_number", 4.0)))

            title = _normalize_str(c.get("title", ""))
            if title:
                title_toks = tokenize_pyvi(title)
                weighted_tokens.extend(title_toks * int(self.field_weights.get("title", 3.0)))

            article = _normalize_str(c.get("article", ""))
            if article:
                art_toks = tokenize_pyvi(article)
                weighted_tokens.extend(art_toks * int(self.field_weights.get("article", 2.0)))

            clause = _normalize_str(c.get("clause", ""))
            if clause:
                clause_toks = tokenize_pyvi(clause)
                weighted_tokens.extend(clause_toks * int(self.field_weights.get("clause", 1.0)))

            link = _normalize_str(c.get("link", ""))
            if link:
                slug = link.rstrip("/").split("/")[-1].replace("-", " ")
                slug_toks = tokenize_pyvi(slug)
                weighted_tokens.extend(slug_toks * int(self.field_weights.get("url_slug", 1.0)))

            doc_len = len(weighted_tokens)
            lens.append(doc_len)
            self.chunk_ids.append(cid)
            self.doc_ids.append(did)

            tf = Counter(weighted_tokens)
            for term, freq in tf.items():
                term_df[term] += 1
                term_postings[term].append((idx, freq))

        N = len(records)
        self.chunk_lens = np.array(lens, dtype=np.float32)
        self.avg_len = float(np.mean(self.chunk_lens)) if N > 0 else 1.0

        # Calculate BM25 IDF
        self.idf = {}
        for term, df in term_df.items():
            self.idf[term] = float(math.log((N - df + 0.5) / (df + 0.5) + 1.0))

        # Convert postings to numpy arrays
        self.postings = {}
        for term, post_list in term_postings.items():
            c_indices = np.array([p[0] for p in post_list], dtype=np.int32)
            t_freqs = np.array([p[1] for p in post_list], dtype=np.float32)
            self.postings[term] = (c_indices, t_freqs)

        return self

    def retrieve(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Retrieve top candidate documents with PyVi tokenized BM25."""
        if not query or self.chunk_lens is None or len(self.chunk_ids) == 0:
            return []

        tokens = tokenize_pyvi(query)
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

            num = t_freqs * (self.k1 + 1.0)
            den = t_freqs + self.k1 * (1.0 - self.b + self.b * (lens / self.avg_len))
            term_scores = (idf_val * qf) * (num / den)

            np.add.at(scores_arr, c_indices, term_scores)
            has_matches = True

        if not has_matches:
            return []

        candidate_count = min(max(top_k * 10, 100), len(scores_arr))
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
                "score": float(agg_score),
                "bm25_score": float(agg_score),
                "bm25_pyvi_score": float(agg_score),
                "bm25_pyvi_best_score": float(best_s),
                "bm25_pyvi_second_score": float(second_s),
                "bm25_pyvi_mean_score": float(mean_s),
                "bm25_pyvi_best_chunk_id": best_cid,
            })

        doc_records.sort(key=lambda x: (-x["score"], str(x["doc_id"])))
        return doc_records[:top_k]

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Alias for retrieve."""
        return self.retrieve(query, top_k=top_k)

    def save(self, file_path: str | Path) -> Path:
        file_path = Path(file_path)
        if file_path.is_dir() or file_path.suffix == "":
            file_path.mkdir(parents=True, exist_ok=True)
            target = file_path / "bm25_pyvi_index.pkl"
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            target = file_path
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25PyViRetriever":
        file_path = Path(file_path)
        if file_path.is_dir():
            target = file_path / "bm25_pyvi_index.pkl"
            if not target.exists():
                candidates = list(file_path.glob("*.pkl"))
                target = candidates[0] if candidates else target
        else:
            target = file_path
        with open(target, "rb") as f:
            return pickle.load(f)
