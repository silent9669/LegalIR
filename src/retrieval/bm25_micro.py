"""Fielded Micro BM25 Retriever with legal-preserving tokenization and legal signal boosting."""

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

from src.dataset.normalize import clean_legal_text, extract_legal_signals

# Legal-preserving regex: matches compound statutory identifiers (123/2020/NĐ-CP, 61/2020/QH14)
# as well as Vietnamese words/numbers
LEGAL_TOKEN_PATTERN = re.compile(
    r'(\d+[\/\-][0-9a-zà-ỹ\-\_\/]+|[a-zà-ỹ0-9]+)',
    re.IGNORECASE | re.UNICODE,
)

DOC_TYPE_PATTERN = re.compile(
    r'\b(luật|bộ luật|nghị định|thông tư liên tịch|thông tư|quyết định|chỉ thị|nghị quyết|tiêu chuẩn|công văn|pháp lệnh)\b',
    re.IGNORECASE,
)


def tokenize_legal(text: str) -> list[str]:
    """Tokenize legal Vietnamese text while strictly preserving legal numbers and identifiers."""
    if not text or not isinstance(text, str):
        return []
    text = unicodedata.normalize("NFC", text).lower()
    return LEGAL_TOKEN_PATTERN.findall(text)


# Backward-compatibility alias
tokenize_vietnamese = tokenize_legal


def _normalize_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return unicodedata.normalize("NFC", str(val)).strip()


class BM25MicroRetriever:
    """Branch A: Lexical BM25 retriever on micro-chunks with legal signal boosting."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        field_weights: dict[str, float] | None = None,
        legal_boost_weights: dict[str, float] | None = None,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.field_weights = field_weights or {
            "legal_number": 5.0,
            "title": 3.0,
            "article": 2.5,
            "clause": 1.5,
            "body": 1.0,
            "url_slug": 1.5,
        }
        self.legal_boost_weights = legal_boost_weights or {
            "legal_number": 6.0,
            "article": 3.0,
            "clause": 1.5,
            "point": 0.75,
            "year": 1.0,
            "doc_type": 1.0,
        }
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.chunk_lens: np.ndarray | None = None
        self.avg_len: float = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Structured metadata for query-time legal signal boosting
        self.chunk_articles: list[str] = []
        self.chunk_clauses: list[str] = []
        self.chunk_points: list[str] = []
        self.doc_legal_numbers: dict[str, set[str]] = defaultdict(set)
        self.doc_years: dict[str, set[str]] = defaultdict(set)
        self.doc_types: dict[str, set[str]] = defaultdict(set)

    @property
    def corpus(self) -> list[str]:
        """Backward-compatibility property returning indexed chunk IDs."""
        return self.chunk_ids

    def fit(self, chunks: Any, show_progress: bool = False) -> "BM25MicroRetriever":
        """Fit BM25 inverted index on micro chunk records or DataFrame."""
        if isinstance(chunks, pd.DataFrame):
            records = chunks.to_dict(orient="records")
        elif isinstance(chunks, Mapping):
            records = [dict(chunks)]
        else:
            records = list(chunks)

        self.chunk_ids = []
        self.doc_ids = []
        self.chunk_articles = []
        self.chunk_clauses = []
        self.chunk_points = []
        self.doc_legal_numbers = defaultdict(set)
        self.doc_years = defaultdict(set)
        self.doc_types = defaultdict(set)

        lens = []
        term_df = Counter()
        term_postings = defaultdict(list)

        iterator = enumerate(records)
        if show_progress:
            iterator = tqdm(iterator, total=len(records), desc="Indexing BM25 micro chunks")

        for idx, c in iterator:
            cid = str(c.get("chunk_id", idx))
            did = str(c.get("doc_id", c.get("document_id", cid)))

            # Extract and store structured metadata for boosting
            legal_num = _normalize_str(c.get("legal_number", ""))
            if legal_num:
                norm_num = legal_num.lower().replace("-", "/")
                self.doc_legal_numbers[did].add(norm_num)
                clean_num = re.sub(r"^[^\d]*", "", norm_num)
                if clean_num:
                    self.doc_legal_numbers[did].add(clean_num)

            year_val = _normalize_str(c.get("year", ""))
            if year_val:
                self.doc_years[did].add(year_val)

            doc_type_val = _normalize_str(c.get("doc_type", ""))
            if doc_type_val:
                self.doc_types[did].add(doc_type_val.lower())

            art_val = _normalize_str(c.get("article", ""))
            self.chunk_articles.append(art_val.lower())

            clause_val = _normalize_str(c.get("clause", ""))
            self.chunk_clauses.append(clause_val.lower())

            point_val = _normalize_str(c.get("point", ""))
            self.chunk_points.append(point_val.lower())

            # Field-weighted tokens
            weighted_tokens: list[str] = []

            body_text = _normalize_str(c.get("text_norm") or c.get("text_raw", ""))
            body_toks = tokenize_legal(body_text)
            weighted_tokens.extend(body_toks * int(self.field_weights.get("body", 1.0)))

            if legal_num:
                num_toks = tokenize_legal(legal_num)
                weighted_tokens.extend(num_toks * int(self.field_weights.get("legal_number", 5.0)))

            title = _normalize_str(c.get("title", ""))
            if title:
                title_toks = tokenize_legal(title)
                weighted_tokens.extend(title_toks * int(self.field_weights.get("title", 3.0)))

            if art_val:
                art_toks = tokenize_legal(art_val)
                weighted_tokens.extend(art_toks * int(self.field_weights.get("article", 2.5)))

            if clause_val:
                clause_toks = tokenize_legal(clause_val)
                weighted_tokens.extend(clause_toks * int(self.field_weights.get("clause", 1.5)))

            link = _normalize_str(c.get("link", ""))
            if link:
                slug = link.rstrip("/").split("/")[-1].replace("-", " ")
                slug_toks = tokenize_legal(slug)
                weighted_tokens.extend(slug_toks * int(self.field_weights.get("url_slug", 1.5)))

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

        # Convert postings to numpy arrays for fast vectorized scoring
        self.postings = {}
        for term, post_list in term_postings.items():
            c_indices = np.array([p[0] for p in post_list], dtype=np.int32)
            t_freqs = np.array([p[1] for p in post_list], dtype=np.float32)
            self.postings[term] = (c_indices, t_freqs)

        return self

    def _extract_query_signals(self, query: str) -> dict[str, list[str]]:
        """Extract statutory signals from query for score boosting."""
        signals = extract_legal_signals(query)
        doc_types = [m.group(0).lower() for m in DOC_TYPE_PATTERN.finditer(query)]
        signals["doc_types"] = list(dict.fromkeys(doc_types))
        return signals

    def retrieve(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Retrieve top candidate documents using legal BM25 and statutory signal boosting."""
        if not query or self.chunk_lens is None or len(self.chunk_ids) == 0:
            return []

        tokens = tokenize_legal(query)
        if not tokens:
            return []

        q_tf = Counter(tokens)
        raw_scores_arr = np.zeros(len(self.chunk_ids), dtype=np.float32)
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

            np.add.at(raw_scores_arr, c_indices, term_scores)
            has_matches = True

        if not has_matches:
            return []

        # Find top candidate chunks
        candidate_count = min(max(top_k * 10, 100), len(raw_scores_arr))
        top_chunk_indices = np.argpartition(raw_scores_arr, -candidate_count)[-candidate_count:]
        top_chunk_indices = top_chunk_indices[raw_scores_arr[top_chunk_indices] > 0]

        # Extract legal signals from query for boosting
        signals = self._extract_query_signals(query)
        q_doc_nums = {num.lower().replace("-", "/") for num in signals.get("doc_numbers", [])}
        q_articles = {art.lower() for art in signals.get("articles", [])}
        q_clauses = {clause.lower() for clause in signals.get("clauses", [])}
        q_points = {pt.lower() for pt in signals.get("points", [])}
        q_years = set(signals.get("years", []))
        q_doc_types = set(signals.get("doc_types", []))

        # Aggregate to document level with boosting
        doc_chunk_scores = defaultdict(list)
        doc_boosts = defaultdict(float)

        for c_idx in top_chunk_indices:
            did = self.doc_ids[c_idx]
            cid = self.chunk_ids[c_idx]
            raw_s = float(raw_scores_arr[c_idx])

            # Compute legal boost for chunk
            boost = 0.0

            # Legal number boost (document-level)
            if q_doc_nums:
                doc_nums = self.doc_legal_numbers.get(did, set())
                if any(qn in doc_nums or any(qn in dn for dn in doc_nums) for qn in q_doc_nums):
                    boost += self.legal_boost_weights.get("legal_number", 6.0)

            # Article boost
            if q_articles:
                art_str = self.chunk_articles[c_idx]
                if any(f"điều {qa}" in art_str or f"điều_{qa}" in art_str or art_str == qa for qa in q_articles):
                    boost += self.legal_boost_weights.get("article", 3.0)

            # Clause boost
            if q_clauses:
                cl_str = self.chunk_clauses[c_idx]
                if any(f"khoản {qc}" in cl_str or f"khoản_{qc}" in cl_str or cl_str == qc for qc in q_clauses):
                    boost += self.legal_boost_weights.get("clause", 1.5)

            # Point boost
            if q_points:
                pt_str = self.chunk_points[c_idx]
                if any(f"điểm {qp}" in pt_str or f"điểm_{qp}" in pt_str or pt_str == qp for qp in q_points):
                    boost += self.legal_boost_weights.get("point", 0.75)

            # Year boost
            if q_years:
                doc_yrs = self.doc_years.get(did, set())
                if q_years.intersection(doc_yrs):
                    boost += self.legal_boost_weights.get("year", 1.0)

            # Doc type boost
            if q_doc_types:
                doc_tps = self.doc_types.get(did, set())
                if q_doc_types.intersection(doc_tps):
                    boost += self.legal_boost_weights.get("doc_type", 1.0)

            total_chunk_s = raw_s + boost
            doc_boosts[did] = max(doc_boosts[did], boost)
            doc_chunk_scores[did].append((cid, total_chunk_s, raw_s))

        doc_records = []
        for did, items in doc_chunk_scores.items():
            sorted_items = sorted(items, key=lambda x: -x[1])
            best_cid, best_s, best_raw_s = sorted_items[0]
            second_s = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
            second_raw_s = sorted_items[1][2] if len(sorted_items) > 1 else 0.0
            mean_s = sum(x[1] for x in items) / len(items)
            agg_score = best_s + 0.1 * second_s
            agg_raw_score = best_raw_s + 0.1 * second_raw_s

            doc_records.append({
                "doc_id": did,
                "score": float(agg_score),
                "bm25_score": float(agg_score),
                "bm25_raw_score": float(agg_raw_score),
                "bm25_best_score": float(best_s),
                "bm25_second_score": float(second_s),
                "bm25_mean_score": float(mean_s),
                "bm25_best_chunk_id": best_cid,
                "bm25_legal_boost": float(doc_boosts[did]),
            })

        # Stable sort: descending score, ascending doc_id
        doc_records.sort(key=lambda x: (-x["score"], str(x["doc_id"])))
        return doc_records[:top_k]

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        """Alias for retrieve for interface compatibility."""
        return self.retrieve(query, top_k=top_k)

    def save(self, file_path: str | Path) -> Path:
        file_path = Path(file_path)
        if file_path.is_dir() or file_path.suffix == "":
            file_path.mkdir(parents=True, exist_ok=True)
            target = file_path / "bm25_micro_index.pkl"
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            target = file_path
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25MicroRetriever":
        file_path = Path(file_path)
        if file_path.is_dir():
            target = file_path / "bm25_micro_index.pkl"
            if not target.exists():
                candidates = list(file_path.glob("*.pkl"))
                target = candidates[0] if candidates else target
        else:
            target = file_path
        with open(target, "rb") as f:
            return pickle.load(f)
