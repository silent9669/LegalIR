"""Deterministic statutory extraction and exact legal matching."""

from collections import defaultdict
import re
from typing import Any, Mapping
import unicodedata
import pandas as pd

from src.dataset.normalize import clean_legal_text, extract_legal_signals

LEGAL_NUM_REGEX = re.compile(
    r'(\d+[\/\-][0-9]+[\/\-][A-ZĐa-z0-9\-\_]+|\d+[\/\-][A-ZĐa-z0-9\-\_]+)',
    re.IGNORECASE,
)
YEAR_REGEX = re.compile(r'\b(19[89]\d|20[012]\d)\b')
DOC_TYPE_REGEX = re.compile(
    r'\b(Luật|Bộ luật|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Chỉ thị|Nghị quyết|Tiêu chuẩn|Công văn|Pháp lệnh)\b',
    re.IGNORECASE,
)
ARTICLE_REGEX = re.compile(r'\bĐiều\s+(\d+[a-zA-Z]?)\b', re.IGNORECASE)
CLAUSE_REGEX = re.compile(r'\bkhoản\s+(\d+[a-zA-Z]?)\b', re.IGNORECASE)
POINT_REGEX = re.compile(r'\bđiểm\s+([a-zA-Z\d]+)\b', re.IGNORECASE)


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip().lower()


def _get_tokens(text: str) -> set[str]:
    if not text:
        return set()
    norm = normalize_text(text)
    return set(re.findall(r'\b[a-zà-ỹ0-9_/-]+\b', norm))


class ExactMatcher:
    """Deterministic statutory extraction producing structured exact matching features."""

    def __init__(
        self,
        documents: list[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        doc_index: dict[str, Any] | None = None,
    ):
        self.doc_by_num: dict[str, list[str]] = defaultdict(list)
        self.doc_by_title: dict[str, list[str]] = defaultdict(list)
        self.doc_metadata: dict[str, dict[str, Any]] = {}
        self.doc_articles: dict[str, set[str]] = defaultdict(set)
        self.doc_clauses: dict[str, set[str]] = defaultdict(set)
        self.doc_points: dict[str, set[str]] = defaultdict(set)
        self.doc_title_tokens: dict[str, set[str]] = {}

        docs = documents if documents is not None else doc_index
        if isinstance(docs, dict):
            doc_list = []
            for did, d in docs.items():
                if isinstance(d, dict):
                    entry = dict(d)
                    entry.setdefault("doc_id", str(did))
                    doc_list.append(entry)
            docs = doc_list
        elif docs is None:
            docs = []

        for d in docs:
            if not isinstance(d, (dict, Mapping)):
                continue
            raw_id = d.get("doc_id", d.get("document_id"))
            if raw_id is None or (isinstance(raw_id, float) and pd.isna(raw_id)):
                continue
            doc_id = str(raw_id)
            self.doc_metadata[doc_id] = dict(d)

            # Index legal number safely
            lnum = d.get("legal_number")
            if lnum is not None and not (isinstance(lnum, float) and pd.isna(lnum)):
                norm_num = normalize_text(lnum).replace("-", "/")
                if norm_num:
                    self.doc_by_num[norm_num].append(doc_id)
                    clean_num = re.sub(r"^[^\d]*", "", norm_num)
                    if clean_num and clean_num != norm_num:
                        self.doc_by_num[clean_num].append(doc_id)

            # Index title safely
            title = d.get("title")
            if title is not None and not (isinstance(title, float) and pd.isna(title)):
                norm_title = normalize_text(title)
                if norm_title:
                    self.doc_by_title[norm_title].append(doc_id)
                    self.doc_title_tokens[doc_id] = _get_tokens(norm_title)
                    title_no_year = re.sub(r"\b(19[89]\d|20[012]\d)\b", "", norm_title).strip()
                    if len(title_no_year) > 6 and title_no_year != norm_title:
                        self.doc_by_title[title_no_year].append(doc_id)

            # Index articles / clauses / points if present in metadata or chunks
            art = d.get("article")
            if art is not None and not (isinstance(art, float) and pd.isna(art)):
                self.doc_articles[doc_id].add(normalize_text(art))

            cl = d.get("clause")
            if cl is not None and not (isinstance(cl, float) and pd.isna(cl)):
                self.doc_clauses[doc_id].add(normalize_text(cl))

            pt = d.get("point")
            if pt is not None and not (isinstance(pt, float) and pd.isna(pt)):
                self.doc_points[doc_id].add(normalize_text(pt))

    def match(self, query: str) -> dict[str, dict[str, Any]]:
        """Extract statutory references and match against indexed documents.

        Returns {doc_id: feature_dict} with structured exact features:
          - exact_legal_number: bool
          - exact_article: bool
          - exact_clause: bool
          - exact_point: bool
          - exact_year: bool
          - exact_doc_type: bool
          - exact_title: bool
          - exact_title_overlap: float (0.0 to 1.0)
          - exact_score: float
          - score: float
        """
        if not query:
            return {}

        norm_query = normalize_text(query)
        q_tokens = _get_tokens(norm_query)

        # Extract legal signals
        signals = extract_legal_signals(query)
        q_doc_nums = [n.lower().replace("-", "/") for n in signals.get("doc_numbers", [])]
        q_articles = [a.lower() for a in signals.get("articles", [])]
        q_clauses = [c.lower() for c in signals.get("clauses", [])]
        q_points = [p.lower() for p in signals.get("points", [])]
        q_years = set(signals.get("years", []))
        doc_types = DOC_TYPE_REGEX.findall(query)
        q_doc_types = {normalize_text(t) for t in doc_types}

        matches: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "score": 0.0,
            "exact_score": 0.0,
            "exact_legal_number": False,
            "exact_article": False,
            "exact_clause": False,
            "exact_point": False,
            "exact_year": False,
            "exact_doc_type": False,
            "exact_title": False,
            "exact_title_overlap": 0.0,
        })

        # 1. Match legal numbers
        found_nums = LEGAL_NUM_REGEX.findall(query)
        for num in found_nums + q_doc_nums:
            clean_num = normalize_text(num).replace("-", "/")
            matched_dids = []
            score_val = 1.0
            if clean_num in self.doc_by_num:
                matched_dids = self.doc_by_num[clean_num]
            else:
                sub_num = re.sub(r"^[^\d]*", "", clean_num)
                if sub_num and sub_num in self.doc_by_num:
                    matched_dids = self.doc_by_num[sub_num]
                    score_val = 0.9

            for did in matched_dids:
                matches[did]["exact_legal_number"] = True
                matches[did]["score"] = max(matches[did]["score"], score_val)

        # 2. Match exact law titles
        for title, dids in self.doc_by_title.items():
            if len(title) > 6 and title in norm_query:
                for did in dids:
                    matches[did]["exact_title"] = True
                    matches[did]["score"] = max(matches[did]["score"], 0.85)

        # 3. Match title + year co-occurrence
        if q_years:
            for target_year in q_years:
                for did, meta in self.doc_metadata.items():
                    doc_yr = meta.get("year")
                    if doc_yr is not None and str(doc_yr).strip() == target_year:
                        doc_title = normalize_text(meta.get("title", ""))
                        doc_title_no_year = re.sub(r"\b(19[89]\d|20[012]\d)\b", "", doc_title).strip()
                        if doc_title_no_year and len(doc_title_no_year) > 6 and doc_title_no_year in norm_query:
                            matches[did]["exact_title"] = True
                            matches[did]["exact_year"] = True
                            matches[did]["score"] = max(matches[did]["score"], 0.95)

        # 4. Enrich matched documents with article, clause, point, year, doc_type, title_overlap
        for did in list(matches.keys()):
            meta = self.doc_metadata.get(did, {})

            # Year check
            doc_yr = meta.get("year")
            if doc_yr is not None and not (isinstance(doc_yr, float) and pd.isna(doc_yr)):
                if str(doc_yr).strip() in q_years:
                    matches[did]["exact_year"] = True

            # Doc type check
            d_type = normalize_text(meta.get("doc_type", ""))
            if d_type and d_type in q_doc_types:
                matches[did]["exact_doc_type"] = True

            # Article check
            d_arts = self.doc_articles.get(did, set())
            if any(any(f"điều {qa}" in da or da == qa for qa in q_articles) for da in d_arts):
                matches[did]["exact_article"] = True

            # Clause check
            d_clauses = self.doc_clauses.get(did, set())
            if any(any(f"khoản {qc}" in dc or dc == qc for qc in q_clauses) for dc in d_clauses):
                matches[did]["exact_clause"] = True

            # Point check
            d_pts = self.doc_points.get(did, set())
            if any(any(f"điểm {qp}" in dp or dp == qp for qp in q_points) for dp in d_pts):
                matches[did]["exact_point"] = True

            # Title overlap
            t_tokens = self.doc_title_tokens.get(did, set())
            if t_tokens and q_tokens:
                overlap = len(t_tokens & q_tokens) / len(t_tokens)
                matches[did]["exact_title_overlap"] = float(overlap)
                if overlap > 0.6:
                    matches[did]["exact_title"] = True

            # Compute composite exact score
            e_num = float(matches[did]["exact_legal_number"])
            e_title = float(matches[did]["exact_title"])
            e_overlap = float(matches[did]["exact_title_overlap"])
            e_art = float(matches[did]["exact_article"])
            e_cl = float(matches[did]["exact_clause"])
            e_pt = float(matches[did]["exact_point"])
            e_yr = float(matches[did]["exact_year"])
            e_tp = float(matches[did]["exact_doc_type"])

            composite_score = (
                3.0 * e_num
                + 1.5 * e_title
                + 1.0 * e_overlap
                + 1.0 * e_art
                + 0.5 * e_cl
                + 0.5 * e_pt
                + 0.5 * e_yr
                + 0.5 * e_tp
            )
            matches[did]["exact_score"] = float(composite_score)
            matches[did]["score"] = max(matches[did]["score"], composite_score)

        return dict(matches)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return ranked list of candidate document hits."""
        match_dict = self.match(query)
        results = []
        for did, data in match_dict.items():
            entry = {
                "doc_id": str(did),
                "score": float(data.get("exact_score", 1.0) * 100.0),
                "exact_score": float(data.get("exact_score", 1.0)),
                "branch": "exact",
            }
            entry.update(data)
            results.append(entry)
        results.sort(key=lambda x: (-x["score"], str(x["doc_id"])))
        return results[:top_k]

    retrieve = search


# Backward-compatibility alias
LegalMatcher = ExactMatcher
