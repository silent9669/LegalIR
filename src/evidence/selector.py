"""Lazy positive localizer and evidence pack builder backed by MacroEvidenceStore."""

from __future__ import annotations

import collections
import math
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from src.dataset.normalize import clean_legal_text, extract_legal_signals, prettify_doc_title
from src.evidence.macro_store import MacroChunk, MacroEvidenceStore

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\bkhoản\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
POINT_RE = re.compile(r"\bđiểm\s+([a-zA-Z\d]+)", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFC", str(text)).lower()
    return TOKEN_PATTERN.findall(text)


def _truncate_to_word_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    target = max_chars - 3
    space_idx = text.rfind(" ", 0, target)
    if space_idx > target // 2:
        return text[:space_idx].rstrip() + "..."
    return text[:target].rstrip() + "..."


class LazyPositiveLocalizer:
    """
    Finds the most relevant, query-aware macro chunk for a positive document lazily.
    Queries the MacroEvidenceStore on the fly, eliminating full-corpus memory retention.
    """

    def __init__(self, evidence_store: MacroEvidenceStore):
        self.evidence_store = evidence_store

    def localize(
        self,
        query: str,
        gold_doc_id: str,
        top_k: Optional[int] = None,
    ) -> Any:
        gold_doc_id = str(gold_doc_id)
        doc_chunks = self.evidence_store.get_doc_chunks(gold_doc_id)

        if not doc_chunks:
            return [] if top_k is not None else None

        if len(doc_chunks) == 1:
            return doc_chunks[:top_k] if top_k is not None else doc_chunks[0]

        clean_q = clean_legal_text(query)
        signals = extract_legal_signals(clean_q)
        q_toks = tokenize(clean_q)
        if not q_toks:
            return doc_chunks[:top_k] if top_k is not None else doc_chunks[0]

        q_tf = collections.Counter(q_toks)
        q_articles = {str(a).lower().strip() for a in signals.get("articles", [])}
        q_clauses = {str(c).lower().strip() for c in signals.get("clauses", [])}
        q_points = {str(p).lower().strip() for p in signals.get("points", [])}
        q_nums = {str(n).lower().strip() for n in signals.get("doc_numbers", [])}

        scored_chunks: List[Tuple[float, int, MacroChunk]] = []

        for idx, chunk in enumerate(doc_chunks):
            score = 0.0
            body = chunk.text or ""
            body_lower = body.lower()
            body_prefix_lower = body[:600].lower()

            art_nums = set(ARTICLE_RE.findall(body_prefix_lower))
            cl_nums = set(CLAUSE_RE.findall(body_prefix_lower))
            pt_nums = set(POINT_RE.findall(body_prefix_lower))

            # Statutory article matching
            if q_articles:
                if q_articles.intersection(art_nums):
                    score += 20.0
                elif any(f"điều {qa}" in body_lower for qa in q_articles):
                    score += 12.0

            # Statutory clause matching
            if q_clauses:
                if q_clauses.intersection(cl_nums):
                    score += 8.0
                elif any(f"khoản {qc}" in body_lower or f"{qc}." in body_lower for qc in q_clauses):
                    score += 5.0

            # Statutory point matching
            if q_points:
                if q_points.intersection(pt_nums):
                    score += 4.0
                elif any(f"điểm {qp}" in body_lower or f"{qp})" in body_lower for qp in q_points):
                    score += 3.0

            # Doc number matching
            if q_nums and any(qn in body_lower for qn in q_nums):
                score += 6.0

            # Lexical TF-IDF overlap
            c_toks = tokenize(body)
            c_tf = collections.Counter(c_toks)
            c_tok_count = len(c_toks)
            if c_tok_count > 0:
                overlap = 0.0
                for term, qf in q_tf.items():
                    if term in c_tf:
                        overlap += qf * (1.0 + math.log(1.0 + c_tf[term]))
                score += (overlap / (math.sqrt(c_tok_count) + 1.0)) * 2.0

            scored_chunks.append((score, -idx, chunk))

        scored_chunks.sort(key=lambda x: (x[0], x[1]), reverse=True)

        k = top_k if top_k is not None else 1
        selected = [c for _, _, c in scored_chunks[:k]]

        if top_k is not None:
            return selected
        return selected[0] if selected else doc_chunks[0]


class LazyEvidencePackBuilder:
    """
    Builds query-aware evidence packs for candidate documents lazily.
    Interacts with MacroEvidenceStore without requiring preloaded corpus dictionaries.
    """

    def __init__(
        self,
        evidence_store: MacroEvidenceStore,
        doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        max_chunks: int = 2,
        max_chars: int = 1200,
        max_tokens: Optional[int] = None,
    ):
        self.evidence_store = evidence_store
        self.doc_metadata = doc_metadata or {}
        self.max_chunks = max_chunks
        self.max_chars = max_chars
        self.max_tokens = max_tokens

    def _select_chunks(self, query: str, doc_id: str, max_chunks: int) -> List[MacroChunk]:
        localizer = LazyPositiveLocalizer(self.evidence_store)
        chunks = localizer.localize(query, doc_id, top_k=max_chunks)
        return chunks if isinstance(chunks, list) else ([chunks] if chunks is not None else [])

    def build_pack(
        self,
        query: str,
        doc_id: str,
        max_chunks: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        k = max_chunks or self.max_chunks
        chars_limit = max_chars or self.max_chars

        selected_chunks = self._select_chunks(query, doc_id, k)

        metadata = self.doc_metadata.get(str(doc_id), {})
        title = metadata.get("title") or metadata.get("name_raw") or f"Văn bản {doc_id}"
        legal_num = metadata.get("legal_number") or ""
        doc_header = f"{title} {legal_num}".strip() if legal_num else title

        sections = [f"[DOCUMENT] {doc_header}"]
        if not selected_chunks:
            sections.append(f"[EVIDENCE 1] {doc_header}")
        else:
            for idx, c in enumerate(selected_chunks, start=1):
                body = c.text.strip() if c and c.text else ""
                if len(body) > chars_limit:
                    body = _truncate_to_word_boundary(body, chars_limit)
                sections.append(f"[EVIDENCE {idx}] {body}")

        return " ".join(sections)
