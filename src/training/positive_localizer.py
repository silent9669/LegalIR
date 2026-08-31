from collections import defaultdict, Counter
from typing import Any
import math
import re
import unicodedata

from src.dataset.normalize import clean_legal_text, extract_legal_signals

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\bkhoản\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
POINT_RE = re.compile(r"\bđiểm\s+([a-zA-Z\d]+)", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFC", str(text)).lower()
    return TOKEN_PATTERN.findall(text)


class PositiveLocalizer:
    """Find the most relevant, query-aware macro chunk(s) inside a document."""

    def __init__(self, macro_chunks: list[dict[str, Any]]):
        self.chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in macro_chunks:
            did = str(c.get("doc_id", ""))
            body = str(c.get("text_raw") or c.get("text_norm") or "")
            art = str(c.get("article") or "")
            cl = str(c.get("clause") or "")
            pt = str(c.get("point") or "")

            art_lower = art.lower()
            cl_lower = cl.lower()
            pt_lower = pt.lower()
            body_prefix_lower = body[:600].lower()

            art_nums = set(ARTICLE_RE.findall(art_lower) + ARTICLE_RE.findall(body_prefix_lower))
            cl_nums = set(CLAUSE_RE.findall(cl_lower) + CLAUSE_RE.findall(body_prefix_lower))
            pt_nums = set(POINT_RE.findall(pt_lower) + POINT_RE.findall(body_prefix_lower))

            toks = tokenize(body)
            tf = Counter(toks)
            if art:
                for a_tok in tokenize(art):
                    tf[a_tok] = tf.get(a_tok, 0) + 2

            processed = dict(c)
            processed.update({
                "body": body,
                "tf": tf,
                "token_set": set(toks),
                "token_count": len(toks),
                "art_nums": art_nums,
                "cl_nums": cl_nums,
                "pt_nums": pt_nums,
                "art_lower": art_lower,
                "cl_lower": cl_lower,
                "pt_lower": pt_lower,
                "body_lower": body.lower(),
            })
            self.chunks_by_doc[did].append(processed)

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
            return doc_chunks[:top_k] if top_k is not None else doc_chunks[0]

        clean_q = clean_legal_text(query)
        signals = extract_legal_signals(clean_q)
        q_toks = tokenize(clean_q)
        if not q_toks:
            return doc_chunks[:top_k] if top_k is not None else doc_chunks[0]

        q_tf = Counter(q_toks)
        q_articles = {str(a).lower().strip() for a in signals.get("articles", [])}
        q_clauses = {str(c).lower().strip() for c in signals.get("clauses", [])}
        q_points = {str(p).lower().strip() for p in signals.get("points", [])}
        q_nums = {str(n).lower().strip() for n in signals.get("doc_numbers", [])}

        scored_chunks: list[tuple[float, int, dict[str, Any]]] = []

        for idx, chunk in enumerate(doc_chunks):
            score = 0.0
            c_art_nums = chunk["art_nums"]
            c_cl_nums = chunk["cl_nums"]
            c_pt_nums = chunk["pt_nums"]
            body_lower = chunk["body_lower"]
            art_lower = chunk["art_lower"]

            # Statutory article matching
            if q_articles:
                if q_articles.intersection(c_art_nums):
                    score += 20.0
                elif any(f"điều {qa}" in art_lower or art_lower == f"điều {qa}" or art_lower == qa for qa in q_articles):
                    score += 18.0
                elif any(f"điều {qa}" in body_lower for qa in q_articles):
                    score += 12.0

            # Statutory clause matching
            if q_clauses:
                if q_clauses.intersection(c_cl_nums):
                    score += 8.0
                elif any(f"khoản {qc}" in body_lower or f"{qc}." in body_lower for qc in q_clauses):
                    score += 5.0

            # Statutory point matching
            if q_points:
                if q_points.intersection(c_pt_nums):
                    score += 4.0
                elif any(f"điểm {qp}" in body_lower or f"{qp})" in body_lower for qp in q_points):
                    score += 3.0

            # Doc number matching
            if q_nums and any(qn in body_lower for qn in q_nums):
                score += 6.0

            # TF-IDF lexical overlap
            c_tf = chunk["tf"]
            c_tok_count = chunk["token_count"]
            if c_tok_count > 0:
                overlap = 0.0
                for term, qf in q_tf.items():
                    if term in c_tf:
                        overlap += qf * (1.0 + math.log(1.0 + c_tf[term]))
                score += (overlap / (math.sqrt(c_tok_count) + 1.0)) * 2.0

            scored_chunks.append((score, -idx, chunk))

        scored_chunks.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Deduplication
        selected: list[dict[str, Any]] = []
        selected_tokens: list[set[str]] = []

        k = top_k if top_k is not None else 1
        for _, _, chunk in scored_chunks:
            c_toks = chunk["token_set"]
            is_dup = False
            for prev_toks in selected_tokens:
                if not prev_toks:
                    continue
                intersection_len = len(c_toks & prev_toks)
                union_len = len(c_toks | prev_toks)
                if (intersection_len / max(1, union_len)) >= 0.85:
                    is_dup = True
                    break

            if not is_dup:
                selected.append(chunk)
                selected_tokens.append(c_toks)
                if len(selected) == k:
                    break

        if len(selected) < k:
            for _, _, chunk in scored_chunks:
                if chunk not in selected:
                    selected.append(chunk)
                    if len(selected) == k:
                        break

        if top_k is not None:
            return selected[:top_k]
        return selected[0] if selected else doc_chunks[0]
