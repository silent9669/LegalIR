from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import math
import re
import unicodedata

from src.dataset.normalize import clean_legal_text, extract_legal_signals, prettify_doc_title

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\bkhoản\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
POINT_RE = re.compile(r"\bđiểm\s+([a-zA-Z\d]+)", re.IGNORECASE)
DOC_NUMBER_RE = re.compile(r"\b\d{1,5}/(?:\d{4}(?:/[A-ZĐa-z\-]+)?|(?:[A-ZĐa-z]+-[A-ZĐa-z]+))\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

CHUNK_COLUMNS = ("doc_id", "chunk_id", "granularity", "chapter", "section", "article", "clause", "point", "text_norm", "text_raw")
DOCUMENT_COLUMNS = ("doc_id", "title", "legal_number", "name_raw", "year", "doc_type")
MISSING_STRINGS = frozenset({"", "nan", "nat", "none", "null"})


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in MISSING_STRINGS
    try:
        import pandas as pd

        missing = pd.isna(value)
        return not bool(missing)
    except (TypeError, ValueError):
        return True


def _text_value(value: Any, default: str = "") -> str:
    if not _is_present(value):
        return default
    return str(value).strip()


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFC", str(text)).lower()
    return TOKEN_PATTERN.findall(text)


def _truncate_to_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text cleanly at word boundary with ellipsis."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    target = max_chars - 3
    space_idx = text.rfind(" ", 0, target)
    if space_idx > target // 2:
        return text[:space_idx].rstrip() + "..."
    return text[:target].rstrip() + "..."


class EvidencePackBuilder:
    """Build concise, query-aware evidence packs for candidate documents.

    Selects the most relevant, complementary chunks within each document based on
    a combination of:
      1. Statutory article/clause/point exact matching (e.g. Điều 61, Khoản 2, Điểm a).
      2. Weighted lexical token overlap / TF-IDF scoring.
      3. Exact n-gram phrase matching.
      4. Prior retrieval signals (e.g. dense_best_chunk_id, bm25_best_chunk_id).
      5. Complementary chunk selection with near-duplicate suppression.
      6. Token budget awareness (configurable max_tokens, max_chars).
    """

    def __init__(
        self,
        macro_chunks: Iterable[Mapping[str, Any]] | Any | str | Path | None = None,
        doc_metadata: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any | str | Path | None = None,
        *,
        max_chunks: int = 2,
        max_chars: int = 1200,
        max_tokens: int | None = None,
        max_chunks_per_doc: int | None = None,
        max_chars_per_chunk: int | None = None,
        chunks_path: str | Path | None = None,
        documents_path: str | Path | None = None,
        chunks_parquet: str | Path | None = None,
        documents_parquet: str | Path | None = None,
    ):
        if max_chunks_per_doc is not None:
            max_chunks = max_chunks_per_doc
        if max_chars_per_chunk is not None:
            max_chars = max_chars_per_chunk

        if chunks_path is not None or chunks_parquet is not None:
            if macro_chunks is not None:
                raise ValueError("provide macro_chunks or chunks_path, not both")
            macro_chunks = chunks_path or chunks_parquet
        if documents_path is not None or documents_parquet is not None:
            if doc_metadata is not None:
                raise ValueError("provide doc_metadata or documents_path, not both")
            doc_metadata = documents_path or documents_parquet

        self.max_chunks = self._validate_limit(max_chunks, "max_chunks")
        self.max_chars = self._validate_limit(max_chars, "max_chars")
        self.max_tokens = int(max_tokens) if max_tokens is not None and max_tokens > 0 else None

        chunk_records = self._records_from_source(macro_chunks, CHUNK_COLUMNS)
        # Canonical chunks contain both granularities. Prefer macro chunks when
        # a full chunks.parquet file is provided, matching reranker semantics.
        if any(record.get("granularity") == "macro" for record in chunk_records):
            chunk_records = [record for record in chunk_records if record.get("granularity") == "macro"]

        self.chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunk_records:
            if not _is_present(chunk.get("doc_id")):
                continue
            did = _text_value(chunk["doc_id"])
            processed = self._preprocess_chunk(chunk, did)
            self.chunks_by_doc[did].append(processed)

        self.doc_metadata = self._metadata_from_source(doc_metadata)

    @staticmethod
    def _validate_limit(value: int, name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if parsed < 1:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    @staticmethod
    def _preprocess_chunk(chunk: Mapping[str, Any], did: str) -> dict[str, Any]:
        """Pre-extract structural signals and terms for sub-millisecond retrieval scoring."""
        body = _text_value(chunk.get("text_raw") or chunk.get("text_norm") or "")
        art = _text_value(chunk.get("article") or "")
        cl = _text_value(chunk.get("clause") or "")
        pt = _text_value(chunk.get("point") or "")

        art_lower = art.lower()
        cl_lower = cl.lower()
        pt_lower = pt.lower()
        body_prefix_lower = body[:600].lower()

        # Extract article, clause, point numbers
        art_nums = set(ARTICLE_RE.findall(art_lower) + ARTICLE_RE.findall(body_prefix_lower))
        cl_nums = set(CLAUSE_RE.findall(cl_lower) + CLAUSE_RE.findall(body_prefix_lower))
        pt_nums = set(POINT_RE.findall(pt_lower) + POINT_RE.findall(body_prefix_lower))

        toks = tokenize(body)
        tf = Counter(toks)
        # Add article terms to TF for term matching
        if art:
            for art_tok in tokenize(art):
                tf[art_tok] = tf.get(art_tok, 0) + 2

        return {
            "chunk_id": _text_value(chunk.get("chunk_id")),
            "doc_id": did,
            "article": art,
            "clause": cl,
            "point": pt,
            "text_raw": _text_value(chunk.get("text_raw")),
            "text_norm": _text_value(chunk.get("text_norm")),
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
        }

    @staticmethod
    def _records_from_source(
        source: Any,
        columns: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        if source is None:
            return []
        if isinstance(source, (str, Path)):
            import pandas as pd
            import pyarrow.parquet as pq

            try:
                parquet_file = pq.ParquetFile(source)
                existing_cols = set(parquet_file.schema.names)
                available = [c for c in columns if c in existing_cols] if columns else None
                frame = pd.read_parquet(source, columns=available)
            except Exception:
                frame = pd.read_parquet(source)
            return [dict(row) for row in frame.to_dict(orient="records")]
        if hasattr(source, "to_dict"):
            try:
                if columns and hasattr(source, "columns"):
                    available = [column for column in columns if column in source.columns]
                    source = source.loc[:, available]
                return [dict(row) for row in source.to_dict(orient="records")]
            except (AttributeError, TypeError):
                pass
        if isinstance(source, Mapping):
            return [dict(source)]
        return [dict(row) for row in source]

    @classmethod
    def _metadata_from_source(cls, source: Any) -> dict[str, dict[str, Any]]:
        if source is None:
            return {}
        if isinstance(source, (str, Path)):
            records = cls._records_from_source(source, DOCUMENT_COLUMNS)
        elif isinstance(source, Mapping):
            if "doc_id" in source or "title" in source or "legal_number" in source:
                records = [dict(source)]
            else:
                return {
                    str(doc_id): {
                        key: meta[key]
                        for key in DOCUMENT_COLUMNS
                        if key in meta and _is_present(meta[key])
                    }
                    for doc_id, meta in source.items()
                    if _is_present(doc_id) and isinstance(meta, Mapping)
                }
        else:
            records = cls._records_from_source(source, DOCUMENT_COLUMNS)

        metadata: dict[str, dict[str, Any]] = {}
        for record in records:
            if not _is_present(record.get("doc_id")):
                continue
            metadata[str(record["doc_id"])] = {
                key: record[key]
                for key in DOCUMENT_COLUMNS
                if key in record and _is_present(record[key])
            }
        return metadata

    @staticmethod
    def _candidate_doc_id(candidate: Any) -> str:
        if isinstance(candidate, Mapping):
            if "doc_id" not in candidate:
                raise ValueError("candidate record must contain doc_id")
            candidate = candidate["doc_id"]
        elif isinstance(candidate, (tuple, list)):
            if not candidate:
                raise ValueError("candidate tuple must contain a document ID")
            candidate = candidate[0]
        if not _is_present(candidate):
            raise ValueError("document ID cannot be null")
        return _text_value(candidate)

    def _metadata_for(self, doc_id: str, candidate_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
        metadata = dict(self.doc_metadata.get(str(doc_id), {}))
        if candidate_record:
            for key in ("title", "legal_number", "name_raw", "year", "doc_type"):
                if key in candidate_record and _is_present(candidate_record[key]):
                    metadata.setdefault(key, candidate_record[key])
        return metadata

    @staticmethod
    def _chunk_body(chunk: Mapping[str, Any]) -> str:
        for key in ("body", "text_raw", "text_norm"):
            if _is_present(chunk.get(key)):
                return _text_value(chunk[key])
        return ""

    def _extract_query_info(self, query: str) -> dict[str, Any]:
        """Extract tokens, TF counter, and legal signals from query."""
        clean_q = clean_legal_text(query)
        signals = extract_legal_signals(clean_q)
        q_toks = tokenize(clean_q)
        q_tf = Counter(q_toks)

        art_set = {str(a).lower().strip() for a in signals.get("articles", [])}
        cl_set = {str(c).lower().strip() for c in signals.get("clauses", [])}
        pt_set = {str(p).lower().strip() for p in signals.get("points", [])}
        num_set = {str(n).lower().strip() for n in signals.get("doc_numbers", [])}
        year_set = {str(y).strip() for y in signals.get("years", [])}

        q_ngrams = []
        if len(q_toks) >= 3:
            for i in range(len(q_toks) - 2):
                q_ngrams.append(" ".join(q_toks[i : i + 3]))

        return {
            "clean_query": clean_q,
            "tokens": q_toks,
            "tf": q_tf,
            "articles": art_set,
            "clauses": cl_set,
            "points": pt_set,
            "doc_numbers": num_set,
            "years": year_set,
            "ngrams": q_ngrams,
        }

    def _score_chunk(
        self,
        chunk: dict[str, Any],
        q_info: dict[str, Any],
        prior_best_cid: str | None,
        candidate_record: Mapping[str, Any] | None,
    ) -> float:
        """Compute query-aware relevance score for a single chunk."""
        score = 0.0
        c_tf = chunk.get("tf") or Counter(tokenize(self._chunk_body(chunk)))
        c_tok_count = chunk.get("token_count") or sum(c_tf.values())
        c_art_nums = chunk.get("art_nums") or set(ARTICLE_RE.findall(chunk.get("art_lower", "")))
        c_cl_nums = chunk.get("cl_nums") or set(CLAUSE_RE.findall(chunk.get("cl_lower", "")))
        c_pt_nums = chunk.get("pt_nums") or set(POINT_RE.findall(chunk.get("pt_lower", "")))
        body_lower = chunk.get("body_lower") or self._chunk_body(chunk).lower()
        art_lower = chunk.get("art_lower") or str(chunk.get("article", "")).lower()

        # 1. Statutory Article Matching (Primary boost)
        q_articles = q_info["articles"]
        if q_articles:
            if q_articles.intersection(c_art_nums):
                score += 20.0
            elif any(f"điều {qa}" in art_lower or art_lower == f"điều {qa}" or art_lower == qa for qa in q_articles):
                score += 18.0
            elif any(f"điều {qa}" in body_lower for qa in q_articles):
                score += 12.0

        # 2. Statutory Clause Matching
        q_clauses = q_info["clauses"]
        if q_clauses:
            if q_clauses.intersection(c_cl_nums):
                score += 8.0
            elif any(f"khoản {qc}" in body_lower or f"{qc}." in body_lower for qc in q_clauses):
                score += 5.0

        # 3. Statutory Point Matching
        q_points = q_info["points"]
        if q_points:
            if q_points.intersection(c_pt_nums):
                score += 4.0
            elif any(f"điểm {qp}" in body_lower or f"{qp})" in body_lower for qp in q_points):
                score += 3.0

        # 4. Document Number / Year Matching
        q_nums = q_info["doc_numbers"]
        if q_nums and any(qn in body_lower for qn in q_nums):
            score += 6.0

        q_years = q_info["years"]
        if q_years and any(qy in body_lower for qy in q_years):
            score += 1.5

        # 5. Token Overlap / Weighted TF Matching
        q_tf = q_info["tf"]
        if c_tok_count > 0:
            overlap = 0.0
            for term, qf in q_tf.items():
                if term in c_tf:
                    overlap += qf * (1.0 + math.log(1.0 + c_tf[term]))
            lexical_score = overlap / (math.sqrt(c_tok_count) + 1.0)
            score += lexical_score * 2.0

        # 6. Exact Phrase N-Gram Matching
        q_ngrams = q_info["ngrams"]
        if q_ngrams:
            for ng in q_ngrams:
                if ng in body_lower:
                    score += 2.5

        # 7. Prior Best Chunk Signal
        cid = str(chunk.get("chunk_id", ""))
        if prior_best_cid and cid == str(prior_best_cid):
            score += 5.0

        if candidate_record and "chunk_scores" in candidate_record:
            chunk_scores_map = candidate_record["chunk_scores"]
            if isinstance(chunk_scores_map, Mapping) and cid in chunk_scores_map:
                try:
                    score += float(chunk_scores_map[cid])
                except (TypeError, ValueError):
                    pass

        return score

    def _select_chunks(
        self,
        query: str,
        doc_id: str,
        candidate_record: Mapping[str, Any] | None,
        max_chunks: int,
    ) -> list[dict[str, Any]]:
        doc_chunks = self.chunks_by_doc.get(str(doc_id), [])
        if not doc_chunks:
            return []
        if len(doc_chunks) == 1:
            return list(doc_chunks)

        q_info = self._extract_query_info(query)
        prior_best_cid = None
        if candidate_record:
            prior_best_cid = (
                candidate_record.get("dense_best_chunk_id")
                or candidate_record.get("bm25_best_chunk_id")
                or candidate_record.get("best_chunk_id")
            )

        scored_chunks: list[tuple[float, int, dict[str, Any]]] = []
        for index, chunk in enumerate(doc_chunks):
            score = self._score_chunk(chunk, q_info, prior_best_cid, candidate_record)
            # Source index tie-breaker
            scored_chunks.append((score, -index, chunk))

        scored_chunks.sort(key=lambda item: (item[0], item[1]), reverse=True)

        # Complementary selection with near-duplicate suppression
        selected: list[dict[str, Any]] = []
        selected_tokens: list[set[str]] = []

        for _, _, chunk in scored_chunks:
            c_toks = chunk.get("token_set") or set(tokenize(self._chunk_body(chunk)))
            if not c_toks:
                continue

            # Deduplication: check token Jaccard similarity against already selected chunks
            is_dup = False
            for prev_toks in selected_tokens:
                if not prev_toks:
                    continue
                intersection_len = len(c_toks & prev_toks)
                union_len = len(c_toks | prev_toks)
                jaccard = intersection_len / max(1, union_len)
                if jaccard >= 0.85:
                    is_dup = True
                    break

            if not is_dup:
                selected.append(chunk)
                selected_tokens.append(c_toks)
                if len(selected) == max_chunks:
                    break

        # Fallback if deduplication filtered too aggressively
        if len(selected) < max_chunks:
            for _, _, chunk in scored_chunks:
                if chunk not in selected:
                    selected.append(chunk)
                    if len(selected) == max_chunks:
                        break

        return selected

    def format_evidence_text(self, chunk: Mapping[str, Any], doc_meta: Mapping[str, Any] | None = None) -> str:
        """Return the legacy per-chunk representation used by older callers."""
        did = str(chunk.get("doc_id", ""))
        meta = dict(doc_meta or self.doc_metadata.get(did, {}))
        title = _text_value(meta.get("title"), _text_value(meta.get("name_raw"), f"Văn bản {did}"))
        legal_num = _text_value(meta.get("legal_number"))
        article = _text_value(chunk.get("article"), "Thông tin văn bản")
        body = self._chunk_body(chunk)

        header = f"[VĂN BẢN]: {title}"
        if legal_num:
            header += f" (Số: {legal_num})"
        return f"{header}\n[ĐIỀU KHOẢN]: {article}\n[NỘI DUNG]:\n{body}"

    def build_evidence_text(
        self,
        query: str,
        doc_info: dict | None,
        chunks: list[dict] | None,
        include_question: bool = False,
    ) -> str:
        """Build multiline evidence text format for cross-encoder reranker.
        Defaults to include_question=False to avoid duplicating sequence A in sequence B.
        """
        doc_info = doc_info or {}
        title = _text_value(doc_info.get("title"), prettify_doc_title(doc_info.get("name_raw", "")))
        legal_number = _text_value(doc_info.get("legal_number"))

        doc_header = f"{title} {legal_number}".strip() if legal_number else title
        if not doc_header:
            doc_header = "Văn bản quy phạm pháp luật"

        sections = []
        if include_question:
            sections.append(f"[QUESTION] {clean_legal_text(query)}")
        sections.append(f"[DOCUMENT] {doc_header}")

        valid_chunks = [c for c in (chunks or []) if c and isinstance(c, dict)][:self.max_chunks]
        if not valid_chunks:
            sections.append(f"[EVIDENCE 1] {doc_header}")
        else:
            for idx, c in enumerate(valid_chunks, start=1):
                art = _text_value(c.get("article"))
                body = _text_value(c.get("text_raw") or c.get("text_norm") or c.get("body", ""))
                if len(body) > self.max_chars:
                    body = _truncate_to_word_boundary(body, self.max_chars)

                chunk_text = f"{art}: {body}".strip() if art and not body.startswith(art) else body
                sections.append(f"[EVIDENCE {idx}] {chunk_text}")

        return "\n".join(sections)

    def build_pack(
        self,
        query: str,
        doc_id: Any,
        candidate_record: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
        max_chunks: int | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        include_question: bool = False,
        include_article_prefix: bool = False,
        *,
        candidate_chunks: Iterable[Mapping[str, Any]] | None = None,
    ) -> str:
        """Build one structured document pack in the canonical format with token budget awareness.
        Defaults to include_question=False so cross-encoders do not duplicate the query in sequence B.
        """
        if isinstance(doc_id, Mapping):
            if candidate_record is None:
                candidate_record = doc_id
            doc_id = self._candidate_doc_id(doc_id)
        else:
            doc_id = self._candidate_doc_id(doc_id)
        if candidate_chunks is None and candidate_record is not None and not isinstance(candidate_record, Mapping):
            candidate_chunks = candidate_record
            candidate_record = None

        max_chunks = self._validate_limit(
            self.max_chunks if max_chunks is None else max_chunks,
            "max_chunks",
        )
        max_chars = self._validate_limit(
            self.max_chars if max_chars is None else max_chars,
            "max_chars",
        )
        eff_max_tokens = max_tokens or self.max_tokens
        if eff_max_tokens is not None:
            # 1 token is roughly 3.5 - 4 chars in Vietnamese. Clamp max_chars to token budget.
            max_chars = min(max_chars, eff_max_tokens * 4)

        selected_chunks = (
            [dict(chunk) for chunk in candidate_chunks][:max_chunks]
            if candidate_chunks is not None
            else self._select_chunks(query, doc_id, candidate_record, max_chunks)
        )
        metadata = self._metadata_for(doc_id, candidate_record)
        title = _text_value(
            metadata.get("title"),
            _text_value(metadata.get("name_raw"), f"Văn bản {doc_id}"),
        )
        legal_number = _text_value(metadata.get("legal_number"))
        document_label = f"{title} {legal_number}".strip() if legal_number else title
        if not document_label:
            document_label = f"Văn bản {doc_id}"

        sections = []
        if include_question:
            sections.append(f"[QUESTION] {str(query).strip()}")
        sections.append(f"[DOCUMENT] {document_label}")

        header_str = " ".join(sections)
        header_len = len(header_str)

        if not selected_chunks:
            selected_chunks = [{"chunk_id": f"{doc_id}_fallback", "body": f"Văn bản pháp luật {doc_id}"}]

        # Budget allocation across evidence chunks
        num_chunks = len(selected_chunks)
        budget_for_evidence = max(100, max_chars - header_len - (num_chunks * 15))

        for index, chunk in enumerate(selected_chunks, start=1):
            art = _text_value(chunk.get("article"))
            cl = _text_value(chunk.get("clause"))
            pt = _text_value(chunk.get("point"))
            body = self._chunk_body(chunk).strip()

            if include_article_prefix:
                label_parts = []
                if art:
                    label_parts.append(art)
                if cl and cl not in art and not art.startswith(cl):
                    label_parts.append(cl)
                if pt and pt not in art and not art.startswith(pt):
                    label_parts.append(pt)

                label_prefix = ", ".join(label_parts)
                if label_prefix and not body.startswith(label_prefix):
                    chunk_content = f"{label_prefix}: {body}".strip()
                else:
                    chunk_content = body
            else:
                chunk_content = body

            # Allocate budget: primary evidence gets larger share
            if num_chunks == 1:
                chunk_budget = budget_for_evidence
            elif index == 1:
                chunk_budget = max(150, int(budget_for_evidence * 0.6))
            else:
                chunk_budget = max(100, int(budget_for_evidence * (0.4 / max(1, num_chunks - 1))))

            if len(chunk_content) > chunk_budget:
                chunk_content = _truncate_to_word_boundary(chunk_content, chunk_budget)

            sections.append(f"[EVIDENCE {index}] {chunk_content}")

        pack = " ".join(sections)

        # If token limit was specified, perform strict token limit trimming
        if eff_max_tokens is not None:
            toks = tokenize(pack)
            if len(toks) > eff_max_tokens:
                word_limit = int(eff_max_tokens * 0.9)
                pack_words = pack.split()
                if len(pack_words) > word_limit:
                    pack = " ".join(pack_words[:word_limit]).rstrip() + "..."

        return pack

    def build(
        self,
        query: str,
        doc_id: Any,
        candidate_record: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
        max_chunks: int | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        include_question: bool = False,
        *,
        candidate_chunks: Iterable[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build evidence records for a candidate document.

        The first record exposes the complete structured pack in ``pack`` and
        ``evidence_text``. ``text`` also carries legacy labels for callers
        that consume the pre-v2 representation. Each record includes a
        one-evidence ``reranker_text`` so chunk-level scores remain distinct.
        """
        if isinstance(doc_id, Mapping):
            if candidate_record is None:
                candidate_record = doc_id
            doc_id = self._candidate_doc_id(doc_id)
        else:
            doc_id = self._candidate_doc_id(doc_id)
        if candidate_chunks is None and candidate_record is not None and not isinstance(candidate_record, Mapping):
            candidate_chunks = candidate_record
            candidate_record = None

        max_chunks = self._validate_limit(
            self.max_chunks if max_chunks is None else max_chunks,
            "max_chunks",
        )
        max_chars = self._validate_limit(
            self.max_chars if max_chars is None else max_chars,
            "max_chars",
        )

        selected_chunks = (
            [dict(chunk) for chunk in candidate_chunks][:max_chunks]
            if candidate_chunks is not None
            else self._select_chunks(query, doc_id, candidate_record, max_chunks)
        )
        pack = self.build_pack(
            query,
            doc_id,
            candidate_record,
            max_chunks=max_chunks,
            max_chars=max_chars,
            max_tokens=max_tokens,
            include_question=include_question,
            candidate_chunks=selected_chunks,
        )
        metadata = self._metadata_for(doc_id, candidate_record)
        title = _text_value(
            metadata.get("title"),
            _text_value(metadata.get("name_raw"), f"Văn bản {doc_id}"),
        )
        legal_number = _text_value(metadata.get("legal_number"))
        document_label = f"{title} {legal_number}".strip() if legal_number else title
        if not document_label:
            document_label = f"Văn bản {doc_id}"

        header_prefix = f"[QUESTION] {str(query).strip()} " if include_question else ""
        header = f"{header_prefix}[DOCUMENT] {document_label}"

        if not selected_chunks:
            selected_chunks = [{"chunk_id": f"{doc_id}_fallback", "body": f"Văn bản pháp luật {doc_id}"}]

        records: list[dict[str, Any]] = []
        for index, chunk in enumerate(selected_chunks, start=1):
            body = self._chunk_body(chunk)[:max_chars].strip()
            reranker_text = f"{header} [EVIDENCE 1] {body}"
            legacy_text = self.format_evidence_text(chunk, metadata)
            # Keep legacy labels in text for API compatibility while exposing
            # the exact structured value through pack/evidence_text.
            text = f"{pack} {legacy_text}" if index == 1 else pack
            records.append({
                "doc_id": doc_id,
                "chunk_id": _text_value(chunk.get("chunk_id"), f"{doc_id}_{index}"),
                "text": text,
                "pack": pack,
                "evidence_text": pack,
                "reranker_text": reranker_text,
                "chunk_text": body,
                "article": _text_value(chunk.get("article")),
                "clause": _text_value(chunk.get("clause")),
                "point": _text_value(chunk.get("point")),
            })
        return records

    def build_evidence(
        self,
        query: str,
        doc_id: Any,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Backwards-compatible helper returning the highest-ranked evidence record."""
        packs = self.build(query, doc_id, max_chunks=1, max_chars=max_chars, max_tokens=max_tokens)
        return packs[0] if packs else {
            "doc_id": self._candidate_doc_id(doc_id),
            "chunk_id": f"{doc_id}_fallback",
            "evidence_text": self.build_pack(query, doc_id, max_chunks=1, max_chars=max_chars, max_tokens=max_tokens),
            "article": "Văn bản",
        }
