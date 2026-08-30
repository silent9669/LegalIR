from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import math
import re
import unicodedata


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
CHUNK_COLUMNS = ("doc_id", "chunk_id", "granularity", "article", "text_norm", "text_raw")
DOCUMENT_COLUMNS = ("doc_id", "title", "legal_number", "name_raw")
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


class EvidencePackBuilder:
    """Build concise, query-aware evidence packs for candidate documents.

    ``macro_chunks`` and ``doc_metadata`` may be supplied as record iterables,
    pandas DataFrames, or Parquet paths.  The record-based constructor remains
    supported for the pipeline and tests, while the path form consumes the
    canonical ``chunks.parquet`` and ``documents.parquet`` files directly.
    """

    def __init__(
        self,
        macro_chunks: Iterable[Mapping[str, Any]] | Any | str | Path | None = None,
        doc_metadata: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any | str | Path | None = None,
        *,
        max_chunks: int = 2,
        max_chars: int = 1200,
        chunks_path: str | Path | None = None,
        documents_path: str | Path | None = None,
        chunks_parquet: str | Path | None = None,
        documents_parquet: str | Path | None = None,
    ):
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

        chunk_records = self._records_from_source(macro_chunks, CHUNK_COLUMNS)
        # Canonical chunks contain both granularities. Prefer macro chunks when
        # a full chunks.parquet file is provided, matching reranker semantics.
        if any(record.get("granularity") == "macro" for record in chunk_records):
            chunk_records = [record for record in chunk_records if record.get("granularity") == "macro"]

        self.chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunk_records:
            if not _is_present(chunk.get("doc_id")):
                continue
            self.chunks_by_doc[_text_value(chunk["doc_id"])].append(dict(chunk))

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
    def _records_from_source(
        source: Any,
        columns: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        if source is None:
            return []
        if isinstance(source, (str, Path)):
            import pandas as pd

            frame = pd.read_parquet(source, columns=list(columns) if columns else None)
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
            # Metadata is normally keyed by document ID. Also accept one row
            # supplied directly as a convenience.
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
            for key in ("title", "legal_number", "name_raw"):
                if key in candidate_record and _is_present(candidate_record[key]):
                    metadata.setdefault(key, candidate_record[key])
        return metadata

    @staticmethod
    def _chunk_body(chunk: Mapping[str, Any]) -> str:
        for key in ("text_raw", "text_norm"):
            if _is_present(chunk.get(key)):
                return _text_value(chunk[key])
        return ""

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

    def _select_chunks(
        self,
        query: str,
        doc_id: str,
        candidate_record: Mapping[str, Any] | None,
        max_chunks: int,
    ) -> list[dict[str, Any]]:
        doc_chunks = self.chunks_by_doc.get(str(doc_id), [])
        if len(doc_chunks) <= max_chunks:
            return list(doc_chunks)

        query_terms = Counter(tokenize(query))
        prior_best_cid = None
        if candidate_record:
            prior_best_cid = (
                candidate_record.get("dense_best_chunk_id")
                or candidate_record.get("bm25_best_chunk_id")
            )

        scored_chunks: list[tuple[float, int, dict[str, Any]]] = []
        for index, chunk in enumerate(doc_chunks):
            chunk_terms = Counter(tokenize(self._chunk_body(chunk)))
            if not chunk_terms:
                score = 0.0
            else:
                overlap = sum(
                    frequency * (1.0 + math.log(1.0 + chunk_terms[term]))
                    for term, frequency in query_terms.items()
                    if term in chunk_terms
                )
                score = overlap / (math.sqrt(sum(chunk_terms.values())) + 1.0)
            if prior_best_cid and str(chunk.get("chunk_id")) == str(prior_best_cid):
                score += 5.0
            # Keep source order as the deterministic tie-breaker.
            scored_chunks.append((score, -index, chunk))

        scored_chunks.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [chunk for _, _, chunk in scored_chunks[:max_chunks]]

    def build_pack(
        self,
        query: str,
        doc_id: Any,
        candidate_record: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
        max_chunks: int | None = None,
        max_chars: int | None = None,
        *,
        candidate_chunks: Iterable[Mapping[str, Any]] | None = None,
    ) -> str:
        """Build one structured document pack in the canonical format."""
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
        metadata = self._metadata_for(doc_id, candidate_record)
        title = _text_value(
            metadata.get("title"),
            _text_value(metadata.get("name_raw"), f"Văn bản {doc_id}"),
        )
        legal_number = _text_value(metadata.get("legal_number"))
        document_label = f"{title} {legal_number}".strip()

        sections = [f"[QUESTION] {str(query).strip()}", f"[DOCUMENT] {document_label}"]
        if not selected_chunks:
            selected_chunks = [{"chunk_id": f"{doc_id}_fallback", "text_norm": f"Văn bản pháp luật {doc_id}"}]
        for index, chunk in enumerate(selected_chunks, start=1):
            body = self._chunk_body(chunk)[:max_chars].strip()
            sections.append(f"[EVIDENCE {index}] {body}")
        return " ".join(sections)

    def build(
        self,
        query: str,
        doc_id: Any,
        candidate_record: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
        max_chunks: int | None = None,
        max_chars: int | None = None,
        *,
        candidate_chunks: Iterable[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build evidence records for a candidate document.

        The first record exposes the complete structured pack in ``pack`` and
        ``evidence_text``. ``text`` also carries the legacy labels for callers
        that still consume the pre-v2 representation. Each record includes a
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
            candidate_chunks=selected_chunks,
        )
        metadata = self._metadata_for(doc_id, candidate_record)
        title = _text_value(
            metadata.get("title"),
            _text_value(metadata.get("name_raw"), f"Văn bản {doc_id}"),
        )
        legal_number = _text_value(metadata.get("legal_number"))
        document_label = f"{title} {legal_number}".strip()
        header = f"[QUESTION] {str(query).strip()} [DOCUMENT] {document_label}"

        if not selected_chunks:
            selected_chunks = [{"chunk_id": f"{doc_id}_fallback", "text_norm": f"Văn bản pháp luật {doc_id}"}]

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
            })
        return records

    def build_evidence(self, query: str, doc_id: Any, max_chars: int | None = None) -> dict[str, Any]:
        """Backwards-compatible helper returning the highest-ranked evidence."""
        packs = self.build(query, doc_id, max_chunks=1, max_chars=max_chars)
        return packs[0] if packs else {
            "doc_id": self._candidate_doc_id(doc_id),
            "chunk_id": f"{doc_id}_fallback",
            "evidence_text": self.build_pack(query, doc_id, max_chunks=1, max_chars=max_chars),
            "article": "Văn bản",
        }
