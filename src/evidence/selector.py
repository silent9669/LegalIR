"""Lazy positive localizer and evidence pack builder backed by MacroEvidenceStore."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

from src.evidence.macro_store import MacroChunk, MacroEvidenceStore
from src.ranking.evidence_pack import EvidencePackBuilder
from src.training.positive_localizer import PositiveLocalizer


class LazyPositiveLocalizer:
    """
    Finds the most relevant, query-aware macro chunk for a positive document lazily.
    Queries the MacroEvidenceStore on the fly, eliminating full-corpus memory retention.
    Guarantees exact mathematical parity with legacy PositiveLocalizer.
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
        chunks = self.evidence_store.get_doc_chunks(gold_doc_id)
        if not chunks:
            return [] if top_k is not None else None

        chunk_dicts = [c.to_dict() for c in chunks]
        if len(chunk_dicts) == 1:
            c = chunk_dicts[0]
            return [c] if top_k is not None else c

        loc = PositiveLocalizer(chunk_dicts)
        return loc.localize(query, gold_doc_id, top_k=top_k)


class LazyEvidencePackBuilder:
    """
    Builds query-aware evidence packs for candidate documents lazily.
    Interacts with MacroEvidenceStore without requiring preloaded corpus dictionaries.
    Guarantees exact mathematical parity with legacy EvidencePackBuilder.
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
        self._legacy_builder = EvidencePackBuilder(
            macro_chunks=[],
            doc_metadata=self.doc_metadata,
            max_chunks=self.max_chunks,
            max_chars=self.max_chars,
            max_tokens=self.max_tokens,
        )

    def build_pack(
        self,
        query: str,
        doc_id: str,
        candidate_record: Optional[Mapping[str, Any]] = None,
        max_chunks: Optional[int] = None,
        max_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        include_question: bool = False,
    ) -> str:
        did = str(doc_id)
        chunks = self.evidence_store.get_doc_chunks(did)
        chunk_dicts = [c.to_dict() for c in chunks]

        return self._legacy_builder.build_pack(
            query=query,
            doc_id=did,
            candidate_record=candidate_record,
            max_chunks=max_chunks or self.max_chunks,
            max_chars=max_chars or self.max_chars,
            max_tokens=max_tokens or self.max_tokens,
            include_question=include_question,
            candidate_chunks=chunk_dicts,
        )

    def build(
        self,
        query: str,
        doc_id: str,
        candidate_record: Optional[Mapping[str, Any]] = None,
        max_chunks: Optional[int] = None,
        max_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        include_question: bool = False,
    ) -> List[Dict[str, Any]]:
        did = str(doc_id)
        chunks = self.evidence_store.get_doc_chunks(did)
        chunk_dicts = [c.to_dict() for c in chunks]

        return self._legacy_builder.build(
            query=query,
            doc_id=did,
            candidate_record=candidate_record,
            max_chunks=max_chunks or self.max_chunks,
            max_chars=max_chars or self.max_chars,
            max_tokens=max_tokens or self.max_tokens,
            include_question=include_question,
            candidate_chunks=chunk_dicts,
        )

    def build_evidence(
        self,
        query: str,
        doc_id: str,
        max_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        records = self.build(query, doc_id, max_chunks=1, max_chars=max_chars, max_tokens=max_tokens)
        return records[0] if records else {
            "doc_id": str(doc_id),
            "chunk_id": f"{doc_id}_fallback",
            "evidence_text": self.build_pack(query, doc_id, max_chunks=1, max_chars=max_chars, max_tokens=max_tokens),
            "article": "Văn bản",
        }
