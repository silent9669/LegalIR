"""Compatibility package re-exporting canonical modules."""

from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.evidence import EvidencePackBuilder
from src.common.legal_parser import parse_legal_structure
from src.common.normalize import (
    clean_legal_text,
    extract_legal_signals,
    normalize_question,
    prettify_doc_title,
    tokenize_vietnamese,
)
from src.common.reranker import BGEReranker
from src.common.rrf import reciprocal_rank_fusion

__all__ = [
    "BGEReranker",
    "BM25Retriever",
    "DEk21Retriever",
    "EvidencePackBuilder",
    "clean_legal_text",
    "extract_legal_signals",
    "normalize_question",
    "parse_legal_structure",
    "prettify_doc_title",
    "reciprocal_rank_fusion",
    "tokenize_vietnamese",
]
