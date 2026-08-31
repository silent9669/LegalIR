"""Compatibility wrapper re-exporting canonical normalization utilities."""

from src.dataset.normalize import (
    ARTICLE_PATTERN,
    CLAUSE_PATTERN,
    DOC_NUMBER_PATTERN,
    POINT_PATTERN,
    YEAR_PATTERN,
    clean_legal_text,
    extract_legal_signals,
    normalize_question,
    prettify_doc_title,
    tokenize_vietnamese,
)

__all__ = [
    "ARTICLE_PATTERN",
    "CLAUSE_PATTERN",
    "DOC_NUMBER_PATTERN",
    "POINT_PATTERN",
    "YEAR_PATTERN",
    "clean_legal_text",
    "extract_legal_signals",
    "normalize_question",
    "prettify_doc_title",
    "tokenize_vietnamese",
]
