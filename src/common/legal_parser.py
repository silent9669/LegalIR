"""Compatibility wrapper re-exporting canonical legal parser utilities."""

from src.dataset.legal_parser import (
    ARTICLE_PATTERN,
    CHAPTER_PATTERN,
    CHƯƠNG_PATTERN,
    CLAUSE_PATTERN,
    ĐIỀU_PATTERN,
    ĐIỂM_PATTERN,
    KHOẢN_PATTERN,
    LegalUnit,
    MỤC_PATTERN,
    POINT_PATTERN,
    SECTION_PATTERN,
    parse_legal_structure,
    parse_legal_units,
)

__all__ = [
    "ARTICLE_PATTERN",
    "CHAPTER_PATTERN",
    "CHƯƠNG_PATTERN",
    "CLAUSE_PATTERN",
    "ĐIỀU_PATTERN",
    "ĐIỂM_PATTERN",
    "KHOẢN_PATTERN",
    "LegalUnit",
    "MỤC_PATTERN",
    "POINT_PATTERN",
    "SECTION_PATTERN",
    "parse_legal_structure",
    "parse_legal_units",
]
