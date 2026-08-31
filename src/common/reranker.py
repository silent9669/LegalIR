"""Compatibility wrapper re-exporting canonical CrossEncoderReranker as BGEReranker."""

from src.ranking.reranker import BGEReranker, CrossEncoderReranker

__all__ = ["BGEReranker", "CrossEncoderReranker"]
