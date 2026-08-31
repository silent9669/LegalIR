"""Compatibility wrapper re-exporting canonical BM25MicroRetriever."""

from src.retrieval.bm25_micro import BM25MicroRetriever

# Backward-compatibility alias
BM25Retriever = BM25MicroRetriever

__all__ = ["BM25Retriever", "BM25MicroRetriever"]
