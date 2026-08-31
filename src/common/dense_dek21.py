"""Compatibility wrapper re-exporting canonical DenseMacroRetriever as DEk21Retriever."""

from src.retrieval.dense_macro import DenseMacroRetriever

# Backward-compatibility alias
DEk21Retriever = DenseMacroRetriever

__all__ = ["DEk21Retriever", "DenseMacroRetriever"]
