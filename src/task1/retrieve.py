"""Compatibility wrapper re-exporting canonical CandidateRetriever and LegalMatcher."""

from src.retrieval.exact_matcher import ExactMatcher, LegalMatcher
from src.retrieval.hybrid_search import CandidateRetriever, HybridSearchEngine

__all__ = ["CandidateRetriever", "ExactMatcher", "HybridSearchEngine", "LegalMatcher"]
