"""Compatibility package re-exporting canonical pipeline components."""

from src.task1.memory import QuestionMemory, TrainQuestionMemory
from src.task1.predict import LegalIRPipeline
from src.task1.rerank import DocumentReranker
from src.task1.retrieve import CandidateRetriever, ExactMatcher, LegalMatcher
from src.task1.selector import TopKSelector

__all__ = [
    "CandidateRetriever",
    "DocumentReranker",
    "ExactMatcher",
    "LegalIRPipeline",
    "LegalMatcher",
    "QuestionMemory",
    "TopKSelector",
    "TrainQuestionMemory",
]
