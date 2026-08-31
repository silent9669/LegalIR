"""Compatibility wrapper re-exporting canonical QuestionMemory and TrainQuestionMemory."""

from src.retrieval.question_memory import QuestionMemory, TrainQuestionMemory

__all__ = ["QuestionMemory", "TrainQuestionMemory"]
