from src.evaluation.benchmark import build_memory_rows
from src.retrieval.question_memory import QuestionMemory


def test_validation_queries_never_enter_question_memory():
    queries = {"train": "câu hỏi train", "val": "câu hỏi val"}
    qrels = {"train": ["1"], "val": ["2"]}
    rows = build_memory_rows(["train"], queries, qrels)
    memory = QuestionMemory(rows, min_similarity=0.82)
    assert memory.training_query_ids == frozenset({"train"})
    assert "val" not in memory.qid_to_docs
