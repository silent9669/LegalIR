import numpy as np
from src.retrieval.question_memory import QuestionMemory


def test_dense_memory_votes_without_validation_rows():
    rows = [
        {"query_id": "train1", "question_norm": "quy định về đất đai", "doc_ids": ["100"]},
        {"query_id": "train2", "question_norm": "quy định về bất động sản", "doc_ids": ["200"]},
    ]

    # Mock encoder: returns 2D vectors
    def mock_encoder(texts):
        vecs = []
        for t in texts:
            if "đất đai" in t:
                vecs.append([1.0, 0.0])
            elif "bất động sản" in t:
                vecs.append([0.0, 1.0])
            else:
                vecs.append([0.5, 0.5])
        return np.array(vecs, dtype=np.float32)

    memory = QuestionMemory(
        train_queries=rows,
        min_similarity=0.70,
        dense_encoder=mock_encoder,
        dense_min_similarity=0.80,
    )

    assert memory.training_query_ids == frozenset({"train1", "train2"})
    assert "val" not in memory.qid_to_docs

    res = memory.retrieve("thông tin luật đất đai", top_k=5)
    assert "100" in res
    assert res["100"]["dense_similarity"] >= 0.8
    assert res["100"]["vote_count"] >= 1
