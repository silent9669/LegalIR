import numpy as np
from pathlib import Path
from src.retrieval.dense_macro import DenseMacroRetriever


def test_exact_dense_search_aggregates_chunks_to_documents(tmp_path: Path):
    emb_path = tmp_path / "embeddings.npy"
    # 3 vectors: dim=2
    np.save(emb_path, np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float16))

    retriever = DenseMacroRetriever.from_arrays(
        embeddings_path=emb_path,
        chunk_ids=["a1", "a2", "b1"],
        doc_ids=["A", "A", "B"],
        query_encoder=lambda _: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    res = retriever.retrieve("query text", top_k=2)
    assert len(res) == 2
    assert res[0]["doc_id"] == "A"
    assert res[0]["dense_best_chunk_id"] == "a1"
    assert res[0]["dense_best_score"] == 1.0
    assert res[0]["dense_second_score"] > 0
    assert res[1]["doc_id"] == "B"
