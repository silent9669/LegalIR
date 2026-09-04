import numpy as np
import pytest
from src.retrieval.dense import DenseIndexManager


def test_dense_matrix_drop_retains_metadata():
    mgr = DenseIndexManager()
    dummy_matrix = np.random.randn(20, 32).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(20)]
    chunk_ids = [f"chunk_{i}" for i in range(20)]

    mgr.load_embeddings(dummy_matrix, doc_ids=doc_ids, chunk_ids=chunk_ids)
    assert mgr.has_matrix() is True
    assert mgr.num_docs == 20

    mgr.build_faiss()
    assert mgr.has_index() is True

    # Drop matrix
    mgr.drop_corpus_matrix()
    assert mgr.has_matrix() is False
    assert mgr.has_index() is True
    assert mgr.num_docs == 20
    assert mgr.get_doc_id(0) == "doc_0"
    assert mgr.get_chunk_id(0) == "chunk_0"


def test_dense_search_parity_after_matrix_drop():
    mgr = DenseIndexManager()
    dummy_matrix = np.random.randn(30, 16).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(30)]

    mgr.load_embeddings(dummy_matrix, doc_ids=doc_ids)
    mgr.build_faiss()

    query_vecs = np.random.randn(2, 16).astype(np.float32)
    res1 = mgr.search(query_vecs, top_k=5)

    mgr.drop_corpus_matrix()
    res2 = mgr.search(query_vecs, top_k=5)

    assert len(res1) == len(res2) == 2
    for q_idx in range(2):
        docs1 = [doc for doc, _ in res1[q_idx]]
        docs2 = [doc for doc, _ in res2[q_idx]]
        assert docs1 == docs2
        scores1 = [s for _, s in res1[q_idx]]
        scores2 = [s for _, s in res2[q_idx]]
        np.testing.assert_allclose(scores1, scores2, rtol=1e-5)


def test_dense_unload():
    mgr = DenseIndexManager()
    dummy_matrix = np.random.randn(5, 8).astype(np.float32)
    mgr.load_embeddings(dummy_matrix, doc_ids=[f"d{i}" for i in range(5)])
    mgr.build_faiss()

    mgr.unload()
    assert mgr.has_matrix() is False
    assert mgr.has_index() is False
