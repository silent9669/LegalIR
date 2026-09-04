"""Dense retrieval indexing, FAISS management, and memory lifecycle."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.memory import release_memory

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FallbackFlatIndex:
    """Pure NumPy fallback for IndexFlatIP when FAISS is unavailable."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: Optional[np.ndarray] = None

    def add(self, x: np.ndarray) -> None:
        if self.vectors is None:
            self.vectors = x.copy()
        else:
            self.vectors = np.vstack([self.vectors, x])

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Inner product
        scores = np.dot(x, self.vectors.T)
        indices = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, indices, axis=1)
        return top_scores, indices

    def reset(self) -> None:
        self.vectors = None


class DenseIndexManager:
    """Manages dense corpus embeddings, FAISS indexing, and memory unloading."""

    def __init__(self):
        self.corpus_matrix: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self.chunk_ids: List[str] = []
        self.index: Optional[Any] = None
        self.dim: int = 0
        self._num_docs: int = 0

    def load_embeddings(
        self,
        matrix: np.ndarray,
        doc_ids: List[str],
        chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """Load corpus embeddings matrix and document/chunk mapping."""
        self.corpus_matrix = matrix.astype(np.float32)
        self.dim = matrix.shape[1]
        self._num_docs = matrix.shape[0]
        self.doc_ids = list(doc_ids)
        self.chunk_ids = list(chunk_ids) if chunk_ids is not None else [f"chunk_{i}" for i in range(self._num_docs)]

    @property
    def num_docs(self) -> int:
        return self._num_docs

    def has_matrix(self) -> bool:
        return self.corpus_matrix is not None

    def has_index(self) -> bool:
        return self.index is not None

    def get_doc_id(self, idx: int) -> str:
        return self.doc_ids[idx]

    def get_chunk_id(self, idx: int) -> str:
        return self.chunk_ids[idx]

    def build_faiss(self, use_gpu: bool = False) -> None:
        """Construct IndexFlatIP from loaded corpus embeddings."""
        if self.corpus_matrix is None:
            raise ValueError("No embeddings loaded to build index.")

        if HAS_FAISS:
            index = faiss.IndexFlatIP(self.dim)
            if use_gpu and hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(self.corpus_matrix)
            self.index = index
        else:
            fb = FallbackFlatIndex(self.dim)
            fb.add(self.corpus_matrix)
            self.index = fb

    def drop_corpus_matrix(self) -> None:
        """
        Drop the Python numpy matrix to free host RAM after building the index.
        Retains metadata: doc_ids, chunk_ids, dim, num_docs, and the FAISS index.
        """
        self.corpus_matrix = None
        release_memory()

    def search(
        self, query_vectors: np.ndarray, top_k: int = 150
    ) -> List[List[Tuple[str, float]]]:
        """Search nearest documents for query vectors."""
        if self.index is None:
            raise ValueError("Index has not been built. Call build_faiss() first.")

        q_vecs = query_vectors.astype(np.float32)
        k = min(top_k, self._num_docs)
        scores, indices = self.index.search(q_vecs, k)

        results: List[List[Tuple[str, float]]] = []
        for i in range(len(q_vecs)):
            q_res: List[Tuple[str, float]] = []
            for j in range(k):
                idx = int(indices[i, j])
                score = float(scores[i, j])
                q_res.append((self.doc_ids[idx], score))
            results.append(q_res)

        return results

    def unload(self) -> None:
        """Release all allocated index objects, matrices, and metadata."""
        self.corpus_matrix = None
        self.index = None
        self.doc_ids.clear()
        self.chunk_ids.clear()
        self._num_docs = 0
        self.dim = 0
        release_memory()
