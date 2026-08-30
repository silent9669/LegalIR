"""Dense semantic retrieval over macro legal-document chunks."""

from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from src.models.device import resolve_device


DEFAULT_MODEL_NAME = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
DEFAULT_DIMENSION = 768


class DenseMacroRetriever:
    """Encode macro chunks and aggregate dense similarities by document.

    The DEk21 v2 model is used by default. Encoded vectors are mean-pooled over
    non-padding tokens and L2-normalized, so inner products are cosine scores.
    FAISS is used when installed; NumPy provides the same search semantics as a
    dependency-free fallback.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dimension: int = DEFAULT_DIMENSION,
        use_pyvi: bool = True,
        device: str | None = None,
        model_name_or_path: str | None = None,
    ):
        if model_name_or_path is not None:
            model_name = model_name_or_path

        self.model_name = str(model_name)
        # Keep the old attribute for callers using the pre-DEk21 interface.
        self.model_name_or_path = self.model_name
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        self.use_pyvi = bool(use_pyvi)
        self.device = resolve_device(device or "auto")

        self.tokenizer = None
        self.model = None
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.corpus: list[dict[str, Any]] = []
        self.query_ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.query_encoder: Callable[[list[str]], np.ndarray] | None = None
        self._faiss_index = None

    @classmethod
    def from_arrays(
        cls,
        embeddings_path: str | Path | None = None,
        chunk_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
        query_encoder: Callable[[list[str]], np.ndarray] | None = None,
        embeddings: np.ndarray | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        use_pyvi: bool = False,
    ) -> "DenseMacroRetriever":
        """Create a retriever from a precomputed embedding matrix.

        This is useful for loading a local index or supplying a deterministic
        encoder in tests without downloading a model.
        """
        if embeddings is not None and embeddings_path is not None:
            raise ValueError("provide embeddings or embeddings_path, not both")

        loaded_embeddings: np.ndarray | None = None
        if embeddings is not None:
            loaded_embeddings = np.asarray(embeddings)
        elif embeddings_path is not None:
            loaded_embeddings = np.load(str(embeddings_path), mmap_mode="r")

        dimension = (
            int(loaded_embeddings.shape[1])
            if loaded_embeddings is not None and loaded_embeddings.ndim == 2
            else DEFAULT_DIMENSION
        )
        retriever = cls(
            model_name=model_name,
            dimension=dimension,
            use_pyvi=use_pyvi,
            device="cpu",
        )
        if loaded_embeddings is not None:
            retriever._set_embeddings(loaded_embeddings)
        retriever.chunk_ids = [str(x) for x in (chunk_ids or [])]
        retriever.doc_ids = [str(x) for x in (doc_ids or [])]
        retriever.query_encoder = query_encoder
        retriever._validate_metadata_lengths()
        return retriever

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = unicodedata.normalize("NFC", str(text))
        text = re.sub(r"[ \t\r\n]+", " ", text)
        return text.strip()

    def preprocess_text(self, text: Any) -> str:
        """Normalize text and optionally apply PyVi Vietnamese segmentation."""
        normalized = self._normalize_text(text)
        if not normalized or not self.use_pyvi:
            return normalized

        try:
            from pyvi import ViTokenizer
        except ImportError as exc:
            raise ImportError(
                "PyVi is required when use_pyvi=True; install it with `pip install pyvi`."
            ) from exc
        return ViTokenizer.tokenize(normalized)

    def _load_model(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading dense model {self.model_name} on {self.device}...")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, Mapping):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def _coerce_embedding_matrix(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("encoder output must be a two-dimensional matrix")
        if matrix.shape[1] != self.dimension:
            # A supplied encoder is intentionally allowed to define its own
            # dimensionality for in-memory/test indexes.
            if self.query_encoder is not None and self.model is None:
                self.dimension = int(matrix.shape[1])
            else:
                raise ValueError(
                    f"encoder dimension {matrix.shape[1]} does not match configured "
                    f"dimension {self.dimension}"
                )
        return self._normalize_embeddings(matrix)

    @staticmethod
    def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)

    def encode_texts(
        self,
        texts: Iterable[Any],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """Encode texts with mean pooling and normalized embeddings."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if isinstance(texts, str):
            texts = [texts]
        normalized_texts = [self.preprocess_text(text) for text in texts]
        if not normalized_texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.query_encoder is not None:
            return self._coerce_embedding_matrix(self.query_encoder(normalized_texts))

        self._load_model()
        import torch
        import torch.nn.functional as F

        all_vectors: list[np.ndarray] = []
        for start in range(0, len(normalized_texts), batch_size):
            batch = normalized_texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = self._move_inputs_to_device(inputs)

            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_state = getattr(outputs, "last_hidden_state", None)
                if hidden_state is None:
                    hidden_state = outputs[0]
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones(
                        hidden_state.shape[:2], dtype=hidden_state.dtype, device=hidden_state.device
                    )
                mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
                pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
                normalized = F.normalize(pooled, p=2, dim=1)
                vectors = normalized.cpu().to(torch.float32).numpy()

            all_vectors.append(vectors)

        return self._coerce_embedding_matrix(np.vstack(all_vectors))

    @staticmethod
    def _records(corpus: Any) -> list[dict[str, Any]]:
        if isinstance(corpus, Path) or (
            isinstance(corpus, str) and Path(corpus).exists()
        ):
            corpus = pd.read_parquet(corpus)
        elif isinstance(corpus, str):
            corpus = [{"text": corpus}]
        if isinstance(corpus, pd.DataFrame):
            records = corpus.to_dict(orient="records")
        elif isinstance(corpus, Mapping):
            records = [dict(corpus)]
        else:
            records = [dict(record) if isinstance(record, Mapping) else {"text": record} for record in corpus]

        if any("granularity" in record for record in records):
            records = [record for record in records if record.get("granularity", "macro") == "macro"]
        return records

    @staticmethod
    def _record_text(record: Mapping[str, Any]) -> str:
        for key in ("text_norm", "text", "text_raw", "body"):
            value = record.get(key)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                return str(value)
        return ""

    def _set_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if matrix.shape[1] != self.dimension:
            self.dimension = int(matrix.shape[1])
        self.embeddings = self._normalize_embeddings(matrix)
        self._build_search_index()
        return self.embeddings

    def _build_search_index(self) -> None:
        self._faiss_index = None
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        try:
            import faiss
        except ImportError:
            return
        index = faiss.IndexFlatIP(self.dimension)
        index.add(self.embeddings.astype(np.float32, copy=False))
        self._faiss_index = index

    def _validate_metadata_lengths(self) -> None:
        if self.embeddings is None:
            return
        count = len(self.embeddings)
        if self.chunk_ids and len(self.chunk_ids) != count:
            raise ValueError("chunk_ids length must match embeddings")
        if self.doc_ids and len(self.doc_ids) != count:
            raise ValueError("doc_ids length must match embeddings")

    def encode_corpus(
        self,
        corpus: Any,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """Encode macro records and retain chunk/document metadata for search."""
        records = self._records(corpus)
        texts = [self._record_text(record) for record in records]
        embeddings = self.encode_texts(texts, batch_size=batch_size, max_length=max_length)

        self.chunk_ids = [str(record.get("chunk_id", index)) for index, record in enumerate(records)]
        self.doc_ids = [
            str(record.get("doc_id", record.get("document_id", self.chunk_ids[index])))
            for index, record in enumerate(records)
        ]
        self.corpus = [
            {
                "chunk_id": self.chunk_ids[index],
                "doc_id": self.doc_ids[index],
                "article": record.get("article", record.get("article_id")),
            }
            for index, record in enumerate(records)
        ]
        self._set_embeddings(embeddings)
        return self.embeddings

    def encode_queries(
        self,
        queries: Any,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """Encode query strings or query records from ``queries_train.parquet``."""
        if isinstance(queries, Path) or (
            isinstance(queries, str) and Path(queries).exists()
        ):
            queries = pd.read_parquet(queries)
        elif isinstance(queries, str):
            queries = [queries]
        if isinstance(queries, pd.DataFrame):
            records = queries.to_dict(orient="records")
        elif isinstance(queries, Mapping):
            records = [dict(queries)]
        else:
            records = list(queries)

        texts: list[Any] = []
        query_ids: list[str] = []
        for index, query in enumerate(records):
            if isinstance(query, Mapping):
                texts.append(
                    query.get("question_norm")
                    or query.get("question_raw")
                    or query.get("text")
                    or query.get("query", "")
                )
                query_ids.append(str(query.get("query_id", index)))
            else:
                texts.append(query)
                query_ids.append(str(index))

        self.query_ids = query_ids
        return self.encode_texts(texts, batch_size=batch_size, max_length=max_length)

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Return top documents with scores aggregated from their best chunks."""
        if top_k <= 0 or not query or self.embeddings is None or len(self.embeddings) == 0:
            return []
        self._validate_metadata_lengths()

        q_vec = self.encode_queries([query], batch_size=1)[0].astype(np.float32, copy=False)
        candidate_count = min(max(top_k * 6, top_k), len(self.embeddings))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec.reshape(1, -1), candidate_count)
            chunk_indices = indices[0]
            similarities = scores[0]
        else:
            similarities_all = np.dot(self.embeddings.astype(np.float32, copy=False), q_vec)
            chunk_indices = np.argsort(-similarities_all, kind="stable")[:candidate_count]
            similarities = similarities_all[chunk_indices]

        doc_chunk_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for chunk_index, similarity in zip(chunk_indices, similarities):
            chunk_index = int(chunk_index)
            if chunk_index < 0:
                continue
            doc_id = self.doc_ids[chunk_index] if self.doc_ids else str(chunk_index)
            chunk_id = self.chunk_ids[chunk_index] if self.chunk_ids else str(chunk_index)
            doc_chunk_scores[doc_id].append((chunk_id, float(similarity)))

        document_records: list[dict[str, Any]] = []
        for doc_id, chunk_scores in doc_chunk_scores.items():
            chunk_scores.sort(key=lambda item: (-item[1], item[0]))
            best_chunk_id, best_score = chunk_scores[0]
            second_score = chunk_scores[1][1] if len(chunk_scores) > 1 else 0.0
            mean_score = sum(score for _, score in chunk_scores) / len(chunk_scores)
            aggregate_score = best_score + 0.1 * second_score
            document_records.append(
                {
                    "doc_id": doc_id,
                    "score": aggregate_score,
                    "dense_score": aggregate_score,
                    "dense_best_score": best_score,
                    "dense_second_score": second_score,
                    "dense_mean_score": mean_score,
                    "dense_best_chunk_id": best_chunk_id,
                }
            )

        document_records.sort(key=lambda item: (-item["score"], item["doc_id"]))
        return document_records[:top_k]

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Backward-compatible alias for :meth:`search`."""
        return self.search(query, top_k=top_k)

    @classmethod
    def build(
        cls,
        chunks: Any,
        output_dir: str | Path,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        dimension: int = DEFAULT_DIMENSION,
        use_pyvi: bool = True,
        model_name_or_path: str | None = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_name_or_path or model_name

        records = cls._records(chunks)
        encoder = cls(
            model_name=model_name,
            dimension=dimension,
            use_pyvi=use_pyvi,
            device=device,
        )
        embeddings = encoder.encode_corpus(records, batch_size=batch_size, max_length=max_length)
        np.save(str(output_dir / "embeddings.npy"), embeddings.astype(np.float16))

        meta_df = pd.DataFrame({"chunk_id": encoder.chunk_ids, "doc_id": encoder.doc_ids})
        meta_df.to_parquet(output_dir / "chunks_meta.parquet", index=False)
        manifest = {
            "model_name": model_name,
            "model_name_or_path": model_name,
            "total_macro_chunks": len(records),
            "embedding_dimension": int(embeddings.shape[1]),
            "dtype": "float16",
            "max_length": max_length,
            "use_pyvi": use_pyvi,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return output_dir

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        use_pyvi: bool = True,
        model_name_or_path: str | None = None,
    ) -> "DenseMacroRetriever":
        index_dir = Path(index_dir)
        embeddings = np.load(str(index_dir / "embeddings.npy"), mmap_mode="r")
        meta_df = pd.read_parquet(index_dir / "chunks_meta.parquet")
        model_name = model_name_or_path or model_name
        retriever = cls(
            model_name=model_name,
            dimension=int(embeddings.shape[1]),
            use_pyvi=use_pyvi,
            device=device,
        )
        retriever._set_embeddings(embeddings)
        retriever.chunk_ids = meta_df["chunk_id"].astype(str).tolist()
        retriever.doc_ids = meta_df["doc_id"].astype(str).tolist()
        retriever._validate_metadata_lengths()
        return retriever
