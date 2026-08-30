from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.dense_macro import DenseMacroRetriever


def normalize_text(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip().lower()


class QuestionMemory:
    def __init__(
        self,
        train_queries: list[dict[str, Any]],
        min_similarity: float = 0.82,
        dense_encoder: Callable[[list[str]], np.ndarray] | None = None,
        dense_min_similarity: float = 0.85,
    ):
        """
        train_queries: list of dicts with:
          - query_id: str
          - question_norm: str
          - doc_ids: list of str (gold document IDs)
        """
        self.min_similarity = min_similarity
        self.dense_min_similarity = dense_min_similarity
        self.dense_encoder = dense_encoder
        self.qids: list[str] = []
        self.texts: list[str] = []
        self.qid_to_docs: dict[str, list[str]] = {}

        for q in train_queries:
            qid = str(q["query_id"])
            text = normalize_text(q.get("question_norm") or q.get("question_raw", ""))
            doc_ids = [str(x) for x in q.get("doc_ids", [])]

            self.qids.append(qid)
            self.texts.append(text)
            self.qid_to_docs[qid] = doc_ids

        self.training_query_ids: frozenset[str] = frozenset(self.qids)

        # Build word and character n-gram TF-IDF vectorizer
        min_df = 1 if len(self.texts) < 20 else 2
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            sublinear_tf=True,
        )
        if self.texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        else:
            self.tfidf_matrix = None

        self.dense_embeddings = None
        if self.dense_encoder and self.texts:
            raw_embs = self.dense_encoder(self.texts)
            # Normalize embeddings for cosine similarity via dot product
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.dense_embeddings = raw_embs / norms

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_sim: float | None = None,
        min_similarity: float | None = None,
        dense_min_similarity: float | None = None,
        exclude_qid: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Returns {doc_id: {"score": sim_score, "lexical_similarity": sim, "dense_similarity": d_sim, "matched_qid": qid, "vote_count": count}}
        """
        effective_lex_sim = min_similarity if min_similarity is not None else (min_sim if min_sim is not None else self.min_similarity)
        effective_dense_sim = dense_min_similarity if dense_min_similarity is not None else self.dense_min_similarity

        if not query or not self.texts:
            return {}

        norm_q = normalize_text(query)
        doc_votes = defaultdict(lambda: {
            "score": 0.0,
            "lexical_similarity": 0.0,
            "dense_similarity": 0.0,
            "matched_qid": None,
            "vote_count": 0,
        })

        # 1. Lexical TF-IDF match
        if self.tfidf_matrix is not None:
            q_vec = self.vectorizer.transform([norm_q])
            sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(sims)[::-1][:top_k * 4]

            for idx in top_indices:
                sim = float(sims[idx])
                matched_qid = self.qids[idx]

                if exclude_qid and str(matched_qid) == str(exclude_qid):
                    continue
                if sim < effective_lex_sim:
                    continue

                for doc_id in self.qid_to_docs.get(matched_qid, []):
                    doc_votes[doc_id]["vote_count"] += 1
                    if sim > doc_votes[doc_id]["lexical_similarity"]:
                        doc_votes[doc_id]["lexical_similarity"] = sim
                        doc_votes[doc_id]["score"] = max(doc_votes[doc_id]["score"], sim)
                        doc_votes[doc_id]["matched_qid"] = matched_qid

        # 2. Dense semantic question similarity match
        if self.dense_encoder is not None and self.dense_embeddings is not None:
            q_emb = self.dense_encoder([norm_q])
            q_norm = np.linalg.norm(q_emb[0])
            if q_norm > 0:
                q_vec_norm = q_emb[0] / q_norm
                dense_sims = np.dot(self.dense_embeddings, q_vec_norm)
                top_dense_indices = np.argsort(dense_sims)[::-1][:top_k * 4]

                for idx in top_dense_indices:
                    d_sim = float(dense_sims[idx])
                    matched_qid = self.qids[idx]

                    if exclude_qid and str(matched_qid) == str(exclude_qid):
                        continue
                    if d_sim < effective_dense_sim:
                        continue

                    for doc_id in self.qid_to_docs.get(matched_qid, []):
                        if doc_votes[doc_id]["vote_count"] == 0:
                            doc_votes[doc_id]["vote_count"] += 1
                        if d_sim > doc_votes[doc_id]["dense_similarity"]:
                            doc_votes[doc_id]["dense_similarity"] = d_sim
                            doc_votes[doc_id]["score"] = max(doc_votes[doc_id]["score"], d_sim)
                            if doc_votes[doc_id]["matched_qid"] is None:
                                doc_votes[doc_id]["matched_qid"] = matched_qid

        return dict(doc_votes)


class TrainQuestionMemory:
    """Fold-local memory over labelled training questions.

    The index is rebuilt by :meth:`fit` from only the supplied questions and
    qrels.  Character n-gram TF-IDF and optional DEk21-compatible question
    embeddings are independent retrieval signals; each matching training
    question casts one similarity-weighted vote for every gold document.

    ``dense_encoder`` is intentionally injectable.  Production callers can
    supply ``DenseMacroRetriever.encode_texts`` (the DEk21 v2 encoder), while
    tests and offline callers can provide a deterministic encoder or precomputed
    embeddings without downloading a model.  Set ``use_dense=True`` to lazily
    construct the default DEk21 retriever when no dense source is supplied.
    """

    DEFAULT_MODEL_NAME = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"

    def __init__(
        self,
        min_similarity: float = 0.82,
        dense_encoder: Callable[[list[str]], Any] | Any | None = None,
        dense_min_similarity: float | None = None,
        dense_embeddings: Mapping[str, Any] | Any | None = None,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        model_name: str = DEFAULT_MODEL_NAME,
        use_dense: bool = True,
        dense_dimension: int = 768,
        dense_use_pyvi: bool = True,
        dense_device: str | None = None,
    ):
        if min_similarity < -1.0 or min_similarity > 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        if dense_min_similarity is not None and not -1.0 <= dense_min_similarity <= 1.0:
            raise ValueError("dense_min_similarity must be between -1 and 1")
        if lexical_weight < 0.0 or dense_weight < 0.0:
            raise ValueError("signal weights must be non-negative")
        if dense_dimension <= 0:
            raise ValueError("dense_dimension must be positive")

        self.min_similarity = float(min_similarity)
        self.dense_min_similarity = float(
            dense_min_similarity if dense_min_similarity is not None else min_similarity
        )
        self.dense_encoder = dense_encoder
        self.dense_embeddings_input = dense_embeddings
        self.use_dense = bool(use_dense or dense_encoder is not None or dense_embeddings is not None)
        self.dense_dimension = int(dense_dimension)
        self.dense_use_pyvi = bool(dense_use_pyvi)
        self.dense_device = dense_device
        self.lexical_weight = float(lexical_weight)
        self.dense_weight = float(dense_weight)
        self.model_name = str(model_name)
        self._clear_index()

    def _clear_index(self) -> None:
        self.qids: list[str] = []
        self.texts: list[str] = []
        self.qid_to_docs: dict[str, list[str]] = {}
        self.training_query_ids: frozenset[str] = frozenset()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        self.tfidf_matrix = None
        self.dense_embeddings: np.ndarray | None = None

    @staticmethod
    def _read_table(value: Any) -> Any:
        if isinstance(value, Path) or (isinstance(value, str) and Path(value).exists()):
            return pd.read_parquet(value)
        return value

    @staticmethod
    def _query_text(record: Mapping[str, Any]) -> str:
        for key in ("question_norm", "question_raw", "text", "query", "question"):
            value = record.get(key)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                return normalize_text(value)
        return ""

    @staticmethod
    def _query_embedding(record: Mapping[str, Any]) -> Any | None:
        for key in ("question_embedding", "query_embedding", "embedding", "q_emb"):
            if key in record and record[key] is not None:
                return record[key]
        return None

    @classmethod
    def _query_records(cls, train_queries: Any) -> list[tuple[str, str, Any | None]]:
        train_queries = cls._read_table(train_queries)
        if isinstance(train_queries, pd.DataFrame):
            records = train_queries.to_dict(orient="records")
        elif isinstance(train_queries, Mapping):
            query_keys = {"query_id", "qid", "question_norm", "question_raw", "text", "query"}
            if query_keys.intersection(train_queries):
                records = [dict(train_queries)]
            else:
                records = []
                for qid, value in train_queries.items():
                    if isinstance(value, Mapping):
                        record = dict(value)
                        record.setdefault("query_id", qid)
                    else:
                        record = {"query_id": qid, "question_norm": value}
                    records.append(record)
        elif isinstance(train_queries, str):
            records = [{"query_id": "0", "question_norm": train_queries}]
        else:
            records = list(train_queries or [])

        parsed: list[tuple[str, str, Any | None]] = []
        for index, record in enumerate(records):
            if isinstance(record, Mapping):
                qid = record.get("query_id", record.get("qid", index))
                text = cls._query_text(record)
                embedding = cls._query_embedding(record)
            else:
                qid, text, embedding = index, normalize_text(record), None
            parsed.append((str(qid), text, embedding))
        return parsed

    @staticmethod
    def _doc_ids(value: Any) -> list[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, (str, bytes)):
            values = [value]
        else:
            try:
                values = list(value)
            except TypeError:
                values = [value]
        result: list[str] = []
        for doc_id in values:
            if doc_id is None or (isinstance(doc_id, float) and pd.isna(doc_id)):
                continue
            doc_id = str(doc_id)
            if doc_id not in result:
                result.append(doc_id)
        return result

    @classmethod
    def _qrel_records(cls, qrels: Any) -> dict[str, list[str]]:
        qrels = cls._read_table(qrels)
        grouped: dict[str, list[str]] = defaultdict(list)

        if isinstance(qrels, pd.DataFrame):
            query_col = "query_id" if "query_id" in qrels.columns else "qid"
            doc_col = "doc_id" if "doc_id" in qrels.columns else "document_id"
            for record in qrels[[query_col, doc_col]].to_dict(orient="records"):
                grouped[str(record[query_col])].extend(cls._doc_ids(record[doc_col]))
        elif isinstance(qrels, Mapping):
            qrel_keys = {"query_id", "qid", "doc_id", "document_id"}
            if qrel_keys.intersection(qrels):
                qid = qrels.get("query_id", qrels.get("qid"))
                doc_id = qrels.get("doc_id", qrels.get("document_id"))
                if qid is not None:
                    grouped[str(qid)].extend(cls._doc_ids(doc_id))
            else:
                for qid, doc_ids in qrels.items():
                    grouped[str(qid)].extend(cls._doc_ids(doc_ids))
        else:
            for record in list(qrels or []):
                if not isinstance(record, Mapping):
                    continue
                qid = record.get("query_id", record.get("qid"))
                doc_id = record.get("doc_id", record.get("document_id"))
                if qid is not None:
                    grouped[str(qid)].extend(cls._doc_ids(doc_id))

        return {
            qid: list(dict.fromkeys(doc_ids))
            for qid, doc_ids in grouped.items()
            if doc_ids
        }

    @staticmethod
    def _normalize_embeddings(values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("question embeddings must be a two-dimensional matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)

    def _ensure_dense_encoder(self) -> Any | None:
        if self.dense_encoder is None and self.use_dense:
            self.dense_encoder = DenseMacroRetriever(
                model_name=self.model_name,
                dimension=self.dense_dimension,
                use_pyvi=self.dense_use_pyvi,
                device=self.dense_device,
            )
        return self.dense_encoder

    def _encode(self, texts: list[str]) -> np.ndarray:
        encoder = self._ensure_dense_encoder()
        if encoder is None:
            raise ValueError("a dense_encoder is required to encode question text")
        if callable(encoder):
            values = encoder(texts)
        elif hasattr(encoder, "encode_texts"):
            values = encoder.encode_texts(texts)
        elif hasattr(encoder, "encode_queries"):
            values = encoder.encode_queries(texts)
        elif hasattr(encoder, "encode"):
            values = encoder.encode(texts)
        else:
            raise TypeError("dense_encoder must be callable or expose an encode method")
        return self._normalize_embeddings(values)

    def _provided_embeddings(
        self,
        qids: list[str],
        record_embeddings: list[Any | None],
    ) -> np.ndarray | None:
        values = self.dense_embeddings_input
        if values is not None:
            if isinstance(values, Mapping):
                rows = [values.get(qid) for qid in qids]
                if any(row is None for row in rows):
                    return None
                return self._normalize_embeddings(rows)
            matrix = self._normalize_embeddings(values)
            if len(matrix) != len(qids):
                raise ValueError("dense_embeddings length must match indexed training questions")
            return matrix

        if any(value is not None for value in record_embeddings):
            if any(value is None for value in record_embeddings):
                raise ValueError("embeddings must be provided for every indexed question")
            return self._normalize_embeddings(record_embeddings)
        return None

    def fit(self, train_queries: Any, qrels: Any) -> "TrainQuestionMemory":
        """Index only the supplied training questions and their gold qrels."""
        self._clear_index()
        qrels_by_qid = self._qrel_records(qrels)
        query_records = self._query_records(train_queries)

        for qid, text, embedding in query_records:
            # Requiring qrels here is the leakage guard: an unlabelled query,
            # including a validation query accidentally passed alongside the
            # fold, cannot become a memory entry.
            if qid not in qrels_by_qid:
                continue
            self.qids.append(qid)
            self.texts.append(text)
            self.qid_to_docs[qid] = qrels_by_qid[qid]

        self.training_query_ids = frozenset(self.qids)
        if self.texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

            record_embeddings = [
                embedding
                for qid, _, embedding in query_records
                if qid in self.training_query_ids
            ]
            self.dense_embeddings = self._provided_embeddings(self.qids, record_embeddings)
            if self.dense_embeddings is None and self.use_dense:
                self._ensure_dense_encoder()
                if self.dense_encoder is not None:
                    self.dense_embeddings = self._encode(self.texts)
                    if len(self.dense_embeddings) != len(self.texts):
                        raise ValueError("dense_encoder output length must match indexed questions")
        return self

    @staticmethod
    def _query_vector(q_emb: Any) -> np.ndarray:
        matrix = TrainQuestionMemory._normalize_embeddings(q_emb)
        if len(matrix) != 1:
            raise ValueError("q_emb must contain exactly one question embedding")
        return matrix[0]

    def search(self, q_text: str, top_k: int = 5, q_emb: Any | None = None) -> list[dict[str, Any]]:
        """Return gold documents voted for by similar indexed questions."""
        if top_k <= 0 or not q_text or not self.texts:
            return []

        normalized_query = normalize_text(q_text)
        lexical_sims = cosine_similarity(
            self.vectorizer.transform([normalized_query]), self.tfidf_matrix
        )[0]
        dense_sims: np.ndarray | None = None
        if self.dense_embeddings is not None:
            query_embedding = self._query_vector(q_emb) if q_emb is not None else None
            if query_embedding is None and self.use_dense:
                self._ensure_dense_encoder()
            if query_embedding is None and self.dense_encoder is not None:
                query_embedding = self._encode([normalized_query])[0]
            if query_embedding is not None:
                if query_embedding.shape[0] != self.dense_embeddings.shape[1]:
                    raise ValueError("q_emb dimension must match indexed question embeddings")
                dense_sims = np.dot(self.dense_embeddings, query_embedding)

        votes: dict[str, dict[str, Any]] = {}
        best_contributions: dict[str, float] = {}
        for index, lexical_sim in enumerate(lexical_sims):
            lexical_sim = float(lexical_sim)
            dense_sim = float(dense_sims[index]) if dense_sims is not None else 0.0
            lexical_match = lexical_sim >= self.min_similarity
            dense_match = dense_sims is not None and dense_sim >= self.dense_min_similarity
            if not lexical_match and not dense_match:
                continue

            lexical_score = lexical_sim if lexical_match else 0.0
            dense_score = dense_sim if dense_match else 0.0
            contribution = self.lexical_weight * lexical_score + self.dense_weight * dense_score
            qid = self.qids[index]
            for doc_id in self.qid_to_docs[qid]:
                vote = votes.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "score": 0.0,
                        "lexical_similarity": 0.0,
                        "dense_similarity": 0.0,
                        "matched_qid": None,
                        "vote_count": 0,
                    },
                )
                vote["score"] += contribution
                vote["lexical_similarity"] = max(vote["lexical_similarity"], lexical_score)
                vote["dense_similarity"] = max(vote["dense_similarity"], dense_score)
                vote["vote_count"] += 1
                if contribution > best_contributions.get(doc_id, float("-inf")):
                    best_contributions[doc_id] = contribution
                    vote["matched_qid"] = qid

        for vote in votes.values():
            vote["positive_frequency"] = vote["vote_count"] / max(len(self.qids), 1)
        return sorted(
            votes.values(),
            key=lambda vote: (-float(vote["score"]), str(vote["doc_id"])),
        )[:top_k]

    query = search
    retrieve = search
