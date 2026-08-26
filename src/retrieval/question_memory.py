from collections import defaultdict
from typing import Any, Callable
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
