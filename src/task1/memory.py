import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from src.common.normalize import normalize_question

class QuestionMemory:
    def __init__(self, min_similarity: float = 0.82, top_k_neighbors: int = 5):
        self.min_similarity = min_similarity
        self.top_k_neighbors = top_k_neighbors
        self.train_queries = []
        self.train_qids = []
        self.train_qrels = defaultdict(list)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.dense_embeddings = None
        self.dense_retriever = None

    def fit(self, queries: dict[str, str], qrels: dict[str, list[str]], dense_retriever=None, encode_dense: bool = True):
        self.train_qids = [str(qid) for qid in queries.keys()]
        self.train_queries = [normalize_question(queries[qid]) for qid in self.train_qids]
        self.train_qrels = {str(k): [str(d) for d in v] for k, v in qrels.items()}
        self.dense_retriever = dense_retriever

        if self.train_queries:
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            self.tfidf_matrix = self.vectorizer.fit_transform(self.train_queries)

            if dense_retriever is not None and encode_dense:
                self.dense_embeddings = dense_retriever.encode_texts(self.train_queries)

    def search(self, query: str, top_k: int = 5, q_dense_emb: np.ndarray = None) -> list[dict]:
        if self.tfidf_matrix is None or len(self.train_queries) == 0:
            return []

        norm_q = normalize_question(query)
        q_vec = self.vectorizer.transform([norm_q])
        tfidf_sims = (self.tfidf_matrix * q_vec.T).toarray().flatten()

        dense_sims = np.zeros(len(self.train_queries), dtype=np.float32)
        if q_dense_emb is not None and self.dense_embeddings is not None:
            dense_sims = np.dot(self.dense_embeddings, q_dense_emb)
        elif self.dense_retriever is not None and self.dense_embeddings is not None:
            q_emb = self.dense_retriever.encode_texts([query])[0]
            dense_sims = np.dot(self.dense_embeddings, q_emb)

        # Combined similarity score
        combined_sims = 0.6 * tfidf_sims + 0.4 * dense_sims if self.dense_embeddings is not None else tfidf_sims

        doc_votes = defaultdict(float)
        top_neighbor_indices = np.argsort(combined_sims)[::-1][:self.top_k_neighbors]

        for idx in top_neighbor_indices:
            sim = float(combined_sims[idx])
            if sim < self.min_similarity:
                continue
            neighbor_qid = self.train_qids[idx]
            gold_docs = self.train_qrels.get(neighbor_qid, [])
            for doc_id in gold_docs:
                doc_votes[doc_id] += sim

        ranked_docs = sorted(doc_votes.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank,
                "branch": "memory"
            })
        return results

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        data = {
            "qids": self.train_qids,
            "queries": self.train_queries,
            "qrels": self.train_qrels
        }
        with open(os.path.join(index_dir, "train_qa.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        if self.dense_embeddings is not None:
            np.save(os.path.join(index_dir, "train_embeddings.npy"), self.dense_embeddings)

    @classmethod
    def load(cls, index_dir: str, dense_retriever=None, min_similarity: float = 0.82):
        qa_path = os.path.join(index_dir, "train_qa.json")
        emb_path = os.path.join(index_dir, "train_embeddings.npy")
        mem = cls(min_similarity=min_similarity)
        if os.path.exists(qa_path):
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            queries = {qid: q for qid, q in zip(data["qids"], data["queries"])}
            qrels = data["qrels"]
            mem.fit(queries, qrels, dense_retriever=dense_retriever, encode_dense=not os.path.exists(emb_path))
            if os.path.exists(emb_path):
                mem.dense_embeddings = np.load(emb_path)
        return mem
