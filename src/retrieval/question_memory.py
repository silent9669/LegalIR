import re
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def normalize_text(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip().lower()

class QuestionMemory:
    def __init__(self, train_queries: list):
        """
        train_queries: list of dicts with:
          - query_id: str
          - question_norm: str
          - doc_ids: list of str (gold document IDs)
        """
        self.qids = []
        self.texts = []
        self.qid_to_docs = {}

        for q in train_queries:
            qid = str(q["query_id"])
            text = normalize_text(q.get("question_norm") or q.get("question_raw", ""))
            doc_ids = [str(x) for x in q.get("doc_ids", [])]

            self.qids.append(qid)
            self.texts.append(text)
            self.qid_to_docs[qid] = doc_ids

        # Build word and character n-gram TF-IDF vectorizer
        min_df = 1 if len(self.texts) < 20 else 2
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            sublinear_tf=True
        )
        if self.texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        else:
            self.tfidf_matrix = None

    def retrieve(self, query: str, top_k: int = 5, min_sim: float = 0.70, exclude_qid: str = None) -> dict:
        """
        Returns {doc_id: {"score": sim_score, "matched_qid": qid, "vote_count": count}}
        """
        if self.tfidf_matrix is None or not query:
            return {}

        norm_q = normalize_text(query)
        q_vec = self.vectorizer.transform([norm_q])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:top_k * 3]
        doc_votes = defaultdict(lambda: {"score": 0.0, "matched_qid": None, "vote_count": 0})

        for idx in top_indices:
            sim = float(sims[idx])
            matched_qid = self.qids[idx]

            # Exclude self during cross-validation / training query evaluation
            if exclude_qid and str(matched_qid) == str(exclude_qid):
                continue

            if sim < min_sim:
                continue

            for doc_id in self.qid_to_docs.get(matched_qid, []):
                doc_votes[doc_id]["vote_count"] += 1
                if sim > doc_votes[doc_id]["score"]:
                    doc_votes[doc_id]["score"] = sim
                    doc_votes[doc_id]["matched_qid"] = matched_qid

        return dict(doc_votes)
