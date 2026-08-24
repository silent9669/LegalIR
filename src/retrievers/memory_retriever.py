import json
import numpy as np
from data_utils import normalize_text

class TrainQuestionMemory:
    def __init__(self, train_file: str = "train.json"):
        with open(train_file, "r", encoding="utf-8") as f:
            self.train_data = json.load(f)

        self.qids = []
        self.questions = []
        self.answers = []
        self.exact_map = {}

        for qid, qobj in self.train_data.items():
            q_text = normalize_text(qobj.get("question", ""))
            ans_list = [str(x) for x in qobj.get("answer", [])]
            self.qids.append(qid)
            self.questions.append(q_text)
            self.answers.append(ans_list)

            norm_key = q_text.lower().strip()
            if norm_key not in self.exact_map:
                self.exact_map[norm_key] = set()
            self.exact_map[norm_key].update(ans_list)

        # Build Char TF-IDF Vectorizer
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
        except ImportError:
            self.vectorizer = None
            self.tfidf_matrix = None

    def retrieve(self, query: str, top_k_neighbors: int = 15, exclude_qid: str = None) -> dict:
        """Transfer positive document IDs from nearest train questions."""
        scores = {}
        q_norm = normalize_text(query).lower().strip()

        # 1. Exact question match
        if q_norm in self.exact_map:
            for did in self.exact_map[q_norm]:
                scores[did] = scores.get(did, 0.0) + 20.0

        # 2. TF-IDF char n-gram similarity
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            q_vec = self.vectorizer.transform([normalize_text(query)])
            # Cosine similarity (since tfidf rows are L2 normalized, dot product = cosine sim)
            sims = (self.tfidf_matrix * q_vec.T).toarray().flatten()

            # If evaluating on train fold, zero out the exact query itself if exclude_qid given
            if exclude_qid is not None:
                try:
                    idx = self.qids.index(str(exclude_qid))
                    sims[idx] = 0.0
                except ValueError:
                    pass

            top_indices = np.argsort(sims)[::-1][:top_k_neighbors]
            for idx in top_indices:
                sim = float(sims[idx])
                if sim < 0.35:
                    break
                weight = (sim ** 3) * 10.0
                for did in self.answers[idx]:
                    scores[did] = scores.get(did, 0.0) + weight

        return scores
