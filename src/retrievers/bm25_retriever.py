import os
import json
import math
import pickle
import re
from collections import Counter
import numpy as np
from data_utils import normalize_text

TOKEN_PATTERN = re.compile(r'\b\w+\b', re.UNICODE)

def tokenize_vietnamese(text: str) -> list:
    """Fast word and syllable tokenization for Vietnamese legal text."""
    if not text:
        return []
    text = normalize_text(text).lower()
    tokens = TOKEN_PATTERN.findall(text)
    return tokens

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = []           # chunk_idx -> doc_id
        self.chunk_ids = []         # chunk_idx -> chunk_id
        self.doc_lens = None        # np.ndarray of doc lengths
        self.avg_doc_len = 0.0
        self.num_docs = 0
        self.idf = {}               # term -> idf
        self.postings = {}          # term -> (chunk_indices: np.ndarray, term_freqs: np.ndarray)
        self.doc_to_chunk_indices = None # doc_id -> list of chunk indices

    def build_index(self, chunks_file: str = "data/processed_chunks.jsonl", save_path: str = "data/bm25_index.pkl"):
        """Build BM25 inverted index from processed chunks."""
        print(f"Building BM25 index from {chunks_file}...")

        chunk_doc_ids = []
        chunk_ids = []
        doc_lens = []
        term_postings = {}

        chunk_idx = 0
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                did = str(c["doc_id"])
                cid = c["chunk_id"]

                title = c.get("title") or ""
                legal_num = c.get("legal_number") or ""
                dieu = c.get("dieu") or ""
                body = c.get("body") or ""

                title_tokens = tokenize_vietnamese(f"{title} {legal_num}")
                dieu_tokens = tokenize_vietnamese(dieu)
                body_tokens = tokenize_vietnamese(body)

                tf_counter = Counter()
                for t in title_tokens: tf_counter[t] += 3.0
                for t in dieu_tokens: tf_counter[t] += 2.0
                for t in body_tokens: tf_counter[t] += 1.0

                length = len(body_tokens) + len(title_tokens) + len(dieu_tokens)
                doc_lens.append(length)
                chunk_doc_ids.append(did)
                chunk_ids.append(cid)

                for term, tf in tf_counter.items():
                    if term not in term_postings:
                        term_postings[term] = []
                    term_postings[term].append((chunk_idx, tf))

                chunk_idx += 1
                if chunk_idx % 100000 == 0:
                    print(f"Indexed {chunk_idx} chunks...")

        self.num_docs = chunk_idx
        self.doc_ids = chunk_doc_ids
        self.chunk_ids = chunk_ids
        self.doc_lens = np.array(doc_lens, dtype=np.float32)
        self.avg_doc_len = float(np.mean(self.doc_lens))

        print(f"Computing IDFs and converting postings for {len(term_postings)} unique terms...")
        self.idf = {}
        self.postings = {}

        for term, plist in term_postings.items():
            df = len(plist)
            idf_val = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            if idf_val > 0:
                self.idf[term] = idf_val
                chunks_arr = np.array([p[0] for p in plist], dtype=np.int32)
                tfs_arr = np.array([p[1] for p in plist], dtype=np.float32)
                self.postings[term] = (chunks_arr, tfs_arr)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.save(save_path)

    def save(self, file_path: str):
        print(f"Saving BM25 index to {file_path}...")
        with open(file_path, "wb") as f:
            pickle.dump({
                "k1": self.k1,
                "b": self.b,
                "num_docs": self.num_docs,
                "avg_doc_len": self.avg_doc_len,
                "doc_ids": self.doc_ids,
                "chunk_ids": self.chunk_ids,
                "doc_lens": self.doc_lens,
                "idf": self.idf,
                "postings": self.postings
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("BM25 index saved successfully.")

    @classmethod
    def load(cls, file_path: str):
        print(f"Loading BM25 index from {file_path}...")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        obj = cls(k1=data["k1"], b=data["b"])
        obj.num_docs = data["num_docs"]
        obj.avg_doc_len = data["avg_doc_len"]
        obj.doc_ids = data["doc_ids"]
        obj.chunk_ids = data["chunk_ids"]
        obj.doc_lens = data["doc_lens"]
        obj.idf = data["idf"]
        obj.postings = data["postings"]

        # Pre-build doc_id -> list of chunk indices mapping for instant aggregation
        obj.doc_to_chunk_indices = {}
        for c_idx, did in enumerate(obj.doc_ids):
            if did not in obj.doc_to_chunk_indices:
                obj.doc_to_chunk_indices[did] = []
            obj.doc_to_chunk_indices[did].append(c_idx)

        print(f"Loaded BM25 index with {obj.num_docs} chunks.")
        return obj

    def retrieve(self, query: str, top_docs: int = 150) -> dict:
        """Fast vectorized BM25 retrieval across all chunks aggregated to documents."""
        tokens = tokenize_vietnamese(query)
        if not tokens:
            return {}

        chunk_scores = np.zeros(self.num_docs, dtype=np.float32)

        for token in set(tokens):
            if token not in self.idf or token not in self.postings:
                continue
            idf = self.idf[token]
            chunk_indices, tfs = self.postings[token]
            lens = self.doc_lens[chunk_indices]

            denom = tfs + self.k1 * (1.0 - self.b + self.b * (lens / self.avg_doc_len))
            term_scores = idf * (tfs * (self.k1 + 1.0)) / denom

            np.add.at(chunk_scores, chunk_indices, term_scores)

        # Non-zero chunk indices
        non_zero_chunks = np.flatnonzero(chunk_scores)
        if len(non_zero_chunks) == 0:
            return {}

        # Aggregate scores by doc_id
        doc_scores = {}
        for c_idx in non_zero_chunks:
            sc = float(chunk_scores[c_idx])
            did = self.doc_ids[c_idx]
            if did not in doc_scores:
                doc_scores[did] = [sc]
            else:
                doc_scores[did].append(sc)

        doc_final_scores = {}
        for did, s_list in doc_scores.items():
            if len(s_list) == 1:
                doc_final_scores[did] = s_list[0]
            else:
                s_list.sort(reverse=True)
                doc_final_scores[did] = s_list[0] + 0.1 * s_list[1]

        return doc_final_scores
