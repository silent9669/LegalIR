import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from src.common.normalize import tokenize_vietnamese

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

class DEk21Retriever:
    def __init__(self, model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", device: str = None, dimension: int = 768):
        self.model_name = model_name
        self.dimension = dimension
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self.corpus = []
        self.corpus_embeddings = None
        self.chunk_to_doc = []

    def _lazy_init(self):
        if self.model is None and self.model_name != "mock" and AutoModel is not None:
            print(f"Loading DEk21 embedding model {self.model_name} on {self.device}...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def encode_texts(self, texts: list[str], batch_size: int = 256, max_length: int = 256, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if self.model_name == "mock" or AutoModel is None:
            np.random.seed(42)
            emb = np.random.randn(len(texts), self.dimension).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / np.maximum(norms, 1e-12)

        self._lazy_init()
        total = len(texts)
        if show_progress:
            print(f"  Tokenizing {total:,} texts with PyVi...", flush=True)
        segmented_texts = [tokenize_vietnamese(t) for t in texts]

        all_embeddings = []
        for i in range(0, total, batch_size):
            batch = segmented_texts[i:i+batch_size]
            encoded = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                hidden_states = outputs.last_hidden_state
                sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())

            if self.device == "mps" and (i // batch_size) % 20 == 0:
                torch.mps.empty_cache()

            if show_progress and (i + batch_size >= total or (i // batch_size) % 50 == 0):
                print(f"  Encoded {min(i + batch_size, total):,}/{total:,} chunks on {self.device}...", flush=True)

        return np.vstack(all_embeddings).astype(np.float32)

    def fit(self, corpus: list[dict], batch_size: int = 256, max_length: int = 256):
        self.corpus = corpus
        self.chunk_to_doc = [str(c.get("doc_id", c.get("chunk_id", ""))) for c in corpus]
        texts = [f"{c.get('article', '')} {c.get('text_raw', '')}".strip() for c in corpus]
        self.corpus_embeddings = self.encode_texts(texts, batch_size=batch_size, max_length=max_length, show_progress=True)

    def search(self, query: str, top_k: int = 60) -> list[dict]:
        if self.corpus_embeddings is None or len(self.corpus) == 0:
            return []

        q_emb = self.encode_texts([query])[0]
        sims = np.dot(self.corpus_embeddings, q_emb)

        # Aggregate chunk scores to document level
        doc_scores_map = defaultdict(list)
        for idx, sc in enumerate(sims):
            doc_id = str(self.chunk_to_doc[idx])
            doc_scores_map[doc_id].append((float(sc), self.corpus[idx]))

        doc_results = []
        for doc_id, chunk_list in doc_scores_map.items():
            sorted_chunks = sorted(chunk_list, key=lambda x: x[0], reverse=True)
            max_sc = sorted_chunks[0][0]
            mean_sc = sum(x[0] for x in sorted_chunks) / len(sorted_chunks)
            total_doc_sc = max_sc + 0.1 * mean_sc
            doc_results.append({
                "doc_id": doc_id,
                "score": float(total_doc_sc),
                "best_chunk": sorted_chunks[0][1]
            })

        doc_results = sorted(doc_results, key=lambda x: x["score"], reverse=True)[:top_k]
        for rank, item in enumerate(doc_results, start=1):
            item["rank"] = rank
        return doc_results

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        if self.corpus_embeddings is not None:
            np.save(os.path.join(index_dir, "embeddings.npy"), self.corpus_embeddings)
        df = pd.DataFrame([{"chunk_id": c.get("chunk_id"), "doc_id": c.get("doc_id"), "article": c.get("article")} for c in self.corpus])
        df.to_parquet(os.path.join(index_dir, "corpus_meta.parquet"), index=False)

    @classmethod
    def load(cls, index_dir: str, model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", device: str = None):
        retriever = cls(model_name=model_name, device=device)
        emb_path = os.path.join(index_dir, "embeddings.npy")
        meta_path = os.path.join(index_dir, "corpus_meta.parquet")
        if os.path.exists(emb_path) and os.path.exists(meta_path):
            retriever.corpus_embeddings = np.load(emb_path)
            df = pd.read_parquet(meta_path)
            retriever.corpus = df.to_dict("records")
            retriever.chunk_to_doc = [str(c.get("doc_id", c.get("chunk_id", ""))) for c in retriever.corpus]
        return retriever
