import os
import numpy as np

class DenseMacroRetriever:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.chunk_ids = []
        self.doc_ids = []
        self.embeddings = None  # np.ndarray of shape (num_chunks, hidden_dim)

    def _init_device(self):
        if self.device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

    def _load_model(self):
        if self.model is None:
            self._init_device()
            import torch
            from transformers import AutoTokenizer, AutoModel
            print(f"Loading dense retriever {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def encode_texts(self, texts: list, batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        self._load_model()
        import torch

        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # CLS token representation normalized
                cls_rep = outputs.last_hidden_state[:, 0, :]
                norm_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
                all_vecs.append(norm_rep.cpu().numpy())

        return np.vstack(all_vecs)

    def retrieve(self, query: str, top_k: int = 50) -> list:
        """Returns list of (doc_id, score) sorted by descending similarity."""
        if self.embeddings is None or not query:
            return []

        q_vec = self.encode_texts([query], batch_size=1)[0]
        # Cosine similarity
        sims = np.dot(self.embeddings, q_vec)

        # Top chunk indices
        top_chunk_indices = np.argsort(sims)[::-1][:top_k * 4]

        doc_scores = {}
        for idx in top_chunk_indices:
            did = self.doc_ids[idx]
            score = float(sims[idx])
            if did not in doc_scores or score > doc_scores[did]:
                doc_scores[did] = score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs
