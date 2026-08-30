import os
import torch
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
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

    def _lazy_init(self):
        if self.model is None and self.model_name != "mock" and AutoModelForSequenceClassification is not None:
            print(f"Loading BGE Cross-Encoder Reranker {self.model_name} on {self.device}...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def rerank_pairs(self, pairs: list[tuple[str, str]], batch_size: int = None, max_length: int = 256) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)

        if self.model_name == "mock" or AutoModelForSequenceClassification is None:
            scores = []
            for q, doc in pairs:
                q_words = set(q.lower().split())
                doc_words = set(doc.lower().split())
                overlap = len(q_words & doc_words)
                scores.append(float(overlap))
            return np.array(scores, dtype=np.float32)

        self._lazy_init()
        bs = batch_size or self.batch_size
        all_scores = []

        for i in range(0, len(pairs), bs):
            batch = pairs[i:i+bs]
            inputs = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1).cpu().numpy()
                else:
                    batch_scores = logits[:, 0].cpu().numpy()
                all_scores.extend(batch_scores.tolist())

            if self.device == "mps" and (i // bs) % 10 == 0:
                torch.mps.empty_cache()

        return np.array(all_scores, dtype=np.float32)

    def rerank_candidates(self, query: str, candidates: list[dict], evidence_texts: list[str] = None, top_k: int = 5) -> list[dict]:
        if not candidates:
            return []

        if evidence_texts is None:
            evidence_texts = [c.get("evidence_text") or c.get("text_raw", "") for c in candidates]

        pairs = [(query, text[:1000]) for text in evidence_texts]
        scores = self.rerank_pairs(pairs)

        scored = []
        for item, sc in zip(candidates, scores):
            entry = dict(item)
            entry["reranker_score"] = float(sc)
            scored.append(entry)

        scored = sorted(scored, key=lambda x: x["reranker_score"], reverse=True)
        for rank, item in enumerate(scored[:top_k], start=1):
            item["final_rank"] = rank
        return scored[:top_k]
