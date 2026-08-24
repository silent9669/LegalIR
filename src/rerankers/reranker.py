import os
import json
import numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        self.model_name = model_name
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading cross-encoder model {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def score_pairs(self, pairs: list, batch_size: int = 32) -> list:
        """Score (query, chunk_text) pairs and return logits."""
        if not pairs:
            return []

        self._load_model()
        import torch

        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]

            inputs = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
                all_scores.extend(scores.cpu().tolist())

        return all_scores
