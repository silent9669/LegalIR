import os
import numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

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
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading reranker model {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def score_pairs(self, pairs: list, batch_size: int = 16, max_length: int = 512) -> list:
        """
        pairs: list of (query, passage_text) tuples
        Returns: list of float scores
        """
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
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.view(-1).float().cpu().numpy()
                all_scores.extend(logits.tolist())

        return all_scores
