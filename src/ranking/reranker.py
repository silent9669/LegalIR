from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
import numpy as np
from src.models.device import resolve_device
from src.ranking.evidence_pack import EvidencePackBuilder
from src.retrieval.types import CandidateRecord


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        score_fn: Callable[[list[tuple[str, str]], int, int], list[float]] | None = None,
    ):
        self.model_name = str(model_name)
        self.device = resolve_device(device or "auto") if device != "cpu" and model_name != "mock" else "cpu"
        self.tokenizer = None
        self.model = None
        self.score_fn = score_fn

    def _load_model(self):
        if self.model is None and self.score_fn is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading reranker model {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 16, max_length: int = 512) -> list[float]:
        """
        pairs: list of (query, passage_text) tuples
        Returns: list of float scores
        """
        if not pairs:
            return []

        if self.score_fn is not None:
            return self.score_fn(pairs, batch_size=batch_size, max_length=max_length)

        self._load_model()
        import torch

        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]

            try:
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
            except RuntimeError as e:
                # Handle MPS OOM by halving batch size
                if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                    if batch_size > 1:
                        half_b = max(1, batch_size // 2)
                        print(f"Reranker OOM: reducing batch size from {batch_size} to {half_b}...")
                        sub_scores = self.score_pairs(batch, batch_size=half_b, max_length=max_length)
                        all_scores.extend(sub_scores)
                    else:
                        raise e
                else:
                    raise e

        return all_scores

    def aggregate_document(
        self,
        doc_id: str,
        chunk_records: list[dict[str, Any]],
        chunk_scores: list[float],
    ) -> dict[str, Any]:
        """Aggregates chunk-level reranker scores to a single document-level record."""
        if not chunk_scores:
            return {
                "doc_id": doc_id,
                "reranker_best_score": 0.0,
                "reranker_second_score": 0.0,
                "reranker_margin": 0.0,
                "reranker_best_chunk_id": None,
                "evidence_chunk_count": 0,
            }

        pairs = list(zip(chunk_records, chunk_scores))
        pairs.sort(key=lambda x: -x[1])

        best_chunk, best_score = pairs[0]
        second_score = pairs[1][1] if len(pairs) > 1 else best_score
        margin = float(best_score - second_score)

        return {
            "doc_id": doc_id,
            "reranker_score": float(best_score),
            "reranker_best_score": float(best_score),
            "reranker_second_score": float(second_score),
            "reranker_margin": margin,
            "reranker_best_chunk_id": str(best_chunk.get("chunk_id", "")),
            "evidence_chunk_count": len(chunk_scores),
        }

    def rerank(
        self,
        query: str,
        candidates: list[CandidateRecord],
        evidence_builder: EvidencePackBuilder,
        top_k: int = 50,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> list[CandidateRecord]:
        """
        Takes top_k candidates from hybrid union, scores their evidence packs with the cross-encoder,
        aggregates document features, and returns stably sorted candidate records.
        """
        if not candidates or not query:
            return candidates

        target_candidates = candidates[:top_k]
        remaining_candidates = candidates[top_k:]

        # 1. Build evidence packs for all candidates
        doc_to_packs = {}
        all_pairs = []
        pair_to_doc = []

        for cand in target_candidates:
            did = str(cand["doc_id"])
            packs = evidence_builder.build(query, did, candidate_record=cand, max_chunks=2)
            doc_to_packs[did] = packs
            for p in packs:
                all_pairs.append((query, p["text"]))
                pair_to_doc.append((did, p))

        # 2. Score all pairs in batched inference
        all_scores = self.score_pairs(all_pairs, batch_size=batch_size, max_length=max_length)

        # 3. Group scores by document
        doc_scores_map = defaultdict(list)
        doc_packs_map = defaultdict(list)
        for (did, pack), score in zip(pair_to_doc, all_scores):
            doc_scores_map[did].append(score)
            doc_packs_map[did].append(pack)

        # 4. Update candidate records
        reranked_target = []
        for cand in target_candidates:
            did = str(cand["doc_id"])
            agg = self.aggregate_document(
                did,
                doc_packs_map.get(did, []),
                doc_scores_map.get(did, []),
            )
            updated_cand = dict(cand)
            updated_cand.update(agg)
            reranked_target.append(updated_cand)

        # Stable sort reranked pool by descending reranker_best_score, ascending doc_id
        reranked_target.sort(key=lambda x: (-x["reranker_best_score"], x["doc_id"]))

        # For remaining candidates beyond top_k, fill default reranker fields
        for cand in remaining_candidates:
            cand["reranker_score"] = -999.0
            cand["reranker_best_score"] = -999.0
            cand["reranker_second_score"] = -999.0
            cand["reranker_margin"] = 0.0
            cand["reranker_best_chunk_id"] = None
            cand["evidence_chunk_count"] = 0

        return reranked_target + remaining_candidates
