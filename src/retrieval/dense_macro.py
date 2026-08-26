from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.device import resolve_device


class DenseMacroRetriever:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        device: str | None = None,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.device = resolve_device(device or "auto")
        self.tokenizer = None
        self.model = None
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.query_encoder: Callable[[list[str]], np.ndarray] | None = None

    @classmethod
    def from_arrays(
        cls,
        embeddings_path: str | Path | None = None,
        chunk_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
        query_encoder: Callable[[list[str]], np.ndarray] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> "DenseMacroRetriever":
        retriever = cls(device="cpu")
        if embeddings is not None:
            retriever.embeddings = embeddings
        elif embeddings_path is not None:
            retriever.embeddings = np.load(str(embeddings_path), mmap_mode="r")
        retriever.chunk_ids = [str(x) for x in (chunk_ids or [])]
        retriever.doc_ids = [str(x) for x in (doc_ids or [])]
        retriever.query_encoder = query_encoder
        return retriever

    def _load_model(self):
        if self.model is None:
            import torch
            from transformers import AutoTokenizer, AutoModel
            print(f"Loading dense model {self.model_name_or_path} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModel.from_pretrained(self.model_name_or_path)
            self.model.to(self.device)
            self.model.eval()

    def encode_texts(self, texts: list[str], batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        if self.query_encoder is not None:
            return self.query_encoder(texts)

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
                all_vecs.append(norm_rep.cpu().to(torch.float32).numpy())

        return np.vstack(all_vecs)

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """
        Returns list of candidate dicts with:
        doc_id, score, dense_score, dense_best_score, dense_second_score, dense_best_chunk_id
        """
        if self.embeddings is None or not query:
            return []

        q_vec = self.encode_texts([query], batch_size=1)[0].astype(np.float32)
        # Cosine similarity via dot product of normalized vectors
        sims = np.dot(self.embeddings.astype(np.float32), q_vec)

        # Top chunk indices via argpartition
        candidate_count = min(top_k * 6, len(sims))
        top_chunk_indices = np.argpartition(sims, -candidate_count)[-candidate_count:]
        top_chunk_indices = top_chunk_indices[np.argsort(-sims[top_chunk_indices])]

        doc_chunk_scores = defaultdict(list)
        for c_idx in top_chunk_indices:
            did = self.doc_ids[c_idx]
            cid = self.chunk_ids[c_idx]
            doc_chunk_scores[did].append((cid, float(sims[c_idx])))

        doc_records = []
        for did, items in doc_chunk_scores.items():
            sorted_items = sorted(items, key=lambda x: -x[1])
            best_cid, best_s = sorted_items[0]
            second_s = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
            mean_s = sum(x[1] for x in items) / len(items)
            agg_score = best_s + 0.1 * second_s

            doc_records.append({
                "doc_id": did,
                "score": agg_score,
                "dense_score": agg_score,
                "dense_best_score": best_s,
                "dense_second_score": second_s,
                "dense_mean_score": mean_s,
                "dense_best_chunk_id": best_cid,
            })

        doc_records.sort(key=lambda x: (-x["score"], x["doc_id"]))
        return doc_records[:top_k]

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        output_dir: str | Path,
        model_name_or_path: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        encoder = cls(model_name_or_path=model_name_or_path, device=device)
        texts = [c.get("text_norm") or c.get("text_raw", "") for c in chunks]
        chunk_ids = [str(c["chunk_id"]) for c in chunks]
        doc_ids = [str(c["doc_id"]) for c in chunks]

        print(f"Encoding {len(texts)} macro chunks with {model_name_or_path}...")
        embeddings = encoder.encode_texts(texts, batch_size=batch_size, max_length=max_length)
        embeddings_fp16 = embeddings.astype(np.float16)

        emb_path = output_dir / "embeddings.npy"
        np.save(str(emb_path), embeddings_fp16)

        meta_df = pd.DataFrame({"chunk_id": chunk_ids, "doc_id": doc_ids})
        meta_df.to_parquet(output_dir / "chunks_meta.parquet", index=False)

        manifest = {
            "model_name_or_path": model_name_or_path,
            "total_macro_chunks": len(chunks),
            "embedding_dimension": int(embeddings.shape[1]),
            "dtype": "float16",
            "max_length": max_length,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Saved dense index to {output_dir}!")
        return output_dir

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        model_name_or_path: str = "BAAI/bge-m3",
        device: str | None = None,
    ) -> "DenseMacroRetriever":
        index_dir = Path(index_dir)
        emb_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "chunks_meta.parquet"

        meta_df = pd.read_parquet(meta_path)
        retriever = cls(model_name_or_path=model_name_or_path, device=device)
        retriever.embeddings = np.load(str(emb_path), mmap_mode="r")
        retriever.chunk_ids = meta_df["chunk_id"].astype(str).tolist()
        retriever.doc_ids = meta_df["doc_id"].astype(str).tolist()
        return retriever
