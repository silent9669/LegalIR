"""
Colab Single-T4 Contract Smoke Pipeline & Verification Utilities.

Authoritative specification: LEGALIR_CI_COLAB_KAGGLE_ARCHITECTURE_SPEC.md
Implementation plan: LEGALIR_CI_COLAB_KAGGLE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import yaml

# ==============================================================================
# Protected Score Keys & Validation
# ==============================================================================

PROTECTED_SCORE_KEYS: tuple[str, ...] = (
    # Candidate retrieval / RRF branch weights
    "weights",
    "branch_weights",
    "rrf_weights",
    "initial_weights",
    "field_weights",
    # Candidate counts & rerank targets
    "candidate_k",
    "top_k_candidates",
    "rerank_k",
    "top_k_for_rerank",
    "top_k_rerank",
    # Model identities
    "reranker_model",
    "model_name",
    "embedding_model",
    # Neural LoRA architecture & hyperparameters
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "loss_type",
    "loss",
    "learning_rate",
    "lr",
    # Fusion ranking policy & features
    "fusion_features",
    "feature_columns",
    "features",
    "fusion_policy",
    "selection_policy",
    "model_type",
    # Top-k selection logic
    "top_5_logic",
    "max_k",
    "min_k",
)

ALLOWED_SMOKE_KEYS: tuple[str, ...] = (
    "seed",
    "train_queries",
    "validation_queries",
    "public_queries",
    "max_documents",
    "max_micro_chunks",
    "max_macro_chunks",
    "folds",
    "reranker_optimizer_steps",
    "dense_batch_size",
    "reranker_batch_size",
    "device",
    "devices",
    "output_dir",
    "work_dir",
    "data_dir",
    "target_sha",
    "telemetry",
    "offline_mode",
    "run_mode",
)


def _extract_all_key_paths(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        key_str = f"{prefix}.{k}" if prefix else str(k)
        items[key_str] = v
        items[str(k)] = v
        if isinstance(v, Mapping):
            items.update(_extract_all_key_paths(v, key_str))
    return items


def validate_smoke_overrides(production: Mapping[str, Any], smoke: Mapping[str, Any]) -> None:
    """
    Validate that smoke configuration does NOT override or alter any protected production score keys.
    Raises ValueError on any protected key violation.
    """
    flat_smoke = _extract_all_key_paths(smoke)
    flat_prod = _extract_all_key_paths(production)

    for full_key, val in flat_smoke.items():
        # Check all segments of full_key
        segments = full_key.split(".")
        is_protected = any(seg in PROTECTED_SCORE_KEYS for seg in segments)

        if is_protected:
            # If it's a dict container (like weights: {...}), check if it defines conflicting values
            if isinstance(val, Mapping):
                # Will be checked at child leaf level
                continue

            prod_val = flat_prod.get(full_key)
            if prod_val is None:
                # Try finding by matching suffix in prod
                for pk, pv in flat_prod.items():
                    if pk.endswith(full_key) or full_key.endswith(pk):
                        prod_val = pv
                        break

            if prod_val is not None and prod_val != val:
                raise ValueError(
                    f"Protected score key '{full_key}' cannot be overridden in smoke configuration. "
                    f"Production value: {prod_val}, Smoke attempted override: {val}"
                )
            elif prod_val is None:
                raise ValueError(
                    f"Protected score key '{full_key}' found in smoke configuration without production baseline. "
                    f"Smoke configs must not define score-affecting hyperparameters."
                )



@dataclasses.dataclass
class ColabSmokeConfig:
    seed: int = 42
    train_queries: int = 64
    validation_queries: int = 32
    public_queries: int = 16
    max_documents: int = 2000
    folds: int = 2
    reranker_optimizer_steps: int = 10
    dense_batch_size: int = 16
    reranker_batch_size: int = 8
    device: str = "cuda:0"
    target_sha: str = ""

    @classmethod
    def from_yaml(cls, path: Path | str) -> ColabSmokeConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Colab smoke config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclasses.dataclass
class ColabSubsetManifest:
    parent_data_dir: str
    seed: int
    train_queries_count: int
    validation_queries_count: int
    public_queries_count: int
    documents_count: int
    chunks_count: int
    qrels_count: int
    selected_train_qids: list[str]
    selected_val_qids: list[str]
    selected_public_qids: list[str]
    selected_doc_ids: list[str]
    manifest_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


def build_colab_subset(
    data_dir: Path | str,
    out_dir: Path | str,
    config: ColabSmokeConfig,
) -> ColabSubsetManifest:
    """
    Build a deterministic, self-contained official-data smoke subset for Colab single-T4 verification.

    Guarantees:
    - Same seed -> exact same QIDs and doc IDs.
    - All qrel-positive documents for selected queries are included.
    - Distractor documents are corpus-valid.
    - No synthetic qrels or labels.
    - Validation QIDs are disjoint from train QIDs.
    - Chunks belong only to selected documents.
    """
    import random
    import numpy as np
    import pandas as pd

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load canonical data
    docs_path = data_dir / "documents.parquet"
    chunks_path = data_dir / "chunks.parquet"
    queries_path = data_dir / "queries_train.parquet"
    qrels_path = data_dir / "qrels_train.parquet"

    if not docs_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(f"Missing required canonical files in {data_dir}")

    docs_df = pd.read_parquet(docs_path)
    chunks_df = pd.read_parquet(chunks_path) if chunks_path.exists() else pd.DataFrame()
    queries_df = pd.read_parquet(queries_path)
    qrels_df = pd.read_parquet(qrels_path)

    docs_df["doc_id"] = docs_df["doc_id"].astype(str)
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)

    rng = random.Random(config.seed)

    # 2. Select Queries across folds
    splits_file = data_dir / "splits" / "random_5fold.json"
    all_query_ids = sorted(list(queries_df["query_id"].unique()))

    if splits_file.exists():
        try:
            fold_splits = json.loads(splits_file.read_text(encoding="utf-8"))
            fold_0 = fold_splits[0]
            fold_0_train = sorted([str(q) for q in fold_0.get("train_query_ids", []) if str(q) in all_query_ids])
            fold_0_val = sorted([str(q) for q in fold_0.get("val_query_ids", []) if str(q) in all_query_ids])

            rng.shuffle(fold_0_train)
            rng.shuffle(fold_0_val)

            selected_train_qids = sorted(fold_0_train[: config.train_queries])
            selected_val_qids = sorted(fold_0_val[: config.validation_queries])
        except Exception:
            shuffled = list(all_query_ids)
            rng.shuffle(shuffled)
            selected_train_qids = sorted(shuffled[: config.train_queries])
            selected_val_qids = sorted(shuffled[config.train_queries : config.train_queries + config.validation_queries])
    else:
        shuffled = list(all_query_ids)
        rng.shuffle(shuffled)
        n_train = min(len(shuffled), config.train_queries)
        n_val = min(max(0, len(shuffled) - n_train), config.validation_queries)
        selected_train_qids = sorted(shuffled[:n_train])
        selected_val_qids = sorted(shuffled[n_train : n_train + n_val])

    # If dataset is small, ensure train and val are disjoint
    if not selected_val_qids and len(selected_train_qids) > 1:
        split_idx = len(selected_train_qids) // 2
        selected_val_qids = selected_train_qids[split_idx:]
        selected_train_qids = selected_train_qids[:split_idx]

    all_selected_qids = sorted(list(set(selected_train_qids) | set(selected_val_qids)))

    # 3. Include all positive documents for selected queries
    pos_qrels = qrels_df[qrels_df["query_id"].isin(all_selected_qids)]
    pos_doc_ids = sorted(list(set(pos_qrels["doc_id"].unique())))

    # 4. Add deterministic corpus distractors
    all_doc_ids = sorted(list(docs_df["doc_id"].unique()))
    non_pos_doc_ids = [d for d in all_doc_ids if d not in set(pos_doc_ids)]
    rng.shuffle(non_pos_doc_ids)

    needed_distractors = max(0, config.max_documents - len(pos_doc_ids))
    distractors = non_pos_doc_ids[:needed_distractors]
    selected_doc_ids = sorted(list(set(pos_doc_ids) | set(distractors)))

    # 5. Filter DataFrames
    sub_docs = docs_df[docs_df["doc_id"].isin(selected_doc_ids)].copy()
    if not chunks_df.empty:
        chunks_df["doc_id"] = chunks_df["doc_id"].astype(str)
        sub_chunks = chunks_df[chunks_df["doc_id"].isin(selected_doc_ids)].copy()
    else:
        sub_chunks = pd.DataFrame()

    sub_queries = queries_df[queries_df["query_id"].isin(all_selected_qids)].copy()
    sub_qrels = qrels_df[
        qrels_df["query_id"].isin(all_selected_qids) & qrels_df["doc_id"].isin(selected_doc_ids)
    ].copy()

    # 6. Sample public queries
    selected_public_qids: list[str] = []
    public_official_path = data_dir / "public-official.json"
    if public_official_path.exists():
        try:
            pub_data = json.loads(public_official_path.read_text(encoding="utf-8"))
            pub_keys = sorted(list(pub_data.keys()))
            rng.shuffle(pub_keys)
            selected_pub_keys = sorted(pub_keys[: config.public_queries])
            selected_public_qids = selected_pub_keys
            sub_pub_dict = {k: pub_data[k] for k in selected_pub_keys}
            (out_dir / "public-official.json").write_text(
                json.dumps(sub_pub_dict, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    # 7. Write subset splits
    fold_0_train_sub = [q for q in selected_train_qids]
    fold_0_val_sub = [q for q in selected_val_qids]
    fold_1_train_sub = [q for q in selected_val_qids] if selected_val_qids else fold_0_train_sub
    fold_1_val_sub = [q for q in selected_train_qids] if selected_val_qids else fold_0_val_sub

    subset_folds = [
        {"fold": 0, "train_query_ids": fold_0_train_sub, "val_query_ids": fold_0_val_sub},
        {"fold": 1, "train_query_ids": fold_1_train_sub, "val_query_ids": fold_1_val_sub},
    ]
    (splits_dir / "random_5fold.json").write_text(
        json.dumps(subset_folds, indent=2), encoding="utf-8"
    )

    doc_disjoint = {
        "train_query_ids": fold_0_train_sub,
        "val_query_ids": fold_0_val_sub,
        "train_doc_ids": selected_doc_ids[: len(selected_doc_ids) // 2],
        "val_doc_ids": selected_doc_ids[len(selected_doc_ids) // 2 :],
    }
    (splits_dir / "doc_disjoint_split.json").write_text(
        json.dumps(doc_disjoint, indent=2), encoding="utf-8"
    )

    # 8. Save Parquet files
    sub_docs.to_parquet(out_dir / "documents.parquet", index=False)
    if not sub_chunks.empty:
        sub_chunks.to_parquet(out_dir / "chunks.parquet", index=False)
    sub_queries.to_parquet(out_dir / "queries_train.parquet", index=False)
    sub_qrels.to_parquet(out_dir / "qrels_train.parquet", index=False)

    # 9. Compute Manifest & Hash
    manifest_data = {
        "parent_data_dir": str(data_dir),
        "seed": config.seed,
        "train_queries_count": len(selected_train_qids),
        "validation_queries_count": len(selected_val_qids),
        "public_queries_count": len(selected_public_qids),
        "documents_count": len(selected_doc_ids),
        "chunks_count": len(sub_chunks),
        "qrels_count": len(sub_qrels),
        "selected_train_qids": selected_train_qids,
        "selected_val_qids": selected_val_qids,
        "selected_public_qids": selected_public_qids,
        "selected_doc_ids": selected_doc_ids,
    }
    raw_json = json.dumps(manifest_data, sort_keys=True)
    manifest_sha = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    manifest_data["manifest_sha256"] = manifest_sha

    (out_dir / "subset_manifest.json").write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8"
    )

    return ColabSubsetManifest(**manifest_data)

