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
