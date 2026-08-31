"""Production Kaggle T4 x2 End-to-End Orchestrator for LegalIR Task 1.

Executes the complete 24-step pipeline:
1. Multi-GPU environment detection and device allocation (GPU 0: Dense, GPU 1: Reranker).
2. Canonical dataset discovery, auto-build from raw archives if needed, and strict validation.
3. Preflight parameter budget audit against the strict <4B rule.
4. Dual BM25 indexing (raw legal + PyVi word-segmented) with document metadata enrichment.
5. DEk21 Dense Macro indexing on GPU 0.
6. Precomputing and caching train query dense embeddings on GPU 0.
7. Fold-isolated hard-negative training pair mining.
8. 5-Fold Out-of-Fold (OOF) cross-validation with fold-specific LoRA rerankers.
9. Full cross-fitted Learned Fusion (LightGBM) vs Tuned RRF evaluation.
10. Document-disjoint robustness split validation.
11. Full 7,000-query question memory indexing with precomputed embeddings.
12. Final LoRA reranker fine-tuning on all 7,000 training queries on GPU 1.
13. Final fusion model fitting on all OOF candidate features (if learned fusion won).
14. Public test query discovery and batch inference with the fully loaded production pipeline.
15. Strict submission invariant validation (bounds, types, no duplicates, valid corpus IDs).
16. Zip packaging containing strictly submission.json at root.
17. Full artifact manifest creation with SHA-256 hashes and parameter audits.
18. Support for run_mode="smoke" and run_mode="full".
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Literal
import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset.build_canonical import build_canonical_package
from src.dataset.validator import validate_canonical_dataset
from src.evaluation.submission import (
    compute_sha256,
    create_submission_manifest,
    package_submission,
    validate_submission,
    validate_submission_zip,
)
from src.models.parameter_audit import (
    MAX_PARAMETER_BUDGET,
    ParameterBudgetExceededError,
    audit_system_parameters,
)
from src.pipeline.oof_runner import OOFRunner
from src.pipeline.predict import LegalIRPipeline
from src.ranking.train_fusion import train_and_evaluate_fusion_cv
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.build_indexes import (
    build_bm25_index,
    build_bm25_pyvi_index,
    enrich_chunks_with_doc_metadata,
)
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.question_memory import TrainQuestionMemory
from src.training.build_pairs import build_training_pairs
from src.training.train_reranker import train_reranker


@dataclass
class KaggleRunResult:
    """Structured result of an end-to-end LegalIR Kaggle pipeline run."""

    is_valid: bool
    submission_path: Path
    submission_zip_path: Path
    manifest_path: Path
    audit_report: dict[str, Any]
    cv_report: dict[str, Any]
    fusion_report: dict[str, Any]
    public_predictions_count: int
    run_mode: str
    execution_time_seconds: float
    artifacts_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_repo_path(value: str | Path | None, repo_root: str | Path) -> Path:
    """Resolve a path relative to the repo root if it is not already absolute."""
    if value is None:
        return Path(repo_root)
    p = Path(value)
    if not p.is_absolute():
        p = Path(repo_root) / p
    return p.resolve()


def resolve_kaggle_devices(devices: list[str] | None = None) -> tuple[str, str]:
    """
    Allocate multi-GPU devices intentionally:
    GPU 0: Dense embedding / index / query encoding
    GPU 1: Reranker training + reranker inference
    """
    if devices and len(devices) >= 2:
        return str(devices[0]), str(devices[1])
    elif devices and len(devices) == 1:
        return str(devices[0]), str(devices[0])

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count >= 2:
        return "cuda:0", "cuda:1"
    elif device_count == 1:
        return "cuda:0", "cuda:0"
    else:
        return "cpu", "cpu"


def get_git_commit(repo_root: Path | None = None) -> str:
    """Retrieve Git commit SHA safely."""
    try:
        cwd = repo_root or Path.cwd()
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("utf-8")
            .strip()
        )
        return commit_sha
    except Exception:
        return "unknown"


CANONICAL_REQUIRED_FILES = {
    "documents.parquet",
    "chunks.parquet",
    "queries_train.parquet",
    "qrels_train.parquet",
}


def discover_data_dir(
    data_dir: str | Path | None = None, repo_root: Path | None = None
) -> Path:
    """Discover canonical dataset or build from raw competition files if missing."""
    if data_dir is not None:
        p = Path(data_dir)
        if p.exists() and all((p / f).exists() for f in CANONICAL_REQUIRED_FILES):
            return p.resolve()
        elif p.exists():
            return p.resolve()

    repo = repo_root or Path.cwd()
    candidate_paths = [
        Path("/kaggle/input/legalir"),
        Path("/kaggle/input/legalir/artifacts/task1/data"),
        Path("/kaggle/input/legalir/artifacts/shared/canonical/v2"),
        Path("/kaggle/input/legalir-task1/artifacts/task1/data"),
        Path("/kaggle/input/legalir-task-1/artifacts/task1/data"),
        Path("/kaggle/input/uit-dsc-2026-task1/artifacts/task1/data"),
        Path("/kaggle/input/legalir-dataset/artifacts/task1/data"),
        Path("/kaggle/input/legalir-canonical/artifacts/task1/data"),
        repo / "artifacts/task1/data",
        repo / "artifacts/shared/canonical/v2",
        Path.cwd() / "artifacts/task1/data",
    ]
    for cand in candidate_paths:
        if cand.exists() and all((cand / f).exists() for f in CANONICAL_REQUIRED_FILES):
            return cand.resolve()

    # Search for raw files
    raw_zip = None
    train_json = None
    for cand_zip in [
        Path("/kaggle/input/legalir/selected-contexts.zip"),
        Path("/kaggle/input/legalir/artifacts/raw/selected-contexts.zip"),
        Path("/kaggle/input/legalir/artifacts/shared/raw/selected-contexts.zip"),
        Path("/kaggle/input/legalir-task1/selected-contexts.zip"),
        Path("/kaggle/input/legalir-task-1/selected-contexts.zip"),
        Path("/kaggle/input/uit-dsc-2026-task1/selected-contexts.zip"),
        repo / "artifacts/shared/raw/selected-contexts.zip",
        repo / "artifacts/raw/selected-contexts.zip",
        repo / "selected-contexts.zip",
    ]:
        if cand_zip.exists():
            raw_zip = cand_zip
            break

    for cand_train in [
        Path("/kaggle/input/legalir/train.json"),
        Path("/kaggle/input/legalir/artifacts/raw/train.json"),
        Path("/kaggle/input/legalir/artifacts/shared/raw/train.json"),
        Path("/kaggle/input/legalir-task1/train.json"),
        Path("/kaggle/input/legalir-task-1/train.json"),
        Path("/kaggle/input/uit-dsc-2026-task1/train.json"),
        repo / "artifacts/shared/raw/train.json",
        repo / "artifacts/raw/train.json",
        repo / "train.json",
    ]:
        if cand_train.exists():
            train_json = cand_train
            break

    target_dir = (
        Path("/kaggle/working/legalir_run/data")
        if Path("/kaggle/working").exists()
        else repo / "artifacts/task1/data"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    if raw_zip and train_json:
        print(
            f"[*] Building canonical dataset from {raw_zip} and {train_json} into {target_dir}..."
        )
        build_canonical_package(
            raw_contexts_dir=raw_zip,
            train_json_path=train_json,
            output_dir=target_dir,
        )
    return target_dir.resolve()


def discover_public_test_file(
    public_json_path: str | Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    """Discover public-official.json test file."""
    if public_json_path is not None:
        p = Path(public_json_path)
        if p.exists():
            return p.resolve()
        return None

    repo = repo_root or Path.cwd()
    for cand in [
        Path("/kaggle/input/legalir/public-official.json"),
        Path("/kaggle/input/legalir/artifacts/raw/public-official.json"),
        Path("/kaggle/input/legalir/artifacts/shared/raw/public-official.json"),
        Path("/kaggle/input/legalir-task1/public-official.json"),
        Path("/kaggle/input/legalir-task-1/public-official.json"),
        Path("/kaggle/input/uit-dsc-2026-task1/public-official.json"),
        Path("/kaggle/input/legalir-dataset/public-official.json"),
        repo / "artifacts/shared/raw/public-official.json",
        repo / "artifacts/raw/public-official.json",
        repo / "public-official.json",
        Path.cwd() / "public-official.json",
    ]:
        if cand.exists():
            return cand.resolve()
    return None


def run_kaggle_pipeline(
    *,
    data_dir: str | Path | None = None,
    working_dir: str | Path = "/kaggle/working/legalir_run",
    run_mode: Literal["smoke", "gpu_smoke", "full"] | str = "full",
    hf_token: str | None = None,
    devices: list[str] | None = None,
    runtime_config_path: str | Path | None = "configs/kaggle.yaml",
    reranker_config_path: str | Path | None = "configs/experiments/reranker_lora.yaml",
    config_path: str | Path | None = None,
    public_json_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    dense_device: str | None = None,
    reranker_device: str | None = None,
    strict_artifacts: bool | None = None,
) -> KaggleRunResult:
    """
    Execute the complete 24-step high-score LegalIR production pipeline on Kaggle.

    Steps executed:
    1. Multi-GPU environment detection and device allocation (GPU 0: Dense, GPU 1: Reranker).
    2. Canonical dataset discovery, validation, and metadata enrichment.
    3. Preflight parameter budget audit (<4B rule).
    4. Dual BM25 index building/loading (raw legal + PyVi segmented) with metadata enrichment.
    5. DEk21 Dense Macro index building/loading on GPU 0.
    6. Precomputing and caching train query dense embeddings on GPU 0.
    7. Mining fold-isolated hard-negative training pairs.
    8. 5-Fold OOF cross-validation with fold-specific LoRA rerankers.
    9. Full cross-fitted Learned Fusion (LightGBM) vs Tuned RRF evaluation.
    10. Document-disjoint robustness split validation with dedicated trained reranker.
    11. Full 7,000-query question memory indexing with precomputed embeddings.
    12. Final LoRA reranker fine-tuning on all 7,000 training queries on GPU 1.
    13. Final fusion model fitting on all OOF candidate features (if learned fusion won).
    14. Public test query discovery and batch inference with the fully loaded production pipeline.
    15. Strict submission invariant validation (bounds, types, no duplicates, valid corpus IDs).
    16. Zip packaging containing strictly submission.json at root.
    17. Full artifact manifest creation with SHA-256 hashes and parameter audits.
    18. Support for run_mode="smoke", "gpu_smoke", and "full".
    """
    t_start = time.time()
    run_mode_str = str(run_mode).lower().strip()
    is_smoke = run_mode_str == "smoke"
    is_gpu_smoke = run_mode_str == "gpu_smoke"
    is_full = run_mode_str == "full"

    if strict_artifacts is None:
        strict_artifacts = is_full or is_gpu_smoke

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except Exception:
                pass

    print("=" * 80)
    print("LEGALIR TASK 1: KAGGLE T4 x2 PRODUCTION ORCHESTRATOR")
    print(f"Execution Mode: {run_mode_str.upper()} | Start Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Setup paths and directories
    root_path = Path(repo_root) if repo_root else Path.cwd()
    working_path = Path(working_dir)
    working_path.mkdir(parents=True, exist_ok=True)
    index_dir = working_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    cv_dir = working_path / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = working_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = working_path / "training_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = working_path / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # 1b. Fail-Fast Early Public Test Check in FULL mode
    public_test_file = discover_public_test_file(public_json_path, repo_root=root_path)
    if is_full and (public_test_file is None or not public_test_file.exists()):
        raise FileNotFoundError("Full mode requires official public-official.json; refusing to generate submission")

    # 2. Hardware and GPU Device Allocation (P1.10)
    from src.models.device import resolve_device

    if dense_device is None or reranker_device is None:
        d_dev, r_dev = resolve_kaggle_devices(devices)
        if dense_device is None:
            dense_device = d_dev
        if reranker_device is None:
            reranker_device = r_dev

    if not is_smoke:
        dense_device = resolve_device(dense_device)
        reranker_device = resolve_device(reranker_device)
    else:
        dense_device = "cpu"
        reranker_device = "cpu"

    print(f"[+] Device Allocation (P1.10 Multi-GPU Utilization):")
    print(f"    - Dense Embedding & Question Encoding : {dense_device}")
    print(f"    - Reranker Training & Neural Inference: {reranker_device}")

    # Set HF token in environment if provided securely
    if hf_token:
        os.environ["HF_TOKEN"] = str(hf_token)

    # 3. Canonical Data Discovery & Validation (P1.5)
    canonical_data_dir = discover_data_dir(data_dir, repo_root=root_path)
    print(f"[+] Canonical Data Directory: {canonical_data_dir}")

    docs_path = canonical_data_dir / "documents.parquet"
    chunks_path = canonical_data_dir / "chunks.parquet"
    queries_train_path = canonical_data_dir / "queries_train.parquet"
    qrels_train_path = canonical_data_dir / "qrels_train.parquet"

    if not (docs_path.exists() and chunks_path.exists()):
        raise FileNotFoundError(
            f"Canonical dataset parquet files missing in {canonical_data_dir}"
        )

    val_report = validate_canonical_dataset(canonical_data_dir)
    print(f"[+] Canonical Dataset Validation: is_valid = {val_report.get('is_valid')}")
    if not val_report.get("is_valid") and is_full:
        raise ValueError(f"Canonical dataset validation failed in FULL mode: {val_report.get('errors')}")

    df_docs = pd.read_parquet(docs_path)
    df_chunks = pd.read_parquet(chunks_path)
    df_queries = pd.read_parquet(queries_train_path) if queries_train_path.exists() else pd.DataFrame()
    df_qrels = pd.read_parquet(qrels_train_path) if qrels_train_path.exists() else pd.DataFrame()
    print(f"[+] Dataset Loaded: {len(df_docs):,} documents | {len(df_chunks):,} chunks | {len(df_queries):,} train queries")

    # 4. Strict Parameter Budget Preflight Audit (<4B Rule)
    if runtime_config_path:
        resolved_runtime_config = resolve_repo_path(runtime_config_path, root_path)
    elif config_path:
        resolved_runtime_config = resolve_repo_path(config_path, root_path)
    else:
        resolved_runtime_config = resolve_repo_path("configs/kaggle.yaml", root_path)

    if not resolved_runtime_config.exists():
        fallback_cfg = resolve_repo_path("configs/pipeline.yaml", root_path)
        if fallback_cfg.exists():
            resolved_runtime_config = fallback_cfg

    if reranker_config_path:
        resolved_reranker_config = resolve_repo_path(reranker_config_path, root_path)
    else:
        resolved_reranker_config = resolve_repo_path("configs/experiments/reranker_lora.yaml", root_path)

    if (is_full or is_gpu_smoke) and not resolved_reranker_config.is_file():
        raise FileNotFoundError(
            f"Reranker training config not found at: {resolved_reranker_config}"
        )

    if not resolved_reranker_config.exists():
        resolved_reranker_config = resolve_repo_path("configs/pipeline.yaml", root_path)

    audit_json_path = working_path / "parameter_audit.json"
    audit_report = audit_system_parameters(
        config_path=resolved_runtime_config if resolved_runtime_config.exists() else None,
        output_json=audit_json_path,
        raise_on_violation=True,
        offline_fallback=True,
    )
    print(
        f"[+] Parameter Budget Preflight: {audit_report['total_learned_parameters']:,} params "
        f"({audit_report['total_parameters_billions']:.4f}B / 4.0B limit, "
        f"{audit_report['budget_utilization_pct']:.2f}% utilization). PASS"
    )

    # Validate explicit reranker training config
    reranker_cfg = yaml.safe_load(resolved_reranker_config.read_text(encoding="utf-8")) if resolved_reranker_config.exists() else {}
    if is_full:
        effective_max_steps = reranker_cfg.get("max_steps", 500)
        if effective_max_steps is None:
            raise ValueError("Full reranker training requires explicit max_steps")
        print(f"[+] Effective Reranker Config: model={reranker_cfg.get('base_model_name', 'BAAI/bge-reranker-v2-m3')}, "
              f"batch_size={reranker_cfg.get('batch_size', 2)}, grad_accum={reranker_cfg.get('gradient_accumulation_steps', 8)}, "
              f"max_length={reranker_cfg.get('max_length', 512)}, max_steps={effective_max_steps}, "
              f"fp16={reranker_cfg.get('fp16', True)}, device={reranker_device}")
    elif is_gpu_smoke:
        effective_max_steps = 3
    else:
        effective_max_steps = 5

    # 5. Build / Load Dual BM25 Indexes with Metadata Enrichment (P1.5)
    micro_chunks = (
        df_chunks[df_chunks["granularity"] == "micro"]
        if "granularity" in df_chunks.columns
        else df_chunks
    )
    if docs_path.exists():
        micro_chunks = enrich_chunks_with_doc_metadata(micro_chunks, docs_path)

    # 5a. Fielded Legal BM25
    bm25_dir = index_dir / "bm25"
    if (bm25_dir / "bm25_micro_index.pkl").exists() or (bm25_dir.exists() and list(bm25_dir.glob("*.pkl"))):
        print(f"[*] Loading cached Legal BM25 index from {bm25_dir}...")
        bm25_legal = BM25MicroRetriever.load(bm25_dir)
    else:
        print(f"[*] Building Legal BM25 index with metadata enrichment...")
        bm25_legal = BM25MicroRetriever(k1=1.5, b=0.75).fit(micro_chunks.to_dict("records"), show_progress=False)
        bm25_legal.save(bm25_dir)
    print(f"[+] Legal BM25 ready ({len(bm25_legal.corpus):,} docs).")

    # 5b. PyVi Segmented BM25
    bm25_pyvi_dir = index_dir / "bm25_pyvi"
    if (bm25_pyvi_dir / "bm25_pyvi_index.pkl").exists() or (bm25_pyvi_dir.exists() and list(bm25_pyvi_dir.glob("*.pkl"))):
        print(f"[*] Loading cached PyVi BM25 index from {bm25_pyvi_dir}...")
        bm25_pyvi = BM25PyViRetriever.load(bm25_pyvi_dir)
    else:
        print(f"[*] Building PyVi BM25 index with metadata enrichment...")
        bm25_pyvi = BM25PyViRetriever(k1=1.5, b=0.75).fit(micro_chunks.to_dict("records"), show_progress=False)
        bm25_pyvi.save(bm25_pyvi_dir)
    print(f"[+] PyVi BM25 ready ({len(bm25_pyvi.corpus):,} docs).")

    # 6. Build / Load DEk21 Dense Macro Index (GPU 0)
    dense_dir = index_dir / "dense_dek21"
    dense_retriever: DenseMacroRetriever | None = None
    if (dense_dir / "embeddings.npy").exists():
        print(f"[*] Loading cached DEk21 Dense index from {dense_dir} on {dense_device}...")
        try:
            dense_retriever = DenseMacroRetriever.load(dense_dir, device=dense_device)
        except Exception as e:
            if is_full or is_gpu_smoke:
                raise RuntimeError(f"Failed to load DEk21 Dense index from {dense_dir} in {run_mode_str.upper()} mode: {e}") from e
            print(f"[-] Warning: Failed to load dense index: {e}")
    elif not is_smoke:
        print(f"[*] Building DEk21 Dense index on {dense_device}...")
        macro_chunks = (
            df_chunks[df_chunks["granularity"] == "macro"]
            if "granularity" in df_chunks.columns
            else df_chunks
        )
        dense_retriever = DenseMacroRetriever(
            model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
            device=dense_device,
            dimension=768,
        )
        dense_batch = 128 if "cuda" in str(dense_device) else 32
        try:
            dense_retriever.fit(macro_chunks.to_dict("records"), batch_size=dense_batch)
            dense_retriever.save(dense_dir)
            print(f"[+] DEk21 Dense Index ready ({len(dense_retriever.doc_ids):,} chunks).")
        except Exception as e:
            if is_full or is_gpu_smoke:
                raise RuntimeError(f"Failed to build DEk21 Dense index on {dense_device} in {run_mode_str.upper()} mode: {e}") from e
            print(f"[-] Warning: Dense index building failed: {e}")
            dense_retriever = None

    # 7. Precompute Train Query Dense Embeddings on GPU 0 (P1.10)
    train_query_embs: dict[str, np.ndarray] = {}
    if not df_queries.empty and dense_retriever is not None:
        q_records = df_queries.to_dict("records")
        qids = [str(r["query_id"]) for r in q_records]
        qtexts = [
            str(r.get("question_norm") or r.get("question_raw") or r.get("question") or "")
            for r in q_records
        ]
        try:
            print(f"[*] Precomputing train query embeddings for {len(qids):,} queries on {dense_device}...")
            embs = dense_retriever.encode_queries(
                qtexts, batch_size=128 if "cuda" in str(dense_device) else 32
            )
            for qid, emb in zip(qids, embs):
                train_query_embs[qid] = emb
            np.save(str(index_dir / "train_query_embeddings.npy"), embs)
            print(f"[+] Precomputed and cached {len(train_query_embs):,} train query dense embeddings.")
        except Exception as e:
            if is_full or is_gpu_smoke:
                raise RuntimeError(f"Failed to precompute train query embeddings on {dense_device} in {run_mode_str.upper()} mode: {e}") from e
            print(f"[-] Warning: query embedding precomputation skipped: {e}")

    # 8. Out-of-Fold (OOF) 5-Fold Cross-Validation with Fold-Trained LoRA Rerankers (P1.1, P1.2, P1.7, P1.10)
    print("\n" + "=" * 70)
    print(f"[*] Starting 5-Fold OOF Cross-Validation (Mode: {run_mode_str.upper()})...")
    print("=" * 70)

    oof_runner = OOFRunner(
        data_dir=canonical_data_dir,
        index_dir=index_dir,
        output_dir=cv_dir,
        splits_path=canonical_data_dir / "splits/random_5fold.json" if (canonical_data_dir / "splits/random_5fold.json").exists() else None,
        doc_disjoint_splits_path=canonical_data_dir / "splits/doc_disjoint_split.json" if (canonical_data_dir / "splits/doc_disjoint_split.json").exists() else None,
        config_path=resolved_runtime_config,
        reranker_config_path=resolved_reranker_config,
        num_folds=2 if (is_smoke or is_gpu_smoke) else 5,
        candidate_k=20 if (is_smoke or is_gpu_smoke) else 150,
        rerank_k=10 if (is_smoke or is_gpu_smoke) else 50,
        use_reranker=True,
        reranker_model="mock" if is_smoke else "BAAI/bge-reranker-v2-m3",
        train_reranker_per_fold=(not is_smoke),
        dense_device=dense_device,
        reranker_device=reranker_device,
        smoke=(is_smoke or is_gpu_smoke),
        smoke_sample_size=20 if (is_smoke or is_gpu_smoke) else 50,
        doc_disjoint=True,
        train_query_embeddings=train_query_embs,
    )
    cv_report = oof_runner.run()

    # 9. Cross-Fitted Learned Fusion vs Tuned RRF Evaluation & Final Model Training (P1.4)
    print("\n" + "=" * 70)
    print("[*] Evaluating Cross-Fitted Learned Fusion vs Tuned RRF (P1.4)...")
    print("=" * 70)

    oof_feat_path = cv_dir / "oof_features.parquet"
    fusion_report: dict[str, Any] = {}
    if oof_feat_path.exists():
        oof_df = pd.read_parquet(oof_feat_path)
        if not oof_df.empty and "fold" in oof_df.columns:
            qrels_dict = defaultdict(list)
            for r in df_qrels.to_dict("records"):
                qrels_dict[str(r["query_id"])].append(str(r["doc_id"]))

            fusion_final_dir = checkpoints_dir / "fusion_final"
            fusion_report = train_and_evaluate_fusion_cv(
                oof_df=oof_df,
                qrels_dict=qrels_dict,
                output_dir=fusion_final_dir,
                num_boost_round=100 if not (is_smoke or is_gpu_smoke) else 10,
            )

    winning_fusion_method = fusion_report.get("winning_method", "reciprocal_rank_fusion")
    use_learned_fusion = bool(winning_fusion_method == "learned_ranker")

    # 10. Build Full 7,000-Query Memory & Train Final LoRA Reranker on All 7,000 Queries (P1.3, P1.8, P1.10, P1.11)
    print("\n" + "=" * 70)
    print(f"[*] Training Final System on All Training Queries (P1.3 / P1.8 / P1.11)...")
    print("=" * 70)

    # 10a. Full Question Memory
    if not df_queries.empty and not df_qrels.empty:
        queries_dict = {
            str(r["query_id"]): str(
                r.get("question_norm") or r.get("question_raw") or r.get("question") or ""
            )
            for r in df_queries.to_dict("records")
        }
        qrels_dict_full = defaultdict(list)
        for r in df_qrels.to_dict("records"):
            qrels_dict_full[str(r["query_id"])].append(str(r["doc_id"]))

        queries_for_memory = [
            (qid, queries_dict[qid], train_query_embs.get(qid))
            if qid in train_query_embs
            else (qid, queries_dict[qid], None)
            for qid in queries_dict.keys()
        ]
        full_memory = TrainQuestionMemory(min_similarity=0.82, dense_encoder=dense_retriever, dense_device=dense_device)
        full_memory.fit(queries_for_memory, qrels_dict_full)
        if len(full_memory.qids) == 0:
            raise ValueError("Final Question Memory has 0 indexed queries")
        full_mem_dir = index_dir / "question_memory"
        full_memory.save(full_mem_dir)
        print(f"[+] Full Question Memory Index saved to {full_mem_dir} ({len(full_memory.qids):,} queries).")

    # 10b. Mine hard-negative training pairs from ALL training queries
    limit_pairs = 100 if (is_smoke or is_gpu_smoke) else None
    print(f"[*] Mining hard-negative training pairs on all training queries (limit={limit_pairs})...")
    build_training_pairs(
        data_dir=canonical_data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        fold=None,
        use_all_queries=True,
        limit=limit_pairs,
        negatives_per_positive=8,
        query_embeddings=train_query_embs,
    )
    final_pairs_file = pairs_dir / "reranker_pairs.parquet"

    # 10c. Train Final LoRA Reranker on GPU 1
    final_reranker_dir = checkpoints_dir / "reranker_final"
    max_final_steps = effective_max_steps
    print(f"[*] Training Final Supervised LoRA Reranker on {reranker_device} (max_steps={max_final_steps})...")
    final_reranker_report = train_reranker(
        pairs_file=final_pairs_file,
        config_path=resolved_reranker_config,
        output_dir=final_reranker_dir,
        fold=None,
        max_steps=max_final_steps,
        base_model_name="mock" if is_smoke else "BAAI/bge-reranker-v2-m3",
        device=reranker_device,
    )
    print(f"[+] Final Reranker Training Status: {final_reranker_report.get('status')} | Checkpoint: {final_reranker_dir}")

    # 11. Public Test Inference with Fully Loaded Final Production Pipeline (P0.3, P0.4, P1.8)
    print("\n" + "=" * 70)
    print("[*] Executing Public Test Batch Inference with Final Production Pipeline (P1.8)...")
    print("=" * 70)

    public_test_file = discover_public_test_file(public_json_path, repo_root=root_path)
    if is_full:
        if public_test_file is None or not public_test_file.exists():
            raise FileNotFoundError("Full mode requires official public-official.json; refusing to generate submission")
        with open(public_test_file, "r", encoding="utf-8") as f:
            public_data = json.load(f)
    elif public_test_file and public_test_file.exists():
        print(f"[+] Found Public Test Queries: {public_test_file}")
        with open(public_test_file, "r", encoding="utf-8") as f:
            public_data = json.load(f)
    else:
        print("[!] public-official.json not found. Using train queries sample for inference verification.")
        sample_records = df_queries.head(20 if (is_smoke or is_gpu_smoke) else 100).to_dict("records")
        public_data = {
            str(r["query_id"]): {
                "question": str(
                    r.get("question_norm") or r.get("question_raw") or r.get("question") or ""
                )
            }
            for r in sample_records
        }

    official_public_qids = set(public_data.keys())
    print(f"[+] Total Public Test Queries: {len(public_data)}")

    # Load fully integrated pipeline
    fusion_model_load_path = (checkpoints_dir / "fusion_final") if use_learned_fusion else None
    reranker_adapter_load_path = final_reranker_dir if final_reranker_dir.exists() else None

    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=canonical_data_dir,
        index_dir=index_dir,
        reranker_adapter_path=reranker_adapter_load_path,
        fusion_model_path=fusion_model_load_path,
        use_reranker=True,
        use_learned_fusion=use_learned_fusion,
        dense_device=dense_device,
        reranker_device=reranker_device,
        strict_artifacts=strict_artifacts,
        audit_preflight=True,
        audit_output_json=working_path / "parameter_audit.json",
        reranker_model_name="mock" if is_smoke else "BAAI/bge-reranker-v2-m3",
    )

    t0_infer = time.time()
    predictions: dict[str, dict[str, list[str]]] = {}
    q_items = list(public_data.items())
    if (is_smoke or is_gpu_smoke) and len(q_items) > 20:
        q_items = q_items[:20]

    # Precompute public query dense embeddings once on GPU0
    public_q_embs: dict[str, np.ndarray] = {}
    dense_ret = getattr(pipeline.hybrid_engine, "dense_retriever", None) or getattr(pipeline.hybrid_engine, "dense", None)
    if dense_ret is not None and len(q_items) > 0:
        print(f"[*] Precomputing public query dense embeddings for {len(q_items)} queries on {dense_device}...")
        try:
            q_texts = [
                (q_val.get("question", "") if isinstance(q_val, dict) else str(q_val))
                for _, q_val in q_items
            ]
            q_embs_array = dense_ret.encode_queries(
                q_texts, batch_size=32 if is_full else 16
            )
            for (qid, _), emb in zip(q_items, q_embs_array):
                public_q_embs[str(qid)] = emb
            print(f"[+] Precomputed {len(public_q_embs):,} public query embeddings for reuse.")
        except Exception as e:
            if is_full or is_gpu_smoke:
                raise RuntimeError(f"Failed to precompute public query embeddings on {dense_device} in {run_mode_str.upper()} mode: {e}") from e
            print(f"[-] Warning: public query embedding precomputation skipped: {e}")

    for idx, (qid, q_val) in enumerate(q_items, start=1):
        q_text = q_val.get("question", "") if isinstance(q_val, dict) else str(q_val)
        q_emb = public_q_embs.get(str(qid))
        pred_docs = pipeline.predict_single(
            query=q_text,
            query_id=str(qid),
            top_k_candidates=150 if is_full else 20,
            top_k_rerank=50 if is_full else 10,
            q_emb=q_emb,
        )
        predictions[str(qid)] = {"answer": pred_docs}
        if idx % 100 == 0 or idx == len(q_items):
            elapsed = time.time() - t0_infer
            print(f"    [{idx:4d}/{len(q_items):4d}] queries predicted ({idx / elapsed:.2f} q/s)")

    print(f"[+] Public inference completed in {time.time() - t0_infer:.2f}s ({len(predictions)} queries predicted).")

    # 12. Strict Submission Invariant Validation & Zip Packaging (P0.5, Invariants 1-24)
    sub_json = submissions_dir / "submission.json"
    sub_zip = submissions_dir / "submission.zip"
    package_submission(predictions, sub_json, sub_zip)

    expected_qids = official_public_qids if is_full else set(predictions.keys())
    if is_full:
        assert set(predictions.keys()) == expected_qids, f"Prediction keys mismatch with official public keys: missing {len(expected_qids - set(predictions.keys()))}, extra {len(set(predictions.keys()) - expected_qids)}"

    official_doc_ids = set(df_docs["doc_id"].astype(str)) if "doc_id" in df_docs.columns else None
    val_res = validate_submission(sub_json, expected_qids=expected_qids, corpus_doc_ids=official_doc_ids)
    zip_val_res = validate_submission_zip(sub_zip)

    is_submission_valid = bool(val_res.get("is_valid") and zip_val_res.get("is_valid"))
    print(f"[+] Submission Validation: JSON = {val_res.get('is_valid')} | ZIP = {zip_val_res.get('is_valid')} | Overall = {is_submission_valid}")

    # 13. Manifest, Hashes, Reports & Parameter Audits
    git_sha = get_git_commit(root_path)
    manifest_path = submissions_dir / "submission_manifest.json"
    manifest = create_submission_manifest(
        submission_json_path=sub_json,
        submission_zip_path=sub_zip,
        output_path=manifest_path,
        git_commit=git_sha,
        parameter_total=audit_report.get("total_learned_parameters", 0),
        model_names_and_revisions=[
            {"name": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", "role": "dense_embedding"},
            {"name": "BAAI/bge-reranker-v2-m3", "role": "cross_encoder_reranker"},
        ],
        metadata={
            "run_mode": run_mode_str,
            "submission_status": "SUBMITTABLE_OFFICIAL" if is_full else f"NON_SUBMITTABLE_{run_mode_str.upper()}",
            "cv_mean_recall@5": cv_report.get("mean_recall@5", 0.0),
            "cv_mean_precision@5": cv_report.get("mean_precision@5", 0.0),
            "candidate_recall@150": cv_report.get("mean_candidate@150", 0.0),
            "fusion_winning_method": winning_fusion_method,
            "devices": {"dense": dense_device, "reranker": reranker_device},
            "submission_valid": is_submission_valid,
        },
    )

    # 13b. Hardware Placement & Peak VRAM Verification (Section 3)
    dense_actual_dev = "cpu"
    if hasattr(pipeline.hybrid_engine, "dense") and pipeline.hybrid_engine.dense is not None:
        dense_m = getattr(pipeline.hybrid_engine.dense, "model", None)
        if dense_m is not None:
            try:
                dense_actual_dev = str(next(dense_m.parameters()).device)
            except Exception:
                dense_actual_dev = str(getattr(pipeline.hybrid_engine.dense, "device", "cpu"))

    reranker_actual_dev = "cpu"
    if pipeline.reranker is not None:
        reranker_m = getattr(pipeline.reranker, "model", None)
        if reranker_m is not None:
            try:
                reranker_actual_dev = str(next(reranker_m.parameters()).device)
            except Exception:
                reranker_actual_dev = str(getattr(pipeline.reranker, "device", "cpu"))

    if is_gpu_smoke and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        if dense_device == "cuda:0" and not dense_actual_dev.startswith("cuda:0"):
            raise RuntimeError(f"Dense model device mismatch: requested {dense_device}, actual {dense_actual_dev}")
        if reranker_device == "cuda:1" and not reranker_actual_dev.startswith("cuda:1"):
            raise RuntimeError(f"Reranker model device mismatch: requested {reranker_device}, actual {reranker_actual_dev}")

    if is_gpu_smoke or is_full:
        gpu0_alloc = torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 0
        gpu1_alloc = torch.cuda.max_memory_allocated(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0
        gpu0_res = torch.cuda.max_memory_reserved(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 0
        gpu1_res = torch.cuda.max_memory_reserved(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0

        gpu_smoke_report = {
            "dense_requested": dense_device,
            "dense_actual": dense_actual_dev,
            "reranker_requested": reranker_device,
            "reranker_actual": reranker_actual_dev,
            "gpu0_peak_allocated_bytes": int(gpu0_alloc),
            "gpu1_peak_allocated_bytes": int(gpu1_alloc),
            "gpu0_peak_reserved_bytes": int(gpu0_res),
            "gpu1_peak_reserved_bytes": int(gpu1_res),
            "optimizer_steps": int(final_reranker_report.get("optimizer_steps", 0)),
            "param_diff": float(final_reranker_report.get("param_diff", 0.0) or 0.0),
            "adapter_checksum": str(final_reranker_report.get("adapter_checksum", "")),
            "strict_artifacts": bool(strict_artifacts),
            "fusion_crossfit_folds": int(oof_runner.num_folds),
            "oom": False,
        }
        report_path = working_path / "gpu_smoke_report.json"
        report_path.write_text(json.dumps(gpu_smoke_report, indent=2), encoding="utf-8")
        print(f"[+] Saved GPU smoke hardware report to {report_path}")

    # Copy files to /kaggle/working root if running in Kaggle environment and in full mode AFTER manifest creation
    if is_full and Path("/kaggle/working").exists() and is_submission_valid:
        try:
            import shutil
            shutil.copy2(sub_json, Path("/kaggle/working/submission.json"))
            shutil.copy2(sub_zip, Path("/kaggle/working/submission.zip"))
            shutil.copy2(manifest_path, Path("/kaggle/working/submission_manifest.json"))
            print("[+] Copied submission.json, submission.zip, and submission_manifest.json to /kaggle/working/ root.")
        except Exception as e:
            print(f"[-] Warning: Failed to copy submission to /kaggle/working: {e}")

    # Save ablation row
    ablation_csv = working_path / "ablation_report.csv"
    ablation_df = pd.DataFrame([
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_mode": run_mode_str,
            "mean_recall@5": cv_report.get("mean_recall@5", 0.0),
            "mean_precision@5": cv_report.get("mean_precision@5", 0.0),
            "mean_mrr": cv_report.get("mean_mrr", 0.0),
            "mean_ndcg@5": cv_report.get("mean_ndcg@5", 0.0),
            "candidate_recall@50": cv_report.get("mean_candidate@50", 0.0),
            "candidate_recall@150": cv_report.get("mean_candidate@150", 0.0),
            "candidate_recall@200": cv_report.get("mean_candidate@200", 0.0),
            "runtime_per_query_ms": cv_report.get("runtime_per_query_ms", 0.0),
            "fusion_winner": winning_fusion_method,
            "total_learned_parameters": audit_report.get("total_learned_parameters", 0),
            "public_predictions_count": len(predictions),
            "git_commit": git_sha,
        }
    ])
    ablation_df.to_csv(ablation_csv, index=False)
    print(f"[+] Saved ablation report to {ablation_csv}")

    fusion_winner_rec5 = fusion_report.get("winner_mean_recall@5", cv_report.get("mean_recall@5", 0.0))
    fusion_winner_prec5 = fusion_report.get("comparison", {}).get("winner_mean_precision@5", cv_report.get("mean_precision@5", 0.0))
    doc_disjoint_rec5 = (
        oof_runner.doc_disjoint_report.get("trained_reranker_system", {}).get("recall@5", 0.0)
        if hasattr(oof_runner, "doc_disjoint_report") and oof_runner.doc_disjoint_report
        else 0.0
    )

    total_runtime = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"LEGALIR TASK 1 PIPELINE RUN COMPLETE in {total_runtime:.2f}s")
    print(f"  - Validation Status                      : {'PASS' if is_submission_valid else 'FAIL'}")
    print(f"  - Submission JSON                        : {sub_json}")
    print(f"  - Submission ZIP                         : {sub_zip} ({sub_zip.stat().st_size:,} bytes)")
    print(f"  - Manifest JSON                          : {manifest_path}")
    print(f"  - Reranker OOF Recall@5                  : {cv_report.get('mean_recall@5', 0.0) * 100:.4f}%")
    print(f"  - Fusion Winner                          : {winning_fusion_method}")
    print(f"  - Fusion Winner Cross-Fitted Recall@5    : {fusion_winner_rec5 * 100:.4f}%")
    print(f"  - Fusion Winner Precision@5              : {fusion_winner_prec5 * 100:.4f}%")
    print(f"  - Candidate Recall@150                   : {cv_report.get('mean_candidate@150', 0.0) * 100:.4f}%")
    if doc_disjoint_rec5 > 0:
        print(f"  - Doc-Disjoint Trained Reranker Recall@5 : {doc_disjoint_rec5 * 100:.4f}%")
    print(f"  - Parameter Utilization                  : {audit_report.get('total_learned_parameters', 0):,} / 4,000,000,000 ({audit_report.get('budget_utilization_pct', 0.0):.2f}%)")
    print(f"  - Submission Status                      : {'SUBMITTABLE_OFFICIAL' if is_full else f'NON_SUBMITTABLE_{run_mode_str.upper()}'}")
    print("=" * 80)

    return KaggleRunResult(
        is_valid=is_submission_valid,
        submission_path=sub_json,
        submission_zip_path=sub_zip,
        manifest_path=manifest_path,
        audit_report=audit_report,
        cv_report=cv_report,
        fusion_report=fusion_report,
        public_predictions_count=len(predictions),
        run_mode=run_mode_str,
        execution_time_seconds=total_runtime,
        artifacts_dir=working_path,
        metadata={
            "git_commit": git_sha,
            "dense_device": dense_device,
            "reranker_device": reranker_device,
            "fusion_winner": winning_fusion_method,
        },
    )
