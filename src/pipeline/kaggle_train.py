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
import resource
import subprocess
import sys
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
class StageTimingEntry:
    """Telemetry entry for a pipeline execution stage."""
    seconds: float
    cache_hit: bool = False


class StageTimingTelemetry:
    """Collects structured performance and cache-hit telemetry across all pipeline stages."""

    def __init__(self) -> None:
        self._stages: dict[str, StageTimingEntry] = {}

    def record(self, stage_name: str, elapsed_seconds: float, cache_hit: bool = False) -> None:
        self._stages[stage_name] = StageTimingEntry(
            seconds=round(float(elapsed_seconds), 4),
            cache_hit=bool(cache_hit),
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"seconds": entry.seconds, "cache_hit": entry.cache_hit}
            for name, entry in self._stages.items()
        }


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


def get_process_rss_mb() -> float:
    """Return process resident set size in megabytes."""
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception:
        return 0.0


def get_peak_process_rss_mb() -> float:
    """Return true historical peak process resident set size in megabytes."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(usage / (1024.0 * 1024.0), 2)
        return round(usage / 1024.0, 2)
    except Exception:
        return 0.0


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
    """Discover canonical dataset or build from raw competition files if missing.

    Prioritizes official Kaggle clean dataset mounts (e.g. phucdangg/legalir-task1-clean-data)
    and recursively scans /kaggle/input when needed.
    """
    if data_dir is not None:
        p = Path(data_dir)
        if not p.exists():
            raise FileNotFoundError(f"Explicitly provided data_dir does not exist: {data_dir}")
        missing_files = [f for f in CANONICAL_REQUIRED_FILES if not (p / f).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Explicitly provided data_dir at {p} is missing required canonical files: {missing_files}"
            )
        return p.resolve()

    repo = repo_root or Path.cwd()
    candidate_paths = [
        # Preferred live clean dataset mounts
        Path("/kaggle/input/legalir-task1-clean-data"),
        Path("/kaggle/input/legalir-task1-clean-data/artifacts/task1/data"),
        Path("/kaggle/input/legalir-task1-clean-data/artifacts/shared/canonical/v2"),
        Path("/kaggle/input/datasets/phucdangg/legalir-task1-clean-data"),
        Path("/kaggle/input/datasets/phucdangg/legalir-task1-clean-data/artifacts/task1/data"),
        Path("/kaggle/input/datasets/phucdangg/legalir-task1-clean-data/artifacts/shared/canonical/v2"),
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

    # Recursive scan over /kaggle/input if present
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matching_dirs: list[Path] = []
        for dirpath in kaggle_input.rglob("*"):
            if dirpath.is_dir() and all((dirpath / f).exists() for f in CANONICAL_REQUIRED_FILES):
                matching_dirs.append(dirpath.resolve())

        if len(matching_dirs) == 1:
            return matching_dirs[0]
        elif len(matching_dirs) > 1:
            # Prioritize matching dirs containing preferred slug
            clean_matches = [d for d in matching_dirs if "legalir-task1-clean-data" in str(d)]
            if len(clean_matches) == 1:
                return clean_matches[0]
            legalir_matches = [d for d in matching_dirs if "legalir" in str(d)]
            if len(legalir_matches) == 1:
                return legalir_matches[0]
            raise ValueError(
                f"Ambiguous canonical dataset discovery in /kaggle/input; found multiple candidate directories: {matching_dirs}"
            )

    # Search for raw files to build on the fly if parquets are not precomputed
    raw_zip = None
    train_json = None
    for cand_zip in [
        Path("/kaggle/input/legalir-task1-clean-data/selected-contexts.zip"),
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
        Path("/kaggle/input/legalir-task1-clean-data/train.json"),
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
    data_dir: str | Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    """Discover public-official.json test file with robust relative and recursive fallback."""
    if public_json_path is not None:
        p = Path(public_json_path)
        if p.exists():
            return p.resolve()
        return None

    # Check relative to data_dir
    if data_dir is not None:
        dp = Path(data_dir)
        for cand in [
            dp / "public-official.json",
            dp.parent / "public-official.json",
            dp.parent.parent / "public-official.json",
            dp.parent / "raw/public-official.json",
            dp.parent / "shared/raw/public-official.json",
        ]:
            if cand.is_file():
                return cand.resolve()

    repo = repo_root or Path.cwd()
    preferred_paths = [
        Path("/kaggle/input/legalir-task1-clean-data/public-official.json"),
        Path("/kaggle/input/legalir-task1-clean-data/artifacts/raw/public-official.json"),
        Path("/kaggle/input/datasets/phucdangg/legalir-task1-clean-data/public-official.json"),
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
    ]
    for cand in preferred_paths:
        if cand.is_file():
            return cand.resolve()

    # Recursive scan in /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        found = list(kaggle_input.rglob("public-official.json"))
        if found:
            return found[0].resolve()
        found_pub = list(kaggle_input.rglob("public.json"))
        if found_pub:
            return found_pub[0].resolve()

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
    allow_nonstandard_production_devices: bool = False,
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
    VALID_RUN_MODES = {"smoke", "gpu_smoke", "full"}
    run_mode_str = str(run_mode).lower().strip()
    if run_mode_str not in VALID_RUN_MODES:
        raise ValueError(
            f"Invalid run_mode: '{run_mode}'. Must be one of {sorted(VALID_RUN_MODES)}"
        )
    is_smoke = run_mode_str == "smoke"
    is_gpu_smoke = run_mode_str == "gpu_smoke"
    is_full = run_mode_str == "full"

    stage_timings = StageTimingTelemetry()

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

    # 1b. Fail-Fast Early Canonical Dataset & Public Test Discovery
    canonical_data_dir = discover_data_dir(data_dir, repo_root=root_path)
    docs_path = canonical_data_dir / "documents.parquet"
    chunks_path = canonical_data_dir / "chunks.parquet"
    queries_train_path = canonical_data_dir / "queries_train.parquet"
    qrels_train_path = canonical_data_dir / "qrels_train.parquet"

    if not (docs_path.exists() and chunks_path.exists()):
        raise FileNotFoundError(
            f"Canonical dataset parquet files missing in {canonical_data_dir}"
        )

    public_test_file = discover_public_test_file(public_json_path, data_dir=canonical_data_dir, repo_root=root_path)
    if (is_full or is_gpu_smoke) and (public_test_file is None or not public_test_file.exists()):
        raise FileNotFoundError(f"{run_mode_str} mode requires official public-official.json; refusing to proceed")

    # 2. Hardware and GPU Device Allocation (P1.10)
    if (is_gpu_smoke or is_full) and not allow_nonstandard_production_devices:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{run_mode_str} mode requires CUDA but torch.cuda.is_available() is False")
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"{run_mode_str} mode requires Kaggle T4 x2 / >=2 CUDA devices (found {torch.cuda.device_count()} device(s))"
            )

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

    if (is_gpu_smoke or is_full) and not allow_nonstandard_production_devices:
        if dense_device != "cuda:0":
            raise RuntimeError(f"{run_mode_str} mode requires dense_device == 'cuda:0', got '{dense_device}'")
        if reranker_device != "cuda:1":
            raise RuntimeError(f"{run_mode_str} mode requires reranker_device == 'cuda:1', got '{reranker_device}'")

    print(f"[+] Device Allocation (P1.10 Multi-GPU Utilization):")
    print(f"    - Dense Embedding & Question Encoding : {dense_device}")
    print(f"    - Reranker Training & Neural Inference: {reranker_device}")

    # Set HF token in environment if provided securely
    if hf_token:
        os.environ["HF_TOKEN"] = str(hf_token)

    # 3. Canonical Dataset Loading & Validation (P1.5)
    print(f"[+] Canonical Data Directory: {canonical_data_dir}")

    t_load0 = time.perf_counter()
    df_docs = pd.read_parquet(docs_path)
    df_chunks = pd.read_parquet(chunks_path)
    df_queries = pd.read_parquet(queries_train_path) if queries_train_path.exists() else pd.DataFrame()
    df_qrels = pd.read_parquet(qrels_train_path) if qrels_train_path.exists() else pd.DataFrame()

    manifest_path = canonical_data_dir / "manifest.json"
    audit_report_path = canonical_data_dir / "audit_report.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    audit_data = json.loads(audit_report_path.read_text(encoding="utf-8")) if audit_report_path.exists() else {}

    if is_full or is_gpu_smoke:
        if len(df_docs) != 8532:
            raise ValueError(
                f"{run_mode_str.upper()} mode requires official Task 1 dataset with exactly 8,532 documents, got {len(df_docs)}"
            )
        if len(df_queries) != 7000:
            raise ValueError(
                f"{run_mode_str.upper()} mode requires official Task 1 dataset with exactly 7,000 training queries, got {len(df_queries)}"
            )
        if manifest_data and manifest_data.get("version") not in (None, "v2", "mock"):
            raise ValueError(
                f"{run_mode_str.upper()} mode requires v2 canonical dataset, got {manifest_data.get('version')}"
            )

    val_report = validate_canonical_dataset(
        canonical_data_dir,
        expected_document_count=8532 if (is_full or is_gpu_smoke) else None,
    )
    print(f"[+] Canonical Dataset Validation: is_valid = {val_report.get('is_valid')}")
    if not val_report.get("is_valid") and (is_full or is_gpu_smoke):
        raise ValueError(f"Canonical dataset validation failed in {run_mode_str.upper()} mode: {val_report.get('errors')}")

    canonical_load_time = max(0.001, time.perf_counter() - t_load0)
    stage_timings.record("canonical_load", elapsed_seconds=canonical_load_time, cache_hit=False)

    print(f"[+] Dataset Loaded: {len(df_docs):,} documents | {len(df_chunks):,} chunks | {len(df_queries):,} train queries")
    rss_after_load_mb = get_process_rss_mb()

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

    preflight_json_path = working_path / "preflight_parameter_audit.json"
    preflight_audit_report = audit_system_parameters(
        config_path=resolved_runtime_config if resolved_runtime_config.exists() else None,
        output_json=preflight_json_path,
        raise_on_violation=True,
        offline_fallback=True,
    )
    print(
        f"[+] Parameter Budget Preflight: {preflight_audit_report['total_learned_parameters']:,} params "
        f"({preflight_audit_report['total_parameters_billions']:.4f}B / 4.0B limit, "
        f"{preflight_audit_report['budget_utilization_pct']:.2f}% utilization). PASS"
    )

    # Validate explicit reranker training config
    from src.training.trainer import compute_coverage_required_steps

    reranker_cfg = yaml.safe_load(resolved_reranker_config.read_text(encoding="utf-8")) if resolved_reranker_config.exists() else {}
    if is_full:
        configured_max_steps = reranker_cfg.get("max_steps", 500)
        if configured_max_steps is None:
            raise ValueError("Full reranker training requires explicit max_steps")
        n_train_queries = len(df_queries) if not df_queries.empty else 7000
        coverage_req_steps = compute_coverage_required_steps(
            eligible_query_count=n_train_queries,
            batch_size=int(reranker_cfg.get("batch_size", 2)),
            gradient_accumulation_steps=int(reranker_cfg.get("gradient_accumulation_steps", 8)),
            target_coverage_pct=1.0,
            require_pos_and_neg=True,
        )
        effective_max_steps = max(int(configured_max_steps), coverage_req_steps)
        print(f"[+] Effective Reranker Config: model={reranker_cfg.get('base_model_name', 'BAAI/bge-reranker-v2-m3')}, "
              f"batch_size={reranker_cfg.get('batch_size', 2)}, grad_accum={reranker_cfg.get('gradient_accumulation_steps', 8)}, "
              f"max_length={reranker_cfg.get('max_length', 512)}, configured_steps={configured_max_steps}, "
              f"coverage_required_steps={coverage_req_steps}, effective_max_steps={effective_max_steps}, "
              f"fp16={reranker_cfg.get('fp16', True)}, device={reranker_device}")
    elif is_gpu_smoke:
        configured_max_steps = 3
        coverage_req_steps = 3
        effective_max_steps = 3
    else:
        configured_max_steps = 5
        coverage_req_steps = 5
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
    t_bm25_0 = time.perf_counter()
    bm25_cached = (bm25_dir / "bm25_micro_index.pkl").exists() or (bm25_dir.exists() and list(bm25_dir.glob("*.pkl")))
    if bm25_cached:
        print(f"[*] Loading cached Legal BM25 index from {bm25_dir}...")
        bm25_legal = BM25MicroRetriever.load(bm25_dir)
    else:
        print(f"[*] Building Legal BM25 index with metadata enrichment...")
        bm25_legal = BM25MicroRetriever(k1=1.5, b=0.75).fit(micro_chunks.to_dict("records"), show_progress=False)
        bm25_legal.save(bm25_dir)
    bm25_legal_time = max(0.001, time.perf_counter() - t_bm25_0)
    stage_timings.record("bm25_legal", elapsed_seconds=bm25_legal_time, cache_hit=bool(bm25_cached))
    print(f"[+] Legal BM25 ready ({len(bm25_legal.corpus):,} docs).")

    # 5b. PyVi Segmented BM25
    bm25_pyvi_dir = index_dir / "bm25_pyvi"
    t_pyvi_0 = time.perf_counter()
    bm25_pyvi_cached = (bm25_pyvi_dir / "bm25_pyvi_index.pkl").exists() or (bm25_pyvi_dir.exists() and list(bm25_pyvi_dir.glob("*.pkl")))
    if bm25_pyvi_cached:
        print(f"[*] Loading cached PyVi BM25 index from {bm25_pyvi_dir}...")
        bm25_pyvi = BM25PyViRetriever.load(bm25_pyvi_dir)
    else:
        print(f"[*] Building PyVi BM25 index with metadata enrichment...")
        bm25_pyvi = BM25PyViRetriever(k1=1.5, b=0.75).fit(micro_chunks.to_dict("records"), show_progress=False)
        bm25_pyvi.save(bm25_pyvi_dir)
    bm25_pyvi_time = max(0.001, time.perf_counter() - t_pyvi_0)
    stage_timings.record("bm25_pyvi", elapsed_seconds=bm25_pyvi_time, cache_hit=bool(bm25_pyvi_cached))
    print(f"[+] PyVi BM25 ready ({len(bm25_pyvi.corpus):,} docs).")

    # 6. Build / Load DEk21 Dense Macro Index (GPU 0)
    dense_dir = index_dir / "dense_dek21"
    dense_retriever: DenseMacroRetriever | None = None
    dense_build_time = 0.001
    dense_cached = (dense_dir / "embeddings.npy").exists()

    t_dense0 = time.perf_counter()
    if dense_cached:
        print(f"[*] Loading cached DEk21 Dense index from {dense_dir} on {dense_device}...")
        try:
            dense_retriever = DenseMacroRetriever.load(dense_dir, device=dense_device)
            dense_build_time = max(0.001, time.perf_counter() - t_dense0)
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
        dense_batch = 32 if "cuda" in str(dense_device) else 32
        try:
            dense_retriever.fit(macro_chunks.to_dict("records"), batch_size=dense_batch, stage_name="corpus")
            dense_retriever.save(dense_dir)
            dense_build_time = max(0.001, time.perf_counter() - t_dense0)
            print(f"[+] DEk21 Dense Index ready ({len(dense_retriever.doc_ids):,} chunks).")
        except Exception as e:
            if is_full or is_gpu_smoke:
                raise RuntimeError(f"Failed to build DEk21 Dense index on {dense_device} in {run_mode_str.upper()} mode: {e}") from e
            print(f"[-] Warning: Dense index building failed: {e}")
            dense_retriever = None
    else:
        dense_build_time = max(0.001, time.perf_counter() - t_dense0)

    stage_timings.record("dense_index", elapsed_seconds=dense_build_time, cache_hit=bool(dense_cached))
    rss_after_dense_mb = get_process_rss_mb()

    if is_full or is_gpu_smoke:
        if dense_retriever is None or getattr(dense_retriever, "_faiss_index", None) is None:
            raise RuntimeError(
                f"FAISS production backend enforcement failed in {run_mode_str.upper()} mode: "
                "dense_retriever._faiss_index is None (NumPy fallback is not allowed in production)."
            )

    # 7. Precompute Train Query Dense Embeddings on GPU 0 (P1.10)
    train_query_embs: dict[str, np.ndarray] = {}
    train_query_enc_time = 0.001
    tq_cached = (index_dir / "train_query_embeddings.npy").exists()
    t_tq0 = time.perf_counter()
    if not df_queries.empty and dense_retriever is not None:
        q_records = df_queries.to_dict("records")
        qids = [str(r["query_id"]) for r in q_records]
        if tq_cached:
            print(f"[*] Loading cached train query dense embeddings from {index_dir / 'train_query_embeddings.npy'}...")
            embs = np.load(str(index_dir / "train_query_embeddings.npy"))
            for qid, emb in zip(qids, embs):
                train_query_embs[qid] = emb
            train_query_enc_time = max(0.001, time.perf_counter() - t_tq0)
        else:
            qtexts = [
                str(r.get("question_norm") or r.get("question_raw") or r.get("question") or "")
                for r in q_records
            ]
            try:
                print(f"[*] Precomputing train query embeddings for {len(qids):,} queries on {dense_device}...")
                embs = dense_retriever.encode_queries(
                    qtexts, batch_size=128 if "cuda" in str(dense_device) else 32, stage_name="train_query"
                )
                for qid, emb in zip(qids, embs):
                    train_query_embs[qid] = emb
                np.save(str(index_dir / "train_query_embeddings.npy"), embs)
                train_query_enc_time = max(0.001, time.perf_counter() - t_tq0)
                print(f"[+] Precomputed and cached {len(train_query_embs):,} train query dense embeddings.")
            except Exception as e:
                train_query_enc_time = max(0.001, time.perf_counter() - t_tq0)
                if is_full or is_gpu_smoke:
                    raise RuntimeError(f"Failed to precompute train query embeddings on {dense_device} in {run_mode_str.upper()} mode: {e}") from e
                print(f"[-] Warning: query embedding precomputation skipped: {e}")
    else:
        train_query_enc_time = max(0.001, time.perf_counter() - t_tq0)

    stage_timings.record("train_query_encoding", elapsed_seconds=train_query_enc_time, cache_hit=bool(tq_cached))

    # 8. Out-of-Fold (OOF) 5-Fold Cross-Validation with Fold-Trained LoRA Rerankers (P1.1, P1.2, P1.7, P1.10)
    print("\n" + "=" * 70)
    print(f"[*] Starting 5-Fold OOF Cross-Validation (Mode: {run_mode_str.upper()})...")
    print("=" * 70)

    t_oof0 = time.perf_counter()
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
    oof_cv_time = max(0.001, time.perf_counter() - t_oof0)
    stage_timings.record("oof_cv", elapsed_seconds=oof_cv_time, cache_hit=False)

    oof_num_folds = int(oof_runner.num_folds)
    doc_disjoint_report = dict(getattr(oof_runner, "doc_disjoint_report", {}) or {})
    rss_after_oof_mb = get_process_rss_mb()
    del oof_runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 9. Cross-Fitted Learned Fusion vs Tuned RRF Evaluation & Final Model Training (P1.4)
    print("\n" + "=" * 70)
    print("[*] Evaluating Cross-Fitted Learned Fusion vs Tuned RRF (P1.4)...")
    print("=" * 70)

    t_fus0 = time.perf_counter()
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
    fusion_time = max(0.001, time.perf_counter() - t_fus0)
    stage_timings.record("fusion_training", elapsed_seconds=fusion_time, cache_hit=False)

    winning_fusion_method = fusion_report.get("winning_method", "reciprocal_rank_fusion")
    use_learned_fusion = bool(winning_fusion_method == "learned_ranker")

    # 10. Build Full 7,000-Query Memory & Train Final LoRA Reranker on All 7,000 Queries (P1.3, P1.8, P1.10, P1.11)
    print("\n" + "=" * 70)
    print(f"[*] Training Final System on All Training Queries (P1.3 / P1.8 / P1.11)...")
    print("=" * 70)

    # 10a. Full Question Memory
    t_qm0 = time.perf_counter()
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
    qm_time = max(0.001, time.perf_counter() - t_qm0)
    stage_timings.record("question_memory", elapsed_seconds=qm_time, cache_hit=False)

    # 10b. Mine hard-negative training pairs from ALL training queries
    t_pm0 = time.perf_counter()
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
    final_pair_mining_time = max(0.001, time.perf_counter() - t_pm0)
    stage_timings.record("final_pair_mining", elapsed_seconds=final_pair_mining_time, cache_hit=False)

    # Audit pair coverage before loading/training final BGE reranker
    from src.training.trainer import audit_pair_coverage
    if final_pairs_file.exists():
        final_pair_df = pd.read_parquet(final_pairs_file)
        expected_train_qids = set(df_queries["query_id"].astype(str)) if not df_queries.empty else None
        pair_coverage_audit = audit_pair_coverage(final_pair_df, expected_qids=expected_train_qids)
        print(f"[+] Final Training Pair Coverage Audit: {pair_coverage_audit.get('eligible_queries_count'):,} eligible queries | "
              f"Positive Coverage: {pair_coverage_audit.get('positive_coverage_pct')}% | "
              f"Negative Coverage: {pair_coverage_audit.get('negative_coverage_pct')}%")
        if is_full:
            if pair_coverage_audit["positive_coverage_pct"] < 100.0:
                raise RuntimeError(f"FULL mode requires 100% positive pair coverage, got {pair_coverage_audit['positive_coverage_pct']}%")
            if pair_coverage_audit["negative_coverage_pct"] < 99.0:
                raise RuntimeError(f"FULL mode requires >=99% negative pair coverage, got {pair_coverage_audit['negative_coverage_pct']}%")

    # 10c. Train Final LoRA Reranker on GPU 1
    final_reranker_dir = checkpoints_dir / "reranker_final"
    max_final_steps = effective_max_steps
    print(f"[*] Training Final Supervised LoRA Reranker on {reranker_device} (max_steps={max_final_steps})...")
    t_tr0 = time.perf_counter()
    final_reranker_report = train_reranker(
        pairs_file=final_pairs_file,
        config_path=resolved_reranker_config,
        output_dir=final_reranker_dir,
        fold=None,
        max_steps=max_final_steps,
        base_model_name="mock" if is_smoke else "BAAI/bge-reranker-v2-m3",
        device=reranker_device,
        enforce_full_coverage_steps=is_full,
    )
    final_training_time = max(0.001, time.perf_counter() - t_tr0)
    stage_timings.record("final_reranker_training", elapsed_seconds=final_training_time, cache_hit=False)
    print(f"[+] Final Reranker Training Status: {final_reranker_report.get('status')} | Checkpoint: {final_reranker_dir}")

    if is_full:
        min_coverage = 99.0
        uniq_cov = final_reranker_report.get("unique_query_coverage_pct", 0.0)
        pos_cov = final_reranker_report.get("positive_query_coverage_pct", 0.0)
        neg_cov = final_reranker_report.get("negative_query_coverage_pct", 0.0)
        actual_cov = final_reranker_report.get("actual_query_coverage_pct", 0.0)

        if uniq_cov < min_coverage:
            raise RuntimeError(
                f"FULL mode requires unique_query_coverage_pct >= {min_coverage}%, got {uniq_cov}% "
                f"({final_reranker_report.get('actual_unique_queries_seen')}/{final_reranker_report.get('eligible_training_queries')} queries seen)."
            )
        if pos_cov < min_coverage:
            raise RuntimeError(
                f"FULL mode requires positive_query_coverage_pct >= {min_coverage}%, got {pos_cov}% "
                f"({final_reranker_report.get('positive_queries_seen')}/{final_reranker_report.get('eligible_training_queries')} queries with positive seen)."
            )
        if neg_cov < min_coverage:
            raise RuntimeError(
                f"FULL mode requires negative_query_coverage_pct >= {min_coverage}%, got {neg_cov}% "
                f"({final_reranker_report.get('queries_with_negative_seen')}/{final_reranker_report.get('eligible_training_queries')} queries with negative seen)."
            )
        if actual_cov < min_coverage:
            raise RuntimeError(
                f"FULL mode requires actual_query_coverage_pct >= {min_coverage}%, got {actual_cov}%."
            )

    # 11. Public Test Inference with Fully Loaded Final Production Pipeline (P0.3, P0.4, P1.8)
    print("\n" + "=" * 70)
    print("[*] Executing Public Test Batch Inference with Final Production Pipeline (P1.8)...")
    print("=" * 70)

    public_test_file = discover_public_test_file(public_json_path, repo_root=root_path)
    if is_full or is_gpu_smoke:
        if public_test_file is None or not public_test_file.exists():
            raise FileNotFoundError(f"{run_mode_str.upper()} mode requires official public-official.json; refusing to generate submission")
        with open(public_test_file, "r", encoding="utf-8") as f:
            public_data = json.load(f)
        if len(public_data) != 999:
            raise ValueError(
                f"{run_mode_str.upper()} mode requires official public-official.json with exactly 999 queries, got {len(public_data)}"
            )
    elif public_test_file and public_test_file.exists():
        print(f"[+] Found Public Test Queries: {public_test_file}")
        with open(public_test_file, "r", encoding="utf-8") as f:
            public_data = json.load(f)
    else:
        print("[!] public-official.json not found. Using train queries sample for inference verification.")
        sample_records = df_queries.head(20).to_dict("records")
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

    t_pipe0 = time.perf_counter()
    final_audit_json = working_path / "parameter_audit.json"
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
        audit_output_json=final_audit_json,
        reranker_model_name="mock" if is_smoke else "BAAI/bge-reranker-v2-m3",
    )
    final_audit_report = pipeline.audit_parameters(
        output_json=final_audit_json,
        raise_on_violation=True,
        require_loaded_models=(is_full or is_gpu_smoke),
    )
    pipe_load_audit_time = max(0.001, time.perf_counter() - t_pipe0)
    stage_timings.record("final_pipeline_load_audit", elapsed_seconds=pipe_load_audit_time, cache_hit=False)

    # Strict final PEFT adapter audit verification
    if is_full or is_gpu_smoke:
        reranker_audit = next((m for m in final_audit_report.get("models", {}).values() if m.get("role") == "cross_encoder_reranker"), {})
        if reranker_adapter_load_path is not None:
            if not reranker_audit.get("is_peft_lora", False):
                raise RuntimeError("Strict runtime audit failed: loaded reranker is not marked as is_peft_lora")
            if reranker_audit.get("adapter_parameters", 0) <= 0:
                raise RuntimeError(f"Strict runtime audit failed: reranker adapter_parameters must be > 0, got {reranker_audit.get('adapter_parameters')}")
            if final_audit_report.get("total_learned_parameters", 0) <= 702_754_049:
                raise RuntimeError("Strict runtime audit failed: total parameters did not increase after adapter loading")

    t0_infer = time.perf_counter()
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
                q_texts, batch_size=32 if is_full else 16, stage_name="public_query"
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
            elapsed = time.perf_counter() - t0_infer
            print(f"    [{idx:4d}/{len(q_items):4d}] queries predicted ({idx / elapsed:.2f} q/s)")

    public_inference_time = max(0.001, time.perf_counter() - t0_infer)
    stage_timings.record("public_inference", elapsed_seconds=public_inference_time, cache_hit=False)
    print(f"[+] Public inference completed in {public_inference_time:.2f}s ({len(predictions)} queries predicted).")
    rss_peak_mb = get_peak_process_rss_mb()

    # 12. Strict Submission Invariant Validation & Zip Packaging (P0.5, Invariants 1-24)
    t_pkg0 = time.perf_counter()
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
    pkg_time = max(0.001, time.perf_counter() - t_pkg0)
    stage_timings.record("submission_packaging", elapsed_seconds=pkg_time, cache_hit=False)
    print(f"[+] Submission Validation: JSON = {val_res.get('is_valid')} | ZIP = {zip_val_res.get('is_valid')} | Overall = {is_submission_valid}")
    if is_full and not is_submission_valid:
        raise RuntimeError(
            f"Final official submission failed validation: JSON errors={val_res.get('errors')} | ZIP errors={zip_val_res.get('errors')}"
        )

    # 13. Manifest, Hashes, Reports & Parameter Audits
    git_sha = get_git_commit(root_path)
    manifest_path = submissions_dir / "submission_manifest.json"
    manifest = create_submission_manifest(
        submission_json_path=sub_json,
        submission_zip_path=sub_zip,
        output_path=manifest_path,
        git_commit=git_sha,
        parameter_total=final_audit_report.get("total_learned_parameters", 0),
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
            "stage_timings": stage_timings.to_dict(),
            "peak_rss_mb": rss_peak_mb,
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

    if (is_gpu_smoke or is_full) and not allow_nonstandard_production_devices and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        if dense_device == "cuda:0" and not dense_actual_dev.startswith("cuda:0"):
            raise RuntimeError(f"Dense model device mismatch in {run_mode_str.upper()}: requested {dense_device}, actual {dense_actual_dev}")
        if reranker_device == "cuda:1" and not reranker_actual_dev.startswith("cuda:1"):
            raise RuntimeError(f"Reranker model device mismatch in {run_mode_str.upper()}: requested {reranker_device}, actual {reranker_actual_dev}")

    if is_gpu_smoke or is_full:
        gpu0_alloc = torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 0
        gpu1_alloc = torch.cuda.max_memory_allocated(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0
        gpu0_res = torch.cuda.max_memory_reserved(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 0
        gpu1_res = torch.cuda.max_memory_reserved(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0

        oof_reranker_oom = sum(f.get("reranker_oom_events", 0) for f in cv_report.get("folds", []))
        final_reranker_oom = int(getattr(pipeline.reranker, "oom_events", 0)) if pipeline.reranker is not None else 0

        stage_telem = getattr(dense_retriever, "stage_telemetry", None)
        corpus_telem = stage_telem.get("corpus") if isinstance(stage_telem, dict) else None
        train_q_telem = stage_telem.get("train_query") if isinstance(stage_telem, dict) else None

        pipeline_dense = getattr(pipeline.hybrid_engine, "dense_retriever", None) or getattr(pipeline.hybrid_engine, "dense", None)
        pipe_stage_telem = getattr(pipeline_dense, "stage_telemetry", None)
        public_q_telem = pipe_stage_telem.get("public_query") if isinstance(pipe_stage_telem, dict) else None

        if corpus_telem is not None:
            dense_corpus_oom = int(getattr(corpus_telem, "oom_events", 0) or 0)
        else:
            dense_corpus_oom = int(getattr(dense_retriever, "dense_oom_events", 0) or 0)

        if train_q_telem is not None:
            dense_train_q_oom = int(getattr(train_q_telem, "oom_events", 0) or 0)
        else:
            dense_train_q_oom = 0

        if public_q_telem is not None:
            dense_public_q_oom = int(getattr(public_q_telem, "oom_events", 0) or 0)
        elif pipeline_dense is not None and pipeline_dense is not dense_retriever:
            dense_public_q_oom = int(getattr(pipeline_dense, "dense_oom_events", 0) or 0)
        else:
            dense_public_q_oom = 0

        dense_oom = dense_corpus_oom + dense_train_q_oom + dense_public_q_oom
        total_reranker_oom = int(oof_reranker_oom or 0) + int(final_reranker_oom or 0)
        total_oom_events = total_reranker_oom + dense_oom

        min_batch = int(getattr(pipeline.reranker, "min_successful_batch_size", 16) or 16) if pipeline.reranker is not None else 16
        reranker_audit_entry = next((m for m in final_audit_report.get("models", {}).values() if m.get("role") == "cross_encoder_reranker"), {})

        gpu_smoke_report = {
            "dense_requested": dense_device,
            "dense_actual": dense_actual_dev,
            "reranker_requested": reranker_device,
            "reranker_actual": reranker_actual_dev,
            "dense_search_backend": "faiss_index_flat_ip" if getattr(dense_retriever, "_faiss_index", None) is not None else "numpy",
            "gpu0_peak_allocated_bytes": int(gpu0_alloc),
            "gpu1_peak_allocated_bytes": int(gpu1_alloc),
            "gpu0_peak_reserved_bytes": int(gpu0_res),
            "gpu1_peak_reserved_bytes": int(gpu1_res),
            "dense_corpus": {
                "requested_batch_size": int(getattr(corpus_telem, "requested_batch_size", getattr(dense_retriever, "dense_initial_batch_size", 32)) or 32),
                "min_successful_batch_size": int(getattr(corpus_telem, "min_successful_batch_size", getattr(dense_retriever, "dense_min_successful_batch_size", 32)) or 32),
                "oom_events": dense_corpus_oom,
                "item_count": int(getattr(corpus_telem, "item_count", len(df_chunks)) or len(df_chunks)),
                "elapsed_seconds": round(float(getattr(corpus_telem, "elapsed_seconds", dense_build_time) or dense_build_time), 2),
            },
            "dense_train_query": {
                "requested_batch_size": int(getattr(train_q_telem, "requested_batch_size", 128) or 128),
                "min_successful_batch_size": int(getattr(train_q_telem, "min_successful_batch_size", 128) or 128),
                "oom_events": dense_train_q_oom,
                "item_count": int(getattr(train_q_telem, "item_count", len(df_queries)) or len(df_queries)),
                "elapsed_seconds": round(float(getattr(train_q_telem, "elapsed_seconds", train_query_enc_time) or train_query_enc_time), 2),
            },
            "dense_public_query": {
                "requested_batch_size": int(getattr(public_q_telem, "requested_batch_size", 32) or 32),
                "min_successful_batch_size": int(getattr(public_q_telem, "min_successful_batch_size", 32) or 32),
                "oom_events": dense_public_q_oom,
                "item_count": int(getattr(public_q_telem, "item_count", len(public_data)) or len(public_data)),
                "elapsed_seconds": round(float(getattr(public_q_telem, "elapsed_seconds", public_inference_time) or public_inference_time), 2),
            },
            "dense_total_oom_events": dense_oom,
            "dense_oom_events": dense_oom,
            "oof_reranker_oom_events": oof_reranker_oom,
            "final_reranker_oom_events": final_reranker_oom,
            "total_reranker_oom_events": total_reranker_oom,
            "total_oom_events": total_oom_events,
            "oom": bool(total_oom_events > 0),
            "stable_reranker_batch_size": min_batch,
            "optimizer_steps": int(final_reranker_report.get("optimizer_steps", 0)),
            "param_diff": float(final_reranker_report.get("param_diff", 0.0) or 0.0),
            "actual_unique_queries_seen": int(final_reranker_report.get("actual_unique_queries_seen", 0)),
            "actual_query_coverage_pct": float(final_reranker_report.get("actual_query_coverage_pct", 0.0)),
            "adapter_checksum": str(final_reranker_report.get("adapter_checksum", "")),
            "adapter_parameters": int(reranker_audit_entry.get("adapter_parameters", 0)),
            "is_peft_lora": bool(reranker_audit_entry.get("is_peft_lora", False)),
            "host_rss_mb": {
                "after_canonical_load": rss_after_load_mb,
                "after_dense_index": rss_after_dense_mb,
                "after_oof": rss_after_oof_mb,
                "peak": rss_peak_mb,
            },
            "stage_timings": stage_timings.to_dict(),
            "strict_artifacts": bool(strict_artifacts),
            "fusion_crossfit_folds": int(oof_num_folds),
        }
        report_path = working_path / "gpu_smoke_report.json"
        report_path.write_text(json.dumps(gpu_smoke_report, indent=2), encoding="utf-8")
        print(f"[+] Saved GPU smoke hardware report to {report_path}")

        # Runtime projection calculation
        q_infer_rate = len(q_items) / max(0.001, public_inference_time)
        total_public_count = float(len(public_data))
        projected_public_infer_sec = total_public_count / max(0.1, q_infer_rate)

        # OOF pure inference throughput
        heldout_qps = float(cv_report.get("heldout_inference_queries_per_second", cv_report.get("queries_per_second", 1.0)))
        total_oof_queries = float(len(df_queries)) if not df_queries.empty else 7000.0
        projected_oof_inference_sec = total_oof_queries / max(0.1, heldout_qps)

        # Training rate per optimizer step
        measured_final_steps = int(final_reranker_report.get("optimizer_steps", final_reranker_report.get("global_steps", 1)))
        final_training_sec = float(final_reranker_report.get("training_time_sec", 10.0))
        sec_per_final_step = final_training_sec / max(1, measured_final_steps)

        fold_opt_steps_total = int(cv_report.get("reranker_optimizer_steps_total", 0))
        fold_train_sec_total = float(cv_report.get("reranker_training_seconds_total", 0.0))
        sec_per_fold_step = (fold_train_sec_total / max(1, fold_opt_steps_total)) if fold_opt_steps_total > 0 else sec_per_final_step

        sec_per_train_step = sec_per_final_step if sec_per_final_step > 0 else (sec_per_fold_step if sec_per_fold_step > 0 else 0.5)

        # Production step counts (scale smoke steps to full production coverage requirements)
        full_final_steps = int(effective_max_steps if is_full else 875)
        full_fold_steps = int(effective_max_steps if is_full else 875)

        projected_final_training_sec = full_final_steps * sec_per_train_step
        avg_pair_mining_sec = float(cv_report.get("pair_mining_seconds_total", 5.0)) / max(1, len(cv_report.get("folds", [])))
        projected_5fold_training_sec = 5.0 * (full_fold_steps * sec_per_train_step + avg_pair_mining_sec)
        projected_5fold_oof_sec = projected_oof_inference_sec + projected_5fold_training_sec

        # Document disjoint projection
        dj_report = doc_disjoint_report
        dj_mining_sec = float(dj_report.get("doc_disjoint_pair_mining_seconds", 5.0))
        dj_training_sec = full_fold_steps * sec_per_train_step
        dj_infer_sec = float(dj_report.get("doc_disjoint_inference_seconds", 5.0))
        projected_doc_disjoint_sec = dj_mining_sec + dj_training_sec + dj_infer_sec

        # Setup and overhead: explicit sum of all cold-start setup stages
        cold_start_setup_sec = round(
            bm25_legal_time
            + bm25_pyvi_time
            + dense_build_time
            + train_query_enc_time
            + canonical_load_time
            + pipe_load_audit_time
            + pkg_time
            + 60.0,
            2,
        )
        warm_cache_setup_sec = 60.0

        cold_start_total_sec = (
            projected_5fold_oof_sec
            + projected_final_training_sec
            + projected_doc_disjoint_sec
            + projected_public_infer_sec
            + cold_start_setup_sec
        )
        warm_cache_total_sec = (
            projected_5fold_oof_sec
            + projected_final_training_sec
            + projected_doc_disjoint_sec
            + projected_public_infer_sec
            + warm_cache_setup_sec
        )

        KAGGLE_MAX_SECONDS = 12 * 3600.0  # 43,200s
        SAFETY_FACTOR = 0.90
        PRODUCTION_RUNTIME_BUDGET = KAGGLE_MAX_SECONDS * SAFETY_FACTOR  # 38,880s (~10.8 hours)

        runtime_proj = {
            "public_queries_per_second": round(q_infer_rate, 2),
            "oof_pure_inference_qps": round(heldout_qps, 2),
            "sec_per_optimizer_step": round(sec_per_train_step, 4),
            "measured_gpu_smoke_fold_steps": fold_opt_steps_total // max(1, len(cv_report.get("folds", []))),
            "projected_full_fold_steps": full_fold_steps,
            "measured_final_steps": measured_final_steps,
            "projected_final_steps": full_final_steps,
            "projected_num_folds": 5,
            "total_oof_validation_queries": int(total_oof_queries),
            "public_queries": int(total_public_count),
            "includes_doc_disjoint": True,
            "includes_dense_build": True,
            "dense_corpus_build_seconds": round(dense_build_time, 2),
            "train_query_encoding_seconds": round(train_query_enc_time, 2),
            "projected_oof_inference_seconds": round(projected_oof_inference_sec, 2),
            "projected_5fold_training_seconds": round(projected_5fold_training_sec, 2),
            "projected_5fold_oof_seconds": round(projected_5fold_oof_sec, 2),
            "projected_5fold_oof_hours": round(projected_5fold_oof_sec / 3600.0, 3),
            "projected_final_training_seconds": round(projected_final_training_sec, 2),
            "projected_doc_disjoint_seconds": round(projected_doc_disjoint_sec, 2),
            "projected_public_inference_seconds": round(projected_public_infer_sec, 2),
            "cold_start_total_seconds": round(cold_start_total_sec, 2),
            "cold_start_total_hours": round(cold_start_total_sec / 3600.0, 3),
            "warm_cache_total_seconds": round(warm_cache_total_sec, 2),
            "warm_cache_total_hours": round(warm_cache_total_sec / 3600.0, 3),
            "production_runtime_budget_seconds": round(PRODUCTION_RUNTIME_BUDGET, 2),
            "production_runtime_budget_hours": round(PRODUCTION_RUNTIME_BUDGET / 3600.0, 2),
            "fits_kaggle_session_limit": bool(cold_start_total_sec < PRODUCTION_RUNTIME_BUDGET),
        }
        proj_path = working_path / "runtime_projection.json"
        proj_path.write_text(json.dumps(runtime_proj, indent=2), encoding="utf-8")
        print(f"[+] Saved runtime projection to {proj_path}")

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
            "total_learned_parameters": final_audit_report.get("total_learned_parameters", 0),
            "public_predictions_count": len(predictions),
            "git_commit": git_sha,
        }
    ])
    ablation_df.to_csv(ablation_csv, index=False)
    print(f"[+] Saved ablation report to {ablation_csv}")

    fusion_winner_rec5 = fusion_report.get("winner_mean_recall@5", cv_report.get("mean_recall@5", 0.0))
    fusion_winner_prec5 = fusion_report.get("comparison", {}).get("winner_mean_precision@5", cv_report.get("mean_precision@5", 0.0))
    doc_disjoint_rec5 = (
        doc_disjoint_report.get("trained_reranker_system", {}).get("recall@5", 0.0)
        if doc_disjoint_report
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
    print(f"  - Parameter Utilization                  : {final_audit_report.get('total_learned_parameters', 0):,} / 4,000,000,000 ({final_audit_report.get('budget_utilization_pct', 0.0):.2f}%)")
    print(f"  - Submission Status                      : {'SUBMITTABLE_OFFICIAL' if is_full else f'NON_SUBMITTABLE_{run_mode_str.upper()}'}")
    print("=" * 80)

    return KaggleRunResult(
        is_valid=is_submission_valid,
        submission_path=sub_json,
        submission_zip_path=sub_zip,
        manifest_path=manifest_path,
        audit_report=final_audit_report,
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
