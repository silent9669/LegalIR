#!/usr/bin/env python3
"""
CLI script to build leak-free fold pair artifacts using static cache and lazy evidence.
Enforces:
- fold-local Question Memory (fitted only on train queries)
- zero validation query leakage in pairs or memory
- duplicate group closure blacklist
- real multi-band hard-negative mining matching legacy semantics
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.core.hashing import sha256_file
from src.core.manifests import Manifest
from src.core.memory import (
    check_memory_guard,
    format_memory_report,
    release_memory,
    take_memory_snapshot,
)
from src.data.canonical import resolve_duplicate_groups_path
from src.data.splits import load_5fold_splits
from src.evaluation.benchmark import build_memory_rows
from src.evidence.macro_store import MacroEvidenceStore
from src.evidence.pair_materializer import PairMaterializer
from src.retrieval.question_memory import QuestionMemory
from src.retrieval.static_cache import StaticCacheReader


def main():
    parser = argparse.ArgumentParser(description="Build leak-free training pair artifacts per fold.")
    parser.add_argument("--fold", type=str, default="0", help="Fold index (0-4) or 'all'")
    parser.add_argument("--dataset-dir", type=str, default="data/task1_canonical_v2", help="Path to dataset directory")
    parser.add_argument("--static-cache", type=str, default="artifacts/factory/static_cache/static_candidates_train.parquet", help="Path to static cache")
    parser.add_argument("--output-dir", type=str, default="artifacts/factory/folds", help="Output root for fold pairs")
    parser.add_argument("--negatives-per-positive", type=int, default=10, help="Hard negatives per positive document")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query limit for testing")
    args = parser.parse_args()

    dataset_p = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_cache_p = Path(args.static_cache)

    print(f"[*] Initializing fold pair builder for fold={args.fold}")
    snap = take_memory_snapshot()
    print(format_memory_report(snap, stage="Pair Builder Pre-init"))

    # Load canonical data
    chunks_p = dataset_p / "chunks.parquet"
    docs_p = dataset_p / "documents.parquet"
    queries_p = dataset_p / "queries_train.parquet"
    qrels_p = dataset_p / "qrels_train.parquet"

    for p in [chunks_p, queries_p, qrels_p]:
        if not p.is_file():
            print(f"[!] Required canonical file missing: {p}")
            sys.exit(1)

    # Resolve duplicate groups
    dup_p = resolve_duplicate_groups_path(dataset_p)
    if not dup_p or not dup_p.is_file():
        print(f"[!] Missing duplicate_groups.json in {dataset_p} or fallback locations")
        sys.exit(1)
    with open(dup_p, "r", encoding="utf-8") as f:
        dup_groups = json.load(f)

    # Load splits
    all_splits = load_5fold_splits(dataset_p)
    if args.fold.lower() == "all":
        folds_to_run = list(range(len(all_splits)))
    else:
        folds_to_run = [int(args.fold)]

    # Load queries dict
    queries_df = pd.read_parquet(queries_p)
    q_col = "question_norm" if "question_norm" in queries_df.columns else ("question_raw" if "question_raw" in queries_df.columns else "text")
    queries_dict = dict(zip(queries_df["query_id"].astype(str), queries_df[q_col].astype(str)))

    # Load qrels dict
    qrels_df = pd.read_parquet(qrels_p)
    qrels_dict: Dict[str, List[str]] = collections.defaultdict(list)
    for _, row in qrels_df.iterrows():
        qrels_dict[str(row["query_id"])].append(str(row["doc_id"]))

    # Load documents metadata
    doc_metadata = {}
    if docs_p.is_file():
        docs_df = pd.read_parquet(docs_p)
        doc_metadata = {str(r["doc_id"]): dict(r) for r in docs_df.to_dict("records")}

    # Initialize lazy evidence store
    print(f"[*] Initializing Arrow MacroEvidenceStore from {chunks_p} ...")
    evidence_store = MacroEvidenceStore(chunks_p)

    # Initialize static cache reader if present
    static_cache_reader = None
    if static_cache_p.is_file():
        print(f"[*] Loading StaticCacheReader from {static_cache_p} ...")
        static_cache_reader = StaticCacheReader(static_cache_p)
    else:
        print(f"[!] Warning: static cache {static_cache_p} not found. Pair builder will only use question memory.")

    for fold_idx in folds_to_run:
        fold_split = all_splits[fold_idx]
        train_qids = fold_split.train_qids
        val_qids = fold_split.val_qids

        if args.max_queries:
            train_qids = set(sorted(train_qids)[:args.max_queries])
            val_qids = set(sorted(val_qids)[:args.max_queries])

        print(f"\n==================== Building Fold {fold_idx} Artifacts ====================")
        print(f"Train queries: {len(train_qids)} | Val queries: {len(val_qids)}")

        # Build fold-local Question Memory strictly from train_qids
        print("[*] Fitting fold-local Question Memory strictly on train queries ...")
        memory_rows = build_memory_rows(train_qids, queries_dict, qrels_dict)
        question_memory = QuestionMemory(memory_rows, min_similarity=0.82)

        # Assert no validation queries in question memory
        mem_qids = set(map(str, getattr(question_memory, "query_ids", [])))
        leak = mem_qids.intersection(val_qids)
        if leak:
            raise ValueError(f"[!] Fatal: Memory leakage detected! {len(leak)} validation query IDs in Question Memory.")

        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_pairs_path = fold_dir / "train_pairs.parquet"
        reranker_pairs_path = fold_dir / "reranker_pairs.parquet"
        val_cands_path = fold_dir / "validation_candidates.parquet"

        materializer = PairMaterializer(
            train_qids=train_qids,
            val_qids=val_qids,
            qrels=qrels_dict,
            duplicate_groups=dup_groups,
            evidence_store=evidence_store,
            doc_metadata=doc_metadata,
            fold=fold_idx,
            question_memory=question_memory,
        )

        t0 = time.time()
        print(f"[*] Materializing train pairs to {train_pairs_path} ...")
        num_train_rows = materializer.materialize_train_pairs(
            output_parquet=train_pairs_path,
            queries_dict=queries_dict,
            static_cache_reader=static_cache_reader,
            negatives_per_positive=args.negatives_per_positive,
        )
        if num_train_rows == 0:
            print(f"[!] Error: zero train pairs materialized for fold {fold_idx}!")
            sys.exit(1)

        # Copy/symlink to reranker_pairs.parquet for legacy compatibility
        import shutil
        shutil.copy2(train_pairs_path, reranker_pairs_path)

        print(f"[*] Materializing validation candidates to {val_cands_path} ...")
        num_val_rows = materializer.materialize_validation_candidates(
            output_parquet=val_cands_path,
            queries_dict=queries_dict,
            static_cache_reader=static_cache_reader,
            candidate_k=150,
        )

        elapsed = time.time() - t0
        print(f"[+] Fold {fold_idx} complete in {elapsed:.1f}s: {num_train_rows} train pairs, {num_val_rows} val candidates.")

        # Write manifest
        manifest_data = {
            "fold": fold_idx,
            "status": "PASS",
            "train_queries_count": len(train_qids),
            "val_queries_count": len(val_qids),
            "train_pairs_count": num_train_rows,
            "validation_candidates_count": num_val_rows,
            "outputs": {
                "train_pairs.parquet": sha256_file(train_pairs_path),
                "reranker_pairs.parquet": sha256_file(reranker_pairs_path),
                "validation_candidates.parquet": sha256_file(val_cands_path),
            },
            "duration_seconds": elapsed,
        }

        with open(fold_dir / "pair_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        release_memory()

    print("\n[+] All requested fold pair artifacts built successfully.")


if __name__ == "__main__":
    main()
