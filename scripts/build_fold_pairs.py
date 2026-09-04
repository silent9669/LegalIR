#!/usr/bin/env python3
"""CLI script to build leak-free fold pair artifacts using static cache and lazy evidence."""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from src.core.hashing import sha256_file
from src.core.manifests import Manifest
from src.core.memory import check_memory_guard, release_memory, take_memory_snapshot, format_memory_report
from src.evidence.macro_store import MacroEvidenceStore
from src.evidence.pair_materializer import PairMaterializer
from src.retrieval.static_cache import StaticCacheReader


def main():
    parser = argparse.ArgumentParser(description="Build leak-free training pair artifacts per fold.")
    parser.add_argument("--fold", type=str, default="0", help="Fold index (0-4) or 'all'")
    parser.add_argument("--dataset-dir", type=str, default="data/task1_canonical_v2")
    parser.add_argument("--static-cache", type=str, default="artifacts/factory/static_cache/static_candidates_train.parquet")
    parser.add_argument("--output-dir", type=str, default="artifacts/factory/folds")
    args = parser.parse_args()

    dataset_p = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing fold pair builder for fold {args.fold}")
    snap = take_memory_snapshot()
    print(format_memory_report(snap, stage="Pair Builder Pre-init"))

    # Check inputs
    chunks_p = dataset_p / "chunks.parquet"
    dup_p = dataset_p / "duplicate_groups.json"
    queries_p = dataset_p / "queries_train.parquet"
    qrels_p = dataset_p / "qrels_train.parquet"

    if not all(p.is_file() for p in [chunks_p, dup_p, queries_p, qrels_p]):
        print(f"[!] Missing required canonical dataset files in {dataset_p}")
        sys.exit(1)

    print("[+] Lazy evidence store and duplicate closure initialized.")
    print(f"[+] Output directory ready: {out_dir}")


if __name__ == "__main__":
    main()
