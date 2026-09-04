#!/usr/bin/env python3
"""CLI script to build static label-free candidate caches for train and public queries."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.core.memory import check_memory_guard, release_memory, take_memory_snapshot, format_memory_report
from src.data.canonical import verify_canonical_dataset
from src.retrieval.static_cache import StaticCacheWriter, StaticCandidateRecord


def main():
    parser = argparse.ArgumentParser(description="Build static retrieval candidate cache.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/task1_canonical_v2",
        help="Path to canonical dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/factory/static_cache",
        help="Path to save static cache parquets",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=150,
        help="Number of candidates to cache per branch",
    )
    args = parser.parse_args()

    dataset_p = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting static cache builder for {dataset_p} -> {out_dir}")
    is_valid, ident, errors = verify_canonical_dataset(dataset_p)
    if not is_valid:
        print(f"[!] Dataset verification failed: {errors}")
        sys.exit(1)

    snap = take_memory_snapshot()
    print(format_memory_report(snap, stage="Pre-build Static Cache"))

    # Output targets
    train_cache_p = out_dir / "static_candidates_train.parquet"
    public_cache_p = out_dir / "static_candidates_public.parquet"

    print(f"[+] Static cache targets: {train_cache_p} and {public_cache_p}")
    print("[+] Static cache will be built without ground-truth labels (zero qrels accepted).")


if __name__ == "__main__":
    main()
