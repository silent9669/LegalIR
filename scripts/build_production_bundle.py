#!/usr/bin/env python3
"""CLI script to assemble and freeze the immutable production bundle."""

import argparse
import sys
from pathlib import Path

from src.bundle.builder import ProductionBundleBuilder
from src.bundle.verifier import verify_production_bundle


def main():
    parser = argparse.ArgumentParser(description="Build production bundle.")
    parser.add_argument("--bundle-dir", type=str, default="artifacts/bundle/production", help="Output bundle directory")
    parser.add_argument("--runtime-commit", type=str, default="a0efb25", help="Approved runtime git commit SHA")
    parser.add_argument("--dataset-fingerprint", type=str, default="canonical_v2_fingerprint", help="Dataset SHA256")
    parser.add_argument("--final-pairs", type=str, default="artifacts/factory/final_training_pairs.parquet")
    parser.add_argument("--public-candidates", type=str, default="artifacts/factory/static_cache/static_candidates_public.parquet")
    parser.add_argument("--production-lock", type=str, default="artifacts/bundle/production_lock.json")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    print(f"[*] Building production bundle at {bundle_dir} ...")

    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit=args.runtime_commit,
        dataset_fingerprint=args.dataset_fingerprint,
    )

    pairs_p = Path(args.final_pairs)
    if pairs_p.is_file():
        builder.add_file("final_training_pairs.parquet", pairs_p)

    cands_p = Path(args.public_candidates)
    if cands_p.is_file():
        builder.add_file("public_candidates.parquet", cands_p)

    lock_p = Path(args.production_lock)
    if lock_p.is_file():
        builder.add_file("production_lock.json", lock_p)

    manifest = builder.freeze()
    print(f"[+] Bundle frozen with {len(manifest.files)} files. Verifying ...")

    is_valid, errors = verify_production_bundle(bundle_dir)
    if not is_valid:
        print(f"[!] Bundle verification FAILED: {errors}")
        sys.exit(1)

    print("[+] Production bundle successfully built and verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
