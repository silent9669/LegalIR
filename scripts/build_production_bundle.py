#!/usr/bin/env python3
"""CLI script to assemble and freeze the immutable production bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bundle.builder import ProductionBundleBuilder, MANDATORY_BUNDLE_FILES
from src.bundle.verifier import verify_production_bundle
from src.core.hashing import sha256_file


def main():
    parser = argparse.ArgumentParser(description="Build production bundle.")
    parser.add_argument("--bundle-dir", type=str, default="artifacts/bundle/production", help="Output bundle directory")
    parser.add_argument("--runtime-commit", type=str, required=True, help="Real 40-char git commit SHA")
    parser.add_argument("--dataset-fingerprint", type=str, required=True, help="Real 64-char dataset SHA256")
    parser.add_argument("--config-sha256", type=str, required=True, help="Real 64-char production config SHA256")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/factory", help="Factory artifacts root")
    parser.add_argument("--no-strict", action="store_true", help="Allow non-mandatory files during testing")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    artifacts_root = Path(args.artifacts_dir)
    print(f"[*] Building production bundle at {bundle_dir} ...")

    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit=args.runtime_commit,
        dataset_fingerprint=args.dataset_fingerprint,
        config_sha256=args.config_sha256,
        strict_mandatory_check=not args.no_strict,
    )

    # Standard locations
    file_candidates = {
        "final_training_pairs.parquet": artifacts_root / "final_training_pairs.parquet",
        "public_candidates.parquet": artifacts_root / "static_cache" / "static_candidates_public.parquet",
        "public_evidence.parquet": artifacts_root / "evidence" / "public_evidence.parquet",
        "production_lock.json": artifacts_root / "production_lock.json",
        "fusion_model.json": artifacts_root / "fusion" / "fusion_model.json",
        "static_cache_provenance.json": artifacts_root / "static_cache" / "static_cache_provenance.json",
        "validation_summary.json": artifacts_root / "validation_summary.json",
        "dataset_provenance.json": artifacts_root / "dataset_provenance.json",
    }

    for name, path in file_candidates.items():
        if path.is_file():
            builder.add_file(name, path)
        elif not args.no_strict:
            print(f"[!] Warning: required file {name} not found at {path}")

    manifest = builder.freeze()
    print(f"[+] Bundle frozen with {len(manifest.files)} files. Verifying ...")

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=not args.no_strict)
    if not is_valid:
        print(f"[!] Bundle verification FAILED:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    print("[+] Production bundle successfully built and verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
