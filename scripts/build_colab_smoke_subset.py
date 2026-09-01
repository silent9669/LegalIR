#!/usr/bin/env python3
"""
CLI to build a deterministic official-data smoke subset for Colab single-T4 verification.

Authoritative specification: LEGALIR_CI_COLAB_KAGGLE_ARCHITECTURE_SPEC.md
Implementation plan: LEGALIR_CI_COLAB_KAGGLE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.colab_smoke import ColabSmokeConfig, build_colab_subset

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "colab_smoke.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic official-data subset for Colab T4 contract smoke."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to official canonical v2 dataset")
    parser.add_argument("--out-dir", type=Path, required=True, help="Destination directory for subset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to colab_smoke.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed (default: from config)")
    parser.add_argument("--max-docs", type=int, default=None, help="Override max documents")
    parser.add_argument("--train-queries", type=int, default=None, help="Override train query count")

    args = parser.parse_args()

    if args.config.exists():
        cfg = ColabSmokeConfig.from_yaml(args.config)
    else:
        cfg = ColabSmokeConfig()

    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_docs is not None:
        cfg.max_documents = args.max_docs
    if args.train_queries is not None:
        cfg.train_queries = args.train_queries

    print("=================================================================")
    print("LegalIR Colab T4 Smoke Subset Builder")
    print(f"  • Source Data Dir: {args.data_dir}")
    print(f"  • Output Dir     : {args.out_dir}")
    print(f"  • Seed           : {cfg.seed}")
    print(f"  • Train Queries  : {cfg.train_queries}")
    print(f"  • Val Queries    : {cfg.validation_queries}")
    print(f"  • Public Queries : {cfg.public_queries}")
    print(f"  • Max Documents  : {cfg.max_documents}")
    print("=================================================================")

    try:
        manifest = build_colab_subset(args.data_dir, args.out_dir, cfg)
    except Exception as exc:
        print(f"[-] ERROR building smoke subset: {exc}", file=sys.stderr)
        return 1

    print("[+] Smoke subset generated successfully:")
    print(f"    - Documents : {manifest.documents_count}")
    print(f"    - Chunks    : {manifest.chunks_count}")
    print(f"    - Train Qs  : {manifest.train_queries_count}")
    print(f"    - Val Qs    : {manifest.validation_queries_count}")
    print(f"    - Public Qs : {manifest.public_queries_count}")
    print(f"    - Qrels     : {manifest.qrels_count}")
    print(f"    - Manifest  : {manifest.manifest_sha256}")
    print(f"    - Location  : {args.out_dir}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
