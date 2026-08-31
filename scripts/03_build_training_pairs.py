"""Build fold-safe hard-negative and positive localized training pairs for LegalIR."""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.build_pairs import build_training_pairs


def main():
    parser = argparse.ArgumentParser(description="LegalIR Training Pairs Generator")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml", help="Path to pipeline.yaml")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for parquet pairs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of train queries (smoke mode)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"LegalIR: Mining Training Pairs for Fold {args.fold}")
    print("=" * 60)

    retriever_df, reranker_df = build_training_pairs(
        config_path=args.config,
        fold=args.fold,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(f"Generated {len(retriever_df)} retriever pairs, {len(reranker_df)} reranker pairs.")


if __name__ == "__main__":
    main()
