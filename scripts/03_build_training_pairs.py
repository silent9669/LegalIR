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
    parser.add_argument("--data-dir", type=str, default="artifacts/task1/data", help="Path to canonical data directory")
    parser.add_argument("--index-dir", type=str, default="artifacts/task1/indexes", help="Path to indexes directory")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--use-all-queries", action="store_true", help="Use all training queries without fold filtering")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for parquet pairs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of train queries (smoke mode)")
    parser.add_argument("--negatives", type=int, default=10, help="Negatives per positive")
    parser.add_argument("--max-chunks", type=int, default=3, help="Max evidence chunks per doc")
    args = parser.parse_args()

    out_dir = args.output_dir or f"artifacts/local/training/pairs/fold_{args.fold if not args.use_all_queries else 'all'}"

    print("=" * 60)
    print(f"LegalIR: Mining Training Pairs (Fold {args.fold}, use_all={args.use_all_queries})")
    print("=" * 60)

    retriever_df, reranker_df = build_training_pairs(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        output_dir=out_dir,
        fold=args.fold if not args.use_all_queries else None,
        use_all_queries=args.use_all_queries,
        limit=args.limit,
        negatives_per_positive=args.negatives,
        max_evidence_chunks=args.max_chunks,
    )
    print(f"Generated {len(retriever_df)} retriever pairs, {len(reranker_df)} reranker pairs.")


if __name__ == "__main__":
    main()
