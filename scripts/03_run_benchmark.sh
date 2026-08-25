#!/bin/bash
set -e
echo "Running DSC 2026 Task 1 Official Dual-Validation Benchmark..."
PYTHONPATH=. .venv/bin/python src/validate_all.py \
  --canonical_dir data/task1_canonical/v1 \
  --bm25_index indexes/bm25_micro_index.pkl \
  --num_folds 1
