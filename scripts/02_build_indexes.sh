#!/bin/bash
set -e
echo "Building BM25 Micro Index..."
PYTHONPATH=. .venv/bin/python src/retrieval/build_indexes.py \
  --canonical_dir data/task1_canonical/v1 \
  --output_dir indexes
