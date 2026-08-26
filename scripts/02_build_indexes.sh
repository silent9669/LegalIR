#!/bin/bash
set -e
echo "Building BM25 Micro Index..."
PYTHONPATH=. python3 -m src.retrieval.build_indexes \
  --config configs/pipeline.yaml \
  --bm25

echo "BM25 index build complete!"
