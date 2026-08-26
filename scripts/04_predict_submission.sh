#!/bin/bash
set -e
INPUT_FILE="${1:-artifacts/shared/raw/public-official.json}"

echo "Running LegalIR Inference on $INPUT_FILE..."
PYTHONPATH=. python3 -m src.pipeline.run_all \
  --config configs/pipeline.yaml \
  --input "$INPUT_FILE" \
  --offline

echo "Inference and submission packaging complete!"
