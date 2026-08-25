#!/bin/bash
set -e
INPUT_FILE="${1:-public-official.json}"
OUTPUT_JSON="${2:-submission.json}"
OUTPUT_ZIP="${3:-submission.zip}"

echo "Running Inference on $INPUT_FILE..."
PYTHONPATH=. .venv/bin/python src/predict_submission.py \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_JSON" \
  --output_zip "$OUTPUT_ZIP" \
  --canonical_dir data/task1_canonical/v1 \
  --bm25_index indexes/bm25_micro_index.pkl
