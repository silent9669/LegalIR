#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_FILE="${1:-artifacts/shared/raw/public-official.json}"

cd "$ROOT_DIR"
echo "Running LegalIR inference on $INPUT_FILE..."
PYTHONPATH=. "$ROOT_DIR/.venv/bin/python" -m src.pipeline.run_all \
  --config configs/pipeline.yaml \
  --input "$INPUT_FILE" \
  --output-dir artifacts/task1/submissions \
  --offline

echo "Inference and submission packaging complete."
