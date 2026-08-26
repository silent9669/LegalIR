#!/bin/bash
set -e
echo "Running DSC 2026 Task 1 Official Strict Benchmark..."
PYTHONPATH=. .venv/bin/python -m src.evaluation.benchmark \
  --config configs/pipeline.yaml \
  --label strict_baseline
