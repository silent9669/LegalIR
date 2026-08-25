#!/bin/bash
set -e
echo "Building Canonical Dataset package v1..."
PYTHONPATH=. .venv/bin/python src/dataset/build_canonical.py \
  --raw_contexts_dir selected-contexts \
  --train_json train.json \
  --output_dir data/task1_canonical/v1

echo "Generating dual validation splits (Random 5-fold CV & Document-Disjoint)..."
PYTHONPATH=. .venv/bin/python src/evaluation/splits.py \
  --canonical_dir data/task1_canonical/v1
