#!/bin/bash
set -e
echo "Building Canonical Dataset package v2..."
PYTHONPATH=. python3 -m src.dataset.build_canonical \
  --config configs/pipeline.yaml

echo "Canonical dataset v2 build and validation complete!"
