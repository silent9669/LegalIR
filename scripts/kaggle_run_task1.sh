#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KERNEL_DIR="kaggle_kernel_task1"
KERNEL_ID="phucdangg/legalir-training"
OUTPUT_DIR="artifacts/task1/submissions/kaggle"

# Locate Kaggle CLI
if command -v kaggle &>/dev/null; then
    KAGGLE_CMD="kaggle"
elif [ -f "$ROOT_DIR/.venv/bin/kaggle" ]; then
    KAGGLE_CMD="$ROOT_DIR/.venv/bin/kaggle"
elif [ -f "$ROOT_DIR/.venv/bin/python" ]; then
    KAGGLE_CMD="$ROOT_DIR/.venv/bin/python -m kaggle"
else
    KAGGLE_CMD="python3 -m kaggle"
fi

COMMAND="${1:-help}"

case "$COMMAND" in
    push)
        echo "============================================================"
        echo "Pushing Task 1 GPU Kernel to Kaggle ($KERNEL_ID)..."
        echo "============================================================"
        $KAGGLE_CMD kernels push -p "$KERNEL_DIR"
        echo ""
        echo "Kernel pushed successfully. Checking status..."
        $KAGGLE_CMD kernels status "$KERNEL_ID"
        ;;

    status)
        echo "============================================================"
        echo "Checking Kaggle Kernel Status ($KERNEL_ID)..."
        echo "============================================================"
        $KAGGLE_CMD kernels status "$KERNEL_ID"
        ;;

    output|retrieve|download)
        echo "============================================================"
        echo "Downloading Output Artifacts from Kaggle ($KERNEL_ID)..."
        echo "============================================================"
        mkdir -p "$OUTPUT_DIR"
        $KAGGLE_CMD kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
        echo ""
        echo "Outputs saved to $OUTPUT_DIR:"
        ls -lh "$OUTPUT_DIR"
        ;;

    files)
        echo "============================================================"
        echo "Listing Output Files for ($KERNEL_ID)..."
        echo "============================================================"
        $KAGGLE_CMD kernels files "$KERNEL_ID"
        ;;

    *)
        echo "UIT-DSC 2026 Task 1 Kaggle Management Helper"
        echo "Usage: $0 {push|status|output|files}"
        echo ""
        echo "Commands:"
        echo "  push    : Push notebook and metadata from $KERNEL_DIR to Kaggle to trigger GPU execution"
        echo "  status  : Monitor current execution status (queued, running, complete, error)"
        echo "  output  : Download produced submission.json and submission.zip into $OUTPUT_DIR"
        echo "  files   : List files generated in the kernel working directory"
        exit 1
        ;;
esac
