#!/usr/bin/env python3
"""Generate the Kaggle Final Production Notebook pinning approved runtime commit."""

import argparse
import json
from pathlib import Path


def generate_kaggle_notebook(output_path: Path, runtime_commit: str = "a0efb25"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LegalIR Kaggle Final Production Trainer\n",
                f"**Pinned Runtime Commit:** `{runtime_commit}`\n",
                "\n",
                "This notebook executes the pure production run:\n",
                "- Verifies approved runtime SHA and canonical dataset identity\n",
                "- Verifies production bundle fingerprints\n",
                "- Trains exactly one final BGE LoRA adapter on all 7,000 queries (effective batch 16)\n",
                "- Reranks public candidates using frozen fusion & top-5 selector\n",
                "- Validates strict submission criteria and packages `submission.zip`\n",
                "*(Does NOT run 5-fold OOF, doc-disjoint validation, or heavy index builds)*\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# K0: Hardware & Environment Preflight\n",
                "!nvidia-smi\n",
                "import torch\n",
                "assert torch.cuda.is_available(), 'CUDA required!'\n",
                "num_gpus = torch.cuda.device_count()\n",
                "print(f'CUDA GPUs available: {num_gpus}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# K0.1: Checkout pinned approved runtime\n",
                "!git clone https://github.com/silent9669/LegalIR.git repo\n",
                "%cd repo\n",
                f"!git checkout {runtime_commit}\n",
                "!pip install -q -r requirements.txt\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# K1 - K9: Execute Kaggle Final Production Runner\n",
                "!python scripts/run_kaggle_final.py \\\n",
                "    --dataset-dir /kaggle/input/task1-canonical-v2 \\\n",
                "    --bundle-dir /kaggle/input/legalir-production-bundle \\\n",
                "    --output-dir /kaggle/working\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Verify output artifact\n",
                "import os\n",
                "assert os.path.exists('/kaggle/working/submission.zip'), 'submission.zip missing!'\n",
                "print('[+] submission.zip successfully generated and ready for competition submission.')\n",
            ],
        },
    ]

    nb_data = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2)
    print(f"[+] Kaggle final notebook generated at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle final notebook.")
    parser.add_argument("--output", type=str, default="notebooks/kaggle_final.ipynb")
    parser.add_argument("--commit", type=str, default="a0efb25")
    args = parser.parse_args()

    generate_kaggle_notebook(Path(args.output), runtime_commit=args.commit)


if __name__ == "__main__":
    main()
