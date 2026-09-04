#!/usr/bin/env python3
"""Generate the Colab T4 Smoke Notebook pinning approved runtime commit."""

import argparse
import json
from pathlib import Path


def generate_colab_notebook(output_path: Path, runtime_commit: str = "a0efb25"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LegalIR Colab T4 Production Contract Smoke\n",
                f"**Pinned Runtime Commit:** `{runtime_commit}`\n",
                "\n",
                "This notebook executes the production contract smoke on Tesla T4:\n",
                "- Environment & hardware verification (Tesla T4)\n",
                "- Canonical dataset verification\n",
                "- DEk21 & FAISS indexing verification\n",
                "- Lazy evidence & pair materialization\n",
                "- Real BGE+LoRA training probe (effective batch 16)\n",
                "- Adapter reload & public reranking path\n",
                "- Host RAM & GPU VRAM telemetry\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Hardware Preflight\n",
                "!nvidia-smi\n",
                "import torch\n",
                "assert torch.cuda.is_available(), 'CUDA required! Enable GPU runtime.'\n",
                "print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Checkout pinned approved runtime\n",
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
                "# 3. Run Colab T4 Contract Smoke\n",
                "!python scripts/run_colab_t4_smoke.py\n",
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
    print(f"[+] Colab T4 smoke notebook generated at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Colab smoke notebook.")
    parser.add_argument("--output", type=str, default="notebooks/colab_t4_smoke.ipynb")
    parser.add_argument("--commit", type=str, default="a0efb25")
    args = parser.parse_args()

    generate_colab_notebook(Path(args.output), runtime_commit=args.commit)


if __name__ == "__main__":
    main()
