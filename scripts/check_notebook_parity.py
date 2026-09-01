#!/usr/bin/env python3
"""
Verify exact byte-for-byte SHA-256 parity between root notebook and kaggle_kernel_task1 notebook.

Authoritative specification: LEGALIR_CI_COLAB_KAGGLE_ARCHITECTURE_SPEC.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_NB_PATH = REPO_ROOT / "legalir_training.ipynb"
KAGGLE_NB_PATH = REPO_ROOT / "kaggle_kernel_task1" / "legalir_training.ipynb"


def compute_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Notebook file not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_notebook_parity(
    root_path: Path = ROOT_NB_PATH, kaggle_path: Path = KAGGLE_NB_PATH
) -> tuple[bool, str, str]:
    sha_root = compute_sha256(root_path)
    sha_kaggle = compute_sha256(kaggle_path)
    return sha_root == sha_kaggle, sha_root, sha_kaggle


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SHA-256 parity between LegalIR notebooks.")
    parser.add_argument("--root-nb", type=Path, default=ROOT_NB_PATH, help="Path to root legalir_training.ipynb")
    parser.add_argument("--kaggle-nb", type=Path, default=KAGGLE_NB_PATH, help="Path to kaggle_kernel_task1/legalir_training.ipynb")
    args = parser.parse_args()

    try:
        is_identical, sha_root, sha_kaggle = check_notebook_parity(args.root_nb, args.kaggle_nb)
    except Exception as exc:
        print(f"[-] Notebook parity check ERROR: {exc}", file=sys.stderr)
        return 1

    print("=================================================================")
    print("LegalIR Notebook Parity Check (SHA-256 Verification)")
    print("=================================================================")
    print(f"  • Root notebook  : {args.root_nb}")
    print(f"    SHA-256        : {sha_root}")
    print(f"  • Kaggle notebook: {args.kaggle_nb}")
    print(f"    SHA-256        : {sha_kaggle}")
    print("=================================================================")

    if is_identical:
        print("[+] SUCCESS: Notebooks are identical byte-for-byte.")
        return 0
    else:
        print("[-] FAILURE: Notebook SHA-256 mismatch detected!", file=sys.stderr)
        print("    Please run `python scripts/generate_kaggle_notebook.py` to regenerate parity.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
