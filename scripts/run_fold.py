#!/usr/bin/env python3
"""CLI script to execute or resume an isolated OOF validation fold."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.fold_job import FoldJobRunner, should_resume_fold


def main():
    parser = argparse.ArgumentParser(description="Run an isolated validation fold.")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    parser.add_argument("--work-dir", type=str, default="artifacts/factory/folds", help="Root folds directory")
    parser.add_argument("--mock", action="store_true", help="Run mock pass for testing")
    args = parser.parse_args()

    fold_dir = Path(args.work_dir) / f"fold_{args.fold}"
    if should_resume_fold(fold_dir):
        print(f"[+] Fold {args.fold} at {fold_dir} already verified PASS. Resuming without re-execution.")
        sys.exit(0)

    runner = FoldJobRunner(fold_id=args.fold, work_dir=args.work_dir)
    manifest = runner.run(mock_run=args.mock)
    if manifest.status == "PASS":
        print(f"[+] Fold {args.fold} execution completed with PASS.")
        sys.exit(0)
    else:
        print(f"[!] Fold {args.fold} execution FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
