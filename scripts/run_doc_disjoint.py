#!/usr/bin/env python3
"""CLI script to execute or resume the document-disjoint robustness validation job."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.doc_disjoint_job import DocDisjointRunner


def main():
    parser = argparse.ArgumentParser(description="Run document-disjoint validation job.")
    parser.add_argument("--work-dir", type=str, default="artifacts/factory/doc_disjoint", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Run mock pass for testing")
    args = parser.parse_args()

    runner = DocDisjointRunner(work_dir=args.work_dir)
    manifest = runner.run(mock_run=args.mock)
    if manifest.status == "PASS":
        print("[+] Document-disjoint validation completed with PASS.")
        sys.exit(0)
    else:
        print("[!] Document-disjoint validation FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
