#!/usr/bin/env python3
"""
CLI tool to verify that release_approval.json satisfies all release provenance and consistency invariants.

Authoritative specification: LEGALIR_76BB_FINAL_RELEASE_PROVENANCE_SMOKE_REPAIR.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.release.provenance import validate_release_approval

DEFAULT_APPROVAL_PATH = REPO_ROOT / "artifacts" / "task1" / "release_approval.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify release approval artifact consistency.")
    parser.add_argument(
        "--approval",
        type=Path,
        default=DEFAULT_APPROVAL_PATH,
        help="Path to release_approval.json",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root path",
    )

    args = parser.parse_args()

    if not args.approval.exists():
        print(f"[-] Release approval file not found: {args.approval}", file=sys.stderr)
        return 1

    try:
        approval_data = json.loads(args.approval.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[-] Failed to parse release approval JSON: {exc}", file=sys.stderr)
        return 1

    print("=================================================================")
    print("LegalIR Release Approval Consistency Gate")
    print(f"  • Approval File : {args.approval}")
    print(f"  • Runtime SHA   : {approval_data.get('runtime_sha')}")
    print(f"  • Release SHA   : {approval_data.get('release_sha')}")
    print(f"  • Kaggle Commit : {approval_data.get('production', {}).get('kaggle_expected_commit')}")
    print("=================================================================")

    is_valid, errors = validate_release_approval(approval_data, git_root=args.repo_root)

    if is_valid:
        print("[+] SUCCESS: Release approval artifact is valid and provenance-consistent.")
        print("[+] Kaggle FULL is authorized on approved runtime commit.")
        return 0
    else:
        print("[-] FAILURE: Release approval validation errors detected:", file=sys.stderr)
        for err in errors:
            print(f"    - {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
