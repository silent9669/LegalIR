#!/usr/bin/env python3
"""CLI script to cryptographically verify an existing production bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bundle.verifier import verify_production_bundle


def main():
    parser = argparse.ArgumentParser(description="Cryptographically verify a production bundle.")
    parser.add_argument("--bundle-dir", type=str, default="artifacts/bundle/production", help="Bundle directory to verify")
    parser.add_argument("--no-strict", action="store_true", help="Skip strict mandatory file check")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    print(f"[*] Verifying production bundle at {bundle_dir} ...")

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=not args.no_strict)
    if not is_valid:
        print("[!] Production bundle verification FAILED:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    print("[+] Production bundle verification PASSED. All digests and files verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
