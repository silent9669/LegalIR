#!/usr/bin/env python3
"""CLI script to cryptographically verify an existing production bundle."""

import argparse
import sys
from pathlib import Path

from src.bundle.verifier import verify_production_bundle


def main():
    parser = argparse.ArgumentParser(description="Cryptographically verify a production bundle.")
    parser.add_argument("--bundle-dir", type=str, default="artifacts/bundle/production", help="Bundle directory to verify")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    print(f"[*] Verifying production bundle at {bundle_dir} ...")

    is_valid, errors = verify_production_bundle(bundle_dir)
    if not is_valid:
        print("[!] Production bundle verification FAILED:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    print("[+] Production bundle verification PASSED. All digests and files verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
