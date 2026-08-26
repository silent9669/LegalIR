import argparse
import json
from pathlib import Path
import sys
from src.artifacts.manifest import build_inventory, verify_inventory


def main():
    parser = argparse.ArgumentParser(description="LegalIR Artifact Inventory and Verification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inv_parser = subparsers.add_parser("inventory", help="Build inventory JSON from directory")
    inv_parser.add_argument("--root", type=str, required=True, help="Root directory to index")
    inv_parser.add_argument("--output", type=str, required=True, help="Output JSON manifest path")

    ver_parser = subparsers.add_parser("verify", help="Verify directory against JSON manifest")
    ver_parser.add_argument("--root", type=str, required=True, help="Root directory to verify")
    ver_parser.add_argument("--manifest", type=str, required=True, help="Expected JSON manifest path")

    args = parser.parse_args()

    if args.command == "inventory":
        root = Path(args.root)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        inv = build_inventory(root)
        output.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote inventory of {len(inv)} files to {output}")
    elif args.command == "verify":
        errors = verify_inventory(Path(args.root), Path(args.manifest))
        if errors:
            print("Artifact verification FAILED:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Artifact verification PASSED.")


if __name__ == "__main__":
    main()
