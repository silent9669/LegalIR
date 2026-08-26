import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from src.artifacts.manifest import build_inventory, verify_inventory

CLEANUP_CANDIDATE_RELATIVE_PATHS = [
    "data/task1_canonical",
    "artifacts/data",
    "artifacts/indexes",
    "indexes",
    "chunks.parquet",
    "validation_benchmark_report.json",
    ".playwright-mcp",
]


def plan_cleanup(repo_root: str | Path) -> list[dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    actions = []

    for rel in CLEANUP_CANDIDATE_RELATIVE_PATHS:
        target = repo_root / rel
        if target.exists():
            is_dir = target.is_dir()
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) if is_dir else target.stat().st_size
            actions.append({
                "path": str(target),
                "relative_path": rel,
                "is_dir": is_dir,
                "size_bytes": size,
                "reason": f"Redundant legacy artifact replaced by canonical v2 in artifacts/shared/canonical/v2 and artifacts/local/",
            })

    return actions


def apply_cleanup(actions: list[dict[str, Any]], confirmation_token: str = "") -> list[str]:
    if confirmation_token != "CONFIRM_CLEANUP":
        raise ValueError("Invalid confirmation token! Pass 'CONFIRM_CLEANUP' to execute.")

    deleted = []
    for action in actions:
        p = Path(action["path"])
        if p.exists():
            if action["is_dir"]:
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            deleted.append(action["relative_path"])
    return deleted


def main():
    parser = argparse.ArgumentParser(description="LegalIR Artifact Inventory and Verification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inv_parser = subparsers.add_parser("inventory", help="Build inventory JSON from directory")
    inv_parser.add_argument("--root", type=str, required=True, help="Root directory to index")
    inv_parser.add_argument("--output", type=str, required=True, help="Output JSON manifest path")

    ver_parser = subparsers.add_parser("verify", help="Verify directory against JSON manifest")
    ver_parser.add_argument("--root", type=str, required=True, help="Root directory to verify")
    ver_parser.add_argument("--manifest", type=str, required=True, help="Expected JSON manifest path")

    clean_parser = subparsers.add_parser("cleanup", help="Safely clean verified redundant legacy files")
    clean_parser.add_argument("--repo", type=str, default=".", help="Repository root directory")
    clean_parser.add_argument("--dry-run", action="store_true", help="Print cleanup actions without executing")
    clean_parser.add_argument("--confirm", type=str, default=None, help="Confirmation token 'CONFIRM_CLEANUP'")
    clean_parser.add_argument("--output", type=str, default=None, help="Save plan JSON to file")

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
    elif args.command == "cleanup":
        repo_root = Path(args.repo)
        actions = plan_cleanup(repo_root)

        if args.output:
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(actions, indent=2) + "\n", encoding="utf-8")
            print(f"Saved cleanup plan to {out_file}")

        total_bytes = sum(a["size_bytes"] for a in actions)
        print(f"Cleanup Plan: {len(actions)} targets, reclaiming ~{total_bytes / (1024*1024):.1f} MB:")
        for a in actions:
            print(f"  - [{('DIR' if a['is_dir'] else 'FILE')}] {a['relative_path']} (~{a['size_bytes'] / (1024*1024):.1f} MB)")

        if args.dry_run or args.confirm != "CONFIRM_CLEANUP":
            print("\nDry run mode. Run with --confirm CONFIRM_CLEANUP to execute deletion.")
        else:
            deleted = apply_cleanup(actions, confirmation_token=args.confirm)
            print(f"\nSuccessfully cleaned {len(deleted)} legacy targets!")


if __name__ == "__main__":
    main()
