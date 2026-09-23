#!/usr/bin/env python3
"""Read-only Hugging Face repo check for fresh accounts (no mutation).

Verifies, WITHOUT creating anything:
  - the token authenticates (whoami, username only),
  - the repo ID format is valid,
  - whether the repo exists, its visibility (private/public), and whether the
    token has write permission (auth_check, no upload, no create_repo).

This is the safe pre-dispatch check: unlike the Modal preflight (which calls
create_repo and therefore MUTATES by creating a private repo), this script
never creates, uploads, or changes visibility. Do not run the real preflight
"just to check" unless repo creation is approved. Never prints token values.

Exit codes: 0 = repo exists, visibility known, AND write access confirmed
(ready for preflight/upload); 1 = anything else (missing token, repo absent,
unknown visibility, NO write access, network) — an automated gate must treat
1 as BLOCKED, never as pass; 2 = invalid repo ID.

Example:
    HF_TOKEN_WRITE=hf_... .venv/bin/python scripts/check_hf_repo.py --repo OWNER/REPO
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_token(explicit: str | None = None) -> str | None:
    for cand in (explicit, os.environ.get("HF_TOKEN_WRITE"), os.environ.get("HF_TOKEN")):
        if cand and str(cand).startswith("hf_"):
            return str(cand)
    return None


def check_repo(repo_id: str, token: str) -> dict:
    """Read-only status probe. Never calls create_repo/upload."""
    from huggingface_hub import HfApi

    from src.release.hf_repo import validate_hf_repo_id

    repo_id = validate_hf_repo_id(repo_id)
    api = HfApi(token=token)
    result: dict = {"repo_id": repo_id}
    user = api.whoami()
    result["authenticated_as"] = str(user.get("name", "unknown"))
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception as exc:  # noqa: BLE001 - class only, never message/tokens
        result["exists"] = False
        result["note"] = (
            f"repo not found or not visible ({type(exc).__name__}); "
            "creation requires approval (preflight creates private repos)."
        )
        return result
    result["exists"] = True
    private = getattr(info, "private", None)
    result["private"] = private
    if private is False:
        result["visibility"] = "public"
    elif private is True:
        result["visibility"] = "private"
    else:
        result["visibility"] = "unknown"
    try:
        api.auth_check(repo_id=repo_id, repo_type="model", write=True)
        result["write_access"] = True
    except Exception as exc:  # noqa: BLE001
        result["write_access"] = False
        result["write_error"] = type(exc).__name__
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo 'owner/repo' to inspect (read-only).")
    ap.add_argument("--token", default=None, help="HF token (default: HF_TOKEN_WRITE/HF_TOKEN env).")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = ap.parse_args(argv)

    token = resolve_token(args.token)
    if not token:
        print("[!] No HF token: set HF_TOKEN_WRITE (or HF_TOKEN) env.", file=sys.stderr)
        return 1
    try:
        result = check_repo(args.repo, token)
    except ValueError as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Cannot verify ({type(exc).__name__}); check network/token.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"repo  : {result.get('repo_id')}")
        print(f"auth  : @{result.get('authenticated_as', '?')}")
        print(f"exists: {result.get('exists')}")
        print(f"visibility: {result.get('visibility', 'n/a')}")
        print(f"write : {result.get('write_access', 'n/a')}")
        if result.get("note"):
            print(f"note  : {result['note']}")
    if result.get("exists") is not True:
        return 1
    if result.get("visibility") == "unknown":
        return 1
    # Write access is required: without it the upload step fails AFTER hours
    # of GPU billing, so absence of write must never exit 0.
    if result.get("write_access") is not True:
        print(f"[!] BLOCKED: no write access to {result.get('repo_id')} "
              f"(visibility={result.get('visibility')}); upload would fail.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
