#!/usr/bin/env python3
"""Teammate preflight report for a fresh machine/account (read-only).

Shows everything verifiable locally plus best-effort read-only live checks,
and marks anything cloud-only as UNVERIFIED/BLOCKED instead of fake PASS:

  Git SHA in use + origin/main SHA, clean tree, Modal binary/profile/account,
  secret NAMES present (never values), dataset/model revisions, HF target repo
  + write permission + visibility, quota/budget to confirm, and the exact
  train command to use.

  Policy: basic GitHub CI green (tests, score-push, audit, drift) is the
  production dispatch gate. Strict release approval and dual-GPU Kaggle
  evidence are not required and are not checked here.

Exit codes: 0 = no BLOCKED items (UNVERIFIED items are listed for the
GO/NO-GO decision, which lives outside this script); 1 = unexpected error;
2 = a fail-closed BLOCKED item (dirty tree, missing/invalid HF target,
missing token, HF definitive deny, fixtures for offline tests).

Read-only guarantees: never creates HF repos, uploads, warms cloud caches,
or launches GPU. Uses the teammate token only for read-only Hub checks
(whoami/repo_info/auth_check); values are never printed. Never writes .env.

Example:
    .venv/bin/python scripts/preflight_teammate.py \\
        --hf-repo dangphuc2109/legalir-task1-reranker --hf-allow-public-repo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TIMEOUT_S = 30
REQUIRED_SECRETS = ("kaggle-secret", "huggingface-secret")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=kw.get("timeout", TIMEOUT_S), cwd=str(kw.get("cwd", REPO_ROOT)))


# --- Git ---

def git_local_head(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        r = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
        sha = r.stdout.strip()
        return sha if r.returncode == 0 and sha else None
    except Exception:
        return None


def git_origin_main(repo_root: Path = REPO_ROOT) -> str | None:
    """Remote SHA via ls-remote (read-only network). None when unreachable."""
    try:
        r = _run(["git", "ls-remote", "origin", "refs/heads/main"], cwd=repo_root)
        parts = r.stdout.strip().split()
        return parts[0] if r.returncode == 0 and parts else None
    except Exception:
        return None


def git_clean(repo_root: Path = REPO_ROOT) -> tuple[bool, list[str]]:
    try:
        r = _run(["git", "status", "--porcelain=v1"], cwd=repo_root)
        if r.returncode != 0:
            return False, ["<git status failed>"]
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        return (len(lines) == 0), lines[:10]
    except Exception:
        return False, ["<git status error>"]


# --- Modal ---

def find_modal_bin(repo_root: Path = REPO_ROOT) -> str | None:
    for cand in (repo_root / ".venv" / "bin" / "modal",):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return shutil.which("modal")


def modal_version(modal_bin: str) -> str | None:
    try:
        r = _run([modal_bin, "--version"])
        out = (r.stdout.strip() or r.stderr.strip()).splitlines()
        return out[0].strip() if r.returncode == 0 and out else None
    except Exception:
        return None


def modal_active_profile(modal_bin: str) -> tuple[str | None, str]:
    """Returns (profile_name_or_None, detail). Read-only; None when unknown."""
    try:
        r = _run([modal_bin, "profile", "current"])
        name = r.stdout.strip().splitlines()
        if r.returncode == 0 and name and name[0].strip():
            return name[0].strip(), "modal profile current"
    except Exception:
        pass
    try:
        r = _run([modal_bin, "profile", "list"])
        if r.returncode == 0 and r.stdout.strip():
            return None, "profiles listed but active marker unparsed"
    except Exception:
        pass
    return None, "unreachable (no auth/network?)"


def modal_secret_names(modal_bin: str) -> tuple[list[str] | None, str]:
    """Secret NAMES only (never values). None when the query fails."""
    try:
        r = _run([modal_bin, "secret", "list", "--json"])
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout)
            names = [str(x.get("name", "")) for x in data if isinstance(x, dict) and x.get("name")]
            return names, "modal secret list --json"
    except Exception:
        pass
    try:
        r = _run([modal_bin, "secret", "list"])
        if r.returncode == 0 and r.stdout.strip():
            toks = {t.strip("│| ") for ln in r.stdout.splitlines() for t in ln.split()}
            return sorted(t for t in toks if t), "modal secret list (text fallback)"
    except Exception:
        pass
    return None, "unreachable (no auth/network?)"


# --- Revisions (local, read-only) ---

def model_registry_info() -> list[dict[str, str]]:
    try:
        from src.models.bootstrap import MODEL_REGISTRY

        return [{"id": mid, "revision": str((meta or {}).get("revision", ""))}
                for mid, meta in MODEL_REGISTRY.items()]
    except Exception as exc:  # noqa: BLE001
        return [{"error": type(exc).__name__}]


def freeze_info(repo_root: Path = REPO_ROOT) -> dict:
    try:
        p = repo_root / "artifacts" / "task1" / "freeze" / "production_freeze.json"
        if not p.is_file():
            return {"present": False}
        data = json.loads(p.read_text(encoding="utf-8"))
        return {"present": True,
                "git_sha": str(data.get("git_sha", "")),
                "manifest_sha256": str((data.get("dataset") or {}).get("manifest_sha256", "")),
                "algorithm_config_sha256": str(data.get("algorithm_config_sha256", ""))}
    except Exception as exc:  # noqa: BLE001
        return {"present": False, "error": type(exc).__name__}


# --- HF target (explicit, fail-closed; read-only live check) ---

def resolve_target_repo(explicit: str | None, repo_root: Path = REPO_ROOT,
                        allow_default: bool = False) -> tuple[str, str]:
    from src.release.hf_repo import resolve_hf_repo_id

    return resolve_hf_repo_id(explicit=explicit, env=os.environ, repo_root=repo_root,
                              allow_default=allow_default)


def read_token(repo_root: Path = REPO_ROOT) -> str | None:
    """Token for read-only Hub checks. Value is used, never printed."""
    for cand in (os.environ.get("HF_TOKEN_WRITE"), os.environ.get("HF_TOKEN")):
        if cand and str(cand).startswith("hf_"):
            return str(cand)
    try:
        from src.release.hf_repo import parse_dotenv_file

        vals = parse_dotenv_file(repo_root / ".env")
        for key in ("HF_TOKEN_WRITE", "HF_TOKEN"):
            cand = (vals.get(key) or "").strip()
            if cand.startswith("hf_"):
                return cand
    except Exception:
        pass
    return None


def hf_live_check(repo_id: str, token: str, allow_public_repo: bool = False) -> dict:
    """Read-only Hub check (whoami/repo_info/auth_check). Never creates/uploads."""
    import scripts.check_hf_repo as checker

    result = checker.check_repo(repo_id, token)
    verdict = "PASS"
    if result.get("exists") is not True:
        verdict = "FAIL"
    elif result.get("visibility") == "unknown":
        verdict = "FAIL"
    elif result.get("visibility") == "public" and not allow_public_repo:
        verdict = "BLOCKED"
    elif result.get("write_access") is not True:
        verdict = "FAIL"
    result["preflight_verdict"] = verdict
    return result


def train_command(repo_id: str, allow_public_repo: bool) -> str:
    cmd = ("MODAL_TIMEOUT_SECONDS=10800 .venv/bin/python scripts/modal/run_full.py "
           f"--warm --private --push-config --detach --hf-repo {repo_id}")
    if allow_public_repo:
        cmd += " --hf-allow-public-repo"
    return cmd


def collect_report(args, repo_root: Path = REPO_ROOT) -> tuple[dict, int]:
    """Build the report dict; returns (report, exit_code). No I/O besides probes."""
    from src.release.hf_repo import default_allowed

    report: dict = {"checks": []}
    blocked = False

    def add(name: str, status: str, detail: str) -> None:
        report["checks"].append({"name": name, "status": status, "detail": detail})

    # Git.
    head = git_local_head(repo_root)
    origin = git_origin_main(repo_root)
    add("git-local-head", "PASS" if head else "FAIL", head or "unreadable")
    report["local_sha"] = head
    if origin is None:
        add("git-origin-main", "UNVERIFIED", "ls-remote unreachable (offline?)")
    elif head and origin == head:
        add("git-origin-main", "PASS", origin)
    else:
        add("git-origin-main", "FAIL", f"origin={origin} local={head}")
    report["origin_sha"] = origin
    clean, dirty = git_clean(repo_root)
    if clean:
        add("working-tree", "PASS", "clean")
    else:
        blocked = True
        add("working-tree", "BLOCKED", f"dirty: {dirty} (remote clones origin; commit/stash first)")

    # Modal.
    modal_bin = os.environ.get("MODAL_BIN") or find_modal_bin(repo_root)
    if not modal_bin:
        blocked = True
        add("modal-cli", "BLOCKED", "modal CLI not found (run scripts/setup.sh)")
        report["modal_bin"] = None
    else:
        report["modal_bin"] = modal_bin
        ver = modal_version(modal_bin)
        add("modal-cli", "PASS" if ver else "FAIL", ver or "version query failed")
        profile, how = modal_active_profile(modal_bin)
        if profile:
            add("modal-profile", "PASS", f"active={profile} ({how})")
        else:
            add("modal-profile", "UNVERIFIED", how)
        report["modal_profile"] = profile
        names, how = modal_secret_names(modal_bin)
        if names is None:
            add("modal-secrets", "UNVERIFIED", f"{how}; verify kaggle-secret + huggingface-secret on the dashboard")
        else:
            missing = [s for s in REQUIRED_SECRETS if s not in names]
            if missing:
                blocked = True
                add("modal-secrets", "BLOCKED", f"missing: {missing} (found names only, values never read)")
            else:
                add("modal-secrets", "PASS", f"present: {REQUIRED_SECRETS} (names only)")
        report["modal_secret_names"] = names

    # Revisions.
    report["models"] = model_registry_info()
    add("model-revisions", "PASS" if all("revision" in m and m["revision"] for m in report["models"]) else "FAIL",
        "; ".join(f"{m.get('id')}@{str(m.get('revision'))[:8]}" for m in report["models"]))
    report["freeze"] = freeze_info(repo_root)
    add("freeze-file", "PASS" if report["freeze"].get("present") else "FAIL",
        f"git_sha={report['freeze'].get('git_sha', 'n/a')}")

    # HF target (fail-closed).
    allow_default = bool(args.allow_default_hf_repo or default_allowed())
    try:
        repo_id, source = resolve_target_repo(args.hf_repo, repo_root, allow_default)
    except ValueError as exc:
        blocked = True
        add("hf-target", "BLOCKED", str(exc))
        repo_id, source = "", "invalid"
    if repo_id:
        add("hf-target", "PASS", f"{repo_id} (source={source})")
        report["hf_repo"] = repo_id
        report["hf_source"] = source
        token = read_token(repo_root)
        if not token:
            blocked = True
            add("hf-credentials", "BLOCKED", "no HF token in env/.env (needed for read-only check)")
        else:
            try:
                live = hf_live_check(repo_id, token, args.hf_allow_public_repo)
                report["hf_live"] = {k: v for k, v in live.items()}
                v = live.get("preflight_verdict")
                if v == "PASS":
                    add("hf-write-visibility", "PASS",
                        f"exists, visibility={live.get('visibility')}, write=True")
                elif v == "BLOCKED":
                    blocked = True
                    add("hf-write-visibility", "BLOCKED",
                        f"public repo without --hf-allow-public-repo (visibility={live.get('visibility')})")
                else:
                    blocked = True
                    add("hf-write-visibility", "BLOCKED",
                        f"check failed: exists={live.get('exists')} visibility={live.get('visibility')} "
                        f"write={live.get('write_access')}")
            except Exception as exc:  # noqa: BLE001 - class only, never token
                add("hf-write-visibility", "UNVERIFIED", f"live check unreachable ({type(exc).__name__})")
    report["hf_allow_public_repo"] = bool(args.hf_allow_public_repo)

    # Quota/budget: no local source — always UNVERIFIED with instructions.
    add("quota-budget", "UNVERIFIED",
        "no CLI source; confirm GPU quota/SKU/region, approved money budget and stop procedure on the account")

    # Train command (only meaningful when nothing is BLOCKED).
    report["train_command"] = train_command(repo_id, args.hf_allow_public_repo) if repo_id and not blocked else ""
    if repo_id and not blocked:
        add("train-command", "PASS", report["train_command"])

    exit_code = 2 if blocked else 0
    report["exit_code"] = exit_code
    report["verdict"] = ("BLOCKED" if blocked else
                         "READY-FOR-DRY-RUN (production GO needs basic CI green + UNVERIFIED items confirmed)")
    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-repo", "--hf-repo-id", dest="hf_repo", default=None)
    ap.add_argument("--hf-allow-public-repo", action="store_true")
    ap.add_argument("--allow-default-hf-repo", action="store_true")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = ap.parse_args(argv)
    report, code = collect_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print("=" * 70)
        print("LegalIR teammate preflight (read-only; UNVERIFIED is not PASS)")
        print("=" * 70)
        for c in report["checks"]:
            print(f"[{c['status']:>10}] {c['name']}: {c['detail']}")
        print("=" * 70)
        print(f"verdict: {report['verdict']} (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
