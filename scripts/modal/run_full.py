#!/usr/bin/env python3
"""Single-command teammate entrypoint for the Modal A100-80GB full run.

No release required. No time/quality gates. Example:

    python scripts/modal/run_full.py --private --push-config --detach

Flags are forwarded to scripts/modal/run_modal_cli.sh:
    --private      2,080 private queries (default: 1,000 public)
    --push-config  max-score reranker config (default: base listwise r=32):
                   v3 = listwise, LoRA r=64 cold start, 2 coverage epochs.
                   LEGALIR_RERANKER_CONFIG env overrides both.
    --detach       app survives client disconnect (record app ID, watch
                   `modal app logs <id>`, stop with `modal app stop <id> --yes`)
    --warm          warm shared Volume on CPU first (models + dataset),
                   then dispatch A100. Recommended: A100 bills zero downloads.
    --warm-only     only warm the shared Volume (no A100 dispatch).
    --hf-allow-public-repo  opt-in to push to an existing PUBLIC HF repo

Preflight (fast, local, CPU-only):
    - score-push coherence validator (configs, fusion, miner, ensemble, budget)
    - .env has HF_TOKEN_WRITE + KAGGLE credentials (or env exports)
    - modal CLI installed + authenticated
    - reranker config file exists, parameter budget < 4B
    - prints run label, config, timeout (default: no limit = 24h platform max)

Rescue path (if a full run trains but dies in inference, like HF Run05):
    re-dispatch is unnecessary — the finished adapter + indexes persist on the
    'legalir-production' Volume under <label>/attempts/<id>/. Re-run with
    LEGALIR_RESCUE_ADAPTER_DIR set (consumed by run_modal_rescue.py) or ask
    the run owner for the attempt path, then run inference-only from it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _check_env() -> list[str]:
    problems: list[str] = []
    if not (os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")):
        env_file = REPO_ROOT / ".env"
        has_hf = env_file.is_file() and "HF_TOKEN" in env_file.read_text(errors="ignore")
        if not has_hf:
            problems.append("HF token missing: set HF_TOKEN_WRITE env or .env (Modal 'huggingface-secret' also required remotely).")
    if not (os.environ.get("KAGGLE_API_TOKEN")
            or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        env_file = REPO_ROOT / ".env"
        has_kg = env_file.is_file() and "KAGGLE" in env_file.read_text(errors="ignore")
        if not has_kg:
            problems.append("Kaggle creds missing: set KAGGLE_API_TOKEN (or USERNAME+KEY) env or .env (Modal 'kaggle-secret' also required).")
    return problems


def _check_modal() -> str | None:
    for cand in (REPO_ROOT / ".venv/bin/modal", Path("modal")):
        try:
            r = subprocess.run([str(cand), "--version"], capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return str(cand)
        except Exception:
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--push-config", action="store_true")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--warm", action="store_true",
                    help="Warm shared Volume on CPU first, then dispatch A100.")
    ap.add_argument("--warm-only", action="store_true",
                    help="Only warm the shared Volume (no A100 dispatch).")
    ap.add_argument("--hf-allow-public-repo", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Preflight only, do not dispatch.")
    args = ap.parse_args(argv)

    cfg = "configs/experiments/reranker_lora_v3_push.yaml" if args.push_config else "configs/experiments/reranker_lora.yaml"
    if not (REPO_ROOT / cfg).is_file():
        print(f"[!] Reranker config missing: {cfg}", file=sys.stderr)
        return 2
    if args.push_config:
        os.environ["LEGALIR_RERANKER_CONFIG"] = cfg

    print("=" * 70)
    print("LegalIR full-run preflight (fast, local, no release needed)")
    print(f"  config   : {cfg}")
    print(f"  phase    : {'private (2080q)' if args.private else 'public (1000q)'}")
    print(f"  timeout  : {os.environ.get('MODAL_TIMEOUT_SECONDS', '86400 (no-limit default)')}s")
    print(f"  strict   : {os.environ.get('LEGALIR_STRICT_GATES', 'off (advisory)')}")
    print("=" * 70)

    for p in _check_env():
        print(f"[!] {p}", file=sys.stderr)
    modal_bin = _check_modal()
    if modal_bin is None:
        print("[!] modal CLI not found/authenticated (.venv/bin/modal). Install + `modal setup`.", file=sys.stderr)
        return 2
    print(f"[+] modal CLI: {modal_bin}")

    # Score-push coherence (fast, offline): configs, fusion, miner, ensemble, budget.
    val = subprocess.run(
        [sys.executable, "scripts/validate_score_push.py"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )
    if val.returncode != 0:
        print("[!] Score-push coherence FAILED — fix before dispatch:", file=sys.stderr)
        print(val.stdout[-3000:], file=sys.stderr)
        return 1
    print("[+] Score-push coherence: ALL CHECKS PASSED")

    # Parameter budget (fast, offline).
    audit = subprocess.run(
        [sys.executable, "scripts/audit_parameters.py", "--check-only"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )
    if audit.returncode != 0:
        print("[!] Parameter audit failed:", file=sys.stderr)
        print(audit.stdout[-2000:], file=sys.stderr)
        return 1
    print("[+] Parameter budget < 4B: PASS")

    try:
        label = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        cwd=str(REPO_ROOT), text=True, timeout=15).strip()
    except Exception:
        label = "dev"
    print(f"[+] Run label: {label} (used for Volume path only, not gated)")

    if args.dry_run:
        print("[*] --dry-run: preflight OK, not dispatching.")
        return 0

    fwd = []
    if args.private:
        fwd.append("--private")
    if args.push_config:
        fwd.append("--push-config")
    if args.detach:
        fwd.append("--detach")
    if args.warm:
        fwd.append("--warm")
    if args.warm_only:
        fwd.append("--warm-only")
    if args.hf_allow_public_repo:
        fwd.append("--hf-allow-public-repo")
    cmd = ["bash", "scripts/modal/run_modal_cli.sh", *fwd]
    print(f"[*] Dispatching: {' '.join(cmd)}")
    os.environ.setdefault("PYTHON_BIN", ".venv/bin/python")
    os.environ.setdefault("MODAL_BIN", modal_bin)
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
