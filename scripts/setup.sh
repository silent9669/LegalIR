#!/usr/bin/env bash
# ==============================================================================
# LegalIR fresh-machine setup (idempotent, local-only).
#
# Creates .venv (if missing), installs Python dependencies plus the Modal and
# Kaggle CLIs, then verifies the required tools. Safe to re-run: existing
# .venv is reused, pip installs are idempotent.
#
# What this script NEVER does:
#   - never creates, overwrites, or prints .env / secrets (it only reports
#     which variable NAMES are present; values are never echoed),
#   - never touches the cloud (no HF repo creation, no upload, no warm, no GPU),
#   - never commits anything to git.
#
# Steps:  1. scripts/setup.sh [--check-only]   (this file: tools only)
#         2. create .env manually from .env.example (never committed)
#         3. scripts/preflight_teammate.py       (read-only preflight report)
#         4. run_full.py --dry-run, then (after GO + spend approval) train.
#
# Test hooks:
#   SETUP_PYTHON (default python3), SETUP_VENV_DIR (default .venv),
#   SETUP_REQUIREMENTS (default requirements.txt), SETUP_MODAL_BIN / SETUP_GIT_BIN.
# ==============================================================================
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SETUP_PYTHON="${SETUP_PYTHON:-python3}"
SETUP_VENV_DIR="${SETUP_VENV_DIR:-.venv}"
SETUP_REQUIREMENTS="${SETUP_REQUIREMENTS:-requirements.txt}"
CHECK_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    -h|--help)
      echo "Usage: scripts/setup.sh [--check-only]"
      echo "  Default: create .venv (if missing), install deps + modal/kaggle CLIs, verify tools."
      echo "  --check-only: verify tools only, install nothing. Never touches .env/secrets/cloud."
      exit 0
      ;;
    *) echo "[!] Unknown argument: $arg (expected --check-only)" >&2; exit 2 ;;
  esac
done

fail() { echo "[!] $*" >&2; exit 1; }

# --- 0. Python present with a supported version (3.10+; remote runs 3.11) ---
if ! command -v "$SETUP_PYTHON" >/dev/null 2>&1; then
  echo "[!] Python not found: $SETUP_PYTHON (install python3.10+ first)." >&2
  exit 2
fi
PY_VER="$("$SETUP_PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
PY_MAJOR="${PY_VER%%.*}"; PY_MINOR="${PY_VER#*.}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "[!] Python $PY_VER too old; need python3.10+ (remote uses 3.11)." >&2
  exit 2
fi
echo "[+] Python: $SETUP_PYTHON ($PY_VER)"

VENV_PY="$SETUP_VENV_DIR/bin/python"
VENV_PIP="$SETUP_VENV_DIR/bin/pip"

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "[*] --check-only: verifying tools, installing nothing."
else
  # --- 1. Create .venv once (reuse when present) ---
  if [ -x "$VENV_PY" ]; then
    echo "[*] Reusing existing venv at $SETUP_VENV_DIR (idempotent, nothing recreated)."
  else
    echo "[*] Creating venv at $SETUP_VENV_DIR..."
    "$SETUP_PYTHON" -m venv "$SETUP_VENV_DIR" || fail "venv creation failed."
  fi
  [ -x "$VENV_PY" ] || fail "venv python missing after setup: $VENV_PY"
  # --- 2. Dependencies (idempotent) + dispatch CLIs (outside requirements) ---
  echo "[*] Installing $SETUP_REQUIREMENTS..."
  "$VENV_PIP" install -r "$SETUP_REQUIREMENTS" || fail "pip install requirements failed."
  echo "[*] Installing modal + kaggle CLIs..."
  "$VENV_PIP" install "modal>=1.0" "kaggle>=1.8,<3" || fail "pip install modal/kaggle failed."
fi

# --- 3. Verify tools (names/versions only, never secret values) ---
[ -x "$VENV_PY" ] || { echo "[!] Missing tool: $VENV_PY (run scripts/setup.sh without --check-only)." >&2; exit 2; }
echo "[+] Tool: $VENV_PY"

MODAL_BIN="${SETUP_MODAL_BIN:-$SETUP_VENV_DIR/bin/modal}"
if [ -x "$MODAL_BIN" ] || command -v modal >/dev/null 2>&1; then
  echo "[+] Tool: modal ($("$MODAL_BIN" --version 2>/dev/null || modal --version 2>/dev/null || echo present))"
else
  echo "[!] Missing tool: modal (install via scripts/setup.sh)." >&2
  exit 2
fi
GIT_BIN="${SETUP_GIT_BIN:-git}"
if command -v "$GIT_BIN" >/dev/null 2>&1; then
  echo "[+] Tool: git ($("$GIT_BIN" --version 2>/dev/null))"
else
  echo "[!] Missing tool: git." >&2
  exit 2
fi

# --- 4. .env presence report (NAMES only; this script never writes/reads values) ---
if [ -f ".env" ]; then
  echo "[*] .env present (not created or modified by setup). Variables detected (names only):"
  for var in HF_TOKEN_WRITE HF_TOKEN KAGGLE_API_TOKEN KAGGLE_USERNAME KAGGLE_KEY HF_REPO_ID; do
    if grep -Eq "^[[:space:]]*(export[[:space:]]+)?${var}[[:space:]]*=" .env 2>/dev/null; then
      echo "    - $var: set"
    else
      echo "    - $var: missing"
    fi
  done
  echo "    Create/fix it manually from .env.example — setup never writes secrets."
else
  echo "[*] No .env file (expected on a fresh machine). Copy .env.example to .env and fill YOUR credentials manually."
fi

echo "[+] Setup OK: tools ready. Next: scripts/preflight_teammate.py, then run_full.py --dry-run."
