#!/usr/bin/env bash
# ==============================================================================
# LegalIR Modal A100 pre-dispatch CPU gate (recommended entrypoint).
#
# Fail before invoking Modal when CPU provenance is wrong, so known-invalid
# commits never reach image build/dispatch. Remote repeats trust checks;
# local checks alone cannot validate remote secret values or GPU.
#
# Order:
#   1. Validate args/tools/SHA before any cloud command.
#   2. Require a clean tracked/untracked tree (ignored artifacts excluded).
#   3. CPU provenance preflight (scripts/colab/bootstrap.py).
#   4. Dispatch via `modal run` with the same SHA in the environment.
#
# Remote order (in run_modal_a100.py): checkout+provenance, then HF access,
# then dataset download/fingerprint, then train. HF failure precedes expensive
# data acquisition. No auto-retry.
#
# Boolean flag spelling confirmed via installed SDK:
#   modal run scripts/modal/run_modal_a100.py --help
# shows `--hf-allow-public-repo / --no-hf-allow-public-repo`.
# Absent consent stays private-only; explicit --hf-allow-public-repo forwards
# once and is recorded in the manifest.
#
# Supervision model (client-disconnect hazard):
#   Default is ATTACHED (`modal run` without --detach). The installed SDK
#   states that disconnecting an ephemeral app terminates its running tasks.
#   A persistent Volume does NOT keep an unfinished training process alive
#   after that cancellation. Keep the supervising client connected (stable
#   network, machine awake, tmux/screen recommended) until the remote job
#   returns, and independently confirm app termination (not just client exit).
#   `--detach` is an explicit opt-in to `modal run --detach`: the app survives
#   client disconnect, but spending supervision changes. Detached runs require
#   durable job tracking (record app ID, attempt path, start UTC, ceiling),
#   active monitoring (`modal app logs <id>`), and an explicit stop procedure
#   (`modal app stop <id> --yes`, syntax verified). Never enable detached
#   silently.
#
# Test hooks:
#   PYTHON_BIN (default .venv/bin/python) for preflight,
#   MODAL_BIN (default .venv/bin/modal) for dispatch.
# ==============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODAL_BIN="${MODAL_BIN:-.venv/bin/modal}"

HF_PUBLIC_FLAG=""
PRIVATE_FLAG=""
DETACH_MODE=0
SHOW_HELP=0
PUSH_CONFIG=""
WARM_MODE=0
WARM_ONLY=0
HF_REPO_FLAG=""
HF_REPO_SOURCE="default"
ALLOW_DEFAULT_HF_REPO=0

dotenv_hf_repo() {
  # Extract HF_REPO_ID from .env without sourcing secrets (never echo tokens).
  _env_file="$REPO_ROOT/.env"
  if [ ! -f "$_env_file" ]; then
    return 1
  fi
  _line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?HF_REPO_ID[[:space:]]*=' "$_env_file" 2>/dev/null | tail -n 1 || true)"
  if [ -z "$_line" ]; then
    return 1
  fi
  _val="$(printf '%s' "$_line" | sed -E 's/^[[:space:]]*(export[[:space:]]+)?HF_REPO_ID[[:space:]]*=[[:space:]]*//' | sed -E "s/[[:space:]]+#.*$//" | sed -E "s/^'(.*)'\$/\1/" | sed -E 's/^"(.*)"$/\1/' | tr -d '[:space:]')"
  if [ -z "$_val" ]; then
    return 1
  fi
  printf '%s' "$_val"
}

valid_hf_repo() {
  printf '%s' "$1" | grep -Eq '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'
}

while [ "$#" -gt 0 ]; do
  arg="$1"
  case "$arg" in
    --hf-allow-public-repo)
      if [ -n "$HF_PUBLIC_FLAG" ]; then
        echo "[!] Duplicate --hf-allow-public-repo flag." >&2
        exit 2
      fi
      HF_PUBLIC_FLAG="--hf-allow-public-repo"
      shift
      ;;
    --no-hf-allow-public-repo)
      if [ -n "$HF_PUBLIC_FLAG" ]; then
        echo "[!] Duplicate public-repo flag." >&2
        exit 2
      fi
      HF_PUBLIC_FLAG="--no-hf-allow-public-repo"
      shift
      ;;
    --hf-repo=*|--hf-repo-id=*)
      if [ -n "$HF_REPO_FLAG" ]; then
        echo "[!] Duplicate --hf-repo flag." >&2
        exit 2
      fi
      if [ -z "${arg#*=}" ]; then
        echo "[!] --hf-repo requires a non-empty value 'owner/repo'." >&2
        exit 2
      fi
      HF_REPO_FLAG="${arg#*=}"
      HF_REPO_SOURCE="flag"
      shift
      ;;
    --hf-repo|--hf-repo-id)
      if [ -n "$HF_REPO_FLAG" ]; then
        echo "[!] Duplicate --hf-repo flag." >&2
        exit 2
      fi
      if [ "$#" -lt 2 ]; then
        echo "[!] --hf-repo requires a value 'owner/repo'." >&2
        exit 2
      fi
      HF_REPO_FLAG="$2"
      HF_REPO_SOURCE="flag"
      shift 2
      ;;
    --allow-default-hf-repo)
      if [ "$ALLOW_DEFAULT_HF_REPO" -ne 0 ]; then
        echo "[!] Duplicate --allow-default-hf-repo flag." >&2
        exit 2
      fi
      ALLOW_DEFAULT_HF_REPO=1
      shift
      ;;
    --private)
      if [ -n "$PRIVATE_FLAG" ]; then
        echo "[!] Duplicate --private flag." >&2
        exit 2
      fi
      PRIVATE_FLAG="--private"
      export LEGALIR_TEST_PHASE="private"
      shift
      ;;
    --push-config)
      if [ -n "$PUSH_CONFIG" ]; then
        echo "[!] Duplicate --push-config flag." >&2
        exit 2
      fi
      PUSH_CONFIG="configs/experiments/reranker_lora_v3_push.yaml"
      export LEGALIR_RERANKER_CONFIG="$PUSH_CONFIG"
      shift
      ;;
    --detach)
      if [ "$DETACH_MODE" -ne 0 ]; then
        echo "[!] Duplicate --detach flag." >&2
        exit 2
      fi
      DETACH_MODE=1
      shift
      ;;
    --warm)
      if [ "$WARM_MODE" -ne 0 ] || [ "$WARM_ONLY" -ne 0 ]; then
        echo "[!] Duplicate warm flag." >&2
        exit 2
      fi
      WARM_MODE=1
      shift
      ;;
    --warm-only)
      if [ "$WARM_MODE" -ne 0 ] || [ "$WARM_ONLY" -ne 0 ]; then
        echo "[!] Duplicate warm flag." >&2
        exit 2
      fi
      WARM_ONLY=1
      shift
      ;;
    -h|--help)
      SHOW_HELP=1
      shift
      ;;
    *)
      echo "[!] Unknown argument: $arg (expected --hf-allow-public-repo, --allow-default-hf-repo, --detach, --private, --push-config, --warm, --warm-only, --hf-repo owner/repo)" >&2
      exit 2
      ;;
  esac
done

if [ "$SHOW_HELP" -eq 1 ]; then
  cat <<'EOF'
Usage: scripts/modal/run_modal_cli.sh [--hf-allow-public-repo] [--allow-default-hf-repo] [--detach] [--private] [--push-config] [--warm] [--warm-only] [--hf-repo owner/repo]

Recommended Modal entrypoint (no release required; SHA/tree checks advisory
unless LEGALIR_STRICT_GATES=1).
  --hf-allow-public-repo   Explicit opt-in to push to an existing PUBLIC HF
                           repo (recorded in manifest). Absent means
                           private-only (fail closed).
  --private                Explicit opt-in to evaluate Private test queries
                           (2,080 queries) instead of default public (1,000 queries).
  --push-config            Use the score-push reranker config
                           (configs/experiments/reranker_lora_v3_push.yaml:
                           listwise, LoRA r=64 cold start, 2 epochs). Default
                           uses configs/experiments/reranker_lora.yaml.
                           LEGALIR_RERANKER_CONFIG env overrides both.
  --warm                   Warm the shared Volume first (CPU-cheap:
                            models + dataset), then dispatch A100. Recommended:
                            the A100 reuses verified cache instead of downloading.
  --warm-only              Only warm the shared Volume (no A100 dispatch).
  --hf-repo owner/repo     Explicit HF repo for artifacts. Wins over
                            HF_REPO_ID env and .env. REQUIRED: without it the
                            wrapper exits 2 instead of silently targeting the
                            previous owner's repo (opt out with
                            --allow-default-hf-repo, offline/dev only).
                            The ID is echoed (not a secret); tokens
                            are never printed.
  --allow-default-hf-repo    Explicit opt-out for offline/dev runs: permit the
                            owner-default HF repo. Never use for a fresh-account
                            production run.
  --detach                 Explicit opt-in to `modal run --detach` (app survives
                           client disconnect). Default is attached: client
                           disconnect terminates remote tasks even with a
                           persistent Volume. Detached changes spending
                           supervision: record app ID, attempt path, start UTC
                           and ceiling; monitor with `modal app logs <id>`;
                           stop with `modal app stop <id> --yes` (verify syntax).
                           Keep the client supervised in attached mode (stable
                           network, machine awake, tmux/screen recommended).
EOF
  exit 0
fi

if [ ! -x "$PYTHON_BIN" ] && [ ! -f "$PYTHON_BIN" ]; then
  echo "[!] PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 2
fi
if [ ! -x "$MODAL_BIN" ] && [ ! -f "$MODAL_BIN" ]; then
  echo "[!] MODAL_BIN not found: $MODAL_BIN (install modal CLI and authenticate)" >&2
  exit 2
fi

# Resolve HF repo ID explicitly BEFORE any cloud/preflight work so a typo
# fails fast: flag > env > .env > owner default. The ID is not a secret;
# echoing it confirms the destination account.
HF_REPO_DEFAULT="dangphuc2109/legalir-task1-reranker"
if [ -z "$HF_REPO_FLAG" ]; then
  if [ -n "${HF_REPO_ID:-}" ]; then
    HF_REPO_FLAG="$HF_REPO_ID"
    HF_REPO_SOURCE="env"
  else
    if _dotenv_val="$(dotenv_hf_repo)"; then
      HF_REPO_FLAG="$_dotenv_val"
      HF_REPO_SOURCE="dotenv"
    else
      HF_REPO_FLAG="$HF_REPO_DEFAULT"
      HF_REPO_SOURCE="default"
    fi
  fi
fi
if ! valid_hf_repo "$HF_REPO_FLAG"; then
  echo "[!] Invalid HF repo ID '$HF_REPO_FLAG' (source=$HF_REPO_SOURCE; expected 'owner/repo')." >&2
  exit 2
fi
if [ "$HF_REPO_SOURCE" = "default" ] && [ "$ALLOW_DEFAULT_HF_REPO" -ne 1 ] && [ "${LEGALIR_ALLOW_DEFAULT_HF_REPO:-}" != "1" ]; then
  echo "[!] BLOCKED: HF repo $HF_REPO_FLAG is the previous owner's default (source=default)." >&2
  echo "    Fresh accounts must pass --hf-repo owner/repo (or set HF_REPO_ID env/.env)." >&2
  echo "    Opt out explicitly with --allow-default-hf-repo only for offline/dev runs." >&2
  exit 2
fi
export HF_REPO_ID="$HF_REPO_FLAG"
echo "[*] HF repo: $HF_REPO_FLAG (source=$HF_REPO_SOURCE)"

# Run label from env or exact local HEAD.
if [ -n "${LEGALIR_COMMIT_SHA:-}" ]; then
  EXPECTED_SHA="$LEGALIR_COMMIT_SHA"
else
  if ! EXPECTED_SHA="$(git rev-parse HEAD 2>/dev/null)"; then
    echo "[!] Unable to resolve local git HEAD. Set LEGALIR_COMMIT_SHA explicitly." >&2
    exit 2
  fi
fi

if ! echo "$EXPECTED_SHA" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "[!] LEGALIR_COMMIT_SHA must be an exact 40-char lowercase SHA, got '$EXPECTED_SHA'." >&2
  exit 2
fi

# Reject mismatch between selected SHA and local HEAD.
LOCAL_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
if [ -n "$LOCAL_HEAD" ] && [ "$LOCAL_HEAD" != "$EXPECTED_SHA" ]; then
  echo "[!] Selected SHA $EXPECTED_SHA does not match local HEAD $LOCAL_HEAD; refusing to dispatch." >&2
  exit 2
fi

# Require clean tracked/untracked tree (ignored files excluded by porcelain; .env is credentials).
DIRTY_FILES="$(git status --porcelain=v1 2>/dev/null | grep -v '^[?][?] \.env$' || true)"
if [ -n "$DIRTY_FILES" ]; then
  echo "[!] Working tree is dirty; commit or stash before Modal dispatch." >&2
  echo "$DIRTY_FILES" >&2
  exit 2
fi

# CPU provenance preflight before any cloud command.
echo "[*] Local CPU provenance preflight for $EXPECTED_SHA..."
if ! "$PYTHON_BIN" scripts/colab/bootstrap.py --expected-sha "$EXPECTED_SHA"; then
  echo "[!] Local CPU provenance failed; aborting before Modal dispatch." >&2
  exit 1
fi

# Forward explicit consent once; absent stays private-only.
MODAL_ARGS=()
if [ -n "$HF_PUBLIC_FLAG" ]; then
  MODAL_ARGS+=("$HF_PUBLIC_FLAG")
else
  MODAL_ARGS+=("--no-hf-allow-public-repo")
fi
if [ -n "$PRIVATE_FLAG" ]; then
  MODAL_ARGS+=("$PRIVATE_FLAG")
fi
if [ -n "$PUSH_CONFIG" ]; then
  MODAL_ARGS+=("--push-config")
fi
MODAL_ARGS+=("--hf-repo" "$HF_REPO_FLAG")

DETACH_OPT=""
if [ "$DETACH_MODE" -eq 1 ]; then
  DETACH_OPT="--detach"
fi

# CPU-cheap Volume warm (models + dataset) before any A100 billing.
# Warm and training jobs receive the same explicit HF repo so adapter warm
# (when enabled) and the final upload target the same account.
if [ "$WARM_MODE" -eq 1 ] || [ "$WARM_ONLY" -eq 1 ]; then
  echo "[*] Warming shared Volume cache on CPU (no GPU billed)..."
  if ! HF_REPO_ID="$HF_REPO_FLAG" LEGALIR_COMMIT_SHA="$EXPECTED_SHA" "$MODAL_BIN" run scripts/modal/warm_volume.py --hf-repo "$HF_REPO_FLAG"; then
    echo "[!] Volume warm failed; aborting before A100 dispatch." >&2
    exit 1
  fi
  echo "[+] Shared Volume cache ready."
fi
if [ "$WARM_ONLY" -eq 1 ]; then
  echo "[*] --warm-only: warm complete, no A100 dispatch."
  exit 0
fi

echo "[*] Dispatching to Modal for $EXPECTED_SHA ${MODAL_ARGS[*]}..."
echo "[*] NOTE: invoking Modal can build an image before remote preflight runs;"
echo "    local checks cannot validate remote secrets or GPU. Remote repeats"
echo "    checkout/provenance, HF access, dataset, then train. No auto-retry."
if [ "$DETACH_MODE" -eq 1 ]; then
  echo '[!] DETACHED mode: app survives client disconnect (modal run --detach).'
  echo '    This changes spending supervision: record app ID, Volume attempt path'
  echo '    (/root/legalir_volume/<sha>/attempts/<id>/), start UTC, and approved ceiling.'
  echo '    Monitor with: modal app logs <app-id>  Stop with: modal app stop <app-id> --yes'
  echo '    (confirm current installed CLI syntax). Independently confirm termination.'
else
  echo "[*] ATTACHED mode (default): keep this client connected until the remote job returns."
  echo "    Disconnecting the client terminates remote tasks (ephemeral app semantics);"
  echo "    a persistent Volume does not keep unfinished training alive."
  echo "    Use stable network, keep the machine awake, tmux/screen recommended."
  echo "    Record app ID, attempt path, start UTC, and ceiling. Confirm app"
  echo "    termination via Modal, not just client exit. No automatic relaunch."
fi
if [ -n "$DETACH_OPT" ]; then
  HF_REPO_ID="$HF_REPO_FLAG" LEGALIR_COMMIT_SHA="$EXPECTED_SHA" "$MODAL_BIN" run --detach scripts/modal/run_modal_a100.py "${MODAL_ARGS[@]}"
else
  HF_REPO_ID="$HF_REPO_FLAG" LEGALIR_COMMIT_SHA="$EXPECTED_SHA" "$MODAL_BIN" run scripts/modal/run_modal_a100.py "${MODAL_ARGS[@]}"
fi
