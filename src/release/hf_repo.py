"""Explicit Hugging Face repo-ID resolution for fresh-account Modal runs.

Problem: local ``.env`` ``HF_REPO_ID`` was never loaded/forwarded while the
remote container defaulted to the previous owner's repo. Setting ``.env``
alone silently pushed (or failed to push) to the wrong repo.

Contract (local -> remote, explicit and testable):
  explicit CLI flag (``--hf-repo``) wins,
  then ``HF_REPO_ID`` env,
  then ``<repo>/.env`` ``HF_REPO_ID`` line,
  then :data:`HF_REPO_DEFAULT` (backwards compatible).

Only the repo ID (``owner/repo``, not a secret) is ever logged. Tokens are
never read or printed here. New repos stay private by default; this module
never opts into public visibility.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

HF_REPO_DEFAULT = "dangphuc2109/legalir-task1-reranker"

# owner/repo, each part 1..96 chars of [A-Za-z0-9_.-]; exactly one slash.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_MAX_LEN = 96


def validate_hf_repo_id(repo_id: str | None) -> str:
    """Return the stripped repo ID or raise ValueError when invalid."""
    cleaned = str(repo_id or "").strip()
    if not cleaned:
        raise ValueError(
            "Invalid HF repo ID '' (expected 'owner/repo', e.g. 'my-user/legalir-task1-reranker')."
        )
    if len(cleaned) > _MAX_LEN:
        raise ValueError(
            f"Invalid HF repo ID '{cleaned}': longer than {_MAX_LEN} chars."
        )
    if cleaned.count("/") != 1:
        raise ValueError(
            f"Invalid HF repo ID '{cleaned}': expected exactly one '/' as 'owner/repo'."
        )
    owner, repo = cleaned.split("/", 1)
    if not owner or not repo:
        raise ValueError(
            f"Invalid HF repo ID '{cleaned}': owner and repo must both be non-empty."
        )
    if not _HF_REPO_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid HF repo ID '{cleaned}': only [A-Za-z0-9_.-] and one '/' are allowed."
        )
    if cleaned != cleaned.strip() or " " in cleaned or "\t" in cleaned:
        raise ValueError(f"Invalid HF repo ID '{cleaned}': whitespace is not allowed.")
    return cleaned


def parse_dotenv_file(path: str | Path) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser for ``.env`` files (no secret logging).

    Handles ``export KEY=...``, surrounding single/double quotes, trailing
    comments outside quotes, and ignores blank/comment lines. Never executes
    the file and never prints values.
    """
    out: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return out
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        val = val.strip()
        # Quoted value possibly followed by a trailing comment:
        #   KEY="value"  # comment  ->  value
        if val and val[0] in ("'", '"'):
            quote = val[0]
            end = val.find(quote, 1)
            if end != -1:
                out[key] = val[1:end]
                continue
            # Unterminated quote: strip the leading quote and continue.
            val = val[1:].strip()
        else:
            # Split on ' #' (hash preceded by whitespace) to keep '#' inside values.
            cut = re.split(r"\s+#", val, maxsplit=1)
            val = cut[0].strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def dotenv_hf_repo_id(repo_root: str | Path | None) -> str | None:
    """Return ``HF_REPO_ID`` from ``<repo_root>/.env`` if present, else None."""
    if not repo_root:
        return None
    vals = parse_dotenv_file(Path(repo_root) / ".env")
    raw = (vals.get("HF_REPO_ID") or "").strip()
    return raw or None


def resolve_hf_repo_id(
    explicit: str | None = None,
    env: dict[str, str] | os._Environ | None = None,
    repo_root: str | Path | None = None,
    default: str = HF_REPO_DEFAULT,
    allow_default: bool = True,
) -> tuple[str, str]:
    """Resolve the HF repo ID with explicit precedence.

    Returns ``(repo_id, source)`` where source is one of
    ``explicit`` / ``env`` / ``dotenv`` / ``default``.

    Non-default sources are validated; invalid values raise ValueError
    mentioning the source. The default is returned as-is for backwards
    compatibility (it is a known-valid ID) unless ``allow_default`` is
    False, in which case falling back to the default raises ValueError —
    local dispatch wrappers use this so a fresh-account dry-run/run fails
    loudly instead of silently targeting the previous owner's repo.
    """
    mapping = os.environ if env is None else env

    def _get(name: str) -> str:
        try:
            return str(mapping.get(name, "") or "")
        except Exception:
            return ""

    if str(explicit or "").strip():
        cleaned = validate_hf_repo_id(explicit)
        return cleaned, "explicit"
    env_raw = _get("HF_REPO_ID").strip()
    if env_raw:
        try:
            return validate_hf_repo_id(env_raw), "env"
        except ValueError as exc:
            raise ValueError(f"Invalid HF_REPO_ID from env: {exc}") from None
    dot_raw = dotenv_hf_repo_id(repo_root)
    if dot_raw:
        try:
            return validate_hf_repo_id(dot_raw), "dotenv"
        except ValueError as exc:
            raise ValueError(f"Invalid HF_REPO_ID from .env: {exc}") from None
    if not allow_default:
        raise ValueError(
            "HF repo ID is not set explicitly (no --hf-repo flag, HF_REPO_ID env, "
            "or .env entry); refusing the previous owner's default repo. "
            "Pass --hf-repo owner/repo (or set HF_REPO_ID) for the new account, "
            "or opt out explicitly with --allow-default-hf-repo."
        )
    return str(default), "default"


def default_allowed(env: dict[str, str] | os._Environ | None = None) -> bool:
    """Escape hatch for offline tests/dev: LEGALIR_ALLOW_DEFAULT_HF_REPO=1."""
    mapping = os.environ if env is None else env
    try:
        return str(mapping.get("LEGALIR_ALLOW_DEFAULT_HF_REPO", "")).strip() == "1"
    except Exception:
        return False
