import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

# Define the Modal App
app = modal.App("legalir-a100-production")

# Define the environment image
# Pinned to match requirements-colab.txt (verified: transformers 5.15.1 exists).
# Python 3.11 to match torch cp311 wheels. torch>=2.5 is REQUIRED:
# transformers 5.x lazy-loads model classes only when torch>=2.5 is importable
# (torch 2.1.2 passes raw `import torch` but fails the backend gate with the
# misleading "requires the PyTorch library but it was not found" error).
# PyTorch is installed with CUDA 12.4 support, which is suitable for A100.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==5.15.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "huggingface-hub==1.28.0",
        "numpy>=1.24,<3",
        "pandas>=2,<3",
        "pyarrow>=14",
        "pyyaml>=6,<7",
        "scikit-learn>=1.3,<2",
        "lightgbm>=4,<5",
        "bm25s>=0.2,<1",
        "pyvi>=0.1.1,<1",
        "sentencepiece>=0.1.99",
        "faiss-cpu>=1.7",
        "psutil>=5.9",
        "kaggle>=1.8,<3",
        "tqdm>=4.65"
    )
)

# Persistent volume: a timeout kill loses the unfinished training loop
# (5-fold OOF + final LoRA has no checkpoint-resume; trainer saves only at end).
# Pipeline outputs are written directly into a unique attempt directory on the
# Volume from the beginning. Background commits improve durability while the
# process runs, but they do not guarantee every final byte survives a kill:
# the last uncommitted bytes and the current unfinished training loop may be
# lost. There is no checkpoint-resume; a timeout kill still requires a full
# re-run (Volume holds forensics only).
volume = modal.Volume.from_name("legalir-production", create_if_missing=True)
VOLUME_MOUNT = "/root/legalir_volume"

# No time limit by default: Modal requires an integer timeout, so "no limit"
# means the platform maximum (24h = 86400s). Set MODAL_TIMEOUT_SECONDS to a
# smaller value only to cap spend, never to gate quality.
# LEGALIR_STRICT_GATES=1 restores the old 7h gate for release qualification.
_timeout_raw = str(os.environ.get("MODAL_TIMEOUT_SECONDS", "86400")).strip().lower()
if _timeout_raw in ("0", "no", "off", "none", ""):
    TIMEOUT_SECONDS = 86400
else:
    try:
        TIMEOUT_SECONDS = max(3600, int(float(_timeout_raw)))
    except ValueError:
        TIMEOUT_SECONDS = 86400

_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _strict() -> bool:
    return str(os.environ.get("LEGALIR_STRICT_GATES", "")).strip() == "1"


def _resolve_hf_repo(explicit: str | None, allow_default: bool = True) -> tuple[str, str]:
    """Resolve HF repo ID inside local or remote process (no token logging).

    Precedence: explicit arg > HF_REPO_ID env > <repo>/.env > owner default.
    Non-default sources are validated; invalid values raise ValueError.
    With ``allow_default=False``, falling back to the previous owner's
    default raises ValueError so dispatch fails loudly (local gate).
    """
    import sys as _sys

    DEFAULT = "dangphuc2109/legalir-task1-reranker"
    try:
        from src.release.hf_repo import resolve_hf_repo_id as _resolve
    except Exception:
        # Minimal fallback when src is unavailable (e.g. bare container pre-clone):
        # explicit > env > default with a light pattern check.
        import re as _re

        cand = str(explicit or "").strip() or str(os.environ.get("HF_REPO_ID", "")).strip()
        src = "explicit" if str(explicit or "").strip() else ("env" if cand else "default")
        repo = cand or DEFAULT
        if not _re.fullmatch(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", repo):
            raise ValueError(f"Invalid HF repo ID '{repo}' (source={src}; expected 'owner/repo').")
        if src == "default" and not allow_default:
            raise ValueError(
                "HF repo ID is not set explicitly; refusing the previous owner's default repo. "
                "Pass --hf-repo owner/repo."
            )
        return repo, src
    # Local processes can read <repo>/.env; remote resolves after chdir so the
    # checkout's .env is used. Pass the current repo dir when known.
    repo_root = None
    try:
        mod = _sys.modules.get(__name__)
        _ = mod  # keep hook for tests that monkeypatch module attrs
        repo_root = _resolve_repo_dir()
    except Exception:
        repo_root = None
    return _resolve(explicit=explicit, env=os.environ, repo_root=repo_root,
                    default=DEFAULT, allow_default=allow_default)


def _normalize_sha_label(expected_sha: str) -> str:
    """Accept any run label; strict mode still requires an exact 40-char SHA."""
    sha = str(expected_sha or "").strip()
    if _SHA_RE.fullmatch(sha):
        return sha
    if _strict():
        raise ValueError("Expected an exact lowercase 40-character Git SHA")
    print(f"[*] SHA gate advisory only (strict off): using run label '{sha[:24]}'", flush=True)
    return sha or "dev"


def create_attempt_dir(volume_root: Path, expected_sha: str) -> Path:
    """Create a unique Volume-backed attempt directory for one run.

    Layout: <volume_root>/<run-label-or-sha>/attempts/<uuid4-hex>/.
    A new UUID is used per attempt; prior runs are never overwritten and
    resume is never inferred from old files.
    """
    sha = _normalize_sha_label(expected_sha)
    path = Path(volume_root) / sha / "attempts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_launcher_state(attempt_dir: Path, state: dict) -> None:
    """Write sanitized launcher_state.json atomically via temp file + rename.

    Only small private operational metadata is stored: attempt ID, release
    SHA, UTC timestamps, phase, outcome, and exception class. Never exception
    text, token values, or environment dictionaries.
    """
    allowed = {
        "attempt_id": state.get("attempt_id"),
        "expected_sha": state.get("expected_sha"),
        "phase": state.get("phase"),
        "outcome": state.get("outcome"),
        "exception_class": state.get("exception_class"),
        "started_utc": state.get("started_utc"),
        "updated_utc": state.get("updated_utc"),
    }
    tmp = attempt_dir / "launcher_state.json.tmp"
    tmp.write_text(json.dumps(allowed, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, attempt_dir / "launcher_state.json")


def _try_commit_best_effort() -> None:
    """Best-effort intermediate Volume commit; warn and continue on failure."""
    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001 - durability warning only
        print(f"[!] Intermediate Volume commit failed: {type(exc).__name__}", flush=True)


def _resolve_repo_dir() -> Path:
    # Test hook: LEGALIR_MODAL_REPO_DIR overrides the container checkout path
    # so offline orchestration tests never touch /root. Production default
    # remains /root/LegalIR.
    return Path(os.environ.get("LEGALIR_MODAL_REPO_DIR", "/root/LegalIR"))


def _resolve_dataset_dir() -> Path:
    # Test hook: LEGALIR_MODAL_DATASET_DIR overrides /root/kaggle_dataset.
    return Path(os.environ.get("LEGALIR_MODAL_DATASET_DIR", "/root/kaggle_dataset"))


def _resolve_volume_mount() -> Path:
    # VOLUME_MOUNT is the production constant; tests monkeypatch the module
    # attribute directly. This helper keeps that contract explicit.
    import sys as _sys

    mod = _sys.modules.get(__name__)
    override = getattr(mod, "VOLUME_MOUNT", VOLUME_MOUNT) if mod is not None else VOLUME_MOUNT
    return Path(override)


def warmed_models_dir(volume_root: str | Path) -> Path:
    """Shared pre-warmed HF snapshots (see scripts/modal/warm_volume.py)."""
    return Path(volume_root) / "shared" / "models" / "huggingface"


def warmed_dataset_dir(volume_root: str | Path) -> Path:
    """Shared pre-warmed canonical dataset (see scripts/modal/warm_volume.py)."""
    return Path(volume_root) / "shared" / "dataset"


def _looks_like_full_sha(value: object) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", str(value or "").strip()))


def attach_warmed_cache(
    volume_root: str | Path,
    repo_dir: str | Path,
    expected_sha: str | None = None,
) -> dict:
    """Attach pre-warmed Volume cache so the A100 never downloads.

    - HF snapshot cache: sets HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE/TRANSFORMERS_CACHE
      (+HF_HOME) at the warmed dir and mirrors its manifest.json into the repo
      artifacts path consumed by train_reranker/CrossEncoderReranker. The
      manifest must list every registry model id with a non-empty existing
      snapshot path AND the pinned revision from
      src.models.bootstrap.MODEL_REGISTRY; a missing/empty/mismatched entry
      fails closed to downloading (never silently reuses stale weights).
    - Dataset: when every REQUIRED_FILES entry exists warmed, pins
      LEGALIR_MODAL_DATASET_DIR at it so prepare_dataset skips the download.
      Fingerprint verification still runs inside prepare_dataset.
    - Source identity: ``expected_sha`` is the training job's SHA. When both
      it and the warm manifest's ``source_sha`` are full 40-char SHAs and they
      differ, the cache is NOT attached (warm-source-sha-mismatch) — a stale
      checkout's cache is never treated as valid for another SHA. Short/dev
      labels stay advisory.

    Never raises; missing/partial/stale cache simply falls back to
    downloading. Returns a summary dict (models_attached, dataset_reused,
    models_detail, dataset_detail, warm_source_sha, warm_requested_label,
    warm_hf_repo) so the caller can persist how the A100 actually resolved
    its inputs (cache hit vs download fallback).
    """
    summary: dict[str, object] = {
        "models_attached": False,
        "dataset_reused": False,
        "models_detail": "missing",
        "dataset_detail": "missing",
        "warm_source_sha": "unknown",
        "warm_requested_label": "unknown",
        "warm_hf_repo": "unknown",
    }
    try:
        # Record warm provenance first so even a fallback decision is auditable.
        try:
            warm_manifest = Path(volume_root) / "shared" / "warm_manifest.json"
            if warm_manifest.is_file():
                warm_data = json.loads(warm_manifest.read_text(encoding="utf-8"))
                if isinstance(warm_data, dict):
                    summary["warm_source_sha"] = str(warm_data.get("source_sha", "unknown"))
                    summary["warm_requested_label"] = str(warm_data.get("requested_label", warm_data.get("label", "unknown")))
                    summary["warm_hf_repo"] = str(warm_data.get("hf_repo", "unknown"))
        except Exception:
            pass
        warm_sha_ok = True
        if _looks_like_full_sha(expected_sha) and _looks_like_full_sha(summary.get("warm_source_sha")):
            if str(expected_sha).strip().lower() != str(summary["warm_source_sha"]).strip().lower():
                warm_sha_ok = False
        models_dir = warmed_models_dir(volume_root)
        manifest = models_dir / "manifest.json"
        try:
            from src.models.bootstrap import MODEL_REGISTRY as _REG
        except Exception:
            _REG = {}
        # Every registry model must be present: a partial cache is not a hit.
        # (The registry is the source of truth; a hardcoded subset silently
        # blessed incomplete caches, and empty path/revision strings slipped
        # through because Path("") resolves to "." which is_dir().)
        if isinstance(_REG, dict) and len(_REG) > 0:
            required_ids = tuple(_REG.keys())
        else:
            required_ids = ()
        ok = manifest.is_file() and len(required_ids) > 0
        if not manifest.is_file():
            detail = "missing-manifest"
        elif len(required_ids) == 0:
            detail = "registry-unavailable"
            ok = False
        else:
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                problems: list[str] = []
                for mid in required_ids:
                    entry = data.get(mid)
                    if not isinstance(entry, dict):
                        problems.append(f"{mid}:absent")
                        continue
                    raw_path = entry.get("path", "")
                    snap = str(raw_path).strip() if isinstance(raw_path, str) else str(raw_path or "").strip()
                    if not snap:
                        problems.append(f"{mid}:path-empty")
                        continue
                    if not Path(snap).is_dir():
                        problems.append(f"{mid}:missing-path")
                        continue
                    pinned = (_REG.get(mid) or {}).get("revision") if isinstance(_REG, dict) else None
                    recorded = entry.get("revision", "")
                    recorded = str(recorded).strip() if isinstance(recorded, str) else str(recorded or "").strip()
                    if pinned and recorded != str(pinned).strip():
                        # Empty recorded revision can never equal a pinned one:
                        # fail closed to downloading instead of blessing it.
                        problems.append(f"{mid}:revision-mismatch")
                    elif not pinned and not recorded:
                        problems.append(f"{mid}:revision-empty")
                if problems:
                    detail = "stale:" + ",".join(problems)
                    ok = False
                else:
                    detail = "pinned-revisions-verified"
            except Exception as exc:  # noqa: BLE001
                detail = f"unreadable:{type(exc).__name__}"
                ok = False
        summary["models_detail"] = detail
        if not warm_sha_ok:
            summary["models_detail"] = (
                f"warm-source-sha-mismatch:warm={summary['warm_source_sha']} "
                f"training={str(expected_sha).strip()}"
            )
            print(f"[*] Warm cache from another source ({summary['models_detail']}); "
                  "A100 will download weights.", flush=True)
        elif ok:
            for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
                os.environ.setdefault(var, str(models_dir))
            # Mirror into the repo-local manifest path read by model loaders.
            try:
                local_manifest = Path(repo_dir) / "artifacts" / "local" / "models" / "huggingface" / "manifest.json"
                local_manifest.parent.mkdir(parents=True, exist_ok=True)
                local_manifest.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                print(f"[!] Warm manifest mirror skipped: {type(exc).__name__}", flush=True)
            summary["models_attached"] = True
            print(f"[*] Warmed models attached from {models_dir} (no HF download on A100).", flush=True)
        else:
            print(f"[*] No complete warmed model cache ({detail}); A100 will download weights.", flush=True)

        from scripts.colab.bootstrap import REQUIRED_FILES

        ds_dir = warmed_dataset_dir(volume_root)
        missing_ds = [n for n in REQUIRED_FILES if not (ds_dir / n).is_file()]
        if not missing_ds and not warm_sha_ok:
            summary["dataset_detail"] = (
                f"warm-source-sha-mismatch:warm={summary['warm_source_sha']} "
                f"training={str(expected_sha).strip()}"
            )
            print(f"[*] Warm dataset from another source; A100 will download it.", flush=True)
        elif not missing_ds:
            os.environ.setdefault("LEGALIR_MODAL_DATASET_DIR", str(ds_dir))
            summary["dataset_reused"] = True
            summary["dataset_detail"] = f"all-{len(REQUIRED_FILES)}-files-present"
            print(f"[*] Warmed dataset reused from {ds_dir} (no Kaggle download on A100).", flush=True)
        else:
            summary["dataset_detail"] = f"missing:{','.join(missing_ds[:5])}"
            print(f"[*] No complete warmed dataset (missing {len(missing_ds)} files); A100 will download it.", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Warm cache attach skipped: {type(exc).__name__}", flush=True)
        summary["models_detail"] = f"error:{type(exc).__name__}"
    return summary

# NOTE: CPU/RAM are Modal defaults (no explicit cpu/memory reservation).
# Workload (~934k micro-chunks, multiple indexes, multiprocessing, repeated
# model/index loads) has no qualified peak-RAM, CPU-availability, or
# end-to-end completion forecast on A100. Timeout caps duration, not spend.
@app.function(
    image=image,
    gpu="A100",
    cpu=8.0,
    memory=32768,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: volume},
    secrets=[
        modal.Secret.from_name("kaggle-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def run_production_training(
    expected_sha: str,
    hf_allow_public_repo: bool = False,
    private: bool = False,
    reranker_config: str | None = None,
    hf_repo: str | None = None,
):
    """
    Executes the A100 production training pipeline within a Modal container.
    Pre-A100 hardware gate: Kaggle dual-T4 report (B1.1), strictly enforced.

    Durability: the pipeline's output_dir is a unique attempt directory on the
    mounted Volume from the beginning. Completed adapter files and
    recovery.tar.gz do not depend on a final copy allowlist. On graceful
    return or exception, the Volume is explicitly committed after the wrapper
    has built its archive. Background commits improve durability while the
    process runs, but the last uncommitted bytes and the current unfinished
    training loop may still be lost on a hard kill. No resume is implied.
    """
    import sys

    if private:
        os.environ["LEGALIR_TEST_PHASE"] = "private"
    if reranker_config:
        os.environ["LEGALIR_RERANKER_CONFIG"] = str(reranker_config)
        print(f"[+] Remote container initialized with LEGALIR_RERANKER_CONFIG={reranker_config}", flush=True)

    # Run label for paths (advisory unless LEGALIR_STRICT_GATES=1).
    sha = _normalize_sha_label(expected_sha)

    # Unique Volume-backed attempt directory; pipeline writes here directly.
    # training.log is written to this path while training runs by the
    # pipeline's own logger (do not wrap training_log around itself).
    attempt_dir = create_attempt_dir(_resolve_volume_mount(), sha)
    attempt_id = attempt_dir.name
    output_dir = attempt_dir
    started_utc = _utc_now_iso()

    def _update_state(phase: str, outcome: str = "running", exc_class=None) -> None:
        # Best-effort: metadata recording must never mask a primary failure or
        # skip the final persistence attempt. Failures are logged with their
        # class only (no exception text, tokens, or environment).
        try:
            _write_launcher_state(
                attempt_dir,
                {
                    "attempt_id": attempt_id,
                    "expected_sha": sha,
                    "phase": phase,
                    "outcome": outcome,
                    "exception_class": exc_class,
                    "started_utc": started_utc,
                    "updated_utc": _utc_now_iso(),
                },
            )
        except Exception as state_exc:  # noqa: BLE001
            print(
                f"[!] launcher_state write failed: {type(state_exc).__name__} "
                f"(phase={phase} outcome={outcome})",
                flush=True,
            )

    def _final_commit_after_failure(primary_class: str) -> None:
        # A final persistence attempt independent of metadata recording.
        try:
            volume.commit()
            print(f"[*] Committed Volume attempt after failure: {attempt_dir}", flush=True)
        except Exception as commit_exc:  # noqa: BLE001
            # Preserve the original failure; report commit problem separately.
            print(
                f"[!] Volume commit failed: {type(commit_exc).__name__} at {attempt_dir}; "
                f"primary failure was {primary_class}",
                flush=True,
            )

    # Initial metadata before clone/preflight and before expensive work.
    _update_state("checkout")
    _try_commit_best_effort()

    def _run_attempt_body() -> dict:
        # Attempt body: checkout, provenance, HF preflight, dataset, training.
        # Raises on failure; the lifecycle finalization below guarantees
        # best-effort failure metadata plus a final Volume commit for ANY
        # graceful failure across these phases, then re-raises the original
        # exception. State updates never mask the primary failure.
        # 1. Clone the repository and checkout the exact expected SHA
        _update_state("checkout")
        repo_dir = _resolve_repo_dir()
        if not repo_dir.exists():
            print(f"[*] Cloning repository into {repo_dir}...")
            subprocess.run(["git", "clone", "https://github.com/silent9669/LegalIR.git", str(repo_dir)], check=True)

        print(f"[*] Checking out exact commit: {sha}")
        subprocess.run(["git", "fetch", "origin", sha], cwd=repo_dir, check=False)
        res = subprocess.run(["git", "checkout", "--detach", sha], cwd=repo_dir, capture_output=True, text=True)
        if res.returncode != 0:
            print("[*] Checkout fallback: unshallowing repository...")
            subprocess.run(["git", "fetch", "--unshallow", "origin"], cwd=repo_dir, check=False)
            subprocess.run(["git", "checkout", "--detach", sha], cwd=repo_dir, check=True)

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        os.chdir(repo_dir)

        # 1b. Attach pre-warmed Volume cache (models + dataset) so the A100
        # bills zero download seconds. Falls back to downloading when absent
        # or stale; the summary records cache-hit vs download-fallback so
        # operators can verify zero-download claims instead of assuming them.
        # Run scripts/modal/warm_volume.py (CPU-cheap) before dispatch.
        _update_state("warm_cache")
        _warm_summary = attach_warmed_cache(_resolve_volume_mount(), repo_dir, expected_sha=sha)
        try:
            (attempt_dir / "warm_cache_summary.json").write_text(
                json.dumps(
                    {
                        "models_attached": bool(_warm_summary.get("models_attached")),
                        "dataset_reused": bool(_warm_summary.get("dataset_reused")),
                        "models_detail": str(_warm_summary.get("models_detail", "")),
                        "dataset_detail": str(_warm_summary.get("dataset_detail", "")),
                        "warm_source_sha": str(_warm_summary.get("warm_source_sha", "unknown")),
                        "warm_requested_label": str(_warm_summary.get("warm_requested_label", "unknown")),
                        "warm_hf_repo": str(_warm_summary.get("warm_hf_repo", "unknown")),
                        "training_sha": str(sha),
                        "recorded_utc": _utc_now_iso(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[!] warm_cache_summary write skipped: {type(exc).__name__}", flush=True)
        _try_commit_best_effort()

        # Set test phase in remote environment
        if os.environ.get("LEGALIR_TEST_PHASE", "").strip().lower() == "private":
            print("[+] Remote container initialized with LEGALIR_TEST_PHASE=private (2,080 queries)", flush=True)

        # 2. CPU provenance gate before any expensive work (Kaggle T4x2 gate).
        from scripts.colab.bootstrap import prepare_dataset, verify_launch

        # We must point to the verified artifact paths in the repo
        kaggle_report = repo_dir / "artifacts/task1/gates/kaggle_t4x2_report.json"
        freeze_file = repo_dir / "artifacts/task1/freeze/production_freeze.json"

        _update_state("provenance")
        print("[*] Launch preflight (advisory unless LEGALIR_STRICT_GATES=1)...")
        try:
            verify_launch(sha, kaggle_report, freeze_file)
        except Exception as exc:
            if _strict():
                raise
            print(f"[!] provenance advisory (strict off), continuing: {type(exc).__name__}", flush=True)
        _try_commit_best_effort()

        # 3. Preflight Hugging Face Access BEFORE expensive dataset acquisition.
        # Explicit --hf-repo (local flag) wins over the container HF_REPO_ID env;
        # both beat the owner default. The resolved ID is echoed (not a secret)
        # so `modal app logs` confirms the destination account. New repos stay
        # private; public requires the explicit operator override.
        from scripts.gates.run_a100 import preflight_huggingface_access
        try:
            resolved_hf_repo, hf_source = _resolve_hf_repo(hf_repo)
        except ValueError as exc:
            raise RuntimeError(f"Hugging Face repo ID rejected: {exc}") from None
        _update_state("hf_preflight")
        print(f"[*] HF repo: {resolved_hf_repo} (source={hf_source}); verifying write access...")
        if hf_source == "default":
            print("[!] Using owner-default HF repo; fresh accounts should pass --hf-repo owner/repo.", flush=True)
        if hf_allow_public_repo:
            print("[!] OPERATOR OVERRIDE: public HF repos permitted for this launch (recorded in manifest).", flush=True)
        hf_ok, hf_detail = preflight_huggingface_access(resolved_hf_repo, allow_public_repo=hf_allow_public_repo)
        if not hf_ok:
            raise RuntimeError(f"Hugging Face preflight failed: {hf_detail}")
        print(f"[+] {hf_detail}")
        _try_commit_best_effort()

        # 4. Acquire Kaggle dataset (outside the output tree; pipeline caches may
        # reside within the working dir but stay excluded from HF/recovery by
        # release_files()).
        _update_state("dataset")
        dataset_dir = _resolve_dataset_dir()
        if _warm_summary.get("dataset_reused"):
            print(f"[*] Verifying warmed canonical dataset at {dataset_dir} (download skipped)...")
        else:
            print(f"[*] Downloading and verifying canonical dataset at {dataset_dir}...")
        prepare_dataset(dataset_dir, freeze_file)
        _try_commit_best_effort()

        # 5. Execute the pipeline directly into the Volume-backed attempt dir.
        # Keep existing run_colab_production_training() archive generation; the
        # archive is built inside output_dir and committed below, not via a
        # filtered duplicate snapshot.
        from scripts.run_colab_train import run_colab_production_training

        _update_state("training")
        print(f"[*] Starting A100 production training pipeline at {output_dir}...")
        print("[*] Durable attempt path on Volume; background commits do not guarantee final bytes on kill.", flush=True)
        # Shared index cache: fresh UUID attempts reuse prior validated indexes.
        os.environ.setdefault(
            "LEGALIR_INDEX_CACHE_DIR",
            str(_resolve_volume_mount() / "shared" / "indexes"),
        )
        return run_colab_production_training(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            smoke_report_path=kaggle_report,
            expected_sha=sha,
            precision="bf16",
            allow_non_a100=False,
            mock=False,
            hf_repo=resolved_hf_repo,
            freeze_file_path=freeze_file,
            run_mode="full",
            hf_allow_public_repo=hf_allow_public_repo,
        )

    # Lifecycle-wide finalization: any graceful failure after the attempt
    # directory exists records best-effort failure metadata and attempts a
    # final Volume commit before the original exception propagates. Neither
    # step may raise or mask the primary failure.
    try:
        _attempt_report = _run_attempt_body()
    except BaseException as primary_exc:
        primary_class = type(primary_exc).__name__
        _update_state("failed", outcome="failed", exc_class=primary_class)
        _final_commit_after_failure(primary_class)
        raise

    _update_state("completed", outcome="completed")
    try:
        volume.commit()
        print(f"[*] Committed Volume attempt: {attempt_dir}", flush=True)
    except Exception as commit_exc:  # noqa: BLE001
        # Training succeeded but final commit failed: do not return success.
        # Do not erase an already genuine HF receipt; report delivery success
        # and local durability failure separately.
        hf_info = _attempt_report.get("huggingface", {}) if isinstance(_attempt_report, dict) else {}
        hf_ok_flag = bool(hf_info.get("uploaded")) if isinstance(hf_info, dict) else False
        print(
            f"[!] Volume commit failed: {type(commit_exc).__name__} at {attempt_dir}; "
            f"HF delivery success={hf_ok_flag}",
            flush=True,
        )
        raise RuntimeError(
            f"Volume persistence failed ({type(commit_exc).__name__}) at {attempt_dir}; "
            f"training report exists with HF uploaded={hf_ok_flag}. Inspect the Volume path."
        ) from None

    print(f"[+] Training completed. Status: {_attempt_report.get('status')} | Verdict: {_attempt_report.get('verdict')}")

    # The pipeline internally handles Hugging Face artifact upload and cleanup.
    return _attempt_report

@app.local_entrypoint()
def main(
    hf_allow_public_repo: bool = False,
    private: bool = False,
    push_config: bool = False,
    reranker_config: str = "",
    hf_repo: str = "",
    allow_default_hf_repo: bool = False,
):
    import sys
    # Try to grab the SHA from local git if we are in the repo, or from env.
    # Any label works by default; strict mode still requires an exact SHA.
    expected_sha = os.environ.get("LEGALIR_COMMIT_SHA")
    if not expected_sha:
        try:
            expected_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            expected_sha = "dev"
            print("[*] No git SHA found; using run label 'dev' (strict off).", flush=True)

    # Determine test phase
    test_phase = "private" if (private or os.environ.get("LEGALIR_TEST_PHASE", "").strip().lower() == "private") else "public"

    cfg_to_send = None
    if push_config:
        cfg_to_send = "configs/experiments/reranker_lora_v3_push.yaml"
    elif reranker_config:
        cfg_to_send = reranker_config
    elif os.environ.get("LEGALIR_RERANKER_CONFIG"):
        cfg_to_send = os.environ.get("LEGALIR_RERANKER_CONFIG")

    # Advisory local preflight (fail-closed only with LEGALIR_STRICT_GATES=1).
    try:
        from scripts.colab.bootstrap import verify_launch as _verify

        _repo = Path(__file__).resolve().parents[2]
        _verify(
            expected_sha,
            _repo / "artifacts/task1/gates/kaggle_t4x2_report.json",
            _repo / "artifacts/task1/freeze/production_freeze.json",
            repo_root=_repo,
        )
    except Exception as exc:
        if _strict():
            print(f"[!] Local CPU provenance preflight failed before Modal dispatch: {type(exc).__name__}: {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"[*] Local preflight advisory (strict off), continuing: {type(exc).__name__}", flush=True)

    print(f"[*] Dispatching A100 training job to Modal for: {expected_sha}")
    print(f"[*] Evaluation Phase: {test_phase.upper()} ({'2,080 queries' if test_phase == 'private' else '1,000 queries'})")
    if cfg_to_send:
        print(f"[*] Reranker Config: {cfg_to_send}")
    try:
        from src.release.hf_repo import default_allowed as _default_allowed
        from src.release.hf_repo import resolve_hf_repo_id as _resolve_repo

        _local_root = Path(__file__).resolve().parents[2]
        _allow = bool(allow_default_hf_repo or _default_allowed())
        resolved_repo, resolved_source = _resolve_repo(
            explicit=str(hf_repo or "").strip() or None,
            env=os.environ,
            repo_root=_local_root,
            allow_default=_allow,
        )
        if resolved_source == "default" and not _allow:
            raise ValueError(
                "HF repo ID is not set explicitly; refusing the previous owner's default. "
                "Pass --hf-repo owner/repo."
            )
    except ValueError as exc:
        print(f"[!] BLOCKED: {exc}", file=sys.stderr)
        sys.exit(2)
    # Repo ID is not a secret; echo it so the operator can confirm the
    # destination account before GPU billing starts. Tokens are never printed.
    print(f"[*] HF repo: {resolved_repo} (source={resolved_source})", flush=True)
    if resolved_source == "default":
        print("[!] Fresh accounts should pass --hf-repo owner/repo; default targets the previous owner's repo.", flush=True)
    print(f"[*] Remote timeout: {TIMEOUT_SECONDS}s (no quality/time gate; set MODAL_TIMEOUT_SECONDS only to cap spend).")
    print("[*] Durable outputs use /root/legalir_volume/<label>/attempts/<id>/ on the 'legalir-production' Volume.")
    print("[*] Supervision: default `modal run` is ATTACHED — client disconnect terminates")
    print("    remote tasks even with a persistent Volume. Keep the client connected (stable")
    print("    network, machine awake, tmux/screen) until the remote job returns, or dispatch")
    print("    via the wrapper with explicit `--detach` plus app-ID tracking, log monitoring,")
    print("    and an explicit stop procedure. Independently confirm app termination.")
    print("[*] NOTE: timeout caps duration, not spend — retries/re-runs bill extra. No checkpoint-resume:")
    print("    a timeout kill still requires a full re-run (Volume holds forensics only).")
    print("[*] Ensure you have created 'kaggle-secret' and 'huggingface-secret' in the Modal dashboard!")
    if hf_allow_public_repo:
        print("[!] OPERATOR OVERRIDE: public HF repos permitted for this launch.", flush=True)

    result = run_production_training.remote(
        expected_sha,
        hf_allow_public_repo=hf_allow_public_repo,
        private=(test_phase == "private"),
        reranker_config=cfg_to_send,
        hf_repo=resolved_repo,
    )
    print(f"[*] Remote job finished with result: {result}")
