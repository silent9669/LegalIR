"""Pre-warm shared Modal Volume with models + dataset BEFORE renting A100 time.

A100 GPUs bill by the second, so every download on the GPU (HF model weights,
616MB Kaggle dataset) is money spent idling. Run this CPU-cheap function first::

    modal run scripts/modal/warm_volume.py
    # or via wrappers:
    scripts/modal/run_modal_cli.sh --warm-only
    python scripts/modal/run_full.py --warm-only

It populates ``<volume>/shared/`` (NOT a per-attempt dir, so every later run
reuses it):

    shared/models/huggingface/   pinned snapshots + manifest.json
                                 (via src.models.bootstrap.download_models)
    shared/dataset/              canonical Kaggle files, fingerprint-verified
    shared/warm_manifest.json    summary (models, revisions, bytes, timestamps)

The A100 function (run_modal_a100.py) detects this cache automatically:
HF_*_CACHE envs point at the snapshots (zero re-download) and the dataset
dir is reused (zero re-download). No release/SHA gate; any run label works.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("legalir-warm-cache")

# Slim CPU image: snapshot_download + kaggle + repo helpers only (no torch).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "huggingface-hub==1.28.0",
        "kaggle>=1.8,<3",
        "pyyaml>=6,<7",
        "tqdm>=4.65",
    )
)

volume = modal.Volume.from_name("legalir-production", create_if_missing=True)
VOLUME_MOUNT = "/root/legalir_volume"

# 2h is ample for ~3GB of weights + 616MB dataset on CPU.
WARM_TIMEOUT_SECONDS = int(os.environ.get("MODAL_WARM_TIMEOUT_SECONDS", 7200))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_volume_mount() -> Path:
    import sys as _sys

    mod = _sys.modules.get(__name__)
    override = getattr(mod, "VOLUME_MOUNT", VOLUME_MOUNT) if mod is not None else VOLUME_MOUNT
    return Path(override)


def _resolve_repo_dir() -> Path:
    return Path(os.environ.get("LEGALIR_MODAL_REPO_DIR", "/root/LegalIR"))


def shared_models_dir(volume_root: Path) -> Path:
    return Path(volume_root) / "shared" / "models" / "huggingface"


def shared_dataset_dir(volume_root: Path) -> Path:
    return Path(volume_root) / "shared" / "dataset"


def shared_warm_manifest(volume_root: Path) -> Path:
    return Path(volume_root) / "shared" / "warm_manifest.json"


def dir_size_bytes(path: Path) -> int:
    """Best-effort recursive size; 0 when the path is missing."""
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


def write_warm_manifest(path: Path, payload: dict) -> Path:
    """Atomically write the warm summary manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def dataset_ready(dataset_dir: Path) -> bool:
    """True when every REQUIRED_FILES entry exists (download can be skipped)."""
    import sys

    repo = _resolve_repo_dir()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.colab.bootstrap import REQUIRED_FILES

    dataset_dir = Path(dataset_dir)
    return all((dataset_dir / name).is_file() for name in REQUIRED_FILES)


def _resolve_warm_hf_repo(explicit: str | None, allow_default: bool = True) -> tuple[str, str]:
    """Resolve the adapter/HF repo for warm (explicit > env > .env > default)."""
    DEFAULT = "dangphuc2109/legalir-task1-reranker"
    try:
        from src.release.hf_repo import resolve_hf_repo_id as _resolve

        repo_root = None
        try:
            repo_root = _resolve_repo_dir()
        except Exception:
            repo_root = None
        return _resolve(explicit=explicit, env=os.environ, repo_root=repo_root,
                        default=DEFAULT, allow_default=allow_default)
    except ImportError:
        import re as _re

        cand = str(explicit or "").strip() or str(os.environ.get("HF_REPO_ID", "")).strip()
        src = "explicit" if str(explicit or "").strip() else ("env" if cand else "default")
        repo = cand or DEFAULT
        if not _re.fullmatch(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", repo):
            raise ValueError(f"Invalid HF repo ID '{repo}' (source={src}).")
        if src == "default" and not allow_default:
            raise ValueError(
                "HF repo ID is not set explicitly; refusing the previous owner's default. "
                "Pass --hf-repo owner/repo."
            )
        return repo, src


def _is_full_sha(value: str) -> bool:
    import re as _re

    return bool(_re.fullmatch(r"[0-9a-f]{40}", str(value or "").strip()))


def _checkout_requested_sha(repo_dir: Path, requested: str) -> str:
    """Checkout the requested SHA/label; returns the VERIFIED actual HEAD.

    Warm previously cloned without checking out the dispatch SHA, so its
    dataset/model helpers could run on a different checkout than training.
    Short/dev labels stay advisory (warn, keep current HEAD) so CPU warm never
    blocks on non-pinned labels. Two fail-closed rules protect provenance:
      1. a full 40-char SHA was requested but HEAD differs afterwards, or HEAD
         cannot be read at all for a pinned SHA → raise;
      2. HEAD cannot be read for an advisory label → return "unknown" (never
         the requested string), so the manifest records unverified provenance
         instead of asserting an unchecked SHA.
    """
    requested = str(requested or "").strip() or "dev"
    try:
        if len(requested) >= 7:
            subprocess.run(["git", "fetch", "origin", requested], cwd=repo_dir, check=False,
                           capture_output=True, text=True, timeout=120)
            res = subprocess.run(["git", "checkout", "--detach", requested], cwd=repo_dir,
                                 capture_output=True, text=True, timeout=120)
            if res.returncode != 0 and requested == "dev":
                print("[*] Warm checkout advisory: 'dev' label kept on default branch.", flush=True)
            elif res.returncode != 0:
                print(f"[!] Warm checkout advisory: could not checkout '{requested[:24]}'; using current HEAD.", flush=True)
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                             capture_output=True, text=True, timeout=30)
        actual = out.stdout.strip() if out.returncode == 0 else ""
        if not actual:
            if _is_full_sha(requested):
                raise RuntimeError(
                    f"Warm source unverifiable: requested pinned SHA {requested[:12]} "
                    "but HEAD cannot be read; refusing to record this cache."
                )
            print("[!] Warm HEAD unreadable; recording source_sha='unknown'.", flush=True)
            return "unknown"
        if _is_full_sha(requested) and _is_full_sha(actual) \
                and requested.lower() != actual.lower():
            raise RuntimeError(
                f"Warm source mismatch: requested {requested[:12]} but HEAD is {actual[:12]}; "
                "refusing to record this cache under the dispatch SHA."
            )
        return actual
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_full_sha(requested):
            raise RuntimeError(
                f"Warm checkout failed for pinned SHA {requested[:12]} "
                f"({type(exc).__name__}); refusing to record this cache."
            ) from None
        print(f"[!] Warm checkout advisory ({type(exc).__name__}); recording source_sha='unknown'.", flush=True)
        return "unknown"


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=WARM_TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: volume},
    secrets=[
        modal.Secret.from_name("kaggle-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def warm_shared_cache(label: str = "dev", warm_adapter: bool = False, hf_repo: str = "",
                      allow_default_hf_repo: bool = False):
    """Download pinned models + canonical dataset into the shared Volume cache.

    ``label`` should be the same commit SHA dispatched to training (forwarded
    via LEGALIR_COMMIT_SHA); the job checks out that SHA and records both the
    requested label and the actual HEAD so operators can verify warm/training
    SHA consistency. ``hf_repo`` is the explicit artifact repo (flag > env >
    default); it only matters when warming the optional LoRA adapter, but is
    always recorded for traceability. Missing/partial downloads still fall
    back at training time; the manifest records what warm actually produced.
    """
    import sys

    label = str(label or "dev").strip() or "dev"
    volume_root = _resolve_volume_mount()
    started_utc = _utc_now_iso()

    # 1. Checkout repo pinned to the dispatch SHA (like the training job).
    repo_dir = _resolve_repo_dir()
    if not repo_dir.exists():
        print(f"[*] Cloning repository into {repo_dir}...")
        subprocess.run(["git", "clone", "https://github.com/silent9669/LegalIR.git", str(repo_dir)], check=True)
    source_sha = _checkout_requested_sha(repo_dir, label)
    print(f"[*] Warm source: requested={label[:24]} actual HEAD={source_sha[:12]}", flush=True)
    if str(label).strip().lower() != str(source_sha).strip().lower() and len(str(label).strip()) >= 7 and str(label).strip().lower() != "dev":
        print(f"[!] Warm SHA advisory: requested {label[:12]} != actual {source_sha[:12]}; training must verify before reuse.", flush=True)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    try:
        from src.release.hf_repo import default_allowed as _default_allowed

        _allow = bool(allow_default_hf_repo or _default_allowed())
        resolved_repo, resolved_source = _resolve_warm_hf_repo(
            str(hf_repo or "").strip() or None, allow_default=_allow)
    except ValueError as exc:
        raise RuntimeError(f"BLOCKED: {exc}") from None
    # Repo ID is not a secret; echo it for account confirmation.
    print(f"[*] Warm HF repo: {resolved_repo} (source={resolved_source})", flush=True)

    from scripts.colab.bootstrap import prepare_dataset
    from src.models.bootstrap import MODEL_REGISTRY, download_models

    models_root = shared_models_dir(volume_root)
    dataset_dir = shared_dataset_dir(volume_root)
    models_root.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 2. Pinned HF snapshots (writes manifest.json mapping id -> snapshot path).
    print(f"[*] Warming {len(MODEL_REGISTRY)} pinned models into {models_root}...")
    downloaded = download_models(model_root=models_root)

    # 3. Optional warm-start adapter (small; skipped for cold-start default).
    # Cold-start production never warms adapters; when enabled, the adapter
    # source is LEGALIR_WARM_START_ADAPTER > explicit --hf-repo > HF_REPO_ID
    # env > owner default. Automatic HF download stays enabled at training
    # time when warm is partial/missing.
    adapter_id = (os.environ.get("LEGALIR_WARM_START_ADAPTER") or "").strip()
    if warm_adapter or adapter_id:
        from huggingface_hub import snapshot_download

        adapter_id = adapter_id or resolved_repo
        print(f"[*] Warming LoRA adapter {adapter_id}...")
        apath = snapshot_download(repo_id=adapter_id, cache_dir=str(models_root))
        downloaded[adapter_id] = Path(apath)
    else:
        print(f"[*] Skipping adapter warm (cold-start default; HF repo {resolved_repo} recorded for training).", flush=True)

    # 4. Canonical Kaggle dataset (skips download when files already present).
    freeze_file = repo_dir / "artifacts/task1/freeze/production_freeze.json"
    print(f"[*] Ensuring canonical dataset at {dataset_dir}...")
    prepare_dataset(dataset_dir, freeze_file if freeze_file.is_file() else None)

    # 5. Summary manifest for operators + the A100 function.
    from scripts.colab.bootstrap import REQUIRED_FILES

    manifest = {
        "label": label,
        "requested_label": label,
        "source_sha": source_sha,
        "hf_repo": resolved_repo,
        "hf_repo_source": resolved_source,
        "started_utc": started_utc,
        "finished_utc": _utc_now_iso(),
        "models": {
            mid: {
                "path": str(p),
                "revision": (MODEL_REGISTRY.get(mid) or {}).get("revision"),
                "bytes": dir_size_bytes(Path(p)),
            }
            for mid, p in downloaded.items()
        },
        "models_bytes": dir_size_bytes(models_root),
        "dataset_dir": str(dataset_dir),
        "dataset_files": [n for n in REQUIRED_FILES if (dataset_dir / n).is_file()],
        "dataset_bytes": dir_size_bytes(dataset_dir),
    }
    write_warm_manifest(shared_warm_manifest(volume_root), manifest)

    try:
        volume.commit()
        print(f"[*] Committed warm cache: {volume_root}/shared", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Volume commit failed: {type(exc).__name__}", flush=True)

    print(f"[+] Warm complete: models {manifest['models_bytes'] / 1e9:.2f}GB, "
          f"dataset {manifest['dataset_bytes'] / 1e9:.2f}GB", flush=True)
    return manifest


@app.local_entrypoint()
def main(warm_adapter: bool = False, hf_repo: str = "", allow_default_hf_repo: bool = False):
    import sys

    label = os.environ.get("LEGALIR_COMMIT_SHA")
    if not label:
        try:
            # Full SHA so the warm job can pin the same checkout as training.
            label = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            try:
                label = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
            except Exception:
                label = "dev"
    # Local gate mirrors the shell wrapper: default repo is BLOCKED, not OK.
    # dotenv is read from THIS checkout (not the Modal container path).
    from src.release.hf_repo import default_allowed as _main_default_allowed
    from src.release.hf_repo import resolve_hf_repo_id as _main_resolve

    from pathlib import Path as _P

    _main_allow = bool(allow_default_hf_repo or _main_default_allowed())
    try:
        local_repo, local_source = _main_resolve(
            explicit=str(hf_repo or "").strip() or None,
            env=os.environ,
            repo_root=_P(__file__).resolve().parents[2],
            allow_default=_main_allow,
        )
    except ValueError as exc:
        print(f"[!] BLOCKED: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"[*] Warming shared Volume cache (label={label}, adapter={warm_adapter}, "
          f"hf_repo={local_repo} source={local_source}) on CPU — no GPU billed.")
    result = warm_shared_cache.remote(label, warm_adapter=warm_adapter, hf_repo=local_repo,
                                      allow_default_hf_repo=_main_allow)
    print(f"[+] Warm result: {len(result.get('models', {}))} models, "
          f"dataset files: {len(result.get('dataset_files', []))}")
    if not result.get("dataset_files"):
        print("[!] Dataset files missing after warm — check kaggle-secret.", file=sys.stderr)
        sys.exit(1)
