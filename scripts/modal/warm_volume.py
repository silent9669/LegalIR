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
def warm_shared_cache(label: str = "dev", warm_adapter: bool = False):
    """Download pinned models + canonical dataset into the shared Volume cache."""
    import sys

    label = str(label or "dev").strip() or "dev"
    volume_root = _resolve_volume_mount()
    started_utc = _utc_now_iso()

    # 1. Checkout repo (advisory label; remote clones origin like training fn).
    repo_dir = _resolve_repo_dir()
    if not repo_dir.exists():
        print(f"[*] Cloning repository into {repo_dir}...")
        subprocess.run(["git", "clone", "https://github.com/silent9669/LegalIR.git", str(repo_dir)], check=True)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

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
    adapter_id = (os.environ.get("LEGALIR_WARM_START_ADAPTER") or "").strip()
    if warm_adapter or adapter_id:
        from huggingface_hub import snapshot_download

        adapter_id = adapter_id or os.environ.get("HF_REPO_ID", "dangphuc2109/legalir-task1-reranker")
        print(f"[*] Warming LoRA adapter {adapter_id}...")
        apath = snapshot_download(repo_id=adapter_id, cache_dir=str(models_root))
        downloaded[adapter_id] = Path(apath)

    # 4. Canonical Kaggle dataset (skips download when files already present).
    freeze_file = repo_dir / "artifacts/task1/freeze/production_freeze.json"
    print(f"[*] Ensuring canonical dataset at {dataset_dir}...")
    prepare_dataset(dataset_dir, freeze_file if freeze_file.is_file() else None)

    # 5. Summary manifest for operators + the A100 function.
    from scripts.colab.bootstrap import REQUIRED_FILES

    manifest = {
        "label": label,
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
def main(warm_adapter: bool = False):
    import sys

    label = os.environ.get("LEGALIR_COMMIT_SHA")
    if not label:
        try:
            label = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        except Exception:
            label = "dev"
    print(f"[*] Warming shared Volume cache (label={label}, adapter={warm_adapter}) on CPU — no GPU billed.")
    result = warm_shared_cache.remote(label, warm_adapter=warm_adapter)
    print(f"[+] Warm result: {len(result.get('models', {}))} models, "
          f"dataset files: {len(result.get('dataset_files', []))}")
    if not result.get("dataset_files"):
        print("[!] Dataset files missing after warm — check kaggle-secret.", file=sys.stderr)
        sys.exit(1)
