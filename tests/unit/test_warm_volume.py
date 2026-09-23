"""Unit tests for the CPU Volume warm path (warm_volume.py + attach helper)."""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path


def _install_fake_modal():
    if "modal" in sys.modules:
        del sys.modules["modal"]
    fake = types.ModuleType("modal")

    class _Chain:
        def __getattr__(self, name):
            def _mk(*a, **k):
                return self
            return _mk

    class _FakeApp:
        def __init__(self, *a, **k):
            pass

        def function(self, *dargs, **dkwargs):
            def deco(fn):
                fn.remote = fn
                return fn
            return deco

        def local_entrypoint(self, *dargs, **dkwargs):
            def deco(fn):
                return fn
            return deco

    class _FakeVolume:
        @classmethod
        def from_name(cls, *a, **k):
            class _V:
                def commit(self):
                    return None
            return _V()

    class _FakeSecret:
        @classmethod
        def from_name(cls, *a, **k):
            return object()

    fake.App = _FakeApp
    fake.Image = _Chain()
    fake.Volume = _FakeVolume
    fake.Secret = _FakeSecret
    sys.modules["modal"] = fake
    return fake


def _load_warm():
    _install_fake_modal()
    for mod in ("scripts.modal.warm_volume", "scripts.modal"):
        sys.modules.pop(mod, None)
    import scripts.modal.warm_volume as w

    return importlib.reload(w)


def _load_launcher():
    _install_fake_modal()
    for mod in ("scripts.modal.run_modal_a100", "scripts.modal"):
        sys.modules.pop(mod, None)
    import scripts.modal.run_modal_a100 as m

    return importlib.reload(m)


def test_shared_layout_paths(tmp_path):
    w = _load_warm()
    root = tmp_path / "v"
    assert w.shared_models_dir(root) == root / "shared" / "models" / "huggingface"
    assert w.shared_dataset_dir(root) == root / "shared" / "dataset"
    assert w.shared_warm_manifest(root) == root / "shared" / "warm_manifest.json"


def test_dir_size_and_manifest_roundtrip(tmp_path):
    w = _load_warm()
    d = tmp_path / "data"
    (d / "sub").mkdir(parents=True)
    (d / "a.bin").write_bytes(b"x" * 100)
    (d / "sub" / "b.bin").write_bytes(b"y" * 50)
    assert w.dir_size_bytes(d) == 150
    assert w.dir_size_bytes(tmp_path / "missing") == 0

    mp = w.write_warm_manifest(tmp_path / "shared" / "warm_manifest.json", {"b": 1, "a": [1, 2]})
    assert mp.is_file()
    assert not mp.with_suffix(".tmp").exists()
    assert json.loads(mp.read_text(encoding="utf-8")) == {"a": [1, 2], "b": 1}


def test_dataset_ready_checks_required_files(tmp_path):
    w = _load_warm()
    d = tmp_path / "ds"
    d.mkdir()
    assert w.dataset_ready(d) is False
    from scripts.colab.bootstrap import REQUIRED_FILES

    for name in REQUIRED_FILES:
        (d / name).write_bytes(b"")
    assert w.dataset_ready(d) is True


def test_attach_warmed_cache_attaches_and_reuses(tmp_path, monkeypatch):
    mod = _load_launcher()
    vol = tmp_path / "volume"
    models = vol / "shared" / "models" / "huggingface"
    models.mkdir(parents=True)
    from src.models.bootstrap import MODEL_REGISTRY as _REG

    snaps = {}
    for mid in _REG:
        d = models / ("snap-" + mid.split("/")[-1].replace("-", "_"))
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"w")
        snaps[mid] = d
    (models / "manifest.json").write_text(json.dumps({
        mid: {"path": str(snaps[mid]), "revision": _REG[mid]["revision"]}
        for mid in _REG
    }), encoding="utf-8")
    ds = vol / "shared" / "dataset"
    ds.mkdir(parents=True)
    from scripts.colab.bootstrap import REQUIRED_FILES
    from src.release.fingerprints import generate_dataset_manifest

    for name in REQUIRED_FILES:
        (ds / name).write_bytes(b"data")
    ds_manifest = generate_dataset_manifest(ds)
    (ds / "dataset_manifest.json").write_text(json.dumps(ds_manifest), encoding="utf-8")
    (vol / "shared" / "warm_manifest.json").write_text(json.dumps({
        "requested_label": "a" * 40, "source_sha": "a" * 40, "hf_repo": "someone/repo",
        "dataset_manifest_sha256": ds_manifest["manifest_sha256"],
    }), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo, expected_sha="a" * 40)
    assert out["models_attached"] is True and out["dataset_reused"] is True
    assert out.get("models_detail") == "pinned-revisions-verified"
    assert os.environ["HF_HUB_CACHE"] == str(models)
    assert os.environ["LEGALIR_MODAL_DATASET_DIR"] == str(ds)
    mirrored = repo / "artifacts" / "local" / "models" / "huggingface" / "manifest.json"
    assert mirrored.is_file()
    assert json.loads(mirrored.read_text(encoding="utf-8"))["BAAI/bge-reranker-v2-m3"]["path"] == str(snaps["BAAI/bge-reranker-v2-m3"])


def test_attach_warmed_cache_falls_back_when_absent(tmp_path, monkeypatch, capsys):
    mod = _load_launcher()
    vol = tmp_path / "emptyvol"
    vol.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo, expected_sha="a" * 40)
    assert out["models_attached"] is False and out["dataset_reused"] is False
    assert out["models_detail"] == "source-gate:warm-manifest-missing"
    assert "models_detail" in out and "dataset_detail" in out
    for var in ("HF_HUB_CACHE", "LEGALIR_MODAL_DATASET_DIR"):
        assert var not in os.environ


def test_attach_warmed_cache_partial_models_falls_back(tmp_path, monkeypatch):
    mod = _load_launcher()
    vol = tmp_path / "vol"
    models = vol / "shared" / "models" / "huggingface"
    models.mkdir(parents=True)
    # Manifest references a snapshot dir that does not exist.
    (models / "manifest.json").write_text(json.dumps({
        "BAAI/bge-reranker-v2-m3": {"path": str(models / "nope"), "revision": "r"},
        "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2": {"path": str(models / "nope2"), "revision": "r"},
    }), encoding="utf-8")
    # Source gate passes so the model-identity check is what rejects.
    (vol / "shared" / "warm_manifest.json").write_text(json.dumps({
        "requested_label": "a" * 40, "source_sha": "a" * 40, "hf_repo": "someone/repo",
    }), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo, expected_sha="a" * 40)
    assert out["models_attached"] is False
    assert "stale:" in str(out["models_detail"])
    assert "HF_HUB_CACHE" not in os.environ


def _write_valid_model_cache(vol: Path) -> Path:
    from src.models.bootstrap import MODEL_REGISTRY as _REG

    models = vol / "shared" / "models" / "huggingface"
    models.mkdir(parents=True, exist_ok=True)
    for mid in _REG:
        d = models / ("snap-" + mid.split("/")[-1].replace("-", "_"))
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"w")
    (models / "manifest.json").write_text(json.dumps({
        mid: {"path": str(models / ("snap-" + mid.split("/")[-1].replace("-", "_"))),
              "revision": _REG[mid]["revision"]}
        for mid in _REG
    }), encoding="utf-8")
    return models


def _write_dataset_files(vol: Path) -> Path:
    from scripts.colab.bootstrap import REQUIRED_FILES
    from src.release.fingerprints import generate_dataset_manifest

    ds = vol / "shared" / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (ds / name).write_bytes(b"data")
    manifest = generate_dataset_manifest(ds)
    (ds / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds


def test_checkout_requested_sha_fails_closed_on_full_sha_mismatch(tmp_path):
    import subprocess

    w = _load_warm()
    repo = tmp_path / "repo"
    repo.mkdir()
    env = dict(__import__("os").environ, GIT_CONFIG_NOSYSTEM="1")
    subprocess.run(["git", "init", "-q"], cwd=str(repo), env=env, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=str(repo), env=env, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(repo), env=env, check=True)
    (repo / "f.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), env=env, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), env=env, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), env=env, text=True).strip()
    assert w._checkout_requested_sha(repo, head) == head
    # Advisory labels never raise.
    assert w._checkout_requested_sha(repo, "dev")
    # A definitive full-SHA mismatch fails closed.
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="mismatch"):
        w._checkout_requested_sha(repo, "0" * 40)


def test_checkout_requested_sha_never_asserts_unreadable_head(tmp_path):
    """Unverifiable provenance: 'unknown', never the requested string."""
    import pytest as _pytest

    w = _load_warm()
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    assert w._checkout_requested_sha(plain, "dev") == "unknown"
    with _pytest.raises(RuntimeError, match="unverifiable|failed"):
        w._checkout_requested_sha(plain, "a" * 40)


def test_attach_rejects_cache_from_another_source_sha(tmp_path, monkeypatch):
    mod = _load_launcher()
    vol = tmp_path / "vol"
    _write_valid_model_cache(vol)
    _write_dataset_files(vol)
    (vol / "shared" / "warm_manifest.json").write_text(json.dumps({
        "requested_label": "b" * 40,
        "source_sha": "b" * 40,
        "hf_repo": "someone/repo",
    }), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo, expected_sha="a" * 40)
    assert out["models_attached"] is False and out["dataset_reused"] is False
    assert "warm-source-sha-mismatch" in str(out["models_detail"])
    assert out["warm_source_sha"] == "b" * 40
    assert "HF_HUB_CACHE" not in os.environ
    assert "LEGALIR_MODAL_DATASET_DIR" not in os.environ


def test_attach_accepts_cache_with_matching_source_sha(tmp_path, monkeypatch):
    mod = _load_launcher()
    vol = tmp_path / "vol"
    _write_valid_model_cache(vol)
    _write_dataset_files(vol)
    (vol / "shared" / "warm_manifest.json").write_text(json.dumps({
        "requested_label": "a" * 40,
        "source_sha": "a" * 40,
        "hf_repo": "someone/repo",
    }), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo, expected_sha="a" * 40)
    assert out["models_attached"] is True and out["dataset_reused"] is True
    assert out["warm_source_sha"] == "a" * 40


def _attach_with_warm_sha(mod, vol, repo, monkeypatch, *, warm_sha, expected_sha):
    """Attach over a valid model+dataset cache with a controlled warm SHA."""
    _write_valid_model_cache(vol)
    _write_dataset_files(vol)
    if warm_sha is not None:
        payload: dict = {"requested_label": warm_sha, "hf_repo": "someone/repo"}
        if warm_sha != "<missing-key>":
            payload["source_sha"] = warm_sha
        (vol / "shared" / "warm_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    repo.mkdir(exist_ok=True)
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    return mod.attach_warmed_cache(vol, repo, expected_sha=expected_sha)


def test_attach_falls_back_when_warm_manifest_missing(tmp_path, monkeypatch):
    mod = _load_launcher()
    out = _attach_with_warm_sha(mod, tmp_path / "vol", tmp_path / "repo", monkeypatch,
                                warm_sha=None, expected_sha="a" * 40)
    assert out["models_attached"] is False and out["dataset_reused"] is False
    assert out["models_detail"] == "source-gate:warm-manifest-missing"
    assert out["dataset_detail"] == "source-gate:warm-manifest-missing"


def test_attach_falls_back_when_warm_sha_key_missing(tmp_path, monkeypatch):
    mod = _load_launcher()
    out = _attach_with_warm_sha(mod, tmp_path / "vol", tmp_path / "repo", monkeypatch,
                                warm_sha="<missing-key>", expected_sha="a" * 40)
    assert out["models_attached"] is False and out["dataset_reused"] is False
    assert out["models_detail"] == "source-gate:warm-source-sha-unknown"


def test_attach_falls_back_when_warm_sha_unknown(tmp_path, monkeypatch):
    mod = _load_launcher()
    out = _attach_with_warm_sha(mod, tmp_path / "vol", tmp_path / "repo", monkeypatch,
                                warm_sha="unknown", expected_sha="a" * 40)
    assert out["models_attached"] is False and out["dataset_reused"] is False
    assert out["models_detail"] == "source-gate:warm-source-sha-unknown"


def test_attach_falls_back_when_warm_sha_malformed(tmp_path, monkeypatch):
    mod = _load_launcher()
    out = _attach_with_warm_sha(mod, tmp_path / "vol", tmp_path / "repo", monkeypatch,
                                warm_sha="not-a-sha", expected_sha="a" * 40)
    assert out["models_attached"] is False and out["dataset_reused"] is False
    assert out["models_detail"] == "source-gate:warm-source-sha-malformed:not-a-sha"


def test_attach_falls_back_when_training_sha_unpinned(tmp_path, monkeypatch):
    mod = _load_launcher()
    out = _attach_with_warm_sha(mod, tmp_path / "vol", tmp_path / "repo", monkeypatch,
                                warm_sha="a" * 40, expected_sha="dev")
    assert out["models_attached"] is False and out["dataset_reused"] is False
    assert out["models_detail"] == "source-gate:training-sha-unpinned:dev"


def test_attach_falls_back_when_training_sha_missing(tmp_path, monkeypatch):
    mod = _load_launcher()
    out = _attach_with_warm_sha(mod, tmp_path / "vol", tmp_path / "repo", monkeypatch,
                                warm_sha="a" * 40, expected_sha=None)
    assert out["models_attached"] is False and out["dataset_reused"] is False
    assert out["models_detail"] == "source-gate:training-sha-missing"


def test_check_warm_source_sha_unit():
    mod = _load_launcher()
    ok, reason = mod._check_warm_source_sha("a" * 40, warm_manifest_present=True,
                                            warm_source_sha="a" * 40)
    assert (ok, reason) == (True, "sha-verified")
    assert mod._check_warm_source_sha(None, warm_manifest_present=True,
                                      warm_source_sha="a" * 40) == (False, "training-sha-missing")
    assert mod._check_warm_source_sha("dev", warm_manifest_present=True,
                                      warm_source_sha="a" * 40) == (False, "training-sha-unpinned:dev")
    assert mod._check_warm_source_sha("a" * 40, warm_manifest_present=False,
                                      warm_source_sha="unknown") == (False, "warm-manifest-missing")


def test_attach_warns_on_preset_cache_env_and_leaves_it(tmp_path, monkeypatch, capsys):
    """Preset env pointing elsewhere must not silently govern the fallback."""
    mod = _load_launcher()
    vol = tmp_path / "vol"
    _write_valid_model_cache(vol)
    _write_dataset_files(vol)
    (vol / "shared" / "warm_manifest.json").write_text(json.dumps({
        "requested_label": "b" * 40, "source_sha": "b" * 40, "hf_repo": "someone/repo",
    }), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    other_models = tmp_path / "elsewhere-models"
    other_models.mkdir()
    other_ds = tmp_path / "elsewhere-ds"
    other_ds.mkdir()
    for var in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(other_models))
    monkeypatch.setenv("LEGALIR_MODAL_DATASET_DIR", str(other_ds))
    out = mod.attach_warmed_cache(vol, repo, expected_sha="a" * 40)
    assert out["models_attached"] is False and out["dataset_reused"] is False
    logged = capsys.readouterr().out
    assert "HF_HUB_CACHE" in logged
    assert "LEGALIR_MODAL_DATASET_DIR" in logged
    # Preset values are left untouched (downloaders resolve them explicitly).
    assert os.environ["HF_HUB_CACHE"] == str(other_models)
    assert os.environ["LEGALIR_MODAL_DATASET_DIR"] == str(other_ds)


def test_warm_completeness_issues_flags_everything_missing():
    w = _load_warm()
    from src.models.bootstrap import MODEL_REGISTRY as _REG
    from scripts.colab.bootstrap import REQUIRED_FILES

    issues = w.warm_completeness_issues({}, [], list(REQUIRED_FILES), dict(_REG))
    assert any(i.startswith("model:") for i in issues)
    assert any(i.startswith("dataset:") for i in issues)


def test_warm_completeness_issues_passes_on_complete_manifest():
    w = _load_warm()
    from src.models.bootstrap import MODEL_REGISTRY as _REG
    from scripts.colab.bootstrap import REQUIRED_FILES

    models = {mid: {"path": f"/snap/{mid}", "revision": meta["revision"]}
              for mid, meta in _REG.items()}
    assert w.warm_completeness_issues(models, list(REQUIRED_FILES),
                                      list(REQUIRED_FILES), dict(_REG)) == []


def test_warm_completeness_issues_catches_revision_mismatch():
    w = _load_warm()
    from src.models.bootstrap import MODEL_REGISTRY as _REG
    from scripts.colab.bootstrap import REQUIRED_FILES

    first = next(iter(_REG))
    models = {mid: {"path": "/snap/x", "revision": "STALE"} for mid in _REG}
    issues = w.warm_completeness_issues(models, list(REQUIRED_FILES),
                                        list(REQUIRED_FILES), dict(_REG))
    assert f"model:{first}:revision-mismatch" in issues
