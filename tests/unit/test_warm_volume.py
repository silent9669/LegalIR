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
    snap_r = models / "snap-reranker"
    snap_d = models / "snap-dense"
    snap_r.mkdir()
    snap_d.mkdir()
    (models / "manifest.json").write_text(json.dumps({
        "BAAI/bge-reranker-v2-m3": {"path": str(snap_r), "revision": "r"},
        "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2": {"path": str(snap_d), "revision": "r"},
    }), encoding="utf-8")
    ds = vol / "shared" / "dataset"
    ds.mkdir(parents=True)
    from scripts.colab.bootstrap import REQUIRED_FILES

    for name in REQUIRED_FILES:
        (ds / name).write_bytes(b"")
    repo = tmp_path / "repo"
    repo.mkdir()

    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo)
    assert out == {"models_attached": True, "dataset_reused": True}
    assert os.environ["HF_HUB_CACHE"] == str(models)
    assert os.environ["LEGALIR_MODAL_DATASET_DIR"] == str(ds)
    mirrored = repo / "artifacts" / "local" / "models" / "huggingface" / "manifest.json"
    assert mirrored.is_file()
    assert json.loads(mirrored.read_text(encoding="utf-8"))["BAAI/bge-reranker-v2-m3"]["path"] == str(snap_r)


def test_attach_warmed_cache_falls_back_when_absent(tmp_path, monkeypatch, capsys):
    mod = _load_launcher()
    vol = tmp_path / "emptyvol"
    vol.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo)
    assert out == {"models_attached": False, "dataset_reused": False}
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
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo)
    assert out["models_attached"] is False
    assert "HF_HUB_CACHE" not in os.environ
