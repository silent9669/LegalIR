"""Unit tests for the read-only HF preflight (no repo creation, ever)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gates.run_a100 import preflight_huggingface_access, upload_artifacts_to_huggingface


def _fake_hub(monkeypatch, *, repo_private=True, repo_missing=False, write_ok=True):
    fake = types.ModuleType("huggingface_hub")
    calls = {"create": 0, "upload": 0, "auth": 0}

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "teammate"}

        def repo_info(self, repo_id=None, repo_type=None):
            if repo_missing:
                raise OSError("404")
            return types.SimpleNamespace(private=repo_private)

        def auth_check(self, repo_id=None, repo_type=None, write=False):
            calls["auth"] += 1
            assert write is True, "must check WRITE on this exact repo"
            if not write_ok:
                raise OSError("403")

        def create_repo(self, *a, **k):
            calls["create"] += 1
            raise AssertionError("must never create repos")

        def upload_folder(self, *a, **k):
            calls["upload"] += 1
            return types.SimpleNamespace(oid="a" * 40)

    fake.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return calls


def test_public_with_opt_in_and_write(monkeypatch):
    calls = _fake_hub(monkeypatch, repo_private=False, write_ok=True)
    ok, detail = preflight_huggingface_access("o/r", token="hf_test", allow_public_repo=True)
    assert ok is True and "(public)" in detail
    assert calls["create"] == 0


def test_public_without_opt_in_blocked(monkeypatch):
    calls = _fake_hub(monkeypatch, repo_private=False, write_ok=True)
    ok, detail = preflight_huggingface_access("o/r", token="hf_test", allow_public_repo=False)
    assert ok is False and "explicit opt-in" in detail
    assert calls["create"] == 0


def test_readonly_token_blocked(monkeypatch):
    calls = _fake_hub(monkeypatch, repo_private=False, write_ok=False)
    ok, detail = preflight_huggingface_access("o/r", token="hf_test", allow_public_repo=True)
    assert ok is False and "WRITE" in detail
    assert calls["create"] == 0


def test_missing_repo_fails_closed_without_create(monkeypatch):
    calls = _fake_hub(monkeypatch, repo_missing=True)
    ok, detail = preflight_huggingface_access("o/r", token="hf_test", allow_public_repo=True)
    assert ok is False and "does not exist" in detail
    assert calls["create"] == 0


def test_private_with_write(monkeypatch):
    calls = _fake_hub(monkeypatch, repo_private=True, write_ok=True)
    ok, detail = preflight_huggingface_access("o/r", token="hf_test")
    assert ok is True and "(private)" in detail
    assert calls["create"] == 0


def test_unknown_visibility_blocked(monkeypatch):
    calls = _fake_hub(monkeypatch, repo_private=None, write_ok=True)
    ok, _ = preflight_huggingface_access("o/r", token="hf_test", allow_public_repo=True)
    assert ok is False
    assert calls["create"] == 0


def test_missing_token_blocked(monkeypatch):
    monkeypatch.delenv("HF_TOKEN_WRITE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_READ", raising=False)
    ok, _ = preflight_huggingface_access("o/r", token="not-a-token")
    assert ok is False


def _make_artifacts(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    adapter = root / "checkpoints" / "reranker_final"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (root / "submission.zip").write_bytes(b"zip")
    (root / "submission.json").write_text("{}", encoding="utf-8")
    (root / "run_manifest.json").write_text(json.dumps({"run_id": "test_run"}), encoding="utf-8")


def test_upload_never_calls_create_repo(tmp_path, monkeypatch):
    _make_artifacts(tmp_path)
    calls = _fake_hub(monkeypatch, repo_private=True, write_ok=True)
    commit = upload_artifacts_to_huggingface(tmp_path, "o/r", token="hf_test")
    assert len(commit) == 40
    assert calls["create"] == 0
    assert calls["upload"] == 1


def test_upload_missing_repo_fails_closed_without_create(tmp_path, monkeypatch):
    _make_artifacts(tmp_path)
    calls = _fake_hub(monkeypatch, repo_missing=True)
    with pytest.raises(RuntimeError, match="does not exist or is not visible"):
        upload_artifacts_to_huggingface(tmp_path, "o/r", token="hf_test")
    assert calls["create"] == 0
    assert calls["upload"] == 0


def test_upload_readonly_token_fails_closed(tmp_path, monkeypatch):
    _make_artifacts(tmp_path)
    calls = _fake_hub(monkeypatch, repo_private=True, write_ok=False)
    with pytest.raises(RuntimeError, match="WRITE permission"):
        upload_artifacts_to_huggingface(tmp_path, "o/r", token="hf_test")
    assert calls["create"] == 0
    assert calls["upload"] == 0


def test_upload_public_without_optin_fails_closed(tmp_path, monkeypatch):
    _make_artifacts(tmp_path)
    calls = _fake_hub(monkeypatch, repo_private=False, write_ok=True)
    with pytest.raises(RuntimeError, match="explicit opt-in"):
        upload_artifacts_to_huggingface(tmp_path, "o/r", token="hf_test", allow_public_repo=False)
    assert calls["create"] == 0
    assert calls["upload"] == 0


def test_upload_public_with_optin_succeeds(tmp_path, monkeypatch):
    _make_artifacts(tmp_path)
    calls = _fake_hub(monkeypatch, repo_private=False, write_ok=True)
    commit = upload_artifacts_to_huggingface(tmp_path, "o/r", token="hf_test", allow_public_repo=True)
    assert len(commit) == 40
    assert calls["create"] == 0
    assert calls["upload"] == 1
