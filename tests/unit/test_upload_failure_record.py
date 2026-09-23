"""Failed-upload manifest records repo ID, opt-in consent, and visibility."""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gates.run_a100 import failed_upload_record


def _fake_hub(monkeypatch, *, private=True, fail=False):
    fake = types.ModuleType("huggingface_hub")

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def repo_info(self, repo_id=None, repo_type=None):
            if fail:
                raise OSError("offline")
            assert repo_id == "o/r"
            return types.SimpleNamespace(private=private)

    fake.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def test_failure_record_public_with_opt_in(monkeypatch):
    _fake_hub(monkeypatch, private=False)
    rec = failed_upload_record("o/r", allow_public_repo=True, token="hf_testtoken")
    assert rec == {"repo_id": "o/r", "uploaded": False,
                   "public_repo_override": True, "visibility": "public"}


def test_failure_record_private_without_opt_in(monkeypatch):
    _fake_hub(monkeypatch, private=True)
    rec = failed_upload_record("o/r", allow_public_repo=False, token="hf_testtoken")
    assert rec["visibility"] == "private"
    assert rec["public_repo_override"] is False
    assert rec["uploaded"] is False


def test_failure_record_visibility_unknown_when_offline(monkeypatch):
    _fake_hub(monkeypatch, fail=True)
    rec = failed_upload_record("o/r", token="hf_testtoken")
    assert rec["visibility"] == "unknown"
    assert rec["repo_id"] == "o/r"


def test_failure_record_never_contains_token(monkeypatch):
    _fake_hub(monkeypatch, private=True)
    rec = failed_upload_record("o/r", token="hf_supersecret_zzz")
    assert "hf_supersecret_zzz" not in str(rec)
