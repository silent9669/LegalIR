"""Unit tests for the read-only HF repo checker (no network, no mutation)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.check_hf_repo as chk


def _install_fake_hub(monkeypatch, *, exists=True, private=True, write=True):
    fake = types.ModuleType("huggingface_hub")
    calls = {"created": 0, "uploaded": 0}

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            assert self.token == "hf_testtoken", "token must be forwarded, never printed"
            return {"name": "testuser"}

        def repo_info(self, repo_id=None, repo_type=None):
            if not exists:
                raise OSError("not found")
            return types.SimpleNamespace(private=private)

        def auth_check(self, repo_id=None, repo_type=None, write=False):
            if not write:
                raise AssertionError("checker must request write=True check")
            if not write:
                return None
            raise OSError("denied")

        def create_repo(self, *a, **k):
            calls["created"] += 1
            raise AssertionError("read-only checker must never create repos")

        def upload_folder(self, *a, **k):
            calls["uploaded"] += 1
            raise AssertionError("read-only checker must never upload")

    fake.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return calls


def test_private_repo_with_write(monkeypatch):
    _install_fake_hub(monkeypatch, exists=True, private=True, write=True)

    class AllowAll:
        def auth_check(self, repo_id=None, repo_type=None, write=False):
            return None

    import huggingface_hub as hub

    hub.HfApi = type("A", (), {
        "__init__": lambda self, token=None: setattr(self, "token", token),
        "whoami": lambda self: {"name": "testuser"},
        "repo_info": lambda self, repo_id=None, repo_type=None: types.SimpleNamespace(private=True),
        "auth_check": lambda self, repo_id=None, repo_type=None, write=False: None,
        "create_repo": lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("no create")),
    })
    out = chk.check_repo("owner/repo", "hf_testtoken")
    assert out["exists"] is True and out["visibility"] == "private"
    assert out["write_access"] is True and out["authenticated_as"] == "testuser"


def test_public_repo_flagged(monkeypatch):
    _install_fake_hub(monkeypatch, exists=True, private=False, write=True)
    import huggingface_hub as hub

    hub.HfApi = type("A", (), {
        "__init__": lambda self, token=None: None,
        "whoami": lambda self: {"name": "testuser"},
        "repo_info": lambda self, repo_id=None, repo_type=None: types.SimpleNamespace(private=False),
        "auth_check": lambda self, repo_id=None, repo_type=None, write=False: (_ for _ in ()).throw(OSError("denied")),
    })
    out = chk.check_repo("owner/repo", "hf_testtoken")
    assert out["visibility"] == "public" and out["write_access"] is False


def test_missing_repo_reported_without_creation(monkeypatch):
    calls = _install_fake_hub(monkeypatch, exists=False)
    out = chk.check_repo("owner/repo", "hf_testtoken")
    assert out["exists"] is False
    assert calls["created"] == 0 and calls["uploaded"] == 0


def test_invalid_repo_id_rejected():
    with pytest.raises(ValueError):
        chk.check_repo("bad id!!", "hf_testtoken")


def test_main_exit_codes(monkeypatch, capsys):
    _install_fake_hub(monkeypatch, exists=False)
    monkeypatch.setenv("HF_TOKEN_WRITE", "hf_testtoken")
    assert chk.main(["--repo", "owner/repo"]) == 1  # missing repo
    assert chk.main(["--repo", "bad id!!"]) == 2  # invalid ID
    monkeypatch.delenv("HF_TOKEN_WRITE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert chk.main(["--repo", "owner/repo", "--token", ""]) == 1  # no token


def test_main_blocks_without_write_access(monkeypatch, capsys):
    """Existing repo + denied write must exit 1, never 0."""
    _install_fake_hub(monkeypatch, exists=True, private=True, write=False)
    import huggingface_hub as hub

    hub.HfApi = type("A", (), {
        "__init__": lambda self, token=None: None,
        "whoami": lambda self: {"name": "testuser"},
        "repo_info": lambda self, repo_id=None, repo_type=None: types.SimpleNamespace(private=True),
        "auth_check": lambda self, repo_id=None, repo_type=None, write=False: (_ for _ in ()).throw(OSError("denied")),
    })
    monkeypatch.setenv("HF_TOKEN_WRITE", "hf_testtoken")
    assert chk.main(["--repo", "owner/repo"]) == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "write" in err


def _public_write_hub(monkeypatch):
    import huggingface_hub as hub

    hub.HfApi = type("A", (), {
        "__init__": lambda self, token=None: None,
        "whoami": lambda self: {"name": "testuser"},
        "repo_info": lambda self, repo_id=None, repo_type=None: types.SimpleNamespace(private=False),
        "auth_check": lambda self, repo_id=None, repo_type=None, write=False: None,
    })
    monkeypatch.setenv("HF_TOKEN_WRITE", "hf_testtoken")


def test_main_blocks_public_repo_without_opt_in(monkeypatch, capsys):
    """Exit 0 must never mean 'public repo with write access is fine'."""
    _install_fake_hub(monkeypatch, exists=True, private=False, write=True)
    _public_write_hub(monkeypatch)
    assert chk.main(["--repo", "owner/repo"]) == 1
    assert "PUBLIC" in capsys.readouterr().err


def test_main_accepts_public_repo_only_with_explicit_opt_in(monkeypatch, capsys):
    _install_fake_hub(monkeypatch, exists=True, private=False, write=True)
    _public_write_hub(monkeypatch)
    assert chk.main(["--repo", "owner/repo", "--allow-public-repo"]) == 0


def test_token_never_printed(monkeypatch, capsys):
    _install_fake_hub(monkeypatch, exists=True)
    monkeypatch.setenv("HF_TOKEN_WRITE", "hf_supersecret_token_xyz")
    chk.main(["--repo", "owner/repo"])
    out = capsys.readouterr()
    assert "hf_supersecret_token_xyz" not in out.out
    assert "hf_supersecret_token_xyz" not in out.err
