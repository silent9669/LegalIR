"""Regression tests for explicit HF_REPO_ID forwarding (fresh-account blocker).

Local .env HF_REPO_ID was never loaded/forwarded while remote defaulted to
the previous owner's repo. These tests pin the explicit contract:

  --hf-repo flag > HF_REPO_ID env > <repo>/.env > owner default,

validated at preflight, echoed (not secret) at dispatch, and confirmed at
the remote. New repos stay private; no public override is added here.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.release.hf_repo import (
    HF_REPO_DEFAULT,
    dotenv_hf_repo_id,
    parse_dotenv_file,
    resolve_hf_repo_id,
    validate_hf_repo_id,
)


# --- validate ---

def test_validate_accepts_owner_slash_repo():
    assert validate_hf_repo_id("my-user/legalir-task1-reranker") == "my-user/legalir-task1-reranker"
    assert validate_hf_repo_id("  a/b  ") == "a/b"
    assert validate_hf_repo_id(HF_REPO_DEFAULT) == HF_REPO_DEFAULT


@pytest.mark.parametrize("bad", [
    "", "   ", "noslash", "/repo", "owner/", "a/b/c", "a//b",
    "owner/repo with space", "owner/repo!", "owner\nrepo",
    "x" * 50 + "/" + "y" * 50,  # >96 chars
])
def test_validate_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        validate_hf_repo_id(bad)


def test_validate_never_echoes_tokens():
    token = "hf_test_secret_12345"
    with pytest.raises(ValueError):
        validate_hf_repo_id("bad id with spaces")
    # Resolver errors mention the repo source, never token values.
    with pytest.raises(ValueError, match="env"):
        resolve_hf_repo_id(explicit=None, env={"HF_REPO_ID": "bad id"}, repo_root=None)
    # Token present in env must not leak into the error text.
    try:
        resolve_hf_repo_id(explicit=None, env={"HF_REPO_ID": "bad id", "HF_TOKEN_WRITE": token}, repo_root=None)
    except ValueError as exc:
        assert token not in str(exc)


# --- dotenv ---

def test_parse_dotenv_handles_quotes_export_and_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "HF_TOKEN_WRITE=hf_secret_should_not_be_logged\n"
        "export HF_REPO_ID=\"my-user/my-repo\"  # trailing comment\n"
        "OTHER='quoted value'\n",
        encoding="utf-8",
    )
    vals = parse_dotenv_file(env)
    assert vals["HF_REPO_ID"] == "my-user/my-repo"
    assert vals["OTHER"] == "quoted value"
    assert dotenv_hf_repo_id(tmp_path) == "my-user/my-repo"


def test_parse_dotenv_missing_file_returns_empty(tmp_path):
    assert parse_dotenv_file(tmp_path / "nope.env") == {}
    assert dotenv_hf_repo_id(tmp_path / "no-dir") is None


# --- resolve precedence ---

def test_resolve_explicit_wins_over_env_and_dotenv(tmp_path):
    (tmp_path / ".env").write_text("HF_REPO_ID=dotenv-user/dotenv-repo\n", encoding="utf-8")
    repo, src = resolve_hf_repo_id(
        explicit="flag-user/flag-repo",
        env={"HF_REPO_ID": "env-user/env-repo"},
        repo_root=tmp_path,
    )
    assert (repo, src) == ("flag-user/flag-repo", "explicit")


def test_resolve_env_wins_over_dotenv(tmp_path):
    (tmp_path / ".env").write_text("HF_REPO_ID=dotenv-user/dotenv-repo\n", encoding="utf-8")
    repo, src = resolve_hf_repo_id(explicit=None, env={"HF_REPO_ID": "env-user/env-repo"}, repo_root=tmp_path)
    assert (repo, src) == ("env-user/env-repo", "env")


def test_resolve_dotenv_used_when_no_flag_or_env(tmp_path):
    (tmp_path / ".env").write_text("HF_REPO_ID=dotenv-user/dotenv-repo\n", encoding="utf-8")
    repo, src = resolve_hf_repo_id(explicit=None, env={}, repo_root=tmp_path)
    assert (repo, src) == ("dotenv-user/dotenv-repo", "dotenv")


def test_resolve_missing_id_falls_back_to_default():
    repo, src = resolve_hf_repo_id(explicit=None, env={}, repo_root=None)
    assert (repo, src) == (HF_REPO_DEFAULT, "default")


def test_resolve_allow_default_false_blocks_fallback():
    with pytest.raises(ValueError, match="not set explicitly"):
        resolve_hf_repo_id(explicit=None, env={}, repo_root=None, allow_default=False)
    # Explicit sources still resolve when the default is blocked.
    repo, src = resolve_hf_repo_id(explicit="a/b", env={}, repo_root=None, allow_default=False)
    assert (repo, src) == ("a/b", "explicit")


def test_check_hf_repo_ready_gate():
    rf = _load_run_full()
    assert rf.check_hf_repo_ready("a/b", "explicit", False) is None
    assert rf.check_hf_repo_ready("a/b", "env", False) is None
    assert rf.check_hf_repo_ready(HF_REPO_DEFAULT, "default", True) is None
    err = rf.check_hf_repo_ready(HF_REPO_DEFAULT, "default", False)
    assert err is not None and "BLOCKED" in err
    assert "--hf-repo" in err and "--allow-default-hf-repo" in err


def test_resolve_invalid_env_reports_source():
    with pytest.raises(ValueError, match="env"):
        resolve_hf_repo_id(explicit=None, env={"HF_REPO_ID": "not valid!!"}, repo_root=None)


def test_resolve_invalid_dotenv_reports_source(tmp_path):
    (tmp_path / ".env").write_text("HF_REPO_ID=bad id here\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.env"):
        resolve_hf_repo_id(explicit=None, env={}, repo_root=tmp_path)


# --- run_full.py forwarding ---

def _load_run_full():
    for mod in ("scripts.modal.run_full", "scripts.modal"):
        sys.modules.pop(mod, None)
    import scripts.modal.run_full as rf

    return importlib.reload(rf)


def test_run_full_build_forward_args_includes_hf_repo():
    rf = _load_run_full()
    args = types.SimpleNamespace(
        private=True, push_config=True, detach=False,
        warm=True, warm_only=False, hf_allow_public_repo=False,
        allow_default_hf_repo=False,
        hf_repo="fresh-user/fresh-repo",
    )
    fwd = rf.build_forward_args(args)
    assert "--hf-repo" in fwd
    assert "fresh-user/fresh-repo" in fwd
    assert "--private" in fwd and "--push-config" in fwd and "--warm" in fwd


def test_run_full_build_forward_args_includes_allow_default_opt_out():
    rf = _load_run_full()
    args = types.SimpleNamespace(
        private=False, push_config=False, detach=False,
        warm=False, warm_only=False, hf_allow_public_repo=False,
        allow_default_hf_repo=True, hf_repo=None,
    )
    assert "--allow-default-hf-repo" in rf.build_forward_args(args)


def test_run_full_build_forward_args_omits_hf_repo_when_unset():
    rf = _load_run_full()
    args = types.SimpleNamespace(
        private=False, push_config=False, detach=False,
        warm=False, warm_only=False, hf_allow_public_repo=False,
        allow_default_hf_repo=False, hf_repo=None,
    )
    assert "--hf-repo" not in rf.build_forward_args(args)


def test_run_full_resolve_dispatch_prefers_flag_env_dotenv(monkeypatch, tmp_path):
    rf = _load_run_full()
    monkeypatch.setattr(rf, "REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text("HF_REPO_ID=dotenv-user/dotenv-repo\n", encoding="utf-8")
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    repo, src = rf.resolve_dispatch_hf_repo("flag-user/flag-repo")
    assert (repo, src) == ("flag-user/flag-repo", "explicit")
    monkeypatch.setenv("HF_REPO_ID", "env-user/env-repo")
    repo, src = rf.resolve_dispatch_hf_repo(None)
    assert (repo, src) == ("env-user/env-repo", "env")


def test_run_full_resolve_dispatch_rejects_invalid(monkeypatch, tmp_path):
    rf = _load_run_full()
    monkeypatch.setattr(rf, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    with pytest.raises(ValueError):
        rf.resolve_dispatch_hf_repo("bad id!!")


# --- run_modal_a100 remote resolution ---

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


def _load_launcher():
    _install_fake_modal()
    for mod in ("scripts.modal.run_modal_a100", "scripts.modal"):
        sys.modules.pop(mod, None)
    import scripts.modal.run_modal_a100 as m

    return importlib.reload(m)


def test_launcher_resolve_hf_repo_precedence(monkeypatch, tmp_path):
    mod = _load_launcher()
    monkeypatch.setenv("LEGALIR_MODAL_REPO_DIR", str(tmp_path))
    (tmp_path / ".env").write_text("HF_REPO_ID=dotenv-user/dotenv-repo\n", encoding="utf-8")
    monkeypatch.setenv("HF_REPO_ID", "env-user/env-repo")
    repo, src = mod._resolve_hf_repo("flag-user/flag-repo")
    assert (repo, src) == ("flag-user/flag-repo", "explicit")
    repo, src = mod._resolve_hf_repo(None)
    assert (repo, src) == ("env-user/env-repo", "env")
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    repo, src = mod._resolve_hf_repo(None)
    assert (repo, src) == ("dotenv-user/dotenv-repo", "dotenv")
    with pytest.raises(ValueError):
        mod._resolve_hf_repo("bad id!!")


def test_launcher_remote_signature_keeps_backward_compat():
    import inspect

    mod = _load_launcher()
    sig = inspect.signature(mod.run_production_training)
    assert "hf_repo" in sig.parameters
    assert sig.parameters["hf_repo"].default is None
    # Old positional calls still bind correctly.
    bound = sig.bind("a" * 40, False, False, None)
    assert bound.arguments["expected_sha"] == "a" * 40
    sig_main = inspect.signature(mod.main)
    assert "hf_repo" in sig_main.parameters
    assert sig_main.parameters["hf_repo"].default == ""
    assert "allow_default_hf_repo" in sig_main.parameters
    assert sig_main.parameters["allow_default_hf_repo"].default is False


def test_launcher_remote_uses_explicit_repo(monkeypatch, tmp_path):
    """Remote must preflight the explicit repo, not the owner default."""
    mod = _load_launcher()
    volume_root = tmp_path / "volume"
    volume_root.mkdir()
    repo_dir = tmp_path / "LegalIR"
    repo_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    monkeypatch.setattr(mod, "VOLUME_MOUNT", str(volume_root))
    monkeypatch.setenv("LEGALIR_MODAL_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("LEGALIR_MODAL_DATASET_DIR", str(dataset_dir))
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.setattr(mod.os, "chdir", lambda *a, **k: None)
    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    import scripts.colab.bootstrap as boot
    import scripts.gates.run_a100 as gate
    import scripts.run_colab_train as wrapper

    seen = {}

    monkeypatch.setattr(boot, "verify_launch", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(boot, "prepare_dataset", lambda *a, **k: Path(a[0]))
    def _preflight(repo_id, token=None, allow_public_repo=False):
        seen["repo_id"] = repo_id
        return True, "ok"
    monkeypatch.setattr(gate, "preflight_huggingface_access", _preflight)
    monkeypatch.setattr(
        wrapper, "run_colab_production_training",
        lambda **kw: {"status": "COMPLETED", "verdict": "PASS",
                      "huggingface": {"uploaded": True}},
    )

    class _V:
        def commit(self):
            return None
    monkeypatch.setattr(mod, "volume", _V())
    mod.run_production_training("a" * 40, hf_repo="fresh-user/fresh-repo")
    assert seen.get("repo_id") == "fresh-user/fresh-repo"
    # Remote echoes the resolved repo for log confirmation.
    # (Validated by the preflight receiving the explicit value.)


def test_launcher_remote_rejects_invalid_repo(monkeypatch, tmp_path):
    mod = _load_launcher()
    volume_root = tmp_path / "volume"
    volume_root.mkdir()
    monkeypatch.setattr(mod, "VOLUME_MOUNT", str(volume_root))
    monkeypatch.setenv("LEGALIR_MODAL_REPO_DIR", str(tmp_path / "repo"))
    monkeypatch.setenv("LEGALIR_MODAL_DATASET_DIR", str(tmp_path / "ds"))
    monkeypatch.setattr(mod.os, "chdir", lambda *a, **k: None)
    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="repo ID"):
        mod.run_production_training("a" * 40, hf_repo="bad id!!")


# --- attach cache identity ---

def test_attach_rejects_revision_mismatch_and_falls_back(tmp_path, monkeypatch):
    mod = _load_launcher()
    vol = tmp_path / "vol"
    models = vol / "shared" / "models" / "huggingface"
    models.mkdir(parents=True)
    snap_r = models / "snap-reranker"
    snap_d = models / "snap-dense"
    snap_r.mkdir()
    snap_d.mkdir()
    import json as _json

    (models / "manifest.json").write_text(_json.dumps({
        "BAAI/bge-reranker-v2-m3": {"path": str(snap_r), "revision": "STALE"},
        "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2": {"path": str(snap_d), "revision": "STALE"},
    }), encoding="utf-8")
    # Source gate passes so the revision check is what rejects.
    (vol / "shared" / "warm_manifest.json").write_text(_json.dumps({
        "requested_label": "a" * 40, "source_sha": "a" * 40, "hf_repo": "someone/repo",
    }), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo, expected_sha="a" * 40)
    assert out["models_attached"] is False
    assert "revision-mismatch" in str(out.get("models_detail", ""))
    assert "HF_HUB_CACHE" not in os.environ


def test_attach_records_fallback_summary_keys(tmp_path, monkeypatch):
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
    assert "dataset_detail" in out and "warm_source_sha" in out


def test_attach_rejects_empty_path_and_revision(tmp_path, monkeypatch):
    """Reviewer repro: path="", revision="" must NOT verify."""
    mod = _load_launcher()
    vol = tmp_path / "vol"
    models = vol / "shared" / "models" / "huggingface"
    models.mkdir(parents=True)
    import json as _json

    from src.models.bootstrap import MODEL_REGISTRY as _REG

    (models / "manifest.json").write_text(_json.dumps({
        mid: {"path": "", "revision": ""} for mid in _REG
    }), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo)
    assert out["models_attached"] is False
    assert out["models_detail"] != "pinned-revisions-verified"
    assert "HF_HUB_CACHE" not in os.environ


def test_attach_requires_every_registry_model(tmp_path, monkeypatch):
    """A manifest missing one registry model is not a verified hit."""
    mod = _load_launcher()
    vol = tmp_path / "vol"
    models = vol / "shared" / "models" / "huggingface"
    models.mkdir(parents=True)
    import json as _json

    from src.models.bootstrap import MODEL_REGISTRY as _REG

    mids = list(_REG.keys())
    assert len(mids) >= 2
    first = mids[0]
    snap = models / "snap"
    snap.mkdir()
    (models / "manifest.json").write_text(_json.dumps({
        first: {"path": str(snap), "revision": _REG[first]["revision"]},
    }), encoding="utf-8")
    # Source gate passes so the missing-model check is what rejects.
    (vol / "shared" / "warm_manifest.json").write_text(_json.dumps({
        "requested_label": "a" * 40, "source_sha": "a" * 40, "hf_repo": "someone/repo",
    }), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "LEGALIR_MODAL_DATASET_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = mod.attach_warmed_cache(vol, repo, expected_sha="a" * 40)
    assert out["models_attached"] is False
    assert "absent" in str(out.get("models_detail", ""))


# --- shell wrapper forwarding ---

REAL_WRAPPER = Path("scripts/modal/run_modal_cli.sh").resolve()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts/modal").mkdir(parents=True)
    (repo / "scripts/colab").mkdir(parents=True)
    shutil.copy2(REAL_WRAPPER, repo / "scripts/modal/run_modal_cli.sh")
    (repo / "scripts/modal/run_modal_cli.sh").chmod(0o755)
    return repo


def _git(repo: Path, *args: str):
    env = os.environ.copy()
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return subprocess.check_output(["git", *args], cwd=str(repo), env=env, text=True).strip()


def _init_git_repo(repo: Path):
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")


def _write_fakes(bin_dir: Path):
    bin_dir.mkdir(parents=True, exist_ok=True)
    log = bin_dir / "modal.log"
    py_stub = bin_dir / "py_stub"
    modal_fake = bin_dir / "modal"
    py_stub.write_text(f"#!/bin/sh\necho \"$@\" >> \"{bin_dir / 'py.log'}\"\nexit 0\n", encoding="utf-8")
    py_stub.chmod(0o755)
    modal_fake.write_text(
        "#!/bin/sh\n"
        f"echo \"$@\" >> \"{log}\"\n"
        "echo \"ENV_SHA:$LEGALIR_COMMIT_SHA\" >> \"" + str(log) + "\"\n"
        "echo \"ENV_HF_REPO:$HF_REPO_ID\" >> \"" + str(log) + "\"\n"
        "echo \"ENV_RERANKER_CFG:$LEGALIR_RERANKER_CONFIG\" >> \"" + str(log) + "\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    modal_fake.chmod(0o755)
    return py_stub, modal_fake, log


def _run(repo: Path, bin_dir: Path, args=(), extra_env=None):
    env = os.environ.copy()
    py_stub, modal_fake, log = _write_fakes(bin_dir)
    env["PYTHON_BIN"] = str(py_stub)
    env["MODAL_BIN"] = str(modal_fake)
    env["GIT_CEILING_DIRECTORIES"] = str(repo.parent)
    env.pop("LEGALIR_COMMIT_SHA", None)
    env.pop("LEGALIR_STRICT_GATES", None)
    env.pop("HF_REPO_ID", None)
    env.pop("LEGALIR_ALLOW_DEFAULT_HF_REPO", None)
    for k, v in (extra_env or {}).items():
        env[k] = v
    script = repo / "scripts/modal/run_modal_cli.sh"
    res = subprocess.run([str(script), *args], cwd=str(repo), env=env,
                         capture_output=True, text=True, timeout=20)
    return res, log


def test_cli_forwards_explicit_hf_repo(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin", args=("--hf-repo", "fresh-user/fresh-repo"))
    assert res.returncode == 0, res.stderr
    text = log.read_text(encoding="utf-8")
    assert "--hf-repo" in text
    assert "fresh-user/fresh-repo" in text
    assert "ENV_HF_REPO:fresh-user/fresh-repo" in text


def test_cli_forwards_equals_form_and_env(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin", args=("--hf-repo=fresh-user/equals-form",))
    assert res.returncode == 0, res.stderr
    assert "fresh-user/equals-form" in log.read_text(encoding="utf-8")

    res, log = _run(repo, tmp_path / "bin2",
                    extra_env={"HF_REPO_ID": "env-user/env-repo"})
    assert res.returncode == 0, res.stderr
    text = log.read_text(encoding="utf-8")
    assert "env-user/env-repo" in text


def test_cli_reads_dotenv_when_no_flag_or_env(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    (repo / ".env").write_text("HF_REPO_ID=dotenv-user/dotenv-repo\n", encoding="utf-8")
    res, log = _run(repo, tmp_path / "bin")
    assert res.returncode == 0, res.stderr
    assert "dotenv-user/dotenv-repo" in log.read_text(encoding="utf-8")


def test_cli_missing_id_is_blocked_not_default(tmp_path):
    """Fresh-account gate: no flag/env/dotenv must FAIL, not silently default."""
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin")
    assert res.returncode == 2
    assert "BLOCKED" in (res.stderr or "")
    # No cloud dispatch happened.
    assert not log.is_file() or "run_modal_a100.py" not in log.read_text(encoding="utf-8")


def test_cli_missing_id_allowed_only_with_explicit_opt_out(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin", args=("--allow-default-hf-repo",))
    assert res.returncode == 0, res.stderr
    assert HF_REPO_DEFAULT in log.read_text(encoding="utf-8")

    res, log = _run(repo, tmp_path / "bin2",
                    extra_env={"LEGALIR_ALLOW_DEFAULT_HF_REPO": "1"})
    assert res.returncode == 0, res.stderr
    assert HF_REPO_DEFAULT in log.read_text(encoding="utf-8")


def test_cli_duplicate_allow_flag_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, _ = _run(repo, tmp_path / "bin",
                  args=("--allow-default-hf-repo", "--allow-default-hf-repo"))
    assert res.returncode == 2


def test_cli_rejects_invalid_hf_repo(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, _ = _run(repo, tmp_path / "bin", args=("--hf-repo", "bad id!!"))
    assert res.returncode == 2
    res, _ = _run(repo, tmp_path / "bin2",
                  extra_env={"HF_REPO_ID": "also bad!!"})
    assert res.returncode == 2
    res, _ = _run(repo, tmp_path / "bin3", args=("--hf-repo=",))
    assert res.returncode == 2


def test_cli_rejects_duplicate_hf_repo(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, _ = _run(repo, tmp_path / "bin",
                  args=("--hf-repo", "a/b", "--hf-repo", "c/d"))
    assert res.returncode == 2


def test_cli_warm_forwards_same_repo(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin",
                    args=("--warm-only", "--hf-repo", "fresh-user/fresh-repo"))
    assert res.returncode == 0, res.stderr
    text = log.read_text(encoding="utf-8")
    assert "warm_volume.py" in text
    assert "fresh-user/fresh-repo" in text
    assert "run_modal_a100.py" not in text


def test_cli_push_config_exports_v3_and_forwards_flag(tmp_path):
    """Production must select the v3 push config, never silently base r32."""
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin",
                    args=("--push-config", "--hf-repo", "fresh-user/fresh-repo"))
    assert res.returncode == 0, res.stderr
    text = log.read_text(encoding="utf-8")
    assert "--push-config" in text
    assert "ENV_RERANKER_CFG:configs/experiments/reranker_lora_v3_push.yaml" in text


def test_cli_default_config_does_not_select_v3(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin", args=("--hf-repo", "fresh-user/fresh-repo"))
    assert res.returncode == 0, res.stderr
    text = log.read_text(encoding="utf-8")
    assert "--push-config" not in text
    assert "reranker_lora_v3_push" not in text
