"""Unit tests for the read-only teammate preflight (all probes stubbed)."""
from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.preflight_teammate as pf


def _args(**kw):
    base = {"hf_repo": "owner/repo", "hf_allow_public_repo": True, "allow_default_hf_repo": False}
    base.update(kw)
    return Namespace(**base)


def _pass_probes(monkeypatch, tmp_path, *, live=None, strict=0):
    monkeypatch.setattr(pf, "git_local_head", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(pf, "git_origin_main", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(pf, "git_clean", lambda *a, **k: (True, []))
    monkeypatch.setattr(pf, "find_modal_bin", lambda *a, **k: "/fake/modal")
    monkeypatch.setattr(pf, "modal_version", lambda *a, **k: "modal 9.9.9")
    monkeypatch.setattr(pf, "modal_active_profile", lambda *a, **k: ("teammate", "fake"))
    monkeypatch.setattr(pf, "modal_secret_names",
                        lambda *a, **k: (["kaggle-secret", "huggingface-secret"], "fake"))
    monkeypatch.setattr(pf, "model_registry_info",
                        lambda *a, **k: [{"id": "m", "revision": "r" * 40}])
    monkeypatch.setattr(pf, "freeze_info",
                        lambda *a, **k: {"present": True, "git_sha": "a" * 40})
    monkeypatch.setattr(pf, "resolve_target_repo", lambda *a, **k: ("owner/repo", "explicit"))
    monkeypatch.setattr(pf, "strict_release_status", lambda *a, **k: (strict, "ok"))
    monkeypatch.setattr(
        pf, "hf_live_check",
        lambda *a, **k: dict(live or {"exists": True, "visibility": "public",
                                      "write_access": True, "preflight_verdict": "PASS"}),
    )
    monkeypatch.setenv("HF_TOKEN_WRITE", "hf_faketoken_for_tests_only")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return tmp_path


def _statuses(report):
    return {c["name"]: c["status"] for c in report["checks"]}


def test_all_pass_reports_train_command(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path)
    report, code = pf.collect_report(_args(), tmp_path)
    assert code == 0
    assert _statuses(report)["hf-write-visibility"] == "PASS"
    assert "--hf-repo owner/repo" in report["train_command"]
    assert "--hf-allow-public-repo" in report["train_command"]
    assert "--push-config" in report["train_command"] and "--private" in report["train_command"]


def test_never_prints_token(tmp_path, monkeypatch, capsys):
    _pass_probes(monkeypatch, tmp_path)
    report, _ = pf.collect_report(_args(), tmp_path)
    blob = json.dumps(report, default=str)
    assert "hf_faketoken_for_tests_only" not in blob
    assert pf.main(["--hf-repo", "owner/repo", "--hf-allow-public-repo", "--json"]) == 0
    out = capsys.readouterr().out
    assert "hf_faketoken_for_tests_only" not in out


def test_dirty_tree_blocked(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "git_clean", lambda *a, **k: (False, [" M foo.py"]))
    _, code = pf.collect_report(_args(), tmp_path)
    assert code == 2


def test_missing_repo_blocked(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path)

    def _boom(*a, **k):
        from src.release.hf_repo import resolve_hf_repo_id

        return resolve_hf_repo_id(explicit=None, env={}, repo_root=None, allow_default=False)

    monkeypatch.setattr(pf, "resolve_target_repo", _boom)
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    _, code = pf.collect_report(_args(hf_repo=None), tmp_path)
    assert code == 2


def test_missing_token_blocked(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path)
    monkeypatch.delenv("HF_TOKEN_WRITE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    report, code = pf.collect_report(_args(), tmp_path)
    assert code == 2
    assert _statuses(report)["hf-credentials"] == "BLOCKED"


def test_hf_deny_blocked(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path,
                 live={"exists": True, "visibility": "private",
                       "write_access": False, "preflight_verdict": "FAIL"})
    report, code = pf.collect_report(_args(hf_allow_public_repo=False), tmp_path)
    assert code == 2
    assert _statuses(report)["hf-write-visibility"] == "BLOCKED"


def test_public_without_optin_blocked(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path,
                 live={"exists": True, "visibility": "public",
                       "write_access": True, "preflight_verdict": "BLOCKED"})
    report, code = pf.collect_report(_args(hf_allow_public_repo=False), tmp_path)
    assert code == 2


def test_strict_fail_reported_not_silent(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path, strict=1)
    report, code = pf.collect_report(_args(), tmp_path)
    assert code == 0  # preflight itself passes; strict FAIL stays a production blocker
    assert _statuses(report)["strict-release"] == "FAIL"


def test_strict_probe_forces_strict_gates_env(tmp_path, monkeypatch):
    """The verifier is advisory without LEGALIR_STRICT_GATES=1; preflight must force it."""
    seen: dict = {}

    class _R:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(*a, **k):
        seen["env"] = k.get("env", {})
        return _R()

    monkeypatch.setattr(pf.subprocess, "run", _fake_run)
    code, _ = pf.strict_release_status(tmp_path)
    assert code == 0
    assert seen["env"].get("LEGALIR_STRICT_GATES") == "1"


def test_modal_missing_and_secrets_gap(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "find_modal_bin", lambda *a, **k: None)
    monkeypatch.delenv("MODAL_BIN", raising=False)
    _, code = pf.collect_report(_args(), tmp_path)
    assert code == 2

    _pass_probes(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "find_modal_bin", lambda *a, **k: "/fake/modal")
    monkeypatch.setattr(pf, "modal_secret_names", lambda *a, **k: (["kaggle-secret"], "fake"))
    report, code = pf.collect_report(_args(), tmp_path)
    assert code == 2
    assert _statuses(report)["modal-secrets"] == "BLOCKED"

    _pass_probes(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "modal_secret_names", lambda *a, **k: (None, "unreachable"))
    report, code = pf.collect_report(_args(), tmp_path)
    assert code == 0
    assert _statuses(report)["modal-secrets"] == "UNVERIFIED"


def test_origin_unreachable_is_unverified(tmp_path, monkeypatch):
    _pass_probes(monkeypatch, tmp_path)
    monkeypatch.setattr(pf, "git_origin_main", lambda *a, **k: None)
    report, code = pf.collect_report(_args(), tmp_path)
    assert code == 0
    assert _statuses(report)["git-origin-main"] == "UNVERIFIED"


def test_hf_live_verdict_mapping():
    assert pf.hf_live_check.__doc__ and "Never creates" in pf.hf_live_check.__doc__


def test_train_command_shape():
    cmd = pf.train_command("o/r", True)
    assert "--private" in cmd and "--push-config" in cmd and "--hf-allow-public-repo" in cmd
    assert "--hf-allow-public-repo" not in pf.train_command("o/r", False)
