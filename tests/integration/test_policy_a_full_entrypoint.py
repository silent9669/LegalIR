"""Policy-A FULL entrypoint: no legacy Kaggle/freeze evidence required.

Runs the real FULL gate entrypoint (scripts/gates/run_a100.py) in mock mode
on a checkout WITHOUT a valid Kaggle report or freeze file. Policy A must
proceed to the pipeline start and record skips; LEGALIR_STRICT_GATES=1 must
still refuse. Mock never becomes production validation (no fake PASS), and
wrong-SHA/dataset/model/config inputs stay fail-closed at their own checks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.gates.run_a100 import run_a100_production_gate
from src.release.fingerprints import get_git_head_sha


@pytest.fixture
def bare_dirs(tmp_path: Path) -> dict[str, Path]:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    out_dir = tmp_path / "out"
    return {"dataset_dir": dataset_dir, "out_dir": out_dir}


def test_policy_a_mock_full_without_legacy_evidence(bare_dirs, monkeypatch):
    """Fresh checkout, no Kaggle report, no freeze file: policy A proceeds."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    report = run_a100_production_gate(
        dataset_dir=bare_dirs["dataset_dir"],
        output_dir=bare_dirs["out_dir"],
        expected_sha=get_git_head_sha(),
        kaggle_report_path=bare_dirs["out_dir"].parent / "no-kaggle.json",
        freeze_file_path=bare_dirs["out_dir"].parent / "no-freeze.json",
        mock=True,
    )
    assert report["verdict"] == "DEBUG_ONLY"
    assert report["status"] == "MOCK_COMPLETED"
    assert report["provenance_policy"] == "A-ci-only"
    assert report["gates"]["kaggle_t4x2"]["verdict"] == "DEBUG_ONLY"
    assert (bare_dirs["out_dir"] / "submission.zip").is_file()
    assert (bare_dirs["out_dir"] / "run_manifest.json").is_file()


def test_policy_a_stale_freeze_is_attachment_not_gate(bare_dirs, tmp_path: Path, monkeypatch):
    """Stale freeze (old runtime) warns and proceeds under policy A."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    freeze = {"git_sha": "0" * 40,
              "dataset": {"manifest_sha256": "stale"},
              "algorithm_config_sha256": "stale"}
    f_path = tmp_path / "stale-freeze.json"
    f_path.write_text(json.dumps(freeze), encoding="utf-8")
    report = run_a100_production_gate(
        dataset_dir=bare_dirs["dataset_dir"],
        output_dir=bare_dirs["out_dir"],
        expected_sha=get_git_head_sha(),
        kaggle_report_path=tmp_path / "no-kaggle.json",
        freeze_file_path=f_path,
        mock=True,
    )
    assert report["verdict"] == "DEBUG_ONLY"
    saved = json.loads((bare_dirs["out_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
    # Mock skips freeze evaluation; non-mock policy A would mark drift-not-used.
    assert saved["freeze_status"] in ("mismatched-not-used", "drift-not-used", "absent",
                                      "present-mock-unchecked")


def test_strict_still_refuses_without_evidence(bare_dirs, tmp_path: Path, monkeypatch):
    """Opt-in strict mode keeps the old fail-closed behavior."""
    monkeypatch.setenv("LEGALIR_STRICT_GATES", "1")
    with pytest.raises(RuntimeError, match="Strict mode requires upstream evidence"):
        run_a100_production_gate(
            dataset_dir=bare_dirs["dataset_dir"],
            output_dir=bare_dirs["out_dir"],
            expected_sha=get_git_head_sha(),
            kaggle_report_path=tmp_path / "no-kaggle.json",
            freeze_file_path=tmp_path / "no-freeze.json",
            mock=True,
        )


def test_strict_refuses_wrong_sha(bare_dirs, tmp_path: Path, monkeypatch):
    """Wrong SHA fails closed under strict opt-in (before expensive work)."""
    monkeypatch.setenv("LEGALIR_STRICT_GATES", "1")
    with pytest.raises(Exception, match="SHA|sha"):
        run_a100_production_gate(
            dataset_dir=bare_dirs["dataset_dir"],
            output_dir=bare_dirs["out_dir"],
            expected_sha="1" * 40,
            kaggle_report_path=tmp_path / "no-kaggle.json",
            freeze_file_path=tmp_path / "no-freeze.json",
            mock=True,
        )


def test_policy_a_refuses_wrong_sha_even_without_strict(bare_dirs, tmp_path: Path, monkeypatch):
    """Policy A: exact source SHA is required independently of LEGALIR_STRICT_GATES."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    with pytest.raises(Exception, match="SHA|sha"):
        run_a100_production_gate(
            dataset_dir=bare_dirs["dataset_dir"],
            output_dir=bare_dirs["out_dir"],
            expected_sha="1" * 40,
            kaggle_report_path=tmp_path / "no-kaggle.json",
            freeze_file_path=tmp_path / "no-freeze.json",
            mock=True,
        )


def test_policy_a_malformed_freeze_warns_and_continues(bare_dirs, tmp_path: Path, monkeypatch, capsys):
    """Malformed freeze JSON warns and proceeds under Policy A."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    bad_freeze = tmp_path / "bad_freeze.json"
    bad_freeze.write_text("{not valid json", encoding="utf-8")
    report = run_a100_production_gate(
        dataset_dir=bare_dirs["dataset_dir"],
        output_dir=bare_dirs["out_dir"],
        expected_sha=get_git_head_sha(),
        kaggle_report_path=tmp_path / "no-kaggle.json",
        freeze_file_path=bad_freeze,
        mock=True,
    )
    assert report["verdict"] == "DEBUG_ONLY"
    assert "freeze file unreadable" in capsys.readouterr().out


def test_strict_malformed_freeze_raises(bare_dirs, tmp_path: Path, monkeypatch):
    """Malformed freeze JSON raises under strict gates."""
    monkeypatch.setenv("LEGALIR_STRICT_GATES", "1")
    bad_freeze = tmp_path / "bad_freeze.json"
    bad_freeze.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="freeze"):
        run_a100_production_gate(
            dataset_dir=bare_dirs["dataset_dir"],
            output_dir=bare_dirs["out_dir"],
            expected_sha=get_git_head_sha(),
            kaggle_report_path=tmp_path / "no-kaggle.json",
            freeze_file_path=bad_freeze,
            mock=True,
        )


def test_policy_a_malformed_kaggle_report_warns_and_continues(bare_dirs, tmp_path: Path, monkeypatch, capsys):
    """Malformed Kaggle report JSON warns and proceeds under Policy A."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    bad_k = tmp_path / "bad_k.json"
    bad_k.write_text("{not valid json", encoding="utf-8")
    report = run_a100_production_gate(
        dataset_dir=bare_dirs["dataset_dir"],
        output_dir=bare_dirs["out_dir"],
        expected_sha=get_git_head_sha(),
        kaggle_report_path=bad_k,
        freeze_file_path=tmp_path / "no-freeze.json",
        mock=True,
    )
    assert report["verdict"] == "DEBUG_ONLY"
    assert "Kaggle report unreadable" in capsys.readouterr().out


def test_explicit_missing_kaggle_path_warns_and_continues(bare_dirs, tmp_path: Path, monkeypatch, capsys):
    """An explicit-but-missing report path is a warning, not a silent default."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    missing = tmp_path / "explicit-missing.json"
    run_a100_production_gate(
        dataset_dir=bare_dirs["dataset_dir"],
        output_dir=bare_dirs["out_dir"],
        expected_sha=get_git_head_sha(),
        kaggle_report_path=missing,
        freeze_file_path=tmp_path / "no-freeze.json",
        mock=True,
    )
    assert "explicit Kaggle report path missing" in capsys.readouterr().out


def test_no_mock_production_claim_without_strict_env(bare_dirs, tmp_path: Path, monkeypatch):
    """Mock runs never claim production PASS/COMPLETED (no fake validation)."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    assert os.environ.get("LEGALIR_STRICT_GATES", "") != "1"
    report = run_a100_production_gate(
        dataset_dir=bare_dirs["dataset_dir"],
        output_dir=bare_dirs["out_dir"],
        expected_sha=get_git_head_sha(),
        kaggle_report_path=tmp_path / "no-kaggle.json",
        freeze_file_path=tmp_path / "no-freeze.json",
        mock=True,
    )
    assert report["status"] == "MOCK_COMPLETED"
    assert report["verdict"] == "DEBUG_ONLY"


def test_freeze_drift_keys_detects_stale_tuple():
    from scripts.gates.run_a100 import _freeze_drift_keys

    assert _freeze_drift_keys(
        {"git_sha": "a" * 40, "dataset": {"manifest_sha256": "m"},
         "algorithm_config_sha256": "c"},
        "a" * 40, "m", "c") == []
    assert _freeze_drift_keys(
        {"git_sha": "b" * 40, "dataset": {"manifest_sha256": "m"},
         "algorithm_config_sha256": "c"},
        "a" * 40, "m", "c") == ["git_sha"]
    assert _freeze_drift_keys(
        {"git_sha": "a" * 40, "dataset": {"manifest_sha256": "stale"},
         "algorithm_config_sha256": "stale"},
        "a" * 40, "m", "c") == ["dataset.manifest_sha256", "algorithm_config_sha256"]
    assert _freeze_drift_keys(
        {"git_sha": "a" * 40, "dataset": "malformed",
         "algorithm_config_sha256": "c"},
        "a" * 40, "m", "c") == ["dataset"]
    assert _freeze_drift_keys({}, "a" * 40, "m", "c") != []


def test_policy_a_non_dict_dataset_in_freeze_proceeds(bare_dirs, tmp_path: Path, monkeypatch):
    """Freeze with non-dict dataset value does not crash Policy A; marked as drift."""
    monkeypatch.delenv("LEGALIR_STRICT_GATES", raising=False)
    bad_freeze = tmp_path / "bad_ds_freeze.json"
    bad_freeze.write_text(json.dumps({"git_sha": get_git_head_sha(), "dataset": "malformed"}), encoding="utf-8")
    report = run_a100_production_gate(
        dataset_dir=bare_dirs["dataset_dir"],
        output_dir=bare_dirs["out_dir"],
        expected_sha=get_git_head_sha(),
        kaggle_report_path=tmp_path / "no-kaggle.json",
        freeze_file_path=bad_freeze,
        mock=True,
    )
    assert report["verdict"] == "DEBUG_ONLY"


def test_strict_non_dict_dataset_in_freeze_raises(bare_dirs, tmp_path: Path, monkeypatch):
    """Freeze with non-dict dataset value raises under strict mode."""
    monkeypatch.setenv("LEGALIR_STRICT_GATES", "1")
    bad_freeze = tmp_path / "bad_ds_freeze.json"
    bad_freeze.write_text(json.dumps({"git_sha": get_git_head_sha(), "dataset": "malformed"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        run_a100_production_gate(
            dataset_dir=bare_dirs["dataset_dir"],
            output_dir=bare_dirs["out_dir"],
            expected_sha=get_git_head_sha(),
            kaggle_report_path=tmp_path / "no-kaggle.json",
            freeze_file_path=bad_freeze,
            mock=True,
        )
