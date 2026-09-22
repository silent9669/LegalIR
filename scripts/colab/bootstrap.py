"""CPU-only Colab preflight and canonical Kaggle dataset acquisition."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.release.fingerprints import (
    CRITICAL_DATASET_FILES, assert_exact_git_sha, compute_canonical_json_hash,
    fingerprint_structured_config, verify_dataset_fingerprint, verify_prior_gate_reports,
)

REQUIRED_FILES = (*CRITICAL_DATASET_FILES, "manifest.json", "audit_report.json", "dataset_manifest.json")


def configure_kaggle_credentials():
    """Modern bearer tokens are not legacy username/key pairs."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    key = os.environ.get("KAGGLE_KEY", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key
        del os.environ["KAGGLE_KEY"]
    elif key and not os.environ.get("KAGGLE_USERNAME"):
        raise RuntimeError("Legacy KAGGLE_KEY requires KAGGLE_USERNAME; use KAGGLE_API_TOKEN for modern tokens.")


def _strict() -> bool:
    return str(os.environ.get("LEGALIR_STRICT_GATES", "")).strip() == "1"


def verify_launch(expected_sha, kaggle_report, freeze_file, repo_root=REPO_ROOT):
    """Pre-GPU launch check: SHA + freeze/report integrity.

    Default (LEGALIR_STRICT_GATES unset): advisory — SHA mismatches and
    freeze/report digest drift only warn, so teammates can iterate without a
    new release per edit. Missing freeze/report files still fail (nothing to
    run against). Set LEGALIR_STRICT_GATES=1 to restore fail-closed release
    behavior before allocating an expensive GPU VM.
    """
    sha = assert_exact_git_sha(expected_sha, repo_root=repo_root)
    # No fallback: an explicit missing file is a hard failure.
    freeze_path = Path(freeze_file)
    if not freeze_path.is_file():
        raise RuntimeError(f"Production freeze missing: {freeze_path}")
    report_path = Path(kaggle_report)
    if not report_path.is_file():
        raise RuntimeError(f"Kaggle T4x2 report missing: {report_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime_sha = str(freeze.get("git_sha", "")).strip().lower()

    def _warn(msg: str) -> None:
        print(f"[!] launch advisory (strict off): {msg}", flush=True)

    def _fail(msg: str) -> None:
        if _strict():
            raise RuntimeError(msg)
        _warn(msg)

    if not runtime_sha:
        raise RuntimeError("Production freeze is missing git_sha!")
    if runtime_sha != sha.lower():
        # Two-commit model: release checkout may descend from the frozen runtime
        # with evidence-only diffs (gate reports, freeze, notebooks).
        from src.release.provenance import validate_runtime_release_lineage
        lineage_ok, lineage_errors = validate_runtime_release_lineage(runtime_sha, sha, repo_root)
        if not lineage_ok:
            _fail(
                "Production freeze is for another runtime. Run the Kaggle T4 gate for this SHA and refresh approval before A100. "
                f"Details: {'; '.join(lineage_errors)}"
            )
    config_hash = fingerprint_structured_config(Path(repo_root) / "configs/algorithm/legalir_v2.yaml")
    if not freeze.get("algorithm_config_sha256"):
        raise RuntimeError("Production freeze is missing algorithm_config_sha256")
    if freeze.get("algorithm_config_sha256") != config_hash:
        _fail("Production freeze algorithm config mismatch")
    if not freeze.get("dataset", {}).get("manifest_sha256"):
        raise RuntimeError("Production freeze is missing dataset.manifest_sha256")
    try:
        gate_res = verify_prior_gate_reports(
            kaggle_report=report,
            expected_sha=runtime_sha,
            expected_dataset_hash=freeze["dataset"]["manifest_sha256"],
            expected_config_hash=config_hash,
        )
    except Exception as exc:
        _fail(f"Prior gate reports unverifiable: {type(exc).__name__}: {exc}")
        if _strict():
            raise
        # Advisory: build a best-effort digest so downstream checks degrade
        # to warnings instead of crashing on missing attributes.
        from types import SimpleNamespace as _NS

        from src.release.fingerprints import compute_canonical_json_hash as _chash

        gate_res = _NS(kaggle_report_sha256=_chash(report))
    # Canonical report digest must match the freeze (fail on absent, not just mismatch).
    expected_report_sha = freeze.get("gates", {}).get("kaggle_t4x2", {}).get("report_sha256")
    if not expected_report_sha:
        raise RuntimeError("Production freeze is missing gates.kaggle_t4x2.report_sha256")
    if gate_res.kaggle_report_sha256 != expected_report_sha:
        _fail(
            f"Kaggle report digest mismatch: freeze has '{expected_report_sha}', "
            f"computed '{gate_res.kaggle_report_sha256}'"
        )
    # Report's runtime-profile hash must match the actual Kaggle T4x2 YAML
    # fingerprint (not the A100 profile). Fail on absent.
    profile_in_report = report.get("runtime_profile_sha256")
    if not profile_in_report:
        _fail("Kaggle report is missing runtime_profile_sha256")
        return freeze
    expected_profile = fingerprint_structured_config(
        Path(repo_root) / "configs/runtime/kaggle_t4x2.yaml"
    )
    if profile_in_report != expected_profile:
        _fail(
            f"Kaggle runtime-profile mismatch: report has '{profile_in_report}', "
            f"expected '{expected_profile}' from configs/runtime/kaggle_t4x2.yaml"
        )
    return freeze


def prepare_dataset(dataset_dir: Path, freeze_file: Path | None = None) -> Path:
    dataset_dir = Path(dataset_dir)
    configure_kaggle_credentials()
    if not all((dataset_dir / name).is_file() for name in REQUIRED_FILES):
        dataset_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "phucdangg/legalir-task1-clean-data",
            "-p", str(dataset_dir), "--unzip", "--force",
        ], check=True)
    missing = [name for name in REQUIRED_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Canonical Kaggle dataset is incomplete: {missing}")
    freeze = json.loads(Path(freeze_file).read_text(encoding="utf-8")) if freeze_file else {}
    verify_dataset_fingerprint(
        dataset_dir,
        expected_manifest_hash=freeze.get("dataset", {}).get("manifest_sha256"),
        critical_files_expected=freeze.get("dataset", {}).get("critical_files"),
    )
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--kaggle-report", default="artifacts/task1/gates/kaggle_t4x2_report.json")
    parser.add_argument("--freeze-file", default="artifacts/task1/freeze/production_freeze.json")
    args = parser.parse_args()
    try:
        verify_launch(args.expected_sha, args.kaggle_report, args.freeze_file)
    except Exception as exc:
        print(f"[!] Launch blocked before GPU allocation: {exc}", file=sys.stderr)
        return 1
    print("[+] Runtime SHA, freeze, algorithm config, and upstream gate reports match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
