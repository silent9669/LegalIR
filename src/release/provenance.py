"""
Release Provenance, SHA Validation, and Release Approval Artifact Governance.

Authoritative specification: LEGALIR_76BB_FINAL_RELEASE_PROVENANCE_SMOKE_REPAIR.md
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

SHA_REGEX = re.compile(r"^[0-9a-fA-F]{40}$")

RELEASE_ONLY_DIFF_ALLOWLIST: tuple[str, ...] = (
    "artifacts/task1/colab_smoke_report.json",
    "artifacts/task1/release_approval.json",
    "parameter_audit.json",
    "scripts/generate_kaggle_notebook.py",
    "legalir_training.ipynb",
    "kaggle_kernel_task1/legalir_training.ipynb",
)


def validate_sha(sha: str) -> bool:
    """Validate that a string is an exact 40-character hexadecimal Git commit SHA."""
    return bool(sha and isinstance(sha, str) and SHA_REGEX.match(sha.strip()))


@dataclasses.dataclass(frozen=True)
class ReleaseApproval:
    schema_version: int
    runtime_sha: str
    release_sha: str
    ci: dict[str, Any]
    colab: dict[str, Any]
    dataset: dict[str, Any]
    production: dict[str, Any]
    approved_for_kaggle_full: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def validate_release_approval(
    approval: Mapping[str, Any],
    git_root: Path | str | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate that a release approval artifact is strictly consistent across Git, CI, Colab, and Kaggle.
    """
    errors: list[str] = []

    runtime_sha = str(approval.get("runtime_sha", "")).strip().lower()
    release_sha = str(approval.get("release_sha", "")).strip().lower()

    if not validate_sha(runtime_sha):
        errors.append(f"Invalid runtime_sha format: '{runtime_sha}'. Must be exact 40-hex Git SHA.")
    if not validate_sha(release_sha):
        errors.append(f"Invalid release_sha format: '{release_sha}'. Must be exact 40-hex Git SHA.")

    ci_info = approval.get("ci", {})
    if not isinstance(ci_info, Mapping):
        errors.append("Missing or invalid 'ci' section in release approval.")
    else:
        ci_runtime_sha = str(ci_info.get("runtime_sha", "")).strip().lower()
        if ci_runtime_sha != runtime_sha:
            errors.append(f"CI runtime SHA mismatch: expected {runtime_sha}, got {ci_runtime_sha}")
        if ci_info.get("conclusion") != "success":
            errors.append(f"CI conclusion must be 'success', got '{ci_info.get('conclusion')}'")

    colab_info = approval.get("colab", {})
    if not isinstance(colab_info, Mapping):
        errors.append("Missing or invalid 'colab' section in release approval.")
    else:
        colab_runtime_sha = str(colab_info.get("runtime_sha", "")).strip().lower()
        if colab_runtime_sha != runtime_sha:
            errors.append(f"Colab runtime SHA mismatch: expected {runtime_sha}, got {colab_runtime_sha}")
        if colab_info.get("result") != "PASS":
            errors.append(f"Colab result must be 'PASS', got '{colab_info.get('result')}'")
        report_sha = str(colab_info.get("report_sha256", "")).strip()
        if not report_sha or len(report_sha) != 64:
            errors.append(f"Invalid colab report_sha256: '{report_sha}'")

    prod_info = approval.get("production", {})
    if not isinstance(prod_info, Mapping):
        errors.append("Missing or invalid 'production' section in release approval.")
    else:
        kaggle_expected = str(prod_info.get("kaggle_expected_commit", "")).strip().lower()
        if kaggle_expected != runtime_sha:
            errors.append(
                f"Kaggle EXPECTED_COMMIT mismatch: must pin approved runtime SHA '{runtime_sha}', got '{kaggle_expected}'"
            )
        if not prod_info.get("dual_gpu_required", False):
            errors.append("Production 'dual_gpu_required' must be True.")

    if not approval.get("approved_for_kaggle_full", False):
        errors.append("'approved_for_kaggle_full' must be True.")

    # Check Git diff allowlist between runtime_sha and release_sha
    if git_root is not None and runtime_sha and release_sha and runtime_sha != release_sha:
        git_path = Path(git_root)
        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "--name-only", f"{runtime_sha}..{release_sha}"],
                cwd=git_path,
                text=True,
            ).strip()
            changed_files = [f.strip() for f in diff_output.splitlines() if f.strip()]
            for f in changed_files:
                if f not in RELEASE_ONLY_DIFF_ALLOWLIST:
                    errors.append(
                        f"Disallowed file changed between approved runtime ({runtime_sha[:8]}) "
                        f"and release ({release_sha[:8]}): '{f}'. "
                        f"Only release artifacts may change without invalidating Colab T4 PASS."
                    )
        except Exception as exc:
            errors.append(f"Git diff inspection error between {runtime_sha} and {release_sha}: {exc}")

    is_valid = len(errors) == 0
    return is_valid, errors
