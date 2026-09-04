"""Cryptographic verifier for immutable production bundles."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from src.core.hashing import sha256_file
from src.core.manifests import BundleManifest


def verify_production_bundle(bundle_dir: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Cryptographically verify all artifacts inside the production bundle against
    the sealed bundle_manifest.json. Returns (is_valid, list_of_errors).
    """
    bundle_p = Path(bundle_dir)
    errors: List[str] = []

    manifest_p = bundle_p / "bundle_manifest.json"
    if not manifest_p.is_file():
        return False, [f"Missing bundle_manifest.json at {manifest_p}"]

    try:
        manifest = BundleManifest.load(manifest_p)
    except Exception as e:
        return False, [f"Failed to load bundle_manifest.json: {e}"]

    if manifest.status != "PASS":
        errors.append(f"Bundle manifest status is not PASS: '{manifest.status}'")

    if not manifest.files:
        errors.append("Bundle manifest contains no file entries.")

    for rel_path, meta in manifest.files.items():
        artifact_p = bundle_p / rel_path
        if not artifact_p.is_file():
            errors.append(f"Missing bundle file: {rel_path}")
            continue

        expected_sha = meta.get("sha256")
        expected_size = meta.get("size_bytes")

        actual_size = artifact_p.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            errors.append(f"Size mismatch on {rel_path}: expected {expected_size}, got {actual_size}")

        actual_sha = sha256_file(artifact_p)
        if expected_sha and actual_sha != expected_sha:
            errors.append(
                f"Digest mismatch on {rel_path}: expected {expected_sha}, got {actual_sha}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors
