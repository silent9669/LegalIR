import json
import pytest
from pathlib import Path
from src.bundle.builder import ProductionBundleBuilder
from src.bundle.verifier import verify_production_bundle


def test_bundle_build_and_verify(tmp_path):
    bundle_dir = tmp_path / "bundle"
    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit="test_commit_sha",
        dataset_fingerprint="test_dataset_sha",
    )

    # Add mock components
    pairs_file = tmp_path / "final_pairs.parquet"
    pairs_file.write_text("dummy pairs content", encoding="utf-8")
    builder.add_file("final_training_pairs.parquet", pairs_file)

    cands_file = tmp_path / "public_cands.parquet"
    cands_file.write_text("dummy cands content", encoding="utf-8")
    builder.add_file("public_candidates.parquet", cands_file)

    lock_file = tmp_path / "lock.json"
    lock_file.write_text('{"status": "LOCKED"}', encoding="utf-8")
    builder.add_file("production_lock.json", lock_file)

    manifest = builder.freeze()
    assert (bundle_dir / "bundle_manifest.json").is_file()
    assert manifest.status == "PASS"

    # Verifier check
    is_valid, errors = verify_production_bundle(bundle_dir)
    assert is_valid is True
    assert len(errors) == 0


def test_bundle_verifier_detects_tampered_file(tmp_path):
    bundle_dir = tmp_path / "tampered_bundle"
    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit="test_commit_sha",
        dataset_fingerprint="test_dataset_sha",
    )

    pairs_file = tmp_path / "final_pairs.parquet"
    pairs_file.write_text("original content", encoding="utf-8")
    builder.add_file("final_training_pairs.parquet", pairs_file)
    builder.freeze()

    # Tamper with file
    (bundle_dir / "final_training_pairs.parquet").write_text("tampered content", encoding="utf-8")

    is_valid, errors = verify_production_bundle(bundle_dir)
    assert is_valid is False
    assert any("Digest mismatch" in e for e in errors)
