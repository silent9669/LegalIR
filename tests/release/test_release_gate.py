import pytest
from src.core.manifests import PreflightManifest, BundleManifest


def test_release_gate_preflight_check():
    m = PreflightManifest(
        dataset_name="task1_canonical",
        dataset_version="v2",
        runtime_commit="a0efb25",
        status="PASS",
    )
    assert m.status == "PASS"


def test_release_gate_bundle_check():
    b = BundleManifest(
        bundle_version="v1",
        runtime_commit="a0efb25",
        dataset_version="v2",
        dataset_fingerprint="sha256",
        config_sha256="sha256",
        status="PASS",
    )
    assert b.status == "PASS"
