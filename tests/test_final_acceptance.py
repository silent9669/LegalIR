import json
from pathlib import Path


def test_final_model_passes_all_acceptance_gates():
    model_path = Path("artifacts/shared/benchmarks/accepted/final_model.json")
    assert model_path.exists(), "final_model.json must exist in artifacts/shared/benchmarks/accepted/"
    report = json.loads(model_path.read_text(encoding="utf-8"))

    assert len(report["random_5fold"]["folds"]) == 5
    assert report["official_scorer_equivalent"] is True
    assert report["leakage_checks_passed"] is True

    # Baseline comparison
    baseline_path = Path("artifacts/shared/benchmarks/accepted/strict_baseline.json")
    assert baseline_path.exists()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert report["random_5fold"]["mean_recall_at_5"] >= baseline["random_5fold"]["mean_recall_at_5"]
    assert report["document_disjoint"]["recall_at_5"] >= baseline["document_disjoint"]["recall_at_5"] - 0.01


def test_final_submission_manifest_integrity():
    sub_manifest_path = Path("artifacts/shared/submissions/accepted/submission_manifest.json")
    assert sub_manifest_path.exists(), "submission_manifest.json must exist in artifacts/shared/submissions/accepted/"
    manifest = json.loads(sub_manifest_path.read_text(encoding="utf-8"))

    assert manifest["total_queries"] == 1000
    assert "submission_json_sha256" in manifest
    assert "submission_zip_sha256" in manifest
    assert manifest["compliance_verified"] is True
