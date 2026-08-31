"""Validate submission.json and submission.zip against strict competition rules and official scoring semantics."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.submission import validate_submission, validate_submission_zip
from src.models.parameter_audit import audit_system_parameters, MAX_PARAMETER_BUDGET


def main():
    parser = argparse.ArgumentParser(description="LegalIR Submission Validator")
    parser.add_argument("--submission-json", type=str, default="artifacts/task1/submissions/submission.json", help="Path to submission.json")
    parser.add_argument("--submission-zip", type=str, default="artifacts/task1/submissions/submission.zip", help="Path to submission.zip")
    parser.add_argument("--parameter-audit", type=str, default="artifacts/task1/submissions/parameter_audit.json", help="Path to parameter_audit.json")
    parser.add_argument("--public-json", type=str, default="public-official.json", help="Path to public-official.json")
    parser.add_argument("--data-dir", type=str, default="artifacts/task1/data", help="Path to canonical data dir")
    args = parser.parse_args()

    print("=" * 60)
    print("LegalIR Task 1: Strict Submission Validation")
    print("=" * 60)

    if not os.path.exists(args.public_json):
        for candidate in ["artifacts/raw/public-official.json", "public-official.json", "artifacts/shared/raw/public-official.json"]:
            if os.path.exists(candidate):
                args.public_json = candidate
                break

    json_path = Path(args.submission_json)
    if json_path.exists():
        print(f"\n[1/3] Validating JSON: {json_path}")
        json_report = validate_submission(
            predictions_or_file=json_path,
            public_json=args.public_json,
            data_dir=args.data_dir,
        )
        print(f"JSON validation status: is_valid = {json_report.get('is_valid')}")
        if not json_report.get("is_valid"):
            print(f"Errors: {json_report.get('errors')}")
            sys.exit(1)
        else:
            print(f"Total valid queries: {json_report.get('total_queries')}")
    else:
        print(f"Warning: JSON submission not found at {json_path}")

    zip_path = Path(args.submission_zip)
    if zip_path.exists():
        print(f"\n[2/3] Validating ZIP: {zip_path}")
        zip_report = validate_submission_zip(zip_path)
        print(f"ZIP validation status: is_valid = {zip_report.get('is_valid')}")
        if not zip_report.get("is_valid"):
            print(f"Errors: {zip_report.get('errors')}")
            sys.exit(1)
        else:
            print(f"ZIP archive structure: valid (contains only submission.json at root)")
    else:
        print(f"Warning: ZIP submission not found at {zip_path}")

    # Parameter Budget Audit
    audit_path = Path(args.parameter_audit)
    print(f"\n[3/3] Validating Parameter Budget (<4B Rule)")
    if audit_path.exists():
        with open(audit_path, "r", encoding="utf-8") as f:
            audit_data = json.load(f)
        total_p = audit_data.get("total_learned_parameters", 0)
        is_comp = audit_data.get("is_compliant", False)
        print(f"Loaded parameter audit: {total_p:,} parameters ({audit_data.get('total_parameters_billions', 0):.4f}B)")
        if not is_comp or total_p >= MAX_PARAMETER_BUDGET:
            print(f"CRITICAL ERROR: Parameter budget exceeded! total={total_p:,} >= {MAX_PARAMETER_BUDGET:,}")
            sys.exit(1)
        print(f"Parameter budget status: COMPLIANT ({audit_data.get('budget_utilization_pct', 0):.2f}% of 4B limit)")
    else:
        print(f"Performing live parameter budget audit...")
        report = audit_system_parameters(output_json=audit_path, raise_on_violation=True)
        print(f"Parameter budget status: COMPLIANT ({report['total_learned_parameters']:,} params, {report['budget_utilization_pct']:.2f}% of 4B limit)")

    print("\n" + "=" * 60)
    print("ALL SUBMISSION VALIDATION CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
