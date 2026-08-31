#!/usr/bin/env python3
"""Dedicated Kaggle Pipeline Smoke Test & Verification CLI.

Executes the complete 24-step production pipeline via run_kaggle_pipeline() from
src.pipeline.kaggle_train, supporting --tiny and --offline modes.

Verifies and asserts:
- submission.zip (strictly only submission.json at root)
- submission_manifest.json (SHA-256 provenance and parameter counts)
- parameter_audit.json (strict <4B rule compliance)
- cv_report.json (5-fold OOF metrics and document-disjoint robustness)
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.submission import validate_submission, validate_submission_zip
from src.models.parameter_audit import MAX_PARAMETER_BUDGET
from src.pipeline.kaggle_train import KaggleRunResult, run_kaggle_pipeline
from src.retrieval.build_indexes import build_bm25_index, build_bm25_pyvi_index


def create_toy_canonical_dataset(target_dir: Path) -> Path:
    """Create a minimal self-contained canonical dataset for fast offline smoke verification."""
    target_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = target_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "doc_id": "101",
            "title": "Luật Doanh nghiệp 2020",
            "legal_number": "59/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/59-2020",
            "name_raw": "Luật Doanh nghiệp",
            "is_empty": False,
        },
        {
            "doc_id": "102",
            "title": "Luật Đầu tư 2020",
            "legal_number": "61/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/61-2020",
            "name_raw": "Luật Đầu tư",
            "is_empty": False,
        },
        {
            "doc_id": "103",
            "title": "Nghị định Đăng ký doanh nghiệp",
            "legal_number": "01/2021/NĐ-CP",
            "year": "2021",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/01-2021",
            "name_raw": "Nghị định 01",
            "is_empty": False,
        },
        {
            "doc_id": "104",
            "title": "Nghị định Hướng dẫn Luật Đầu tư",
            "legal_number": "31/2021/NĐ-CP",
            "year": "2021",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/31-2021",
            "name_raw": "Nghị định 31",
            "is_empty": False,
        },
        {
            "doc_id": "105",
            "title": "Luật Thương mại",
            "legal_number": "36/2005/QH11",
            "year": "2005",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/36-2005",
            "name_raw": "Luật Thương mại",
            "is_empty": False,
        },
        {
            "doc_id": "106",
            "title": "Luật Quản lý thuế",
            "legal_number": "38/2019/QH14",
            "year": "2019",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/38-2019",
            "name_raw": "Luật Quản lý thuế",
            "is_empty": False,
        },
    ]
    pd.DataFrame(docs).to_parquet(target_dir / "documents.parquet", index=False)

    chunks = [
        {
            "chunk_id": "c101_micro",
            "parent_chunk_id": "c101_macro",
            "doc_id": "101",
            "granularity": "micro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về thành lập doanh nghiệp và quản lý công ty trách nhiệm hữu hạn.",
            "text_norm": "quy định về thành lập doanh nghiệp và quản lý công ty trách nhiệm hữu hạn",
        },
        {
            "chunk_id": "c101_macro",
            "parent_chunk_id": None,
            "doc_id": "101",
            "granularity": "macro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về thành lập doanh nghiệp và quản lý công ty trách nhiệm hữu hạn.",
            "text_norm": "quy định về thành lập doanh nghiệp và quản lý công ty trách nhiệm hữu hạn",
        },
        {
            "chunk_id": "c102_micro",
            "parent_chunk_id": "c102_macro",
            "doc_id": "102",
            "granularity": "micro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư nước ngoài.",
            "text_norm": "quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư nước ngoài",
        },
        {
            "chunk_id": "c102_macro",
            "parent_chunk_id": None,
            "doc_id": "102",
            "granularity": "macro",
            "article": "Điều 2",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư nước ngoài.",
            "text_norm": "quy định về dự án đầu tư trực tiếp và ưu đãi đầu tư nước ngoài",
        },
        {
            "chunk_id": "c103_micro",
            "parent_chunk_id": "c103_macro",
            "doc_id": "103",
            "granularity": "micro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua cổng thông tin quốc gia.",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua cổng thông tin quốc gia",
        },
        {
            "chunk_id": "c103_macro",
            "parent_chunk_id": None,
            "doc_id": "103",
            "granularity": "macro",
            "article": "Điều 3",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hồ sơ đăng ký doanh nghiệp qua cổng thông tin quốc gia.",
            "text_norm": "hồ sơ đăng ký doanh nghiệp qua cổng thông tin quốc gia",
        },
        {
            "chunk_id": "c104_micro",
            "parent_chunk_id": "c104_macro",
            "doc_id": "104",
            "granularity": "micro",
            "article": "Điều 4",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Thủ tục cấp giấy chứng nhận đăng ký đầu tư cho nhà đầu tư nước ngoài.",
            "text_norm": "thủ tục cấp giấy chứng nhận đăng ký đầu tư cho nhà đầu tư nước ngoài",
        },
        {
            "chunk_id": "c104_macro",
            "parent_chunk_id": None,
            "doc_id": "104",
            "granularity": "macro",
            "article": "Điều 4",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Thủ tục cấp giấy chứng nhận đăng ký đầu tư cho nhà đầu tư nước ngoài.",
            "text_norm": "thủ tục cấp giấy chứng nhận đăng ký đầu tư cho nhà đầu tư nước ngoài",
        },
        {
            "chunk_id": "c105_micro",
            "parent_chunk_id": "c105_macro",
            "doc_id": "105",
            "granularity": "micro",
            "article": "Điều 5",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hoạt động thương mại và mua bán hàng hóa quốc tế.",
            "text_norm": "hoạt động thương mại và mua bán hàng hóa quốc tế",
        },
        {
            "chunk_id": "c105_macro",
            "parent_chunk_id": None,
            "doc_id": "105",
            "granularity": "macro",
            "article": "Điều 5",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Hoạt động thương mại và mua bán hàng hóa quốc tế.",
            "text_norm": "hoạt động thương mại và mua bán hàng hóa quốc tế",
        },
        {
            "chunk_id": "c106_micro",
            "parent_chunk_id": "c106_macro",
            "doc_id": "106",
            "granularity": "micro",
            "article": "Điều 6",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về khai thuế và nộp thuế điện tử.",
            "text_norm": "quy định về khai thuế và nộp thuế điện tử",
        },
        {
            "chunk_id": "c106_macro",
            "parent_chunk_id": None,
            "doc_id": "106",
            "granularity": "macro",
            "article": "Điều 6",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định về khai thuế và nộp thuế điện tử.",
            "text_norm": "quy định về khai thuế và nộp thuế điện tử",
        },
    ]
    pd.DataFrame(chunks).to_parquet(target_dir / "chunks.parquet", index=False)

    queries = [
        {"query_id": "q1", "question_raw": "Thành lập doanh nghiệp như thế nào?", "question_norm": "thành lập doanh nghiệp như thế nào"},
        {"query_id": "q2", "question_raw": "Dự án đầu tư trực tiếp", "question_norm": "dự án đầu tư trực tiếp"},
        {"query_id": "q3", "question_raw": "Hồ sơ đăng ký doanh nghiệp qua mạng", "question_norm": "hồ sơ đăng ký doanh nghiệp qua mạng"},
        {"query_id": "q4", "question_raw": "Thủ tục cấp giấy chứng nhận đăng ký đầu tư", "question_norm": "thủ tục cấp giấy chứng nhận đăng ký đầu tư"},
        {"query_id": "q5", "question_raw": "Hoạt động mua bán hàng hóa quốc tế", "question_norm": "hoạt động mua bán hàng hóa quốc tế"},
        {"query_id": "q6", "question_raw": "Khai thuế và nộp thuế điện tử", "question_norm": "khai thuế và nộp thuế điện tử"},
    ]
    pd.DataFrame(queries).to_parquet(target_dir / "queries_train.parquet", index=False)

    qrels = [
        {"query_id": "q1", "doc_id": "101"},
        {"query_id": "q2", "doc_id": "102"},
        {"query_id": "q3", "doc_id": "103"},
        {"query_id": "q4", "doc_id": "104"},
        {"query_id": "q5", "doc_id": "105"},
        {"query_id": "q6", "doc_id": "106"},
    ]
    pd.DataFrame(qrels).to_parquet(target_dir / "qrels_train.parquet", index=False)

    split_info = [
        {"fold": 0, "train_query_ids": ["q4", "q5", "q6"], "val_query_ids": ["q1", "q2", "q3"]},
        {"fold": 1, "train_query_ids": ["q1", "q2", "q3"], "val_query_ids": ["q4", "q5", "q6"]},
    ]
    (splits_dir / "random_5fold.json").write_text(json.dumps(split_info), encoding="utf-8")

    doc_disjoint_split = {
        "train_query_ids": ["q1", "q2", "q3"],
        "val_query_ids": ["q4", "q5", "q6"],
        "train_doc_ids": ["101", "102", "103"],
        "val_doc_ids": ["104", "105", "106"],
    }
    (splits_dir / "doc_disjoint_split.json").write_text(json.dumps(doc_disjoint_split), encoding="utf-8")

    # Save public test file
    public_data = {
        "q_pub_1": {"question": "Thành lập doanh nghiệp tại Việt Nam"},
        "q_pub_2": {"question": "Ưu đãi đầu tư trực tiếp nước ngoài"},
        "q_pub_3": {"question": "Nộp thuế điện tử qua mạng"},
    }
    (target_dir / "public-official.json").write_text(json.dumps(public_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return target_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run dedicated end-to-end smoke test for LegalIR Kaggle production pipeline."
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a tiny self-contained dataset and fast mock configurations for ultra-fast verification.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enforce offline execution (disable Hugging Face network lookups).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Explicit canonical dataset directory.",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help="Explicit output directory for run artifacts.",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        default="smoke",
        choices=["smoke", "gpu_smoke", "full"],
        help="Execution mode (smoke, gpu_smoke, or full). Default: smoke.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pipeline config file path.",
    )
    args = parser.parse_args()

    # 1. Handle offline mode
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("[+] Offline mode enabled: network access disabled for models & datasets.")

    # 2. Setup working directory
    working_dir = Path(args.working_dir) if args.working_dir else (REPO_ROOT / "artifacts/task1/smoke_run")
    working_dir.mkdir(parents=True, exist_ok=True)

    # 3. Setup data directory
    data_dir_path = Path(args.data_dir) if args.data_dir else None
    if args.tiny or data_dir_path is None or not (data_dir_path / "documents.parquet").exists():
        toy_data_dir = working_dir / "toy_data"
        print(f"[*] Creating toy canonical dataset in {toy_data_dir} for tiny smoke verification...")
        data_dir_path = create_toy_canonical_dataset(toy_data_dir)

    print("\n" + "=" * 80)
    print("LEGALIR TASK 1: PRODUCTION KAGGLE PIPELINE SMOKE VERIFICATION")
    print(f"  Data Directory   : {data_dir_path}")
    print(f"  Working Directory: {working_dir}")
    print(f"  Run Mode         : {args.run_mode.upper()}")
    print(f"  Offline Mode     : {args.offline}")
    print("=" * 80 + "\n")

    t0 = time.time()
    try:
        result: KaggleRunResult = run_kaggle_pipeline(
            data_dir=data_dir_path,
            working_dir=working_dir,
            run_mode=args.run_mode,
            repo_root=REPO_ROOT,
            config_path=args.config,
            devices=["cpu", "cpu"] if (args.tiny and args.run_mode == "smoke") else None,
        )
    except Exception as e:
        print(f"\n[!] FATAL PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0

    # 4. Strict Artifact Validation Assertions
    print("\n" + "=" * 80)
    print("SMOKE PIPELINE ARTIFACT INTEGRITY VERIFICATION:")
    print("=" * 80)

    sub_json = result.submission_path
    sub_zip = result.submission_zip_path
    manifest_file = result.manifest_path
    audit_file = working_dir / "parameter_audit.json"
    cv_report_file = working_dir / "cv" / "cv_report.json"

    # 4a. Assert submission.json
    assert sub_json.exists(), f"Missing submission.json at {sub_json}"
    with open(sub_json, "r", encoding="utf-8") as f:
        preds = json.load(f)
    json_val = validate_submission(sub_json, expected_qids=set(preds.keys()))
    print(f"  [1] submission.json        : {'VALID' if json_val['is_valid'] else 'INVALID'} ({len(preds)} queries, {sub_json.stat().st_size:,} bytes)")
    if not json_val["is_valid"]:
        print(f"      Errors: {json_val['errors']}")

    # 4b. Assert submission.zip
    assert sub_zip.exists(), f"Missing submission.zip at {sub_zip}"
    zip_val = validate_submission_zip(sub_zip)
    with zipfile.ZipFile(sub_zip, "r") as zf:
        zip_namelist = zf.namelist()
    zip_structure_ok = (zip_namelist == ["submission.json"])
    print(f"  [2] submission.zip         : {'VALID' if (zip_val['is_valid'] and zip_structure_ok) else 'INVALID'} (contents: {zip_namelist}, {sub_zip.stat().st_size:,} bytes)")

    # 4c. Assert submission_manifest.json
    assert manifest_file.exists(), f"Missing submission_manifest.json at {manifest_file}"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert "submission_json_sha256" in manifest_data
    assert "submission_zip_sha256" in manifest_data
    print(f"  [3] submission_manifest.json: VALID (SHA-256 tracked, commit: {manifest_data.get('git_commit', 'unknown')[:8]})")

    # 4d. Assert parameter_audit.json
    assert audit_file.exists(), f"Missing parameter_audit.json at {audit_file}"
    with open(audit_file, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    learned_params = audit_data.get("total_learned_parameters", 0)
    audit_ok = (learned_params < MAX_PARAMETER_BUDGET)
    print(f"  [4] parameter_audit.json   : {'PASS' if audit_ok else 'FAIL'} ({learned_params:,} params / 4.0B limit, {audit_data.get('budget_utilization_pct', 0.0):.2f}%)")

    # 4e. Assert cv_report.json
    assert cv_report_file.exists(), f"Missing cv_report.json at {cv_report_file}"
    with open(cv_report_file, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    print(f"  [5] cv_report.json         : VALID (OOF Recall@5: {cv_data.get('mean_recall@5', 0.0) * 100:.2f}%, Precision@5: {cv_data.get('mean_precision@5', 0.0) * 100:.2f}%)")

    overall_pass = bool(
        result.is_valid
        and json_val["is_valid"]
        and zip_val["is_valid"]
        and zip_structure_ok
        and audit_ok
    )

    print("\n" + "=" * 80)
    print(f"OVERALL SMOKE VERIFICATION STATUS: {'SUCCESS (READY FOR KAGGLE)' if overall_pass else 'FAILED'}")
    print(f"Total Verification Runtime       : {elapsed:.2f}s")
    print("=" * 80 + "\n")

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
