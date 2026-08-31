"""Generate predictions and create submission artifacts for LegalIR Task 1."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.predict import LegalIRPipeline
from src.evaluation.submission import validate_submission, package_submission


def generate_submission(
    public_json: str = "public-official.json",
    data_dir: str = "artifacts/task1/data",
    index_dir: str = "artifacts/task1/indexes",
    out_dir: str = "artifacts/task1/submissions",
    use_reranker: bool = True,
    device: str | None = None,
    top_k_candidates: int = 150,
    top_k_rerank: int = 50,
):
    print("=" * 60, flush=True)
    print("UIT-DSC 2026 Task 1: Generating High-Output Submission", flush=True)
    print("=" * 60, flush=True)

    if not os.path.exists(public_json):
        for candidate in ["artifacts/raw/public-official.json", "public-official.json", "artifacts/shared/raw/public-official.json"]:
            if os.path.exists(candidate):
                public_json = candidate
                break

    print(f"Loading public test queries from {public_json}...", flush=True)
    with open(public_json, "r", encoding="utf-8") as f:
        public_data = json.load(f)

    total_queries = len(public_data)
    print(f"Total queries to predict: {total_queries}", flush=True)

    print("\nInitializing Canonical 4-Branch LegalIR Pipeline...", flush=True)
    t0 = time.time()
    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        use_reranker=use_reranker,
        device=device,
    )
    print(f"Pipeline initialized in {time.time() - t0:.2f}s", flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Preflight] Running Strict Parameter Budget Audit (<4B Rule)...", flush=True)
    audit_json_path = out_dir / "parameter_audit.json"
    audit_report = pipeline.audit_parameters(output_json=audit_json_path, raise_on_violation=True)
    print(
        f"Parameter audit passed: {audit_report['total_learned_parameters']:,} params "
        f"({audit_report['total_parameters_billions']:.4f}B / 4.0B, "
        f"{audit_report['budget_utilization_pct']:.2f}% utilization). Saved to {audit_json_path}",
        flush=True,
    )

    interim_path = out_dir / "interim_predictions.json"
    predictions = {}
    if interim_path.exists():
        try:
            with open(interim_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"Resuming from {len(predictions)} cached predictions in {interim_path}...", flush=True)
        except Exception:
            predictions = {}

    print("\nRunning Inference on Public Test Queries...", flush=True)
    t0 = time.time()

    qids = list(public_data.keys())
    processed_count = len(predictions)

    for idx, qid in enumerate(qids, start=1):
        str_qid = str(qid)
        if str_qid in predictions:
            continue

        item = public_data[qid]
        q_text = item.get("question", "") if isinstance(item, dict) else str(item)

        pred_ids = pipeline.predict_single(
            query=q_text,
            query_id=str_qid,
            top_k_candidates=top_k_candidates,
            top_k_rerank=top_k_rerank,
        )
        predictions[str_qid] = {"answer": pred_ids}
        processed_count += 1

        if processed_count % 50 == 0 or processed_count == total_queries:
            elapsed = max(0.1, time.time() - t0)
            completed_in_session = processed_count - len(predictions) + (50 if processed_count % 50 == 0 else 0)
            rate = max(0.01, completed_in_session / elapsed)
            remaining = (total_queries - processed_count) / rate / 60 if rate > 0 else 0
            print(f"[{processed_count:4d}/{total_queries:4d}] queries processed ({rate:.2f} q/s, est remaining: {remaining:.1f}m)", flush=True)

            with open(interim_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False)

    print(f"\nInference completed in {time.time() - t0:.2f}s", flush=True)

    # Invariant Validation
    print("\nValidating Submission Invariants...", flush=True)
    sub_json = out_dir / "submission.json"
    with open(sub_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    val_res = validate_submission(sub_json, public_json=public_json)
    print(f"Validation result: is_valid = {val_res.get('is_valid')}")

    # Package into submission.zip
    sub_zip = out_dir / "submission.zip"
    package_submission(sub_json, sub_zip)
    print(f"\nSubmission packaged successfully at: {sub_zip}")


def main():
    parser = argparse.ArgumentParser(description="LegalIR Submission Generator")
    parser.add_argument("--public-json", type=str, default="public-official.json")
    parser.add_argument("--data-dir", type=str, default="artifacts/task1/data")
    parser.add_argument("--index-dir", type=str, default="artifacts/task1/indexes")
    parser.add_argument("--out-dir", type=str, default="artifacts/task1/submissions")
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--candidate-k", type=int, default=150)
    parser.add_argument("--rerank-k", type=int, default=50)
    args = parser.parse_args()

    generate_submission(
        public_json=args.public_json,
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        out_dir=args.out_dir,
        use_reranker=args.use_reranker,
        device=args.device,
        top_k_candidates=args.candidate_k,
        top_k_rerank=args.rerank_k,
    )


if __name__ == "__main__":
    main()
