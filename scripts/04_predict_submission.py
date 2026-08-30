import os
import json
import zipfile
import time
from src.task1.predict import LegalIRPipeline

def generate_submission(
    public_json: str = "public-official.json",
    data_dir: str = "artifacts/task1/data",
    index_dir: str = "artifacts/task1/indexes",
    out_dir: str = "artifacts/task1/submissions",
    use_reranker: bool = True,
    device: str = None,
    top_k_candidates: int = 50
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

    print("\nInitializing 4-Branch LegalIR Pipeline...", flush=True)
    t0 = time.time()
    pipeline = LegalIRPipeline.load_pipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        use_reranker=use_reranker,
        device=device
    )
    print(f"Pipeline initialized in {time.time() - t0:.2f}s", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    interim_path = os.path.join(out_dir, "interim_predictions.json")
    predictions = {}
    if os.path.exists(interim_path):
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
            top_k_candidates=top_k_candidates,
            top_k_rerank=5
        )
        predictions[str_qid] = {"answer": pred_ids}
        processed_count += 1

        if processed_count % 10 == 0 or processed_count == total_queries:
            elapsed = max(0.1, time.time() - t0)
            completed_in_session = processed_count - len(predictions) + (10 if processed_count % 10 == 0 else 0)
            rate = max(0.01, completed_in_session / elapsed)
            remaining = (total_queries - processed_count) / rate / 60 if rate > 0 else 0
            print(f"[{processed_count:4d}/{total_queries:4d}] queries processed ({rate:.2f} q/s, est remaining: {remaining:.1f}m)", flush=True)

            with open(interim_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False)

    print(f"\nInference completed in {time.time() - t0:.2f}s", flush=True)

    # Invariant Validation
    print("\nValidating Submission Invariants...", flush=True)
    assert len(predictions) == total_queries, f"Expected {total_queries} queries, got {len(predictions)}"

    valid_ids = pipeline.valid_doc_ids or set()
    for qid, res in predictions.items():
        ans = res.get("answer", [])
        assert 1 <= len(ans) <= 5, f"Query {qid} has invalid length {len(ans)}"
        assert len(ans) == len(set(ans)), f"Query {qid} has duplicates: {ans}"
        if valid_ids:
            for doc_id in ans:
                assert doc_id in valid_ids, f"Query {qid} contains non-corpus doc_id: {doc_id}"

    print("All submission invariants PASSED (100% compliant)!", flush=True)

    # Save final artifacts
    out_json = os.path.join(out_dir, "submission.json")
    out_zip = os.path.join(out_dir, "submission.zip")
    top_json = "submission.json"
    top_zip = "submission.zip"

    print(f"\nWriting submission files:", flush=True)
    print(f" - {out_json}", flush=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with open(top_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f" - {out_zip}", flush=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="submission.json")

    with zipfile.ZipFile(top_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(top_json, arcname="submission.json")

    # Generate submission manifest
    import hashlib
    with open(out_json, "rb") as f:
        json_bytes = f.read()
    with open(out_zip, "rb") as f:
        zip_bytes = f.read()

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    manifest = {
        "run_id": f"submission_task1_{timestamp_str}",
        "timestamp_utc": timestamp_str,
        "total_queries": total_queries,
        "submission_json_sha256": hashlib.sha256(json_bytes).hexdigest(),
        "submission_zip_sha256": hashlib.sha256(zip_bytes).hexdigest(),
        "compliance_verified": True,
        "json_size_bytes": len(json_bytes),
        "zip_size_bytes": len(zip_bytes)
    }
    manifest_path = os.path.join(out_dir, "submission_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    accepted_manifest_path = os.path.join(out_dir, "accepted", "submission_manifest.json")
    os.makedirs(os.path.dirname(accepted_manifest_path), exist_ok=True)
    with open(accepted_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print(f"Submission Package Created Successfully: {top_zip}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    generate_submission()
