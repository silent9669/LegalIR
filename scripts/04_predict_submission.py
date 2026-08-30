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
        public_json = "artifacts/shared/raw/public-official.json"

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

    print("\nRunning Inference on Public Test Queries...", flush=True)
    t0 = time.time()
    predictions = {}

    qids = list(public_data.keys())
    for idx, qid in enumerate(qids, start=1):
        item = public_data[qid]
        q_text = item.get("question", "") if isinstance(item, dict) else str(item)

        pred_ids = pipeline.predict_single(
            query=q_text,
            top_k_candidates=top_k_candidates,
            top_k_rerank=5
        )
        predictions[str(qid)] = {"answer": pred_ids}

        if idx % 20 == 0 or idx == total_queries:
            elapsed = time.time() - t0
            rate = idx / elapsed
            print(f"[{idx:4d}/{total_queries:4d}] queries processed ({rate:.2f} q/s, est remaining: {(total_queries - idx) / rate / 60:.1f}m)", flush=True)

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

    # Save to artifacts/task1/submissions and top-level
    os.makedirs(out_dir, exist_ok=True)
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

    print("\n" + "=" * 60, flush=True)
    print(f"Submission Package Created Successfully: {top_zip}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    generate_submission()
