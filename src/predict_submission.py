import os
import json
import zipfile
import argparse
import pandas as pd
from tqdm import tqdm

from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.question_memory import QuestionMemory
from src.retrieval.hybrid_search import HybridSearchEngine
from src.ranking.fusion import ReciprocalRankFusion
from src.ranking.selector import TopKSelector

def generate_submission(
    input_json_path: str,
    output_json_path: str = "submission.json",
    output_zip_path: str = "submission.zip",
    canonical_dir: str = "data/task1_canonical/v1",
    bm25_index_path: str = "indexes/bm25_micro_index.pkl",
    top_k_candidates: int = 50
):
    print("=" * 60)
    print(f"GENERATING SUBMISSION FOR {input_json_path}")
    print("=" * 60)

    # 1. Load input queries
    with open(input_json_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    print(f"Loaded {len(test_queries)} test queries from {input_json_path}")

    # 2. Load canonical data
    docs_df = pd.read_parquet(os.path.join(canonical_dir, "documents.parquet"))
    queries_df = pd.read_parquet(os.path.join(canonical_dir, "queries_train.parquet"))
    qrels_df = pd.read_parquet(os.path.join(canonical_dir, "qrels_train.parquet"))

    valid_corpus_doc_ids = set(docs_df["doc_id"].astype(str))
    docs = docs_df.to_dict(orient="records")

    train_queries_dict = {str(r["query_id"]): r["question_norm"] for r in queries_df.to_dict(orient="records")}
    qrels_dict = {}
    for r in qrels_df.to_dict(orient="records"):
        qid = str(r["query_id"])
        did = str(r["doc_id"])
        if qid not in qrels_dict:
            qrels_dict[qid] = []
        qrels_dict[qid].append(did)

    train_queries_for_memory = [
        {
            "query_id": str(qid),
            "question_norm": train_queries_dict[qid],
            "doc_ids": qrels_dict[qid]
        }
        for qid in train_queries_dict if qid in qrels_dict
    ]

    # 3. Initialize engine
    print(f"Loading BM25 index from {bm25_index_path}...")
    bm25 = BM25MicroRetriever.load(bm25_index_path)
    exact = ExactMatcher(docs)
    memory = QuestionMemory(train_queries_for_memory)

    hybrid_engine = HybridSearchEngine(
        bm25_retriever=bm25,
        exact_matcher=exact,
        question_memory=memory
    )
    fuser = ReciprocalRankFusion()
    selector = TopKSelector(max_k=5)

    # 4. Generate predictions
    predictions = {}
    for qid, qobj in tqdm(test_queries.items(), desc="Inference"):
        q_text = qobj.get("question", "")

        cands = hybrid_engine.search_candidates(q_text, exclude_qid=None, top_k=top_k_candidates)
        ranked = fuser.rank_candidates(cands)
        top5 = selector.select(ranked)

        # Fallback if no candidate returned
        if not top5:
            # Fallback to most frequent high-coverage official documents
            top5 = ["2113"]

        predictions[str(qid)] = {
            "answer": [str(x) for x in top5]
        }

    # 5. Strict Submission Invariant Validation
    print("\nRunning submission compliance check...")
    assert len(predictions) == len(test_queries), f"Query count mismatch: {len(predictions)} vs {len(test_queries)}"

    for qid, pred in predictions.items():
        assert "answer" in pred, f"Missing answer key for query {qid}"
        ans = pred["answer"]
        assert isinstance(ans, list), f"Answer must be a list for query {qid}"
        assert 1 <= len(ans) <= 5, f"Answer length {len(ans)} invalid for query {qid}"
        assert len(set(ans)) == len(ans), f"Duplicate IDs in answer for query {qid}: {ans}"
        for did in ans:
            assert did in valid_corpus_doc_ids, f"Invalid doc ID {did} not in official corpus!"

    print("Submission compliance check PASSED! 100% verified.")

    # 6. Save submission.json and submission.zip
    print(f"Writing {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Writing {output_zip_path}...")
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json_path, arcname="submission.json")

    print(f"Submission packaging completed successfully: {output_zip_path} ({os.path.getsize(output_zip_path):,} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="public-official.json")
    parser.add_argument("--output_file", type=str, default="submission.json")
    parser.add_argument("--output_zip", type=str, default="submission.zip")
    parser.add_argument("--canonical_dir", type=str, default="data/task1_canonical/v1")
    parser.add_argument("--bm25_index", type=str, default="indexes/bm25_micro_index.pkl")
    args = parser.parse_args()

    generate_submission(
        args.input_file,
        args.output_file,
        args.output_zip,
        args.canonical_dir,
        args.bm25_index
    )
