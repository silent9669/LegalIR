import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.question_memory import QuestionMemory
from src.retrieval.hybrid_search import HybridSearchEngine
from src.ranking.fusion import ReciprocalRankFusion
from src.ranking.selector import TopKSelector
from src.evaluation.evaluator import evaluate_predictions, compute_candidate_recall

def run_split_evaluation(
    split_name: str,
    val_query_ids: list,
    queries_dict: dict,
    qrels_dict: dict,
    hybrid_engine: HybridSearchEngine,
    fuser: ReciprocalRankFusion,
    selector: TopKSelector,
    candidate_k: int = 50
) -> dict:
    print(f"\nEvaluating on {split_name} ({len(val_query_ids)} queries)...")

    predictions = {}
    candidate_map = {}
    ground_truths = {}

    for qid in tqdm(val_query_ids, desc=f"Evaluating {split_name}"):
        qid = str(qid)
        q_text = queries_dict.get(qid, "")
        gold_docs = qrels_dict.get(qid, [])
        ground_truths[qid] = gold_docs

        # Multi-branch retrieval with self-exclusion
        cands = hybrid_engine.search_candidates(q_text, exclude_qid=qid, top_k=candidate_k)
        candidate_map[qid] = [c["doc_id"] for c in cands]

        # Fusion ranking
        ranked = fuser.rank_candidates(cands)

        # Top-5 Selection
        top5 = selector.select(ranked)
        predictions[qid] = {"answer": top5}

    # Evaluate metrics
    metrics = evaluate_predictions(predictions, ground_truths)
    cand_rec_20 = compute_candidate_recall(candidate_map, ground_truths, k=20)
    cand_rec_50 = compute_candidate_recall(candidate_map, ground_truths, k=50)

    metrics["Candidate_Recall@20"] = cand_rec_20
    metrics["Candidate_Recall@50"] = cand_rec_50

    return metrics

def run_validation_suite(
    canonical_dir: str = "data/task1_canonical/v1",
    bm25_index_path: str = "indexes/bm25_micro_index.pkl",
    num_folds: int = 5
):
    print("=" * 60)
    print("DSC 2026 TASK 1 — OFFICIAL DUAL-VALIDATION BENCHMARK")
    print("=" * 60)

    # 1. Load canonical data
    print(f"Loading canonical dataset from {canonical_dir}...")
    docs_df = pd.read_parquet(os.path.join(canonical_dir, "documents.parquet"))
    queries_df = pd.read_parquet(os.path.join(canonical_dir, "queries_train.parquet"))
    qrels_df = pd.read_parquet(os.path.join(canonical_dir, "qrels_train.parquet"))

    docs = docs_df.to_dict(orient="records")
    queries_dict = {str(r["query_id"]): r["question_norm"] for r in queries_df.to_dict(orient="records")}

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
            "question_norm": queries_dict[qid],
            "doc_ids": qrels_dict[qid]
        }
        for qid in queries_dict if qid in qrels_dict
    ]

    # 2. Initialize retrieval branches
    print("Initializing Multi-Branch Retrieval Engine...")
    print(f"Loading BM25 index from {bm25_index_path}...")
    bm25 = BM25MicroRetriever.load(bm25_index_path)

    print("Initializing Exact Matcher...")
    exact = ExactMatcher(docs)

    print("Initializing Train-Question Memory...")
    memory = QuestionMemory(train_queries_for_memory)

    hybrid_engine = HybridSearchEngine(
        bm25_retriever=bm25,
        exact_matcher=exact,
        question_memory=memory
    )
    fuser = ReciprocalRankFusion()
    selector = TopKSelector(max_k=5)

    # 3. Protocol 1: Random 5-Fold Cross Validation
    splits_dir = os.path.join(canonical_dir, "splits")
    with open(os.path.join(splits_dir, "random_5fold.json"), "r", encoding="utf-8") as f:
        random_5fold = json.load(f)

    fold_metrics = []
    for f_info in random_5fold[:num_folds]:
        f_idx = f_info["fold"]
        val_qids = f_info["val_query_ids"]
        m = run_split_evaluation(
            f"Random Fold {f_idx + 1}/5",
            val_qids,
            queries_dict,
            qrels_dict,
            hybrid_engine,
            fuser,
            selector
        )
        fold_metrics.append(m)
        print(f"  Fold {f_idx + 1} -> Recall@5: {m['recall']:.4f}, Precision@5: {m['precision']:.4f}, Candidate_Recall@50: {m['Candidate_Recall@50']:.4f}")

    avg_random_recall = float(np.mean([m["recall"] for m in fold_metrics]))
    avg_random_prec = float(np.mean([m["precision"] for m in fold_metrics]))
    avg_random_cand50 = float(np.mean([m["Candidate_Recall@50"] for m in fold_metrics]))

    # 4. Protocol 2: Document-Disjoint Validation Split
    with open(os.path.join(splits_dir, "doc_disjoint_split.json"), "r", encoding="utf-8") as f:
        doc_disjoint = json.load(f)

    disjoint_val_qids = doc_disjoint["val_query_ids"]
    disjoint_metrics = run_split_evaluation(
        "Document-Disjoint Validation (Unseen Documents)",
        disjoint_val_qids,
        queries_dict,
        qrels_dict,
        hybrid_engine,
        fuser,
        selector
    )

    # 5. Print comprehensive diagnostic benchmark report
    print("\n" + "=" * 60)
    print("FINAL OFFICIAL BENCHMARK REPORT")
    print("=" * 60)
    print("PROTOCOL 1: RANDOM 5-FOLD CROSS VALIDATION (Seen Legal Documents & Memory)")
    print(f"  * Official Codabench Recall@5:    {avg_random_recall:.4f}")
    print(f"  * Official Codabench Precision@5: {avg_random_prec:.4f}")
    print(f"  * Candidate Recall@50:            {avg_random_cand50:.4f}")
    print("-" * 60)
    print("PROTOCOL 2: DOCUMENT-DISJOINT VALIDATION (Unseen Legal Documents Generalization)")
    print(f"  * Official Codabench Recall@5:    {disjoint_metrics['recall']:.4f}")
    print(f"  * Official Codabench Precision@5: {disjoint_metrics['precision']:.4f}")
    print(f"  * Recall@1:                       {disjoint_metrics['Recall@1']:.4f}")
    print(f"  * Recall@3:                       {disjoint_metrics['Recall@3']:.4f}")
    print(f"  * Candidate Recall@50:            {disjoint_metrics['Candidate_Recall@50']:.4f}")
    print("=" * 60)

    # Save summary report
    report = {
        "random_5fold_cv": {
            "mean_recall_at_5": avg_random_recall,
            "mean_precision_at_5": avg_random_prec,
            "mean_candidate_recall_at_50": avg_random_cand50,
            "folds": fold_metrics
        },
        "document_disjoint": disjoint_metrics
    }
    with open("validation_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Report saved to validation_benchmark_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical_dir", type=str, default="data/task1_canonical/v1")
    parser.add_argument("--bm25_index", type=str, default="indexes/bm25_micro_index.pkl")
    parser.add_argument("--num_folds", type=int, default=5)
    args = parser.parse_args()

    run_validation_suite(args.canonical_dir, args.bm25_index, args.num_folds)
