import os
import json
import time
import pandas as pd
from collections import defaultdict

from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.evidence import EvidencePackBuilder
from src.common.reranker import BGEReranker
from src.task1.memory import QuestionMemory
from src.task1.retrieve import CandidateRetriever, LegalMatcher
from src.task1.rerank import DocumentReranker
from src.task1.selector import TopKSelector
from src.evaluation.evaluator import compute_candidate_recall, evaluate_predictions

def run_evaluation(
    data_dir: str = "artifacts/task1/data",
    index_dir: str = "artifacts/task1/indexes",
    splits_dir: str = "artifacts/shared/canonical/v2/splits",
    use_reranker: bool = False,
    num_folds: int = 5,
    sample_size: int = None,
    device: str = None
):
    print("=" * 60)
    print("UIT-DSC 2026 Task 1: Dual-Validation Benchmark")
    print("=" * 60)

    docs_df = pd.read_parquet(os.path.join(data_dir, "documents.parquet"))
    queries_df = pd.read_parquet(os.path.join(data_dir, "queries_train.parquet"))
    qrels_df = pd.read_parquet(os.path.join(data_dir, "qrels_train.parquet"))

    doc_map = {str(r["doc_id"]): r for r in docs_df.to_dict("records")}
    queries_map = {str(r["query_id"]): str(r["question_raw"]) for r in queries_df.to_dict("records")}

    qrels_map = defaultdict(list)
    for r in qrels_df.to_dict("records"):
        qrels_map[str(r["query_id"])].append(str(r["doc_id"]))

    # Load BM25 and Dense
    print("Loading BM25 and DEk21 Indexes...")
    bm25 = BM25Retriever.load(os.path.join(index_dir, "bm25"))
    dense = DEk21Retriever.load(os.path.join(index_dir, "dense_dek21"), device=device)
    exact = LegalMatcher(doc_index=doc_map)

    # 1. 5-Fold Cross-Validation
    split_path = os.path.join(splits_dir, "random_5fold.json")
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            fold_splits = json.load(f)
    else:
        fold_splits = []

    fold_metrics = []
    print(f"\n--- Running 5-Fold Cross-Validation ({len(fold_splits)} Folds) ---")

    for f_idx, fold in enumerate(fold_splits[:num_folds]):
        train_ids = set(str(x) for x in fold.get("train_query_ids", []))
        val_ids = [str(x) for x in fold.get("val_query_ids", [])]

        if sample_size is not None:
            val_ids = val_ids[:sample_size]

        # Fold-isolated question memory
        fold_queries = {qid: queries_map[qid] for qid in train_ids if qid in queries_map}
        fold_qrels = {qid: qrels_map[qid] for qid in train_ids if qid in qrels_map}

        memory = QuestionMemory(min_similarity=0.82)
        memory.fit(fold_queries, fold_qrels, dense_retriever=dense)

        retriever = CandidateRetriever(bm25=bm25, dense=dense, memory=memory, exact=exact)
        selector = TopKSelector(max_k=5)

        predictions = {}
        candidate_pools = {}

        t0 = time.time()
        for qid in val_ids:
            q_text = queries_map[qid]
            candidates = retriever.retrieve_candidates(q_text, top_k=60)
            candidate_pools[qid] = [c["doc_id"] for c in candidates]
            predictions[qid] = selector.select(candidates)

        cand_20 = compute_candidate_recall(candidate_pools, qrels_map, k=20)
        cand_50 = compute_candidate_recall(candidate_pools, qrels_map, k=50)
        eval_res = evaluate_predictions(predictions, qrels_map)

        rec5 = eval_res.get("recall@5", 0.0)
        prec5 = eval_res.get("precision@5", 0.0)
        fold_metrics.append({"recall@5": rec5, "precision@5": prec5, "cand@20": cand_20, "cand@50": cand_50})

        print(f"Fold {f_idx}: Recall@5 = {rec5*100:.2f}%, Prec@5 = {prec5*100:.2f}%, Cand@50 = {cand_50*100:.2f}% ({time.time()-t0:.1f}s)")

    if fold_metrics:
        avg_rec5 = sum(m["recall@5"] for m in fold_metrics) / len(fold_metrics)
        avg_prec5 = sum(m["precision@5"] for m in fold_metrics) / len(fold_metrics)
        avg_cand50 = sum(m["cand@50"] for m in fold_metrics) / len(fold_metrics)
        print(f"\n>> 5-Fold CV Mean: Recall@5 = {avg_rec5*100:.2f}%, Prec@5 = {avg_prec5*100:.2f}%, Cand@50 = {avg_cand50*100:.2f}%")

    # 2. Document-Disjoint Evaluation
    doc_disjoint_path = os.path.join(splits_dir, "doc_disjoint_split.json")
    if os.path.exists(doc_disjoint_path):
        with open(doc_disjoint_path, "r", encoding="utf-8") as f:
            dd_split = json.load(f)

        train_ids = set(str(x) for x in dd_split.get("train_query_ids", []))
        val_ids = [str(x) for x in dd_split.get("val_query_ids", [])]

        if sample_size is not None:
            val_ids = val_ids[:sample_size]

        print(f"\n--- Running Document-Disjoint Generalization Evaluation ({len(val_ids)} queries) ---")
        fold_queries = {qid: queries_map[qid] for qid in train_ids if qid in queries_map}
        fold_qrels = {qid: qrels_map[qid] for qid in train_ids if qid in qrels_map}

        memory = QuestionMemory(min_similarity=0.82)
        memory.fit(fold_queries, fold_qrels, dense_retriever=dense)

        retriever = CandidateRetriever(bm25=bm25, dense=dense, memory=memory, exact=exact)
        selector = TopKSelector(max_k=5)

        predictions = {}
        candidate_pools = {}
        for qid in val_ids:
            q_text = queries_map[qid]
            candidates = retriever.retrieve_candidates(q_text, top_k=60)
            candidate_pools[qid] = [c["doc_id"] for c in candidates]
            predictions[qid] = selector.select(candidates)

        dd_cand50 = compute_candidate_recall(candidate_pools, qrels_map, k=50)
        dd_eval = evaluate_predictions(predictions, qrels_map)
        print(f">> Doc-Disjoint: Recall@5 = {dd_eval.get('recall@5', 0)*100:.2f}%, Cand@50 = {dd_cand50*100:.2f}%")

    print("\n" + "=" * 60)
    print("Dual Validation Benchmark Complete!")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation(sample_size=200)
