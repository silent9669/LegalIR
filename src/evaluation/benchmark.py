from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths
from src.dataset.validator import validate_canonical_dataset
from src.evaluation.codabench_compat import assert_official_equivalence
from src.evaluation.evaluator import compute_candidate_cutoffs, evaluate_predictions
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import QuestionMemory
from src.ranking.fusion import ReciprocalRankFusion
from src.ranking.selector import TopKSelector


def build_memory_rows(
    train_query_ids: list[str] | set[str],
    queries_dict: dict[str, str],
    qrels_dict: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = []
    for qid in sorted(map(str, train_query_ids)):
        if qid in queries_dict and qid in qrels_dict:
            rows.append({
                "query_id": qid,
                "question_norm": queries_dict[qid],
                "doc_ids": qrels_dict[qid],
            })
    return rows


def run_split_eval(
    split_name: str,
    val_query_ids: list[str],
    queries_dict: dict[str, str],
    qrels_dict: dict[str, list[str]],
    hybrid_engine: HybridSearchEngine,
    fuser: ReciprocalRankFusion,
    selector: TopKSelector,
    candidate_cutoffs: list[int] = [20, 50, 100, 150],
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]]]:
    val_predictions = {}
    val_candidates = {}
    val_ground_truth = {}

    for qid in tqdm(val_query_ids, desc=f"Evaluating {split_name}", leave=False):
        qid_str = str(qid)
        q_text = queries_dict.get(qid_str, "")
        gold_docs = qrels_dict.get(qid_str, [])
        val_ground_truth[qid_str] = gold_docs

        cands = hybrid_engine.search_candidates(
            query=q_text,
            exclude_qid=qid_str,
            top_k=max(candidate_cutoffs),
        )
        val_candidates[qid_str] = [c["doc_id"] for c in cands]

        ranked = fuser.rank_candidates(cands)
        top5 = selector.select(ranked)
        val_predictions[qid_str] = top5

    metrics = evaluate_predictions(val_predictions, val_ground_truth)
    cand_metrics = compute_candidate_cutoffs(val_candidates, val_ground_truth, cutoffs=candidate_cutoffs)
    metrics.update(cand_metrics)

    # Official scorer assertion
    assert_official_equivalence(val_predictions, val_ground_truth)

    return metrics, val_predictions, val_candidates


def run_benchmark(
    config_path: str | Path = "configs/pipeline.yaml",
    fold_limit: int | None = None,
    label: str = "strict_baseline",
) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_pipeline_config(config_path)
    paths = ProjectPaths.from_repo()

    canonical_dir = paths.repo / cfg.get("paths", {}).get("canonical", "artifacts/shared/canonical/v2")
    if not (canonical_dir / "documents.parquet").exists():
        raise FileNotFoundError(f"Canonical dataset missing at {canonical_dir}")

    # Validate dataset before benchmark
    val_report = validate_canonical_dataset(canonical_dir)
    if not val_report["is_valid"]:
        raise ValueError(f"Canonical dataset invalid: {val_report['errors']}")

    print("Loading canonical data...")
    docs_df = pd.read_parquet(canonical_dir / "documents.parquet")
    chunks_df = pd.read_parquet(canonical_dir / "chunks.parquet")
    queries_df = pd.read_parquet(canonical_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(canonical_dir / "qrels_train.parquet")

    queries_dict = dict(zip(queries_df["query_id"].astype(str), queries_df["question_norm"]))
    qrels_dict = qrels_df.groupby("query_id")["doc_id"].apply(lambda s: [str(x) for x in s]).to_dict()

    splits_dir = canonical_dir / "splits"
    random_5fold = json.loads((splits_dir / "random_5fold.json").read_text(encoding="utf-8"))
    doc_disjoint = json.loads((splits_dir / "doc_disjoint_split.json").read_text(encoding="utf-8"))

    # Fit shared micro-BM25 retriever & exact matcher
    bm25_index_path = paths.local_indexes / "bm25" / "bm25_micro_index.pkl"
    if bm25_index_path.exists():
        print(f"Loading cached BM25 micro-chunk index from {bm25_index_path}...")
        bm25 = BM25MicroRetriever.load(bm25_index_path)
    else:
        print("Fitting BM25 micro-chunk index...")
        micro_chunks_df = chunks_df[chunks_df["granularity"] == "micro"]
        chunks_records = micro_chunks_df.to_dict(orient="records")
        bm25 = BM25MicroRetriever()
        bm25.fit(chunks_records, show_progress=True)
        bm25.save(bm25_index_path)

    print("Initializing ExactMatcher...")
    docs_records = docs_df.to_dict(orient="records")
    exact_matcher = ExactMatcher(docs_records)

    fuser = ReciprocalRankFusion()
    selector = TopKSelector(max_k=5)

    candidate_cutoffs = cfg.get("evaluation", {}).get("candidate_cutoffs", [20, 50, 100, 150])
    num_folds_to_run = fold_limit if fold_limit is not None else len(random_5fold)

    print(f"\n=======================================================")
    print(f"Running LegalIR Benchmark ({label}) — {num_folds_to_run}/5 folds")
    print(f"=======================================================\n")

    fold_metrics_list = []
    all_fold_predictions = {}
    all_fold_candidates = {}

    for fold_idx in range(num_folds_to_run):
        fold_data = random_5fold[fold_idx]
        train_qids = [str(x) for x in fold_data.get("train_query_ids", fold_data.get("train", []))]
        val_qids = [str(x) for x in fold_data.get("val_query_ids", fold_data.get("val", []))]

        # Strict fold isolation: Build memory ONLY from fold's training queries
        memory_rows = build_memory_rows(train_qids, queries_dict, qrels_dict)
        memory = QuestionMemory(memory_rows, min_similarity=0.82)

        # Leakage guard assertion
        assert memory.training_query_ids == frozenset(train_qids), f"Fold {fold_idx} memory contains leaked query IDs!"
        assert set(val_qids).isdisjoint(memory.training_query_ids), f"Fold {fold_idx} validation queries leaked into memory!"

        hybrid_engine = HybridSearchEngine(
            bm25_retriever=bm25,
            exact_matcher=exact_matcher,
            question_memory=memory,
            dense_retriever=None,
        )

        metrics, preds, cands = run_split_eval(
            split_name=f"Random Fold {fold_idx + 1}",
            val_query_ids=val_qids,
            queries_dict=queries_dict,
            qrels_dict=qrels_dict,
            hybrid_engine=hybrid_engine,
            fuser=fuser,
            selector=selector,
            candidate_cutoffs=candidate_cutoffs,
        )

        fold_metrics_list.append(metrics)
        all_fold_predictions[f"fold_{fold_idx}"] = preds
        all_fold_candidates[f"fold_{fold_idx}"] = cands

        print(f"Fold {fold_idx + 1}: Recall@5={metrics['recall_at_5']:.4f} | Prec@5={metrics['precision_at_5']:.4f} | CandRec@50={metrics.get('candidate_recall@50', 0):.4f}")

    # Aggregate 5-fold metrics
    mean_rec_5 = float(np.mean([m["recall_at_5"] for m in fold_metrics_list]))
    std_rec_5 = float(np.std([m["recall_at_5"] for m in fold_metrics_list]))
    mean_prec_5 = float(np.mean([m["precision_at_5"] for m in fold_metrics_list]))
    mean_r1 = float(np.mean([m["recall_at_1"] for m in fold_metrics_list]))
    mean_r3 = float(np.mean([m["recall_at_3"] for m in fold_metrics_list]))

    cand_rec_means = {
        f"candidate_recall@{k}": float(np.mean([m.get(f"candidate_recall@{k}", 0.0) for m in fold_metrics_list]))
        for k in candidate_cutoffs
    }

    # Run document-disjoint evaluation
    print("\nRunning Document-Disjoint Generalization Evaluation...")
    disjoint_train_qids = [str(x) for x in doc_disjoint.get("train_query_ids", doc_disjoint.get("train", []))]
    disjoint_val_qids = [str(x) for x in doc_disjoint.get("val_query_ids", doc_disjoint.get("val", []))]

    disjoint_memory_rows = build_memory_rows(disjoint_train_qids, queries_dict, qrels_dict)
    disjoint_memory = QuestionMemory(disjoint_memory_rows, min_similarity=0.82)
    assert set(disjoint_val_qids).isdisjoint(disjoint_memory.training_query_ids)

    disjoint_engine = HybridSearchEngine(
        bm25_retriever=bm25,
        exact_matcher=exact_matcher,
        question_memory=disjoint_memory,
        dense_retriever=None,
    )

    doc_disjoint_metrics, doc_disjoint_preds, doc_disjoint_cands = run_split_eval(
        split_name="Doc-Disjoint",
        val_query_ids=disjoint_val_qids,
        queries_dict=queries_dict,
        qrels_dict=qrels_dict,
        hybrid_engine=disjoint_engine,
        fuser=fuser,
        selector=selector,
        candidate_cutoffs=candidate_cutoffs,
    )

    print(f"Doc-Disjoint: Recall@5={doc_disjoint_metrics['recall_at_5']:.4f} | Prec@5={doc_disjoint_metrics['precision_at_5']:.4f} | CandRec@50={doc_disjoint_metrics.get('candidate_recall@50', 0):.4f}")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp_str}_{label}"
    run_dir = paths.local_runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_report = {
        "run_id": run_id,
        "label": label,
        "timestamp_utc": timestamp_str,
        "leakage_checks_passed": True,
        "official_scorer_equivalent": True,
        "candidate_cutoffs": candidate_cutoffs,
        "random_5fold": {
            "num_folds_run": num_folds_to_run,
            "mean_recall_at_5": mean_rec_5,
            "std_recall_at_5": std_rec_5,
            "mean_precision_at_5": mean_prec_5,
            "mean_recall_at_1": mean_r1,
            "mean_recall_at_3": mean_r3,
            "candidate_recalls": cand_rec_means,
            "folds": fold_metrics_list,
        },
        "document_disjoint": doc_disjoint_metrics,
    }

    # Save run artifacts
    (run_dir / "metrics.json").write_text(json.dumps(summary_report, indent=2) + "\n", encoding="utf-8")
    (run_dir / "config.snapshot.yaml").write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "predictions.json").write_text(json.dumps({
        "random_5fold": all_fold_predictions,
        "document_disjoint": doc_disjoint_preds,
    }, indent=2) + "\n", encoding="utf-8")
    (run_dir / "candidate_metrics.json").write_text(json.dumps({
        "random_5fold_means": cand_rec_means,
        "document_disjoint": {k: doc_disjoint_metrics.get(k) for k in doc_disjoint_metrics if "candidate" in k},
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\n=======================================================")
    print(f"Benchmark Summary ({label}):")
    print(f"  Random 5-Fold Mean Recall@5: {mean_rec_5:.4f} ± {std_rec_5:.4f}")
    print(f"  Random 5-Fold Mean Prec@5:   {mean_prec_5:.4f}")
    print(f"  Doc-Disjoint Recall@5:       {doc_disjoint_metrics['recall_at_5']:.4f}")
    print(f"  Saved run artifacts to:      {run_dir}")
    print(f"=======================================================\n")

    return summary_report


def main():
    parser = argparse.ArgumentParser(description="LegalIR Benchmark Runner")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--fold-limit", type=int, default=None, help="Limit number of folds to run (e.g. 1 for smoke)")
    parser.add_argument("--label", type=str, default="strict_baseline")
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        fold_limit=args.fold_limit,
        label=args.label,
    )


if __name__ == "__main__":
    main()
