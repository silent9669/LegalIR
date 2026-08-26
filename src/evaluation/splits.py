from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import random
import pandas as pd


def generate_random_5fold_split(queries: list, seed: int = 42) -> list[dict[str, Any]]:
    """Generate 5-fold cross-validation query splits."""
    random.seed(seed)
    qids = sorted([str(q["query_id"]) if isinstance(q, dict) else str(q) for q in queries])
    random.shuffle(qids)

    n = len(qids)
    fold_size = n // 5
    folds = []

    for i in range(5):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < 4 else n
        val_qids = qids[val_start:val_end]
        train_qids = [qid for qid in qids if qid not in set(val_qids)]

        folds.append({
            "fold": i,
            "train": train_qids,
            "val": val_qids,
            "train_query_ids": train_qids,
            "val_query_ids": val_qids,
        })

    return folds


def generate_document_disjoint_split(queries: list, qrels: list, val_ratio: float = 0.2, seed: int = 42) -> dict[str, Any]:
    """Generate a document-disjoint split where validation gold documents never appear in the train set."""
    random.seed(seed)

    # Map query -> gold docs and doc -> queries
    q2d = defaultdict(set)
    d2q = defaultdict(set)
    for r in qrels:
        qid = str(r["query_id"])
        did = str(r["doc_id"])
        q2d[qid].add(did)
        d2q[did].add(qid)

    # Build connected components of queries that share gold documents
    visited_q = set()
    clusters = []  # each cluster is a set of query_ids

    all_qids = sorted(list(q2d.keys()))
    for q in all_qids:
        if q in visited_q:
            continue
        cluster = set()
        queue = [q]
        visited_q.add(q)

        while queue:
            curr_q = queue.pop()
            cluster.add(curr_q)
            for doc in q2d[curr_q]:
                for neighbor_q in d2q[doc]:
                    if neighbor_q not in visited_q:
                        visited_q.add(neighbor_q)
                        queue.append(neighbor_q)
        clusters.append(cluster)

    # Shuffle clusters and greedily assign to val up to val_ratio
    random.shuffle(clusters)
    total_queries = len(all_qids)
    target_val_count = int(total_queries * val_ratio)

    val_qids = []
    train_qids = []

    for c in clusters:
        if len(val_qids) + len(c) <= target_val_count or len(val_qids) == 0:
            val_qids.extend(list(c))
        else:
            train_qids.extend(list(c))

    # Safety assertion
    val_docs = set()
    for q in val_qids:
        val_docs.update(q2d[q])

    train_docs = set()
    for q in train_qids:
        train_docs.update(q2d[q])

    overlap = val_docs & train_docs
    assert len(overlap) == 0, f"Document-disjoint split violation! Overlap: {overlap}"

    return {
        "train": sorted(train_qids),
        "val": sorted(val_qids),
        "train_query_ids": sorted(train_qids),
        "val_query_ids": sorted(val_qids),
        "train_doc_count": len(train_docs),
        "val_doc_count": len(val_docs),
    }


def create_all_splits(canonical_dir: str | Path):
    canonical_dir = Path(canonical_dir)
    splits_dir = canonical_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    queries_df = pd.read_parquet(canonical_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(canonical_dir / "qrels_train.parquet")

    queries = queries_df.to_dict(orient="records")
    qrels = qrels_df.to_dict(orient="records")

    print(f"Generating 5-fold random CV splits for {len(queries)} queries...")
    random_5fold = generate_random_5fold_split(queries, seed=42)
    (splits_dir / "random_5fold.json").write_text(json.dumps(random_5fold, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Generating document-disjoint split...")
    doc_disjoint = generate_document_disjoint_split(queries, qrels, val_ratio=0.2, seed=42)
    (splits_dir / "doc_disjoint_split.json").write_text(json.dumps(doc_disjoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Splits successfully created in {splits_dir}!")
    print(f"  Random 5-fold: 5 folds x {len(random_5fold[0]['val_query_ids'])} val queries")
    print(f"  Doc-disjoint: {len(doc_disjoint['train_query_ids'])} train queries ({doc_disjoint['train_doc_count']} docs) | {len(doc_disjoint['val_query_ids'])} val queries ({doc_disjoint['val_doc_count']} docs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical_dir", type=str, default="artifacts/shared/canonical/v2")
    args = parser.parse_args()
    create_all_splits(args.canonical_dir)
