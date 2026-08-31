from pathlib import Path
from typing import Any
import pandas as pd
from src.dataset.schema import (
    CANONICAL_DOCUMENTS_COLUMNS,
    CANONICAL_CHUNKS_COLUMNS,
    CANONICAL_QUERIES_COLUMNS,
    CANONICAL_QRELS_COLUMNS,
)


def validate_canonical_dataset(canonical_dir: str | Path, expected_document_count: int | None = None) -> dict[str, Any]:
    canonical_dir = Path(canonical_dir)
    docs_path = canonical_dir / "documents.parquet"
    chunks_path = canonical_dir / "chunks.parquet"
    queries_path = canonical_dir / "queries_train.parquet"
    qrels_path = canonical_dir / "qrels_train.parquet"

    errors: list[str] = []

    for name, p in [
        ("documents.parquet", docs_path),
        ("chunks.parquet", chunks_path),
        ("queries_train.parquet", queries_path),
        ("qrels_train.parquet", qrels_path),
    ]:
        if not p.exists():
            errors.append(f"Missing required file: {name}")

    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "total_documents": 0,
            "total_chunks": 0,
            "total_micro_chunks": 0,
            "total_macro_chunks": 0,
            "total_queries": 0,
            "total_qrels": 0,
            "empty_documents_count": 0,
        }

    docs_df = pd.read_parquet(docs_path)
    chunks_df = pd.read_parquet(chunks_path)
    queries_df = pd.read_parquet(queries_path)
    qrels_df = pd.read_parquet(qrels_path)

    # 1. Check required columns
    for name, df, required in [
        ("documents.parquet", docs_df, CANONICAL_DOCUMENTS_COLUMNS),
        ("chunks.parquet", chunks_df, CANONICAL_CHUNKS_COLUMNS),
        ("queries_train.parquet", queries_df, CANONICAL_QUERIES_COLUMNS),
        ("qrels_train.parquet", qrels_df, CANONICAL_QRELS_COLUMNS),
    ]:
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"{name} missing columns: {sorted(missing)}")

    # 2. Document checks
    docs_df["doc_id"] = docs_df["doc_id"].astype(str)
    dup_docs = docs_df[docs_df.duplicated(subset=["doc_id"])]
    if len(dup_docs) > 0:
        errors.append(f"Found {len(dup_docs)} duplicate document IDs in documents.parquet")

    if expected_document_count is not None and len(docs_df) != expected_document_count:
        errors.append(f"Expected {expected_document_count} documents, found {len(docs_df)}")

    doc_ids = set(docs_df["doc_id"])

    # 3. Chunk checks
    chunks_df["chunk_id"] = chunks_df["chunk_id"].astype(str)
    chunks_df["doc_id"] = chunks_df["doc_id"].astype(str)

    dup_chunks = chunks_df[chunks_df.duplicated(subset=["chunk_id"])]
    if len(dup_chunks) > 0:
        errors.append(f"Found {len(dup_chunks)} duplicate chunk IDs in chunks.parquet")

    invalid_granularity = set(chunks_df["granularity"]) - {"macro", "micro"}
    if invalid_granularity:
        errors.append(f"Invalid granularity values: {invalid_granularity}")

    orphans = set(chunks_df["doc_id"]) - doc_ids
    if orphans:
        errors.append(f"Found {len(orphans)} chunk doc_ids not in documents.parquet")

    if "is_empty" in docs_df.columns:
        non_empty_docs = set(docs_df[~docs_df["is_empty"]]["doc_id"])
    else:
        non_empty_docs = doc_ids
    docs_without_chunks = non_empty_docs - set(chunks_df["doc_id"])
    if docs_without_chunks:
        errors.append(f"Found {len(docs_without_chunks)} non-empty documents with 0 chunks")

    # Parent relationship checks
    macros = chunks_df[chunks_df["granularity"] == "macro"]
    macro_id_to_doc = dict(zip(macros["chunk_id"], macros["doc_id"]))
    macro_chunk_ids = set(macro_id_to_doc.keys())

    micros = chunks_df[chunks_df["granularity"] == "micro"]
    if "parent_chunk_id" in micros.columns:
        micros_with_parent = micros[micros["parent_chunk_id"].notna()]
        missing_parents = set(micros_with_parent["parent_chunk_id"]) - macro_chunk_ids
        if missing_parents:
            errors.append(f"Found {len(missing_parents)} micro chunks referencing missing parent macro chunks")

        # Cross-document parent check
        for _, row in micros_with_parent.iterrows():
            parent_id = str(row["parent_chunk_id"])
            if parent_id in macro_id_to_doc and macro_id_to_doc[parent_id] != str(row["doc_id"]):
                errors.append(f"cross-document parent detected: micro {row['chunk_id']} (doc {row['doc_id']}) has parent {parent_id} (doc {macro_id_to_doc[parent_id]})")
                break

    # 4. Query & Qrel checks
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)

    dup_queries = queries_df[queries_df.duplicated(subset=["query_id"])]
    if len(dup_queries) > 0:
        errors.append(f"Found {len(dup_queries)} duplicate query IDs in queries_train.parquet")

    dup_qrels = qrels_df[qrels_df.duplicated(subset=["query_id", "doc_id"])]
    if len(dup_qrels) > 0:
        errors.append(f"Found {len(dup_qrels)} duplicate (query_id, doc_id) pairs in qrels_train.parquet")

    query_ids = set(queries_df["query_id"])
    unknown_qids = set(qrels_df["query_id"]) - query_ids
    if unknown_qids:
        errors.append(f"Found {len(unknown_qids)} unknown query IDs in qrels_train.parquet not in queries_train.parquet")

    invalid_qrel_docs = set(qrels_df["doc_id"]) - doc_ids
    if invalid_qrel_docs:
        errors.append(f"Found {len(invalid_qrel_docs)} qrel doc_ids not in documents.parquet")

    if "relevance" in qrels_df.columns and (qrels_df["relevance"] != 1).any():
        errors.append("Found qrels with relevance != 1")

    # Gold count consistency
    qrel_counts = qrels_df.groupby("query_id").size().to_dict()
    for _, qrow in queries_df.iterrows():
        qid = qrow["query_id"]
        if "gold_count" in qrow and not pd.isna(qrow["gold_count"]):
            expected_cnt = int(qrow["gold_count"])
            actual_cnt = qrel_counts.get(qid, 0)
            if expected_cnt != actual_cnt:
                errors.append(f"Gold count mismatch for query {qid}: expected {expected_cnt}, found {actual_cnt}")
                break

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "total_documents": len(docs_df),
        "total_chunks": len(chunks_df),
        "total_micro_chunks": int((chunks_df["granularity"] == "micro").sum()),
        "total_macro_chunks": int((chunks_df["granularity"] == "macro").sum()),
        "total_queries": len(queries_df),
        "total_qrels": len(qrels_df),
        "empty_documents_count": int(docs_df["is_empty"].sum()),
        "errors": errors,
    }
