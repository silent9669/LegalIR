import os
import json
import pandas as pd

def validate_canonical_dataset(canonical_dir: str) -> dict:
    docs_path = os.path.join(canonical_dir, "documents.parquet")
    chunks_path = os.path.join(canonical_dir, "chunks.parquet")
    queries_path = os.path.join(canonical_dir, "queries_train.parquet")
    qrels_path = os.path.join(canonical_dir, "qrels_train.parquet")

    assert os.path.exists(docs_path), f"Missing {docs_path}"
    assert os.path.exists(chunks_path), f"Missing {chunks_path}"
    assert os.path.exists(queries_path), f"Missing {queries_path}"
    assert os.path.exists(qrels_path), f"Missing {qrels_path}"

    docs_df = pd.read_parquet(docs_path)
    chunks_df = pd.read_parquet(chunks_path)
    queries_df = pd.read_parquet(queries_path)
    qrels_df = pd.read_parquet(qrels_path)

    doc_ids = set(docs_df["doc_id"].astype(str))
    chunk_doc_ids = set(chunks_df["doc_id"].astype(str))
    qrel_doc_ids = set(qrels_df["doc_id"].astype(str))

    errors = []

    # Invariant 1: Every chunk doc_id must exist in documents
    orphans = chunk_doc_ids - doc_ids
    if orphans:
        errors.append(f"Found {len(orphans)} chunk doc_ids not in documents.parquet")

    # Invariant 2: Every non-empty document must have at least one chunk
    non_empty_docs = set(docs_df[~docs_df["is_empty"]]["doc_id"].astype(str))
    docs_without_chunks = non_empty_docs - chunk_doc_ids
    if docs_without_chunks:
        errors.append(f"Found {len(docs_without_chunks)} non-empty documents with 0 chunks")

    # Invariant 3: Every qrel doc_id must exist in documents
    invalid_qrels = qrel_doc_ids - doc_ids
    if invalid_qrels:
        errors.append(f"Found {len(invalid_qrels)} qrel doc_ids not in documents.parquet")

    # Invariant 4: Micro chunks with parent_chunk_id must map to valid macro chunk_id
    macro_chunk_ids = set(chunks_df[chunks_df["granularity"] == "macro"]["chunk_id"])
    micro_chunks = chunks_df[chunks_df["granularity"] == "micro"]
    missing_parents = set(micro_chunks["parent_chunk_id"].dropna()) - macro_chunk_ids
    if missing_parents:
        errors.append(f"Found {len(missing_parents)} micro chunks referencing missing parent macro chunks")

    is_valid = len(errors) == 0
    report = {
        "is_valid": is_valid,
        "total_documents": len(docs_df),
        "total_chunks": len(chunks_df),
        "total_micro_chunks": int((chunks_df["granularity"] == "micro").sum()),
        "total_macro_chunks": int((chunks_df["granularity"] == "macro").sum()),
        "total_queries": len(queries_df),
        "total_qrels": len(qrels_df),
        "empty_documents_count": int(docs_df["is_empty"].sum()),
        "errors": errors
    }
    return report
