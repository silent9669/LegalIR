import os
import json
import zipfile
import pandas as pd
from src.common.normalize import clean_legal_text, prettify_doc_title
from src.common.legal_parser import parse_legal_structure

def build_dataset(
    raw_zip: str = "selected-contexts.zip",
    train_json: str = "train.json",
    public_json: str = "public-official.json",
    out_dir: str = "artifacts/task1/data"
):
    print("=" * 60)
    print("UIT-DSC 2026 Task 1: Building Canonical Dataset")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    # 1. Read Documents from ZIP
    print(f"Reading legal contexts from {raw_zip}...")
    documents = []
    chunks = []

    with zipfile.ZipFile(raw_zip, "r") as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            with z.open(name) as f:
                data = json.load(f)
                doc_id = str(data.get("id", "")).strip()
                name_raw = data.get("name", "")
                passage = data.get("passage", "")
                title = prettify_doc_title(name_raw)

                doc_entry = {
                    "doc_id": doc_id,
                    "name_raw": name_raw,
                    "title": title,
                    "link": data.get("link", ""),
                    "passage_raw": passage,
                    "passage_norm": clean_legal_text(passage),
                    "legal_number": "",
                    "year": "",
                    "doc_type": "",
                    "is_empty": bool(not passage)
                }
                documents.append(doc_entry)

                # Parse hierarchy into micro/macro chunks
                parsed = parse_legal_structure(passage, doc_id=doc_id)
                for idx, p in enumerate(parsed, start=1):
                    p["chunk_id"] = f"{doc_id}_macro_{idx:03d}"
                    p["granularity"] = "macro"
                    p["title"] = title
                    chunks.append(p)

    df_docs = pd.DataFrame(documents)
    df_chunks = pd.DataFrame(chunks)

    df_docs.to_parquet(os.path.join(out_dir, "documents.parquet"), index=False)
    df_chunks.to_parquet(os.path.join(out_dir, "chunks.parquet"), index=False)
    print(f"Saved {len(df_docs):,} documents and {len(df_chunks):,} chunks.")

    # 2. Read train.json queries & qrels
    print(f"\nProcessing train labels from {train_json}...")
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    queries = []
    qrels = []
    for qid, val in train_data.items():
        q_text = val.get("question", "")
        answers = val.get("answer", [])
        queries.append({
            "query_id": str(qid),
            "question_raw": q_text,
            "question_norm": clean_legal_text(q_text),
            "gold_count": len(answers)
        })
        for ans_doc_id in answers:
            qrels.append({
                "query_id": str(qid),
                "doc_id": str(ans_doc_id),
                "relevance": 1
            })

    df_queries = pd.DataFrame(queries)
    df_qrels = pd.DataFrame(qrels)

    df_queries.to_parquet(os.path.join(out_dir, "queries_train.parquet"), index=False)
    df_qrels.to_parquet(os.path.join(out_dir, "qrels_train.parquet"), index=False)
    print(f"Saved {len(df_queries):,} queries and {len(df_qrels):,} exploded qrels.")

    print("\n" + "=" * 60)
    print("Canonical Dataset Build Complete!")
    print("=" * 60)

if __name__ == "__main__":
    build_dataset()
