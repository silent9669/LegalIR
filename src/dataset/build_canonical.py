from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any
import argparse
import glob
import json
import os
import re
import unicodedata
import pandas as pd
from tqdm import tqdm

from src.core.config import load_pipeline_config
from src.dataset.chunker import ChunkConfig, build_document_chunks, normalize_text
from src.dataset.source_reader import iter_official_contexts
from src.dataset.validator import validate_canonical_dataset
from src.evaluation.splits import create_all_splits

LEGAL_NUM_PATTERN = re.compile(
    r'(?:Số|Số\s*:)\s*([0-9]+(?:\/[0-9]+)?(?:\/[A-ZĐ0-9\-\_]+)?)',
    re.IGNORECASE
)
YEAR_PATTERN = re.compile(r'\b(19[89]\d|20[012]\d)\b')
DOC_TYPE_PATTERN = re.compile(
    r'\b(Luật|Bộ luật|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Chỉ thị|Nghị quyết|Tiêu chuẩn|Công văn|Pháp lệnh)\b',
    re.IGNORECASE
)


def extract_metadata(doc_id: str, name: str, link: str, passage: str) -> dict[str, Any]:
    name_str = name or ""
    link_str = link or ""
    passage_sample = (passage or "")[:2000]

    # Extract title
    if name_str:
        clean_name = re.sub(r'-\d+$', '', name_str)
        clean_name = clean_name.replace('-', ' ')
        title = clean_name.strip()
    elif link_str:
        slug = link_str.rstrip('/').split('/')[-1]
        slug = re.sub(r'-\d+$', '', slug).replace('-', ' ')
        title = slug.strip()
    else:
        title = f"Văn bản pháp luật {doc_id}"

    # Extract legal number
    legal_num = None
    m_num = LEGAL_NUM_PATTERN.search(passage_sample)
    if m_num:
        legal_num = m_num.group(1).strip()
    else:
        m_name_num = re.search(r'(\d+[-/][0-9A-Za-zĐ-]+)', name_str)
        if m_name_num:
            legal_num = m_name_num.group(1).replace('-', '/')

    # Extract year
    year = None
    m_year = YEAR_PATTERN.findall(passage_sample)
    if m_year:
        year = m_year[0]
    else:
        m_name_year = YEAR_PATTERN.findall(name_str)
        if m_name_year:
            year = m_name_year[0]

    # Extract document type
    doc_type = "Văn bản"
    m_type = DOC_TYPE_PATTERN.search(passage_sample)
    if m_type:
        doc_type = m_type.group(1).capitalize()
    else:
        m_name_type = DOC_TYPE_PATTERN.search(title)
        if m_name_type:
            doc_type = m_name_type.group(1).capitalize()

    return {
        "title": title,
        "legal_number": legal_num,
        "year": str(year) if year else None,
        "doc_type": doc_type,
    }


def build_canonical_package(
    raw_contexts_dir: str | Path,
    train_json_path: str | Path,
    output_dir: str | Path,
    chunk_config: ChunkConfig = ChunkConfig(),
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(raw_contexts_dir)
    if raw_path.is_file() and raw_path.suffix.lower() == ".zip":
        contexts_iter = list(iter_official_contexts(raw_path))
    else:
        context_files = sorted(glob.glob(os.path.join(str(raw_contexts_dir), "context_*.json")))
        if not context_files:
            raise FileNotFoundError(f"No context_*.json files found in {raw_contexts_dir}")
        contexts_iter = []
        for fpath in context_files:
            with open(fpath, "r", encoding="utf-8") as f:
                row = json.load(f)
                row["id"] = str(row["id"])
                contexts_iter.append(row)

    print(f"Processing {len(contexts_iter)} official contexts...")

    docs_records = []
    chunks_records = []
    passage_hashes = defaultdict(list)
    empty_context_ids = []

    for data in tqdm(contexts_iter, desc="Parsing documents and chunks"):
        doc_id = str(data["id"])
        name_raw = data.get("name") or ""
        link = data.get("link") or ""
        passage_raw = data.get("passage") or ""

        is_empty = not bool(passage_raw and passage_raw.strip())
        if is_empty:
            empty_context_ids.append(doc_id)

        passage_norm = normalize_text(passage_raw)
        meta = extract_metadata(doc_id, name_raw, link, passage_raw)

        if passage_norm:
            p_hash = sha256(passage_norm.encode("utf-8")).hexdigest()
            passage_hashes[p_hash].append(doc_id)

        doc_record = {
            "doc_id": doc_id,
            "name_raw": name_raw,
            "title": meta["title"],
            "link": link,
            "passage_raw": passage_raw,
            "passage_norm": passage_norm,
            "legal_number": meta["legal_number"],
            "year": meta["year"],
            "doc_type": meta["doc_type"],
            "is_empty": is_empty,
        }
        docs_records.append(doc_record)

        # Chunk generation
        doc_chunks = build_document_chunks(doc_record, chunk_config)
        chunks_records.extend(doc_chunks)

    # Save duplicate groups & empty context IDs
    duplicate_groups = {
        h: doc_list for h, doc_list in passage_hashes.items() if len(doc_list) > 1
    }
    (output_dir / "duplicate_groups.json").write_text(
        json.dumps(duplicate_groups, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "empty_context_ids.json").write_text(
        json.dumps(empty_context_ids, indent=2) + "\n", encoding="utf-8"
    )

    # Save documents.parquet & chunks.parquet
    docs_df = pd.DataFrame(docs_records)
    chunks_df = pd.DataFrame(chunks_records)

    docs_path = output_dir / "documents.parquet"
    chunks_path = output_dir / "chunks.parquet"

    print(f"Saving {len(docs_df)} documents to {docs_path}...")
    docs_df.to_parquet(docs_path, index=False)

    print(f"Saving {len(chunks_df)} chunks to {chunks_path}...")
    chunks_df.to_parquet(chunks_path, index=False)

    # Process training queries & qrels
    train_json_path = Path(train_json_path)
    print(f"Processing training data from {train_json_path}...")
    train_data = json.loads(train_json_path.read_text(encoding="utf-8"))

    queries_records = []
    qrels_records = []

    for qid, qobj in sorted(train_data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        q_raw = qobj.get("question", "")
        q_norm = normalize_text(q_raw)
        answers = [str(x) for x in qobj.get("answer", [])]

        queries_records.append({
            "query_id": str(qid),
            "question_raw": q_raw,
            "question_norm": q_norm,
            "gold_count": len(answers),
        })

        for doc_id in answers:
            qrels_records.append({
                "query_id": str(qid),
                "doc_id": str(doc_id),
                "relevance": 1,
            })

    queries_df = pd.DataFrame(queries_records)
    qrels_df = pd.DataFrame(qrels_records)

    queries_path = output_dir / "queries_train.parquet"
    qrels_path = output_dir / "qrels_train.parquet"

    print(f"Saving {len(queries_df)} queries to {queries_path}...")
    queries_df.to_parquet(queries_path, index=False)
    print(f"Saving {len(qrels_df)} qrels to {qrels_path}...")
    qrels_df.to_parquet(qrels_path, index=False)

    # Create splits
    create_all_splits(output_dir)

    # Validate invariants
    report = validate_canonical_dataset(output_dir, expected_document_count=len(docs_df))
    print("Dataset Audit Report:\n", json.dumps(report, indent=2, ensure_ascii=False))

    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest = {
        "dataset": "task1_canonical",
        "version": "v2",
        "total_documents": report["total_documents"],
        "total_chunks": report["total_chunks"],
        "total_micro_chunks": report["total_micro_chunks"],
        "total_macro_chunks": report["total_macro_chunks"],
        "total_queries": report["total_queries"],
        "total_qrels": report["total_qrels"],
        "total_duplicate_groups": len(duplicate_groups),
        "empty_documents_count": len(empty_context_ids),
        "schema": "hierarchical_micro_macro_v2",
        "normalization": "nfc_whitespace_preserve_legal_ids",
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not report["is_valid"]:
        raise ValueError(f"Canonical dataset failed validation: {report['errors']}")

    print(f"Canonical dataset v2 successfully built and verified in {output_dir}!")
    return report


def main():
    parser = argparse.ArgumentParser(description="LegalIR Canonical Dataset v2 Builder")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml", help="Pipeline configuration path")
    parser.add_argument("--raw-zip", type=str, default=None, help="Override raw selected-contexts.zip path")
    parser.add_argument("--train-json", type=str, default=None, help="Override train.json path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    cfg = {}
    if Path(args.config).exists():
        cfg = load_pipeline_config(Path(args.config))

    raw_zip = args.raw_zip or cfg.get("dataset", {}).get("raw_zip", "artifacts/shared/raw/selected-contexts.zip")
    train_json = args.train_json or cfg.get("dataset", {}).get("train_json", "artifacts/shared/raw/train.json")
    output_dir = args.output_dir or cfg.get("paths", {}).get("canonical", "artifacts/shared/canonical/v2")

    c_cfg_data = cfg.get("dataset", {}).get("chunking", {})
    chunk_config = ChunkConfig(
        macro_min_tokens=c_cfg_data.get("macro_min_tokens", 400),
        macro_max_tokens=c_cfg_data.get("macro_max_tokens", 800),
        micro_min_tokens=c_cfg_data.get("micro_min_tokens", 100),
        micro_max_tokens=c_cfg_data.get("micro_max_tokens", 250),
        fallback_min_tokens=c_cfg_data.get("fallback_min_tokens", 700),
        fallback_max_tokens=c_cfg_data.get("fallback_max_tokens", 1200),
        overlap_tokens=c_cfg_data.get("overlap_tokens", 150),
    )

    build_canonical_package(
        raw_contexts_dir=raw_zip,
        train_json_path=train_json,
        output_dir=output_dir,
        chunk_config=chunk_config,
    )


if __name__ == "__main__":
    main()
