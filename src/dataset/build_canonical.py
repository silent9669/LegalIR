import os
import re
import json
import glob
import unicodedata
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.dataset.validator import validate_canonical_dataset
from src.dataset.source_reader import iter_official_contexts

LEGAL_NUM_PATTERN = re.compile(
    r'(?:Số|Số\s*:)\s*([0-9]+(?:\/[0-9]+)?(?:\/[A-ZĐ0-9\-\_]+)?)',
    re.IGNORECASE
)
YEAR_PATTERN = re.compile(r'\b(19[89]\d|20[012]\d)\b')
DOC_TYPE_PATTERN = re.compile(
    r'\b(Luật|Bộ luật|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Chỉ thị|Nghị quyết|Tiêu chuẩn|Công văn|Pháp lệnh)\b',
    re.IGNORECASE
)
CHƯƠNG_PATTERN = re.compile(r'(?:\n|\A)\s*(Chương\s+[IVXLCDM\d]+[^\n]*)', re.IGNORECASE)
MỤC_PATTERN = re.compile(r'(?:\n|\A)\s*(Mục\s+\d+[^\n]*)', re.IGNORECASE)
ĐIỀU_PATTERN = re.compile(r'(?:\n|\A)\s*(Điều\s+\d+[\.\:\s][^\n]*)', re.IGNORECASE)
KHOẢN_PATTERN = re.compile(r'(?:\n|\A)\s*(\d+\.\s+[^\n]*)')

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()

def extract_metadata(doc_id: str, name: str, link: str, passage: str) -> dict:
    name_str = name or ""
    link_str = link or ""
    passage_sample = (passage or "")[:2000]

    # Extract title
    title = ""
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
        "doc_type": doc_type
    }

def split_sliding_windows(text: str, max_chars: int = 800, overlap: int = 150) -> list:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    curr = []
    curr_len = 0
    for p in paragraphs:
        if curr_len + len(p) > max_chars and curr:
            chunks.append("\n".join(curr))
            curr = [p]
            curr_len = len(p)
        else:
            curr.append(p)
            curr_len += len(p)
    if curr:
        chunks.append("\n".join(curr))
    return chunks

def build_canonical_package(raw_contexts_dir: str, train_json_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read all official contexts
    raw_path = Path(raw_contexts_dir)
    if raw_path.is_file() and raw_path.suffix.lower() == ".zip":
        contexts_iter = list(iter_official_contexts(raw_path))
    else:
        context_files = sorted(glob.glob(os.path.join(raw_contexts_dir, "context_*.json")))
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

    for data in tqdm(contexts_iter, desc="Parsing documents"):

        doc_id = str(data["id"])
        name_raw = data.get("name") or ""
        link = data.get("link") or ""
        passage_raw = data.get("passage") or ""

        is_empty = not bool(passage_raw and passage_raw.strip())
        passage_norm = normalize_text(passage_raw)
        meta = extract_metadata(doc_id, name_raw, link, passage_raw)

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
            "is_empty": is_empty
        }
        docs_records.append(doc_record)

        # Build chunks
        if is_empty:
            macro_id = f"{doc_id}_macro_001"
            micro_id = f"{doc_id}_micro_001"
            fallback_text = f"[VĂN BẢN]: {meta['title']} (Số: {meta['legal_number'] or ''}, Năm: {meta['year'] or ''})\n[NỘI DUNG]: Văn bản không có nội dung chi tiết."

            chunks_records.append({
                "chunk_id": macro_id,
                "doc_id": doc_id,
                "granularity": "macro",
                "article": "Thông tin văn bản",
                "clause": None,
                "text_raw": fallback_text,
                "text_norm": normalize_text(fallback_text),
                "parent_chunk_id": None,
                "token_count": len(fallback_text.split())
            })
            chunks_records.append({
                "chunk_id": micro_id,
                "doc_id": doc_id,
                "granularity": "micro",
                "article": "Thông tin văn bản",
                "clause": None,
                "text_raw": fallback_text,
                "text_norm": normalize_text(fallback_text),
                "parent_chunk_id": macro_id,
                "token_count": len(fallback_text.split())
            })
            continue

        # Parse structure with Điều
        dieu_matches = list(ĐIỀU_PATTERN.finditer(passage_norm))

        if len(dieu_matches) >= 1:
            # Document has explicit Điều structure
            # Check pre-dieu text
            pre_dieu = passage_norm[:dieu_matches[0].start()].strip()
            if pre_dieu and len(pre_dieu) > 100:
                pre_macro_id = f"{doc_id}_macro_pre"
                pre_micro_id = f"{doc_id}_micro_pre"
                pre_header = f"[VĂN BẢN]: {meta['title']} (Số: {meta['legal_number'] or ''})\n[CĂN CỨ]:\n{pre_dieu[:1000]}"
                chunks_records.append({
                    "chunk_id": pre_macro_id,
                    "doc_id": doc_id,
                    "granularity": "macro",
                    "article": "Căn cứ ban hành",
                    "clause": None,
                    "text_raw": pre_dieu,
                    "text_norm": normalize_text(pre_header),
                    "parent_chunk_id": None,
                    "token_count": len(pre_header.split())
                })
                chunks_records.append({
                    "chunk_id": pre_micro_id,
                    "doc_id": doc_id,
                    "granularity": "micro",
                    "article": "Căn cứ ban hành",
                    "clause": None,
                    "text_raw": pre_dieu,
                    "text_norm": normalize_text(pre_header),
                    "parent_chunk_id": pre_macro_id,
                    "token_count": len(pre_header.split())
                })

            for idx, d_match in enumerate(dieu_matches, 1):
                start_p = d_match.start()
                end_p = dieu_matches[idx].start() if idx < len(dieu_matches) else len(passage_norm)
                dieu_text = passage_norm[start_p:end_p].strip()
                lines = dieu_text.split('\n')
                dieu_header = lines[0].strip()
                dieu_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0].strip()

                macro_id = f"{doc_id}_macro_{idx:03d}"
                macro_content = f"[VĂN BẢN]: {meta['title']} (Số: {meta['legal_number'] or ''})\n[ĐIỀU KHOẢN]: {dieu_header}\n[NỘI DUNG]:\n{dieu_body}"

                chunks_records.append({
                    "chunk_id": macro_id,
                    "doc_id": doc_id,
                    "granularity": "macro",
                    "article": dieu_header,
                    "clause": None,
                    "text_raw": dieu_text,
                    "text_norm": normalize_text(macro_content),
                    "parent_chunk_id": None,
                    "token_count": len(macro_content.split())
                })

                # Extract micro chunks by clause
                khoan_matches = list(KHOẢN_PATTERN.finditer(dieu_body))
                if len(khoan_matches) >= 2:
                    for k_idx, k_match in enumerate(khoan_matches, 1):
                        k_start = k_match.start()
                        k_end = khoan_matches[k_idx].start() if k_idx < len(khoan_matches) else len(dieu_body)
                        clause_text = dieu_body[k_start:k_end].strip()
                        micro_id = f"{doc_id}_micro_{idx:03d}_{k_idx:02d}"
                        micro_content = f"[VĂN BẢN]: {meta['title']} (Số: {meta['legal_number'] or ''})\n[ĐIỀU KHOẢN]: {dieu_header}\n[KHOẢN]:\n{clause_text}"

                        chunks_records.append({
                            "chunk_id": micro_id,
                            "doc_id": doc_id,
                            "granularity": "micro",
                            "article": dieu_header,
                            "clause": f"Khoản {k_idx}",
                            "text_raw": clause_text,
                            "text_norm": normalize_text(micro_content),
                            "parent_chunk_id": macro_id,
                            "token_count": len(micro_content.split())
                        })
                else:
                    # Single clause or small body -> 1 micro chunk
                    micro_id = f"{doc_id}_micro_{idx:03d}_01"
                    chunks_records.append({
                        "chunk_id": micro_id,
                        "doc_id": doc_id,
                        "granularity": "micro",
                        "article": dieu_header,
                        "clause": None,
                        "text_raw": dieu_body,
                        "text_norm": normalize_text(macro_content),
                        "parent_chunk_id": macro_id,
                        "token_count": len(macro_content.split())
                    })
        else:
            # Fallback sliding window for unstructured document
            windows = split_sliding_windows(passage_norm, max_chars=800)
            for w_idx, win in enumerate(windows, 1):
                macro_id = f"{doc_id}_macro_{w_idx:03d}"
                micro_id = f"{doc_id}_micro_{w_idx:03d}"
                w_content = f"[VĂN BẢN]: {meta['title']} (Số: {meta['legal_number'] or ''})\n[ĐOẠN {w_idx}]:\n{win}"

                chunks_records.append({
                    "chunk_id": macro_id,
                    "doc_id": doc_id,
                    "granularity": "macro",
                    "article": f"Đoạn {w_idx}",
                    "clause": None,
                    "text_raw": win,
                    "text_norm": normalize_text(w_content),
                    "parent_chunk_id": None,
                    "token_count": len(w_content.split())
                })
                chunks_records.append({
                    "chunk_id": micro_id,
                    "doc_id": doc_id,
                    "granularity": "micro",
                    "article": f"Đoạn {w_idx}",
                    "clause": None,
                    "text_raw": win,
                    "text_norm": normalize_text(w_content),
                    "parent_chunk_id": macro_id,
                    "token_count": len(w_content.split())
                })

    # Save documents.parquet & chunks.parquet
    docs_df = pd.DataFrame(docs_records)
    chunks_df = pd.DataFrame(chunks_records)

    docs_path = os.path.join(output_dir, "documents.parquet")
    chunks_path = os.path.join(output_dir, "chunks.parquet")

    print(f"Saving {len(docs_df)} documents to {docs_path}...")
    docs_df.to_parquet(docs_path, index=False)

    print(f"Saving {len(chunks_df)} chunks to {chunks_path}...")
    chunks_df.to_parquet(chunks_path, index=False)

    # 2. Process train.json -> queries_train.parquet, qrels_train.parquet
    print(f"Processing training data from {train_json_path}...")
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    queries_records = []
    qrels_records = []

    for qid, qobj in train_data.items():
        q_raw = qobj.get("question", "")
        q_norm = normalize_text(q_raw)
        answers = [str(x) for x in qobj.get("answer", [])]

        queries_records.append({
            "query_id": str(qid),
            "question_raw": q_raw,
            "question_norm": q_norm,
            "gold_count": len(answers)
        })

        for doc_id in answers:
            qrels_records.append({
                "query_id": str(qid),
                "doc_id": str(doc_id),
                "relevance": 1
            })

    queries_df = pd.DataFrame(queries_records)
    qrels_df = pd.DataFrame(qrels_records)

    queries_path = os.path.join(output_dir, "queries_train.parquet")
    qrels_path = os.path.join(output_dir, "qrels_train.parquet")

    print(f"Saving {len(queries_df)} queries to {queries_path}...")
    queries_df.to_parquet(queries_path, index=False)
    print(f"Saving {len(qrels_df)} qrels to {qrels_path}...")
    qrels_df.to_parquet(qrels_path, index=False)

    # 3. Validate invariants & write manifest and audit report
    report = validate_canonical_dataset(output_dir)
    print("Dataset Audit Report:\n", json.dumps(report, indent=2, ensure_ascii=False))

    with open(os.path.join(output_dir, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    manifest = {
        "dataset": "task1_canonical",
        "version": "v1",
        "total_documents": report["total_documents"],
        "total_chunks": report["total_chunks"],
        "total_micro_chunks": report["total_micro_chunks"],
        "total_macro_chunks": report["total_macro_chunks"],
        "total_queries": report["total_queries"],
        "total_qrels": report["total_qrels"],
        "schema": "hierarchical_micro_macro",
        "normalization": "nfc_whitespace_preserve_legal_ids"
    }

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if not report["is_valid"]:
        raise ValueError(f"Canonical dataset failed validation: {report['errors']}")

    print(f"Canonical dataset v1 successfully built and verified in {output_dir}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_contexts_dir", type=str, default="selected-contexts")
    parser.add_argument("--train_json", type=str, default="train.json")
    parser.add_argument("--output_dir", type=str, default="data/task1_canonical/v1")
    args = parser.parse_args()

    build_canonical_package(args.raw_contexts_dir, args.train_json, args.output_dir)
