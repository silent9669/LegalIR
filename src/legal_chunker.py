import os
import json
import re
import glob
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        total = len(iterable) if hasattr(iterable, "__len__") else None
        for idx, item in enumerate(iterable):
            if idx % 1000 == 0:
                print(f"[{desc}] Processed {idx}/{total or '?'}")
            yield item
from data_utils import normalize_text, extract_metadata_from_doc, clean_dieu_header

CHƯƠNG_PATTERN = re.compile(r'(?:\n|\A)\s*(Chương\s+[IVXLCDM\d]+[^\n]*)', re.IGNORECASE)
MỤC_PATTERN = re.compile(r'(?:\n|\A)\s*(Mục\s+\d+[^\n]*)', re.IGNORECASE)
ĐIỀU_PATTERN = re.compile(r'(?:\n|\A)\s*(Điều\s+\d+[\.\:\s][^\n]*)', re.IGNORECASE)
KHOẢN_PATTERN = re.compile(r'(?:\n|\A)\s*(\d+\.\s+[^\n]*)')

def split_text_by_window(text: str, max_chars: int = 900, overlap: int = 150) -> list:
    """Split long text into sliding windows respecting sentence/paragraph boundaries."""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0

    for p in paragraphs:
        if current_len + len(p) > max_chars and current_chunk:
            chunk_str = "\n".join(current_chunk)
            chunks.append(chunk_str)
            # keep last paragraph if it fits in overlap
            if len(p) < max_chars:
                current_chunk = [p]
                current_len = len(p)
            else:
                # If a single paragraph is longer than max_chars, split by sentence
                sentences = re.split(r'(?<=[\.\?\!])\s+', p)
                sub_chunk = []
                sub_len = 0
                for s in sentences:
                    if sub_len + len(s) > max_chars and sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = [s]
                        sub_len = len(s)
                    else:
                        sub_chunk.append(s)
                        sub_len += len(s)
                if sub_chunk:
                    current_chunk = [" ".join(sub_chunk)]
                    current_len = sub_len
        else:
            current_chunk.append(p)
            current_len += len(p)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def format_chunk_content(meta: dict, chuong_muc: str, dieu_title: str, body: str) -> str:
    """Construct contextualized text for optimal BM25 and dense embedding."""
    header_parts = []

    # Title & Legal Number & Year
    title_str = meta.get("title", "")
    num_str = meta.get("legal_number")
    year_str = meta.get("year")
    meta_info = []
    if num_str: meta_info.append(f"Số: {num_str}")
    if year_str: meta_info.append(f"Năm: {year_str}")

    if meta_info:
        header_parts.append(f"[VĂN BẢN]: {title_str} ({', '.join(meta_info)})")
    else:
        header_parts.append(f"[VĂN BẢN]: {title_str}")

    if chuong_muc:
        header_parts.append(f"[CHƯƠNG / MỤC]: {chuong_muc}")

    if dieu_title:
        header_parts.append(f"[ĐIỀU KHOẢN]: {dieu_title}")

    header_parts.append("[NỘI DUNG]:\n" + body.strip())

    return "\n".join(header_parts)

def chunk_document(doc: dict, max_chunk_chars: int = 1000) -> list:
    """Chunk a legal document with full hierarchy and 100% coverage."""
    doc_id = str(doc.get("id"))
    name = doc.get("name")
    link = doc.get("link", "")
    passage = doc.get("passage", "") or ""

    meta = extract_metadata_from_doc(doc_id, name, link, passage)
    passage = normalize_text(passage)

    # If passage is empty, return 1 metadata chunk
    if not passage or len(passage.strip()) == 0:
        meta_body = f"Văn bản {meta.get('title')} ({meta.get('legal_number') or ''}) không có nội dung văn bản chi tiết."
        content = format_chunk_content(meta, "", "Thông tin văn bản", meta_body)
        return [{
            "chunk_id": f"{doc_id}_meta",
            "doc_id": doc_id,
            "id": doc_id,
            "title": meta.get("title"),
            "legal_number": meta.get("legal_number"),
            "year": meta.get("year"),
            "doc_type": meta.get("doc_type"),
            "link": link,
            "structure": "meta_fallback",
            "chuong_muc": None,
            "dieu": None,
            "khoan": None,
            "part": 1,
            "n_parts": 1,
            "body": meta_body,
            "content": content,
            "char_len": len(content)
        }]

    # Try to split by Điều
    dieu_matches = list(ĐIỀU_PATTERN.finditer(passage))

    chunks = []

    if len(dieu_matches) >= 2:
        # Document is structured with Điều
        # Extract pre-dieu text (preamble / general info)
        pre_dieu_text = passage[:dieu_matches[0].start()].strip()
        if pre_dieu_text and len(pre_dieu_text) > 100:
            pre_windows = split_text_by_window(pre_dieu_text, max_chars=max_chunk_chars)
            for idx, w in enumerate(pre_windows, 1):
                content = format_chunk_content(meta, "", "Phần mở đầu / Căn cứ ban hành", w)
                chunks.append({
                    "chunk_id": f"{doc_id}_pre_p{idx}",
                    "doc_id": doc_id,
                    "id": doc_id,
                    "title": meta.get("title"),
                    "legal_number": meta.get("legal_number"),
                    "year": meta.get("year"),
                    "doc_type": meta.get("doc_type"),
                    "link": link,
                    "structure": "preamble",
                    "chuong_muc": None,
                    "dieu": "Phần mở đầu",
                    "khoan": None,
                    "part": idx,
                    "n_parts": len(pre_windows),
                    "body": w,
                    "content": content,
                    "char_len": len(content)
                })

        # Process each Điều
        current_chuong = None
        current_muc = None

        for i in range(len(dieu_matches)):
            start_pos = dieu_matches[i].start()
            end_pos = dieu_matches[i+1].start() if i + 1 < len(dieu_matches) else len(passage)
            dieu_raw = passage[start_pos:end_pos].strip()

            # Check if Chương or Mục appeared in text leading up to this Điều
            preceding_slice = passage[max(0, start_pos - 300):start_pos]
            ch_match = CHƯƠNG_PATTERN.findall(preceding_slice)
            if ch_match: current_chuong = ch_match[-1].strip()
            m_match = MỤC_PATTERN.findall(preceding_slice)
            if m_match: current_muc = m_match[-1].strip()

            chuong_muc_str = " - ".join(filter(None, [current_chuong, current_muc]))

            # Extract header and body of Điều
            lines = dieu_raw.split('\n')
            dieu_header = clean_dieu_header(lines[0])
            dieu_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0]

            if not dieu_body:
                dieu_body = dieu_header

            if len(dieu_body) <= max_chunk_chars:
                content = format_chunk_content(meta, chuong_muc_str, dieu_header, dieu_body)
                chunks.append({
                    "chunk_id": f"{doc_id}_dieu{i+1}_p1",
                    "doc_id": doc_id,
                    "id": doc_id,
                    "title": meta.get("title"),
                    "legal_number": meta.get("legal_number"),
                    "year": meta.get("year"),
                    "doc_type": meta.get("doc_type"),
                    "link": link,
                    "structure": "dieu",
                    "chuong_muc": chuong_muc_str or None,
                    "dieu": dieu_header,
                    "khoan": None,
                    "part": 1,
                    "n_parts": 1,
                    "body": dieu_body,
                    "content": content,
                    "char_len": len(content)
                })
            else:
                # Split large Điều into sub-parts (clauses or sliding window)
                sub_parts = split_text_by_window(dieu_body, max_chars=max_chunk_chars)
                for p_idx, sp in enumerate(sub_parts, 1):
                    content = format_chunk_content(meta, chuong_muc_str, f"{dieu_header} (Phần {p_idx}/{len(sub_parts)})", sp)
                    chunks.append({
                        "chunk_id": f"{doc_id}_dieu{i+1}_p{p_idx}",
                        "doc_id": doc_id,
                        "id": doc_id,
                        "title": meta.get("title"),
                        "legal_number": meta.get("legal_number"),
                        "year": meta.get("year"),
                        "doc_type": meta.get("doc_type"),
                        "link": link,
                        "structure": "dieu",
                        "chuong_muc": chuong_muc_str or None,
                        "dieu": dieu_header,
                        "khoan": f"Phần {p_idx}/{len(sub_parts)}",
                        "part": p_idx,
                        "n_parts": len(sub_parts),
                        "body": sp,
                        "content": content,
                        "char_len": len(content)
                    })
    else:
        # Plain text without clear Điều structure
        windows = split_text_by_window(passage, max_chars=max_chunk_chars)
        for idx, w in enumerate(windows, 1):
            content = format_chunk_content(meta, "", f"Đoạn {idx}/{len(windows)}", w)
            chunks.append({
                "chunk_id": f"{doc_id}_plain_p{idx}",
                "doc_id": doc_id,
                "id": doc_id,
                "title": meta.get("title"),
                "legal_number": meta.get("legal_number"),
                "year": meta.get("year"),
                "doc_type": meta.get("doc_type"),
                "link": link,
                "structure": "plain",
                "chuong_muc": None,
                "dieu": None,
                "khoan": None,
                "part": idx,
                "n_parts": len(windows),
                "body": w,
                "content": content,
                "char_len": len(content)
            })

    return chunks

def process_all_contexts(contexts_dir: str = "selected-contexts", output_file: str = "data/processed_chunks.jsonl", meta_file: str = "data/doc_metadata.json"):
    """Process all JSON files in selected-contexts and output clean, contextualized chunks."""
    files = sorted(glob.glob(os.path.join(contexts_dir, "*.json")))
    print(f"Found {len(files)} context files in {contexts_dir}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_chunks = []
    doc_metadata = {}
    seen_doc_ids = set()

    with open(output_file, "w", encoding="utf-8") as f_out:
        for fp in tqdm(files, desc="Processing documents"):
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)

            doc_id = str(doc.get("id"))
            seen_doc_ids.add(doc_id)

            # Extract doc metadata for global registry
            meta = extract_metadata_from_doc(doc_id, doc.get("name"), doc.get("link", ""), doc.get("passage", "") or "")
            doc_metadata[doc_id] = meta

            chunks = chunk_document(doc)
            for c in chunks:
                f_out.write(json.dumps(c, ensure_ascii=False) + "\n")
                all_chunks.append(c)

    with open(meta_file, "w", encoding="utf-8") as f_meta:
        json.dump(doc_metadata, f_meta, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete!")
    print(f"- Total documents processed: {len(seen_doc_ids)}")
    print(f"- Total chunks generated: {len(all_chunks)}")
    print(f"- Saved chunks to: {output_file}")
    print(f"- Saved metadata to: {meta_file}")

    return seen_doc_ids

if __name__ == "__main__":
    process_all_contexts()
