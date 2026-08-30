import re
import pandas as pd
from src.common.normalize import clean_legal_text, prettify_doc_title

def clean_value(val) -> str:
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null") else s

class EvidencePackBuilder:
    def __init__(self, max_chunks_per_doc: int = 2, max_chars_per_chunk: int = 800):
        self.max_chunks_per_doc = max_chunks_per_doc
        self.max_chars_per_chunk = max_chars_per_chunk

    def build_evidence_text(self, query: str, doc_info: dict, chunks: list[dict]) -> str:
        doc_info = doc_info or {}
        title = clean_value(doc_info.get("title") or prettify_doc_title(doc_info.get("name_raw", "")))
        legal_number = clean_value(doc_info.get("legal_number"))

        doc_header = f"{title} {legal_number}".strip() if legal_number else title
        if not doc_header:
            doc_header = "Văn bản quy phạm pháp luật"

        sections = [
            f"[QUESTION] {clean_legal_text(query)}",
            f"[DOCUMENT] {doc_header}"
        ]

        valid_chunks = [c for c in (chunks or []) if c and isinstance(c, dict)][:self.max_chunks_per_doc]
        if not valid_chunks:
            sections.append(f"[EVIDENCE 1] {doc_header}")
        else:
            for idx, c in enumerate(valid_chunks, start=1):
                art = clean_value(c.get("article"))
                body = clean_value(c.get("text_raw") or c.get("text_norm"))
                if len(body) > self.max_chars_per_chunk:
                    body = body[:self.max_chars_per_chunk] + "..."

                chunk_text = f"{art}: {body}".strip() if art else body
                sections.append(f"[EVIDENCE {idx}] {chunk_text}")

        return "\n".join(sections)
