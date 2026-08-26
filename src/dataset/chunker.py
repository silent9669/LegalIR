from dataclasses import dataclass
from typing import Any
import unicodedata
import re
from src.dataset.legal_parser import parse_legal_units, LegalUnit, ĐIỀU_PATTERN


@dataclass(frozen=True)
class ChunkConfig:
    macro_min_tokens: int = 400
    macro_max_tokens: int = 800
    micro_min_tokens: int = 100
    micro_max_tokens: int = 250
    fallback_min_tokens: int = 700
    fallback_max_tokens: int = 1200
    overlap_tokens: int = 150


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def sliding_token_windows(tokens: list[str], max_tokens: int, overlap: int) -> list[list[str]]:
    if not tokens:
        return []
    if len(tokens) <= max_tokens:
        return [tokens]
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    step = max_tokens - overlap
    windows = []
    for start in range(0, len(tokens), step):
        chunk_toks = tokens[start:start + max_tokens]
        if chunk_toks:
            windows.append(chunk_toks)
        if start + max_tokens >= len(tokens):
            break
    return windows


def build_document_chunks(doc: dict[str, Any], config: ChunkConfig = ChunkConfig()) -> list[dict[str, Any]]:
    doc_id = str(doc["doc_id"])
    title = doc.get("title") or f"Văn bản pháp luật {doc_id}"
    legal_num = doc.get("legal_number") or ""
    is_empty = doc.get("is_empty", False)
    passage = normalize_text(doc.get("passage_norm") or doc.get("passage_raw") or "")

    # Header prefix
    header_prefix = f"[VĂN BẢN]: {title}"
    if legal_num:
        header_prefix += f" (Số: {legal_num})"

    if is_empty or not passage:
        macro_id = f"{doc_id}_macro_001"
        fallback_text = f"{header_prefix}\n[NỘI DUNG]: Văn bản không có nội dung chi tiết."
        return [{
            "chunk_id": macro_id,
            "doc_id": doc_id,
            "granularity": "macro",
            "chapter": None,
            "section": None,
            "article": "Thông tin văn bản",
            "clause": None,
            "point": None,
            "text_raw": "",
            "text_norm": normalize_text(fallback_text),
            "parent_chunk_id": None,
            "token_count": len(fallback_text.split()),
            "is_empty": True,
        }]

    # Structure check with Điều
    dieu_matches = list(ĐIỀU_PATTERN.finditer(passage))
    chunks: list[dict[str, Any]] = []

    if dieu_matches:
        macro_idx = 1
        # Pre-dieu preamble
        pre_dieu = passage[:dieu_matches[0].start()].strip()
        if pre_dieu and len(pre_dieu.split()) >= 30:
            macro_id = f"{doc_id}_macro_{macro_idx:03d}"
            macro_idx += 1
            content = f"{header_prefix}\n[CĂN CỨ]:\n{pre_dieu}"
            norm_content = normalize_text(content)
            chunks.append({
                "chunk_id": macro_id,
                "doc_id": doc_id,
                "granularity": "macro",
                "chapter": None,
                "section": None,
                "article": "Căn cứ ban hành",
                "clause": None,
                "point": None,
                "text_raw": pre_dieu,
                "text_norm": norm_content,
                "parent_chunk_id": None,
                "token_count": len(norm_content.split()),
                "is_empty": False,
            })
            # Derived micro chunk
            micro_id = f"{doc_id}_micro_{macro_idx-1:03d}_01"
            chunks.append({
                "chunk_id": micro_id,
                "doc_id": doc_id,
                "granularity": "micro",
                "chapter": None,
                "section": None,
                "article": "Căn cứ ban hành",
                "clause": None,
                "point": None,
                "text_raw": pre_dieu,
                "text_norm": norm_content,
                "parent_chunk_id": macro_id,
                "token_count": len(norm_content.split()),
                "is_empty": False,
            })

        for d_i, d_match in enumerate(dieu_matches):
            start_p = d_match.start()
            end_p = dieu_matches[d_i + 1].start() if d_i + 1 < len(dieu_matches) else len(passage)
            dieu_full_text = passage[start_p:end_p].strip()
            lines = dieu_full_text.split("\n")
            dieu_header = lines[0].strip()
            dieu_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            # Check if article exceeds macro max tokens (800)
            tokens = dieu_full_text.split()
            header_overhead = len(f"{header_prefix}\n[ĐIỀU KHOẢN]: {dieu_header} (Phần 99)\n[NỘI DUNG]:\n".split())
            effective_macro_max = max(config.macro_min_tokens, config.macro_max_tokens - header_overhead)

            if len(tokens) > effective_macro_max:
                # Split oversized article with token windows
                sub_windows = sliding_token_windows(tokens, effective_macro_max, config.overlap_tokens)
                for w_i, win_toks in enumerate(sub_windows, 1):
                    win_text = " ".join(win_toks)
                    macro_id = f"{doc_id}_macro_{macro_idx:03d}"
                    macro_idx += 1
                    content = f"{header_prefix}\n[ĐIỀU KHOẢN]: {dieu_header} (Phần {w_i})\n[NỘI DUNG]:\n{win_text}"
                    norm_content = normalize_text(content)
                    chunks.append({
                        "chunk_id": macro_id,
                        "doc_id": doc_id,
                        "granularity": "macro",
                        "chapter": None,
                        "section": None,
                        "article": f"{dieu_header} (Phần {w_i})",
                        "clause": None,
                        "point": None,
                        "text_raw": win_text,
                        "text_norm": norm_content,
                        "parent_chunk_id": None,
                        "token_count": len(norm_content.split()),
                        "is_empty": False,
                    })

                    # Micro windows for this macro chunk
                    micro_overhead = len(f"{header_prefix}\n[ĐIỀU KHOẢN]: {dieu_header}\n[ĐOẠN 99]:\n".split())
                    effective_micro_max = max(config.micro_min_tokens, config.micro_max_tokens - micro_overhead)
                    micro_toks_list = sliding_token_windows(win_toks, effective_micro_max, 50)
                    for m_i, m_toks in enumerate(micro_toks_list, 1):
                        m_text = " ".join(m_toks)
                        micro_id = f"{doc_id}_micro_{macro_idx-1:03d}_{m_i:02d}"
                        m_content = f"{header_prefix}\n[ĐIỀU KHOẢN]: {dieu_header}\n[ĐOẠN {m_i}]:\n{m_text}"
                        norm_m = normalize_text(m_content)
                        chunks.append({
                            "chunk_id": micro_id,
                            "doc_id": doc_id,
                            "granularity": "micro",
                            "chapter": None,
                            "section": None,
                            "article": dieu_header,
                            "clause": f"Đoạn {m_i}",
                            "point": None,
                            "text_raw": m_text,
                            "text_norm": norm_m,
                            "parent_chunk_id": macro_id,
                            "token_count": len(norm_m.split()),
                            "is_empty": False,
                        })

            else:
                macro_id = f"{doc_id}_macro_{macro_idx:03d}"
                macro_idx += 1
                content = f"{header_prefix}\n[ĐIỀU KHOẢN]: {dieu_header}\n[NỘI DUNG]:\n{dieu_body or dieu_header}"
                norm_content = normalize_text(content)
                chunks.append({
                    "chunk_id": macro_id,
                    "doc_id": doc_id,
                    "granularity": "macro",
                    "chapter": None,
                    "section": None,
                    "article": dieu_header,
                    "clause": None,
                    "point": None,
                    "text_raw": dieu_full_text,
                    "text_norm": norm_content,
                    "parent_chunk_id": None,
                    "token_count": len(norm_content.split()),
                    "is_empty": False,
                })

                # Micro chunks by clause if available
                legal_units = parse_legal_units(dieu_full_text)
                if legal_units and len(legal_units) > 1:
                    for u_i, unit in enumerate(legal_units, 1):
                        micro_id = f"{doc_id}_micro_{macro_idx-1:03d}_{u_i:02d}"
                        u_clause = unit.clause or f"Khoản {u_i}"
                        u_content = f"{header_prefix}\n[ĐIỀU KHOẢN]: {unit.article}\n[{u_clause}]:\n{unit.text}"
                        norm_u = normalize_text(u_content)
                        chunks.append({
                            "chunk_id": micro_id,
                            "doc_id": doc_id,
                            "granularity": "micro",
                            "chapter": unit.chapter,
                            "section": unit.section,
                            "article": unit.article,
                            "clause": unit.clause,
                            "point": unit.point,
                            "text_raw": unit.text,
                            "text_norm": norm_u,
                            "parent_chunk_id": macro_id,
                            "token_count": len(norm_u.split()),
                            "is_empty": False,
                        })
                else:
                    # Single micro chunk
                    micro_id = f"{doc_id}_micro_{macro_idx-1:03d}_01"
                    chunks.append({
                        "chunk_id": micro_id,
                        "doc_id": doc_id,
                        "granularity": "micro",
                        "chapter": None,
                        "section": None,
                        "article": dieu_header,
                        "clause": None,
                        "point": None,
                        "text_raw": dieu_full_text,
                        "text_norm": norm_content,
                        "parent_chunk_id": macro_id,
                        "token_count": len(norm_content.split()),
                        "is_empty": False,
                    })
    else:
        # Fallback sliding window for unstructured document
        tokens = passage.split()
        fallback_overhead = len(f"{header_prefix}\n[ĐOẠN 99]:\n".split())
        effective_fallback_max = max(config.fallback_min_tokens, config.fallback_max_tokens - fallback_overhead)
        windows = sliding_token_windows(tokens, effective_fallback_max, config.overlap_tokens)
        for w_idx, win_toks in enumerate(windows, 1):
            win_text = " ".join(win_toks)
            macro_id = f"{doc_id}_macro_{w_idx:03d}"
            w_content = f"{header_prefix}\n[ĐOẠN {w_idx}]:\n{win_text}"
            norm_w = normalize_text(w_content)
            chunks.append({
                "chunk_id": macro_id,
                "doc_id": doc_id,
                "granularity": "macro",
                "chapter": None,
                "section": None,
                "article": f"Đoạn {w_idx}",
                "clause": None,
                "point": None,
                "text_raw": win_text,
                "text_norm": norm_w,
                "parent_chunk_id": None,
                "token_count": len(norm_w.split()),
                "is_empty": False,
            })

            # Derive micro chunks for this fallback window
            micro_overhead = len(f"{header_prefix}\n[ĐOẠN 99.99]:\n".split())
            effective_micro_max = max(config.micro_min_tokens, config.micro_max_tokens - micro_overhead)
            micro_windows = sliding_token_windows(win_toks, effective_micro_max, 50)
            for m_i, m_toks in enumerate(micro_windows, 1):
                m_text = " ".join(m_toks)
                micro_id = f"{doc_id}_micro_{w_idx:03d}_{m_i:02d}"
                m_content = f"{header_prefix}\n[ĐOẠN {w_idx}.{m_i}]:\n{m_text}"
                norm_m = normalize_text(m_content)
                chunks.append({
                    "chunk_id": micro_id,
                    "doc_id": doc_id,
                    "granularity": "micro",
                    "chapter": None,
                    "section": None,
                    "article": f"Đoạn {w_idx}",
                    "clause": f"Đoạn nhỏ {m_i}",
                    "point": None,
                    "text_raw": m_text,
                    "text_norm": norm_m,
                    "parent_chunk_id": macro_id,
                    "token_count": len(norm_m.split()),
                    "is_empty": False,
                })


    return chunks
