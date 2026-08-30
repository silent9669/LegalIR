import re
from src.common.normalize import clean_legal_text

CHAPTER_PATTERN = re.compile(r'^(Chương\s+[IVXLCDM\d]+[.:\s-]*[^\n]*)', re.IGNORECASE | re.MULTILINE)
SECTION_PATTERN = re.compile(r'^(Mục\s+\d+[.:\s-]*[^\n]*)', re.IGNORECASE | re.MULTILINE)
ARTICLE_PATTERN = re.compile(r'^(Điều\s+\d+[a-zA-Z]?[.:\s-]*[^\n]*)', re.IGNORECASE | re.MULTILINE)
CLAUSE_PATTERN = re.compile(r'^(\d+)\.\s+([^\n]+)', re.MULTILINE)
POINT_PATTERN = re.compile(r'^([a-zđ])\)\s+([^\n]+)', re.IGNORECASE | re.MULTILINE)

def parse_legal_structure(passage: str, doc_id: str = "") -> list[dict]:
    if not passage:
        return []

    passage_clean = clean_legal_text(passage)
    lines = passage.split('\n')

    chunks = []
    current_chapter = ""
    current_section = ""
    current_article = ""
    current_clause = ""
    current_body = []

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue

        chap_m = CHAPTER_PATTERN.match(line_s)
        if chap_m:
            current_chapter = chap_m.group(1)
            continue

        sec_m = SECTION_PATTERN.match(line_s)
        if sec_m:
            current_section = sec_m.group(1)
            continue

        art_m = ARTICLE_PATTERN.match(line_s)
        if art_m:
            if current_article and current_body:
                text_raw = "\n".join(current_body).strip()
                chunks.append({
                    "doc_id": str(doc_id),
                    "chapter": current_chapter,
                    "section": current_section,
                    "article": current_article,
                    "clause": current_clause,
                    "text_raw": text_raw,
                    "text_norm": clean_legal_text(text_raw)
                })
            current_article = art_m.group(1)
            current_clause = ""
            current_body = [line_s]
            continue

        cl_m = CLAUSE_PATTERN.match(line_s)
        if cl_m and current_article:
            current_clause = f"Khoản {cl_m.group(1)}"

        current_body.append(line_s)

    if current_article and current_body:
        text_raw = "\n".join(current_body).strip()
        chunks.append({
            "doc_id": str(doc_id),
            "chapter": current_chapter,
            "section": current_section,
            "article": current_article,
            "clause": current_clause,
            "text_raw": text_raw,
            "text_norm": clean_legal_text(text_raw)
        })
    elif not chunks and passage_clean:
        chunks.append({
            "doc_id": str(doc_id),
            "chapter": "",
            "section": "",
            "article": "Toàn văn",
            "clause": "",
            "text_raw": passage,
            "text_norm": passage_clean
        })

    return chunks
