import re
import urllib.parse
import unicodedata

# Regex patterns for legal numbers and types
LEGAL_NUMBER_PATTERN = re.compile(
    r'(\d+[\w\d\-\./]*(?:/\d{2,4})?/(?:NĐ-CP|TT-BCA|TT-BTC|TT-BYT|TT-BGDĐT|TT-BNNPTNT|TT-BGTVT|TT-BTP|TT-BKHĐT|TT-BTTTT|TT-BQP|TT-BLĐTBXH|TT-NHNN|QĐ-TTg|QĐ-BYT|QĐ-BTC|QĐ-BCA|QĐ-UBND|QH\d+|NQ-CP|NQ-HĐND|TTLT|CT-TTg|VBHN-[\w\d]+|[\w\d\-]+))',
    re.IGNORECASE
)

LEGAL_YEAR_PATTERN = re.compile(r'\b(19\d\d|20[0-3]\d)\b')

DOC_TYPE_MAP = {
    'luat': 'Luật',
    'bo-luat': 'Bộ luật',
    'nghi-dinh': 'Nghị định',
    'thong-tu': 'Thông tư',
    'quyet-dinh': 'Quyết định',
    'nghi-quyet': 'Nghị quyết',
    'chi-thi': 'Chỉ thị',
    'cong-van': 'Công văn',
    'van-ban-hop-nhat': 'Văn bản hợp nhất',
    'phap-lenh': 'Pháp lệnh',
    'tcvn': 'TCVN',
    'qcvn': 'QCVN',
}

def normalize_text(text: str) -> str:
    """Normalize Vietnamese unicode text to NFC and clean extra whitespaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Replace carriage returns and normalize multiple spaces/newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def slug_to_title(slug: str) -> str:
    """Convert URL slug to human-readable title."""
    if not slug:
        return ""
    # Unquote URL encodings
    slug = urllib.parse.unquote(slug)
    # Remove file extension like .aspx
    slug = re.sub(r'\.aspx$', '', slug, flags=re.IGNORECASE)
    # Remove trailing document ID numbers if format is ...-123456
    slug = re.sub(r'-[0-9]{5,}$', '', slug)
    # Replace hyphens with spaces
    title = slug.replace('-', ' ')
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def extract_metadata_from_doc(doc_id: str, name: str, link: str, passage: str) -> dict:
    """Extract and normalize metadata for a legal document."""
    # 1. Recover title
    title = name
    if not title or title.strip() in ("", "None", "null"):
        if link:
            # Extract last segment of link before .aspx
            parsed_link = urllib.parse.urlparse(link)
            path_parts = [p for p in parsed_link.path.split('/') if p]
            if path_parts:
                last_part = path_parts[-1]
                title = slug_to_title(last_part)
        if not title or title.strip() in ("", "None", "null"):
            title = f"Văn bản pháp luật {doc_id}"

    title = normalize_text(title)

    # 2. Extract legal number from title, link, or passage
    search_corpus = f"{name or ''} {link or ''} {passage[:1000] if passage else ''}"
    legal_numbers = LEGAL_NUMBER_PATTERN.findall(search_corpus)
    legal_number = legal_numbers[0].strip() if legal_numbers else None

    # 3. Extract year
    years = LEGAL_YEAR_PATTERN.findall(title or "")
    if not years and legal_number:
        years = LEGAL_YEAR_PATTERN.findall(legal_number)
    if not years and passage:
        years = LEGAL_YEAR_PATTERN.findall(passage[:500])
    year = years[0] if years else None

    # 4. Extract doc type
    doc_type = "Văn bản"
    title_lower = title.lower()
    for k, v in DOC_TYPE_MAP.items():
        if k in title_lower or v.lower() in title_lower:
            doc_type = v
            break

    return {
        "doc_id": str(doc_id),
        "title": title,
        "raw_name": name,
        "link": link,
        "legal_number": legal_number,
        "year": year,
        "doc_type": doc_type
    }

def clean_dieu_header(header: str) -> str:
    """Normalize and clean Điều header."""
    header = normalize_text(header)
    # Remove leading dots, hyphens, colons
    header = re.sub(r'^[\.\-:\s]+', '', header)
    return header.strip()
