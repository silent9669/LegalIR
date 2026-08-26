from dataclasses import dataclass
import re


@dataclass
class LegalUnit:
    chapter: str | None
    section: str | None
    article: str
    clause: str | None
    point: str | None
    text: str


CHƯƠNG_PATTERN = re.compile(r'(?:^|\n)\s*(Chương\s+[IVXLCDM\d]+[^\n]*)', re.IGNORECASE)
MỤC_PATTERN = re.compile(r'(?:^|\n)\s*(Mục\s+\d+[^\n]*)', re.IGNORECASE)
ĐIỀU_PATTERN = re.compile(r'(?:^|\n)\s*(Điều\s+\d+[\.\:\s][^\n]*)', re.IGNORECASE)
KHOẢN_PATTERN = re.compile(r'(?:^|\n)\s*(\d+\.\s+[^\n]*)')
ĐIỂM_PATTERN = re.compile(r'(?:^|\n)\s*([a-zđ]\)\s+[^\n]*)', re.IGNORECASE)


def parse_legal_units(text: str) -> list[LegalUnit]:
    if not text or not text.strip():
        return []

    dieu_matches = list(ĐIỀU_PATTERN.finditer(text))
    if not dieu_matches:
        return []

    units: list[LegalUnit] = []
    curr_chapter = None
    curr_section = None

    for idx, d_match in enumerate(dieu_matches):
        start_pos = d_match.start()
        end_pos = dieu_matches[idx + 1].start() if idx + 1 < len(dieu_matches) else len(text)

        # Check for chapter / section before this article
        pre_text = text[:start_pos] if idx == 0 else text[dieu_matches[idx - 1].end():start_pos]
        ch_match = list(CHƯƠNG_PATTERN.finditer(pre_text))
        if ch_match:
            curr_chapter = ch_match[-1].group(1).strip()
            curr_section = None  # Reset section on new chapter
        sec_match = list(MỤC_PATTERN.finditer(pre_text))
        if sec_match:
            curr_section = sec_match[-1].group(1).strip()

        article_full_text = text[start_pos:end_pos].strip()
        lines = article_full_text.split("\n")
        article_header = lines[0].strip()
        article_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        khoan_matches = list(KHOẢN_PATTERN.finditer(article_body))
        if khoan_matches:
            for k_idx, k_match in enumerate(khoan_matches):
                k_start = k_match.start()
                k_end = khoan_matches[k_idx + 1].start() if k_idx + 1 < len(khoan_matches) else len(article_body)
                clause_text = article_body[k_start:k_end].strip()
                k_lines = clause_text.split("\n")
                k_header = k_lines[0].strip()
                k_num_match = re.match(r'^(\d+)\.', k_header)
                clause_label = f"Khoản {k_num_match.group(1)}" if k_num_match else f"Khoản {k_idx + 1}"

                diem_matches = list(ĐIỂM_PATTERN.finditer(clause_text))
                if diem_matches:
                    for d_idx, dm in enumerate(diem_matches):
                        dm_start = dm.start()
                        dm_end = diem_matches[d_idx + 1].start() if d_idx + 1 < len(diem_matches) else len(clause_text)
                        point_text = clause_text[dm_start:dm_end].strip()
                        p_match = re.match(r'^([a-zđ])\)', point_text, re.IGNORECASE)
                        point_label = f"Điểm {p_match.group(1)}" if p_match else f"Điểm {d_idx + 1}"

                        units.append(LegalUnit(
                            chapter=curr_chapter,
                            section=curr_section,
                            article=article_header,
                            clause=clause_label,
                            point=point_label,
                            text=point_text,
                        ))
                else:
                    units.append(LegalUnit(
                        chapter=curr_chapter,
                        section=curr_section,
                        article=article_header,
                        clause=clause_label,
                        point=None,
                        text=clause_text,
                    ))
        else:
            units.append(LegalUnit(
                chapter=curr_chapter,
                section=curr_section,
                article=article_header,
                clause=None,
                point=None,
                text=article_body or article_header,
            ))

    return units
