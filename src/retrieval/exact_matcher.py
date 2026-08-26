from collections import defaultdict
from typing import Any
import re
import unicodedata
import pandas as pd

LEGAL_NUM_REGEX = re.compile(
    r'(\d+[\/\-][0-9]+[\/\-][A-ZĐa-z0-9\-\_]+|\d+[\/\-][A-ZĐa-z0-9\-\_]+)',
    re.IGNORECASE
)
YEAR_REGEX = re.compile(r'\b(19[89]\d|20[012]\d)\b')
DOC_TYPE_REGEX = re.compile(
    r'\b(Luật|Bộ luật|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Chỉ thị|Nghị quyết|Tiêu chuẩn|Công văn|Pháp lệnh)\b',
    re.IGNORECASE
)


def normalize_text(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip().lower()


class ExactMatcher:
    def __init__(self, documents: list[dict[str, Any]]):
        self.doc_by_num = defaultdict(list)
        self.doc_by_title = defaultdict(list)
        self.doc_metadata = {}

        for d in documents:
            doc_id = str(d["doc_id"])
            self.doc_metadata[doc_id] = d

            # Index legal number
            lnum = d.get("legal_number")
            if lnum is not None and not (isinstance(lnum, float) and pd.isna(lnum)):
                norm_num = normalize_text(lnum).replace('-', '/')
                self.doc_by_num[norm_num].append(doc_id)
                # stripped version without prefix
                clean_num = re.sub(r'^[^\d]*', '', norm_num)
                if clean_num:
                    self.doc_by_num[clean_num].append(doc_id)

            # Index title
            title = d.get("title")
            if title is not None and not (isinstance(title, float) and pd.isna(title)):
                norm_title = normalize_text(title)
                self.doc_by_title[norm_title].append(doc_id)
                # Also index title stripped of year
                title_no_year = re.sub(r'\b(19[89]\d|20[012]\d)\b', '', norm_title).strip()
                if len(title_no_year) > 6 and title_no_year != norm_title:
                    self.doc_by_title[title_no_year].append(doc_id)

    def match(self, query: str) -> dict[str, dict[str, Any]]:
        """Returns {doc_id: {score, exact_score, exact_legal_number, exact_title, exact_year, exact_doc_type}}."""
        if not query:
            return {}

        norm_query = normalize_text(query)
        matches: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "score": 0.0,
            "exact_score": 0.0,
            "exact_legal_number": False,
            "exact_title": False,
            "exact_year": False,
            "exact_doc_type": False,
        })

        # 1. Match legal numbers
        found_nums = LEGAL_NUM_REGEX.findall(query)
        for num in found_nums:
            clean_num = normalize_text(num).replace('-', '/')
            matched_dids = []
            if clean_num in self.doc_by_num:
                matched_dids = self.doc_by_num[clean_num]
                score_val = 1.0
            else:
                sub_num = re.sub(r'^[^\d]*', '', clean_num)
                if sub_num and sub_num in self.doc_by_num:
                    matched_dids = self.doc_by_num[sub_num]
                    score_val = 0.9

            for did in matched_dids:
                matches[did]["exact_legal_number"] = True
                matches[did]["score"] = max(matches[did]["score"], score_val)
                matches[did]["exact_score"] = matches[did]["score"]

        # 2. Match exact law titles
        for title, dids in self.doc_by_title.items():
            if len(title) > 6 and title in norm_query:
                for did in dids:
                    matches[did]["exact_title"] = True
                    matches[did]["score"] = max(matches[did]["score"], 0.85)
                    matches[did]["exact_score"] = matches[did]["score"]

        # 3. Match law title + year co-occurrence
        years = YEAR_REGEX.findall(query)
        if years:
            target_year = str(years[0])
            for did, meta in self.doc_metadata.items():
                if str(meta.get("year")) == target_year:
                    doc_title = normalize_text(meta.get("title", ""))
                    doc_title_no_year = re.sub(r'\b(19[89]\d|20[012]\d)\b', '', doc_title).strip()
                    if doc_title_no_year and len(doc_title_no_year) > 6 and doc_title_no_year in norm_query:
                        matches[did]["exact_title"] = True
                        matches[did]["exact_year"] = True
                        matches[did]["score"] = max(matches[did]["score"], 0.95)
                        matches[did]["exact_score"] = matches[did]["score"]

        # 4. Match document type if matched on title or number
        doc_types = DOC_TYPE_REGEX.findall(query)
        if doc_types:
            q_types = {normalize_text(t) for t in doc_types}
            for did in list(matches.keys()):
                d_type = normalize_text(self.doc_metadata.get(did, {}).get("doc_type", ""))
                if d_type in q_types:
                    matches[did]["exact_doc_type"] = True

        return dict(matches)
