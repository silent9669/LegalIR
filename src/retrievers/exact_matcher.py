import json
import re
from data_utils import LEGAL_NUMBER_PATTERN, LEGAL_YEAR_PATTERN, normalize_text

class ExactMatcher:
    def __init__(self, meta_file: str = "data/doc_metadata.json"):
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Build index maps:
        # 1. legal_number -> set of doc_ids
        self.num_to_docs = {}
        # 2. normalized_title -> set of doc_ids
        self.title_to_docs = {}
        # 3. doc_id -> metadata
        for doc_id, meta in self.metadata.items():
            num = meta.get("legal_number")
            if num:
                num_norm = self._norm_num(num)
                self.num_to_docs.setdefault(num_norm, set()).add(doc_id)

            title = meta.get("title", "")
            if title:
                t_norm = normalize_text(title).lower()
                self.title_to_docs.setdefault(t_norm, set()).add(doc_id)

    def _norm_num(self, num_str: str) -> str:
        if not num_str: return ""
        return re.sub(r'[\s\.\-]+', '', num_str).lower()

    def match(self, query: str) -> dict:
        """Find exact document matches for numbers, law names, and patterns in query."""
        scores = {}
        q_norm = normalize_text(query)
        q_lower = q_norm.lower()

        # 1. Check legal numbers in query
        q_nums = LEGAL_NUMBER_PATTERN.findall(query)
        for qn in q_nums:
            qn_norm = self._norm_num(qn)
            if qn_norm in self.num_to_docs:
                for did in self.num_to_docs[qn_norm]:
                    scores[did] = scores.get(did, 0.0) + 15.0
            else:
                # Substring matching on legal numbers
                for stored_num, dids in self.num_to_docs.items():
                    if qn_norm in stored_num or stored_num in qn_norm:
                        for did in dids:
                            scores[did] = scores.get(did, 0.0) + 8.0

        # 2. Check title exact phrases
        for stored_title, dids in self.title_to_docs.items():
            if len(stored_title) > 15 and stored_title in q_lower:
                for did in dids:
                    scores[did] = scores.get(did, 0.0) + 10.0

        return scores
