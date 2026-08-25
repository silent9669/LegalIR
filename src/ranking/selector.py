class TopKSelector:
    def __init__(self, max_k: int = 5):
        self.max_k = max_k

    def select(self, ranked_candidates: list) -> list:
        """
        Takes ranked candidate records and returns a list of unique doc_id strings (up to max_k).
        """
        seen = set()
        selected = []

        for c in ranked_candidates:
            did = str(c["doc_id"]) if isinstance(c, dict) else str(c)
            if did not in seen:
                seen.add(did)
                selected.append(did)
                if len(selected) >= self.max_k:
                    break

        return selected
