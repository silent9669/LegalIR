class TopKSelector:
    def __init__(self, max_k: int = 5, min_k: int = 1, fallback_doc_ids: list[str] = None):
        self.max_k = max_k
        self.min_k = min_k
        self.fallback_doc_ids = [str(x) for x in (fallback_doc_ids or ["2113", "740", "280282", "165290", "200355"])]

    def select(self, ranked_items: list[dict], valid_doc_ids: set[str] = None) -> list[str]:
        selected = []
        seen = set()

        for item in ranked_items:
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id or doc_id in seen:
                continue
            if valid_doc_ids is not None and doc_id not in valid_doc_ids:
                continue

            seen.add(doc_id)
            selected.append(doc_id)
            if len(selected) >= self.max_k:
                break

        # If empty or fewer than min_k, backfill from fallbacks or valid IDs
        if len(selected) < self.min_k:
            pool = self.fallback_doc_ids
            if valid_doc_ids is not None:
                pool = [d for d in pool if d in valid_doc_ids]
                if not pool:
                    pool = sorted(list(valid_doc_ids))[:self.max_k]

            for fb in pool:
                if fb not in seen:
                    seen.add(fb)
                    selected.append(fb)
                if len(selected) >= self.max_k:
                    break

        return selected[:self.max_k]
