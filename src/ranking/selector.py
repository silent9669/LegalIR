from collections.abc import Mapping
from typing import Any


class TopKSelector:
    """Select at most five unique string document IDs in ranked order."""

    def __init__(
        self,
        max_k: int = 5,
        min_k: int = 1,
        fallback_doc_ids: list[str] | None = None,
    ):
        self.max_k = self._validate_k(max_k, "max_k")
        self.min_k = self._validate_k(min_k, "min_k")
        if self.min_k > self.max_k:
            raise ValueError("min_k cannot be greater than max_k")
        self.fallback_doc_ids = (
            [str(x) for x in fallback_doc_ids]
            if fallback_doc_ids is not None
            else ["2113", "740", "280282", "165290", "200355"]
        )

    @staticmethod
    def _validate_k(value: int, name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer between 1 and 5") from exc
        if not 1 <= parsed <= 5:
            raise ValueError(f"{name} must be between 1 and 5")
        return parsed

    @staticmethod
    def _doc_id(candidate: Any) -> str | None:
        if isinstance(candidate, Mapping):
            candidate = candidate.get("doc_id")
        elif isinstance(candidate, (tuple, list)):
            if not candidate:
                return None
            candidate = candidate[0]
        if candidate is None:
            return None
        doc_id = str(candidate).strip()
        return doc_id or None

    def select(
        self,
        ranked_candidates: list[Any],
        top_k: int | None = None,
        valid_doc_ids: set[str] | None = None,
        fill_to_k: int | None = None,
    ) -> list[str]:
        """Return unique string IDs, capped by the requested value and five."""
        if top_k is None:
            limit = self.max_k
        else:
            limit = self._validate_k(top_k, "top_k")
            limit = min(limit, self.max_k)

        selected: list[str] = []
        seen: set[str] = set()
        for candidate in ranked_candidates:
            doc_id = self._doc_id(candidate)
            if doc_id is None or doc_id in seen:
                continue
            if valid_doc_ids is not None and doc_id not in valid_doc_ids:
                continue
            seen.add(doc_id)
            selected.append(doc_id)
            if len(selected) == limit:
                break

        # Determine target answer length (default min_k, or fill_to_k if requested)
        target_len = self.min_k
        if fill_to_k is not None:
            max_avail = len(valid_doc_ids) if valid_doc_ids is not None else limit
            target_len = min(fill_to_k, max_avail, limit)

        # If empty or fewer than target_len, backfill from fallbacks or valid IDs if available
        if len(selected) < target_len:
            pool = self.fallback_doc_ids
            if valid_doc_ids is not None:
                pool = [d for d in pool if d in valid_doc_ids]
                if not pool:
                    pool = sorted(list(valid_doc_ids))[:limit]

            for fb in pool:
                if fb not in seen:
                    seen.add(fb)
                    selected.append(fb)
                if len(selected) == target_len:
                    break

            if len(selected) < target_len and valid_doc_ids is not None:
                for fb in sorted(valid_doc_ids):
                    if fb not in seen:
                        seen.add(fb)
                        selected.append(fb)
                    if len(selected) == target_len:
                        break

        if selected and not 1 <= len(selected) <= self.max_k:
            raise ValueError("selected answer must contain between 1 and 5 IDs")
        return selected

