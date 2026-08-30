from collections.abc import Mapping
from typing import Any


class TopKSelector:
    """Select at most five unique string document IDs in ranked order."""

    def __init__(self, max_k: int = 5, min_k: int = 1):
        self.max_k = self._validate_k(max_k, "max_k")
        self.min_k = self._validate_k(min_k, "min_k")
        if self.min_k > self.max_k:
            raise ValueError("min_k cannot be greater than max_k")

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

    def select(self, ranked_candidates: list[Any], top_k: int | None = None) -> list[str]:
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
            seen.add(doc_id)
            selected.append(doc_id)
            if len(selected) == limit:
                break

        # An empty candidate pool cannot produce a valid submission answer;
        # callers can use their configured fallback document IDs. For every
        # non-empty result, these invariants hold by construction.
        if selected and not self.min_k <= len(selected) <= self.max_k:
            raise ValueError("selected answer must contain between 1 and 5 IDs")
        return selected
