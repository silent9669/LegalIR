from collections import defaultdict
from typing import Any, Mapping


def get_difficulty_band(rank: int) -> str:
    """Classify retrieval rank into difficulty bands."""
    if rank <= 5:
        return "very_hard"
    elif rank <= 20:
        return "hard"
    else:
        return "medium"


class HardNegativeMiner:
    """
    Multi-band, multi-branch hard negative miner for LegalIR cross-encoder training.
    Excludes gold document IDs and duplicate group members to avoid false negatives.
    Samples across branches (exact, bm25, bm25_pyvi, dense, memory, hybrid) and
    difficulty bands (ranks 1-5, 6-20, 21-80).
    """

    def __init__(self, false_negative_blacklist: dict[str, set[str]] | None = None):
        """
        false_negative_blacklist: optional mapping {qid: set_of_doc_ids_to_avoid}
        """
        self.blacklist: dict[str, set[str]] = (
            defaultdict(set, false_negative_blacklist) if false_negative_blacklist else defaultdict(set)
        )
        self.excluded_golds_count = 0
        self.excluded_duplicates_count = 0
        self.mined_counts_by_source: dict[str, int] = defaultdict(int)
        self.mined_counts_by_band: dict[str, int] = defaultdict(int)

    def reset_stats(self) -> None:
        self.excluded_golds_count = 0
        self.excluded_duplicates_count = 0
        self.mined_counts_by_source = defaultdict(int)
        self.mined_counts_by_band = defaultdict(int)

    def get_stats(self) -> dict[str, Any]:
        return {
            "excluded_golds_count": self.excluded_golds_count,
            "excluded_duplicates_count": self.excluded_duplicates_count,
            "mined_counts_by_source": dict(self.mined_counts_by_source),
            "mined_counts_by_band": dict(self.mined_counts_by_band),
        }

    def _extract_doc_info(self, cand: Any) -> tuple[str, str, int, float, str]:
        """Extract (doc_id, source, rank, score, band) from candidate representation."""
        if isinstance(cand, Mapping):
            doc_id = str(cand.get("doc_id", ""))
            source = str(cand.get("negative_source") or cand.get("source") or "hybrid")
            rank = int(cand.get("retrieval_rank") or cand.get("rank") or 0)
            score = float(cand.get("retrieval_score") or cand.get("score") or cand.get("rrf_score") or 0.0)
            band = str(cand.get("difficulty_band") or cand.get("band") or get_difficulty_band(rank if rank > 0 else 1))
            return doc_id, source, rank, score, band
        elif isinstance(cand, (tuple, list)):
            doc_id = str(cand[0])
            score = float(cand[1]) if len(cand) > 1 else 0.0
            return doc_id, "candidate", 0, score, "medium"
        else:
            return str(cand), "candidate", 0, 0.0, "medium"

    def mine_negatives(
        self,
        *args: Any,
        max_negatives: int = 15,
        return_records: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Flexible signature:
        mine_negatives(query_id, candidates, gold_doc_ids, max_negatives=15)
        OR
        mine_negatives(candidates, gold_doc_ids, query_id=None, max_negatives=15)
        """
        if len(args) >= 3:
            if isinstance(args[0], (list, tuple)):
                candidates, gold_doc_ids, query_id = args[0], args[1], args[2]
            else:
                query_id, candidates, gold_doc_ids = args[0], args[1], args[2]
        elif len(args) == 2:
            candidates, gold_doc_ids = args[0], args[1]
            query_id = kwargs.get("query_id")
        else:
            candidates = kwargs.get("candidates", [])
            gold_doc_ids = kwargs.get("gold_doc_ids", [])
            query_id = kwargs.get("query_id")

        gold_set = set(str(x) for x in gold_doc_ids) if gold_doc_ids else set()
        qid_blacklist = set(str(x) for x in self.blacklist.get(str(query_id), set())) if query_id else set()

        mined = []
        seen_dids = set()

        for idx, cand in enumerate(candidates, start=1):
            did, source, rank, score, band = self._extract_doc_info(cand)
            if not did:
                continue

            if did in gold_set:
                self.excluded_golds_count += 1
                continue

            if did in qid_blacklist:
                self.excluded_duplicates_count += 1
                continue

            if did in seen_dids:
                continue

            effective_rank = rank if rank > 0 else idx
            effective_band = band if band in ("very_hard", "hard", "medium") else get_difficulty_band(effective_rank)
            seen_dids.add(did)
            self.mined_counts_by_source[source] += 1
            self.mined_counts_by_band[effective_band] += 1

            if return_records:
                mined.append({
                    "doc_id": did,
                    "negative_source": source,
                    "retrieval_rank": effective_rank,
                    "retrieval_score": score,
                    "difficulty_band": effective_band,
                })
            else:
                mined.append(did)

            if len(mined) >= max_negatives:
                break

        return mined

    def mine_multi_band_negatives(
        self,
        query_id: str,
        candidates_by_source: dict[str, list[Any]],
        gold_doc_ids: list[str] | set[str],
        per_source_limits: dict[str, int] | None = None,
        max_total: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Mine hard negatives across multiple branches (exact, bm25, bm25_pyvi, dense, memory, hybrid, medium_neg)
        and difficulty bands (ranks 1-5, ranks 6-20, ranks 21-80).
        Returns structured records with doc_id, negative_source, retrieval_rank, retrieval_score, difficulty_band.
        """
        default_limits = {
            "exact": 2,
            "bm25": 4,
            "bm25_pyvi": 3,
            "dense": 4,
            "memory": 2,
            "hybrid": 4,
            "medium_neg": 3,
        }
        limits = per_source_limits or default_limits

        gold_set = set(str(x) for x in gold_doc_ids)
        qid_blacklist = set(str(x) for x in self.blacklist.get(str(query_id), set()))

        mined: list[dict[str, Any]] = []
        seen_dids: set[str] = set()

        # Iterate over sources in priority order
        source_order = ["exact", "bm25", "bm25_pyvi", "dense", "memory", "hybrid", "medium_neg"]
        all_sources = [s for s in source_order if s in candidates_by_source] + [
            s for s in candidates_by_source if s not in source_order
        ]

        for source in all_sources:
            cands = candidates_by_source.get(source, [])
            source_limit = limits.get(source, 3)
            source_count = 0

            for idx, cand in enumerate(cands, start=1):
                did, cand_source, rank, score, band = self._extract_doc_info(cand)
                if not did or did in seen_dids:
                    continue

                if did in gold_set:
                    self.excluded_golds_count += 1
                    continue

                if did in qid_blacklist:
                    self.excluded_duplicates_count += 1
                    continue

                effective_rank = rank if rank > 0 else idx
                effective_source = source if source != "medium_neg" else (cand_source if cand_source != "hybrid" else "medium_neg")
                effective_band = band if band in ("very_hard", "hard", "medium") else get_difficulty_band(effective_rank)

                seen_dids.add(did)
                self.mined_counts_by_source[effective_source] += 1
                self.mined_counts_by_band[effective_band] += 1
                source_count += 1

                mined.append({
                    "doc_id": did,
                    "negative_source": effective_source,
                    "retrieval_rank": effective_rank,
                    "retrieval_score": score,
                    "difficulty_band": effective_band,
                })

                if source_count >= source_limit or len(mined) >= max_total:
                    break

            if len(mined) >= max_total:
                break

        return mined

    def mine_by_difficulty_bands(
        self,
        query_id: str,
        candidates_by_source: dict[str, list[Any]],
        gold_doc_ids: list[str] | set[str],
        band_limits: dict[str, int] | None = None,
        max_total: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Sample hard negatives across difficulty bands:
        - very_hard: ranks 1-5
        - hard: ranks 6-20
        - medium: ranks 21-80
        """
        default_band_limits = {
            "very_hard": 5,
            "hard": 5,
            "medium": 5,
        }
        band_quotas = band_limits or default_band_limits

        gold_set = set(str(x) for x in gold_doc_ids)
        qid_blacklist = set(str(x) for x in self.blacklist.get(str(query_id), set()))

        mined: list[dict[str, Any]] = []
        seen_dids: set[str] = set()
        band_counts: dict[str, int] = defaultdict(int)

        # Collect and categorize all valid candidates across all sources
        candidates_by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source, cands in candidates_by_source.items():
            for idx, cand in enumerate(cands, start=1):
                did, cand_source, rank, score, band = self._extract_doc_info(cand)
                if not did or did in gold_set or did in qid_blacklist:
                    continue
                effective_rank = rank if rank > 0 else idx
                effective_band = get_difficulty_band(effective_rank)
                effective_source = source if source != "medium_neg" else "medium_neg"
                candidates_by_band[effective_band].append({
                    "doc_id": did,
                    "negative_source": effective_source,
                    "retrieval_rank": effective_rank,
                    "retrieval_score": score,
                    "difficulty_band": effective_band,
                })

        # Sample sequentially from very_hard -> hard -> medium
        for band_name in ["very_hard", "hard", "medium"]:
            band_cands = candidates_by_band.get(band_name, [])
            quota = band_quotas.get(band_name, 5)

            for cand_rec in band_cands:
                did = cand_rec["doc_id"]
                if did in seen_dids:
                    continue

                seen_dids.add(did)
                self.mined_counts_by_source[cand_rec["negative_source"]] += 1
                self.mined_counts_by_band[band_name] += 1
                band_counts[band_name] += 1
                mined.append(cand_rec)

                if band_counts[band_name] >= quota or len(mined) >= max_total:
                    break

            if len(mined) >= max_total:
                break

        return mined
