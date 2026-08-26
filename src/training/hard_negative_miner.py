from collections import defaultdict
from typing import Any


class HardNegativeMiner:
    def __init__(self, false_negative_blacklist: dict[str, set[str]] | None = None):
        """
        false_negative_blacklist: optional mapping {qid: set_of_doc_ids_to_avoid}
        """
        self.blacklist: dict[str, set[str]] = false_negative_blacklist or defaultdict(set)

    def mine_negatives(
        self,
        *args,
        max_negatives: int = 15,
        **kwargs,
    ) -> list[str]:
        """
        Flexible signature:
        mine_negatives(query_id, candidates, gold_doc_ids, max_negatives=15)
        OR
        mine_negatives(candidates, gold_doc_ids, query_id=None, max_negatives=15)
        """
        if len(args) >= 3:
            # signature: (query_id, candidates, gold_doc_ids) or (candidates, gold_doc_ids, query_id)
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

        gold_set = set(str(x) for x in gold_doc_ids)
        qid_blacklist = set(str(x) for x in self.blacklist.get(str(query_id), set())) if query_id else set()

        mined = []
        for cand in candidates:
            did = str(cand["doc_id"]) if isinstance(cand, dict) else str(cand)

            if did in gold_set:
                continue

            if did in qid_blacklist:
                continue

            mined.append(did)
            if len(mined) >= max_negatives:
                break

        return mined
