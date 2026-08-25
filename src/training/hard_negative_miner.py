from collections import defaultdict

class HardNegativeMiner:
    def __init__(self, false_negative_blacklist: dict = None):
        """
        false_negative_blacklist: optional mapping {qid: set_of_doc_ids_to_avoid}
        """
        self.blacklist = false_negative_blacklist or defaultdict(set)

    def mine_negatives(
        self,
        candidates: list,
        gold_doc_ids: list,
        query_id: str = None,
        max_negatives: int = 15
    ) -> list:
        """
        candidates: list of dicts with 'doc_id' or list of doc_ids
        gold_doc_ids: list of gold document IDs
        """
        gold_set = set(str(x) for x in gold_doc_ids)
        qid_blacklist = self.blacklist.get(str(query_id), set()) if query_id else set()

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
