def reciprocal_rank_fusion(run_list: list[list[dict]], k: int = 60, weights: list[float] = None, key: str = "doc_id") -> list[dict]:
    if not run_list:
        return []
    if weights is None:
        weights = [1.0 / len(run_list)] * len(run_list)

    scores = {}
    item_map = {}

    for run_idx, run in enumerate(run_list):
        w = weights[run_idx] if run_idx < len(weights) else 1.0
        seen_in_run = set()
        for rank, item in enumerate(run, start=1):
            elem_key = str(item.get(key) or item.get("chunk_id") or "")
            if not elem_key or elem_key in seen_in_run:
                continue
            seen_in_run.add(elem_key)

            if elem_key not in item_map:
                item_map[elem_key] = dict(item)
            scores[elem_key] = scores.get(elem_key, 0.0) + w / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for rank, (elem_key, score) in enumerate(ranked, start=1):
        elem = item_map[elem_key]
        elem["rrf_score"] = float(score)
        elem["rank"] = rank
        fused.append(elem)
    return fused
