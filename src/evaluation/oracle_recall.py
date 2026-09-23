"""Candidate oracle@K and fit/eval leakage guards for P2 score work.

``oracle_recall_at_k`` separates retrieval errors (gold never in the pool)
from ranking errors (gold pooled but ranked out of the top-5): a query the
oracle misses at depth K cannot be fixed by any reranker/fusion at that K.

Selection/confirmation discipline (P2):
  - choose among a few PREDECLARED hypotheses on selection fold(s) only,
  - confirm the winner on held-out fold(s) never used for selection,
  - never report selection-fold numbers as independent confirmation.

``assert_disjoint_fit_eval`` enforces the leakage bans at runtime:
  - never score the 7,000 train queries with the final adapter fitted on them,
  - never score a query with a fold adapter fitted on that same query.

These helpers are intentionally post-hoc: run them on a completed attempt's
candidate pools (cv ``candidates.parquet``) plus qrels. They are NOT called by
the training pipeline, so scored outputs stay untouched.
"""

from __future__ import annotations

from typing import Mapping, Sequence

DEFAULT_ORACLE_KS = (50, 100, 150, 200)


def oracle_recall_at_k(
    candidate_pools: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Sequence[str]],
    ks: Sequence[int] = DEFAULT_ORACLE_KS,
) -> dict:
    """Per-K oracle recall + per-query retrieval misses.

    A query counts as oracle-hit at K when ANY gold doc appears in the first
    K pooled candidates. Queries with no golds are excluded (reported).
    """
    ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not ks:
        raise ValueError("oracle_recall_at_k requires at least one positive K")
    per_k_hits: dict[int, int] = {k: 0 for k in ks}
    per_k_total: dict[int, int] = {k: 0 for k in ks}
    misses_at_max: list[str] = []
    no_gold: list[str] = []
    max_k = ks[-1]
    for qid, golds in qrels.items():
        qid = str(qid)
        gold_set = {str(g) for g in (golds or [])}
        if not gold_set:
            no_gold.append(qid)
            continue
        pool = [str(d) for d in (candidate_pools.get(qid) or [])]
        for k in ks:
            per_k_total[k] += 1
            if gold_set.intersection(pool[:k]):
                per_k_hits[k] += 1
        if not gold_set.intersection(pool[:max_k]):
            misses_at_max.append(qid)
    recall = {k: (per_k_hits[k] / per_k_total[k] if per_k_total[k] else 0.0) for k in ks}
    return {
        "ks": list(ks),
        "oracle_recall": recall,
        "evaluated_queries": per_k_total[ks[0]] if ks else 0,
        "retrieval_misses_at_max_k": sorted(misses_at_max),
        "retrieval_miss_count": len(misses_at_max),
        "no_gold_queries": sorted(no_gold),
    }


def assert_disjoint_fit_eval(
    fit_query_ids: Sequence[str] | set[str],
    eval_query_ids: Sequence[str] | set[str],
    context: str = "adapter-eval",
) -> None:
    """Raise when any eval query was in the adapter's fitting population."""
    fit = {str(x) for x in fit_query_ids}
    ev = {str(x) for x in eval_query_ids}
    leaked = sorted(fit & ev)
    if leaked:
        raise AssertionError(
            f"Leakage in {context}: {len(leaked)} eval queries were in the fitting "
            f"population (e.g. {leaked[:5]}). Score them only with adapters fitted "
            "on disjoint queries."
        )
