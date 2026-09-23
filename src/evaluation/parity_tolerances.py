"""Predeclared parity tolerances for expected score-neutral runtime changes.

Any optimization that claims "same ranking, just faster" must pass these
comparisons on a MATCHED workload (same queries, candidates, code path inputs)
before it is classified P1 (output-equivalent) instead of P2 (score-changing):

  - candidate IDs and their order: EXACT (no tolerance),
  - pair texts and token IDs: EXACT,
  - cross-encoder logits: close within BF16 tolerance (ATOL=RTOL=1e-2;
    bf16 has ~3 decimal digits, so kernel nondeterminism below this cannot
    be distinguished from dtype noise),
  - top-5 doc IDs with deterministic tie-break: EXACT,
  - per-fold query coverage counts: EXACT.

CPU runs must additionally be bitwise-exact on logits (deterministic kernels);
the BF16 tolerance applies to GPU comparisons only. A mismatch on any exact
field reclassifies the change as P2 and requires the selection/confirmation
protocol (separate folds), never a re-report on the selection folds.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

BF16_LOGIT_ATOL = 1e-2
BF16_LOGIT_RTOL = 1e-2


def assert_token_ids_equal(a: object, b: object) -> None:
    """Token-ID sequences must match exactly (tokenizer identity)."""
    la = _to_nested_lists(a)
    lb = _to_nested_lists(b)
    if la != lb:
        raise AssertionError(
            f"token-ID parity failed: shapes {[len(x) for x in la]} vs {[len(x) for x in lb]}"
        )


def _to_nested_lists(v: object) -> list:
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, Mapping):
        return sorted((str(k), _to_nested_lists(x)) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return [_to_nested_lists(x) for x in v]
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(v)


def assert_logits_close(
    a: Sequence[float],
    b: Sequence[float],
    *,
    atol: float = BF16_LOGIT_ATOL,
    rtol: float = BF16_LOGIT_RTOL,
    exact_on_cpu: bool = False,
) -> dict[str, Any]:
    """Compare score vectors; returns a mismatch report (empty when passing).

    Raises AssertionError on length mismatch always, and on value mismatch
    beyond tolerance (exact equality when ``exact_on_cpu`` is True).
    """
    la = [float(x) for x in a]
    lb = [float(x) for x in b]
    if len(la) != len(lb):
        raise AssertionError(f"logit parity failed: lengths {len(la)} vs {len(lb)}")
    bad = 0
    worst = 0.0
    for x, y in zip(la, lb):
        diff = abs(x - y)
        worst = max(worst, diff)
        allowed = atol + rtol * abs(y) if not exact_on_cpu else 0.0
        if diff > allowed:
            bad += 1
    report = {"n": len(la), "mismatches": bad, "worst_abs_diff": worst,
              "atol": atol, "rtol": rtol, "exact_on_cpu": exact_on_cpu}
    if bad:
        raise AssertionError(f"logit parity failed: {bad}/{len(la)} beyond tolerance (worst={worst:.5f})")
    return report


def assert_top5_equal(a: Mapping[str, Sequence[str]], b: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Top-5 lists per query must match exactly (order + tie-break)."""
    if set(map(str, a.keys())) != set(map(str, b.keys())):
        missing = sorted(set(map(str, b.keys())) - set(map(str, a.keys())))[:5]
        extra = sorted(set(map(str, a.keys())) - set(map(str, b.keys())))[:5]
        raise AssertionError(f"top-5 coverage parity failed: missing={missing} extra={extra}")
    mismatched = [qid for qid in a if [str(x) for x in a[qid]] != [str(x) for x in b[qid]]]
    report = {"queries": len(a), "mismatched_queries": len(mismatched),
              "mismatched_qids": sorted(map(str, mismatched))[:10]}
    if mismatched:
        raise AssertionError(f"top-5 parity failed on {len(mismatched)} queries (e.g. {report['mismatched_qids'][:3]})")
    return report


def assert_candidate_order_equal(
    a: Mapping[str, Sequence[str]], b: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Full candidate ID lists per query must match exactly (order included)."""
    if set(map(str, a.keys())) != set(map(str, b.keys())):
        raise AssertionError("candidate-pool query coverage differs")
    mismatched = [qid for qid in a if [str(x) for x in a[qid]] != [str(x) for x in b[qid]]]
    if mismatched:
        raise AssertionError(f"candidate order parity failed on {len(mismatched)} queries")
    return {"queries": len(a), "mismatched_queries": 0}
