#!/usr/bin/env python3
"""
Evaluate whether a candidate pipeline run is eligible for production promotion based on leakage-safe OOF evidence.

Authoritative specification: LEGALIR_CI_COLAB_KAGGLE_ARCHITECTURE_SPEC.md
Implementation plan: LEGALIR_CI_COLAB_KAGGLE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GUARD_PATH = REPO_ROOT / "configs" / "production_score_guard.json"


def _extract_metric(d: Mapping[str, Any], *keys: str, default: float | None = None) -> float | None:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (ValueError, TypeError):
                pass
        # Check in nested dictionaries
        for sub_k in ("random_5fold", "baseline_metrics", "metrics", "cv_metrics", "document_disjoint"):
            if sub_k in d and isinstance(d[sub_k], Mapping) and k in d[sub_k]:
                try:
                    return float(d[sub_k][k])
                except (ValueError, TypeError):
                    pass
    return default


def evaluate_score_promotion(
    candidate: Mapping[str, Any],
    baseline_or_guard: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], list[str]]:
    """
    Evaluate candidate OOF metrics against accepted production baseline.

    Promotion Rules:
    1. Leakage checks must be passed.
    2. Higher official Recall@5 -> Eligible.
    3. Equal Recall@5 + higher Precision@5 -> Eligible.
    4. Lower Recall@5 -> Rejected.
    5. Candidate Recall@50 / Recall@150 must not regress beyond tolerance.
    6. Doc-disjoint Recall@5 must be present and not regress beyond tolerance.
    7. Learned parameters must be strictly < 4B.

    Returns:
        tuple[bool, dict[str, Any], list[str]]: (is_promoted, delta_metrics, reasons)
    """
    guardrails = baseline_or_guard.get("guardrails", {})
    require_leakage = guardrails.get("require_leakage_checks_passed", True)
    require_doc_disjoint = guardrails.get("require_doc_disjoint_eval", True)
    max_cand_reg = float(guardrails.get("max_candidate_recall_regression", 0.005))
    max_dd_reg = float(guardrails.get("max_doc_disjoint_recall_regression", 0.02))
    max_params = int(guardrails.get("max_total_parameters", 4_000_000_000))

    reasons: list[str] = []
    deltas: dict[str, Any] = {}

    # 1. Leakage check
    cand_leakage_passed = candidate.get("leakage_checks_passed", candidate.get("leakage_checks", True))
    if require_leakage and not cand_leakage_passed:
        reasons.append("REJECT: Candidate failed leakage verification checks (OOF leakage detected).")

    # 2. Extract baseline metrics
    base_r5 = _extract_metric(baseline_or_guard, "oof_recall_at_5", "mean_recall_at_5", "recall_at_5")
    base_p5 = _extract_metric(baseline_or_guard, "oof_precision_at_5", "mean_precision_at_5", "precision_at_5")
    base_c50 = _extract_metric(baseline_or_guard, "candidate_recall_at_50", "candidate_recall@50", default=0.940)
    base_c150 = _extract_metric(baseline_or_guard, "candidate_recall_at_150", "candidate_recall@150", default=0.970)
    base_dd_r5 = _extract_metric(baseline_or_guard, "doc_disjoint_recall_at_5", "recall_at_5", default=0.660)

    # 3. Extract candidate metrics
    cand_r5 = _extract_metric(candidate, "oof_recall_at_5", "mean_recall_at_5", "recall_at_5")
    cand_p5 = _extract_metric(candidate, "oof_precision_at_5", "mean_precision_at_5", "precision_at_5")
    cand_c50 = _extract_metric(candidate, "candidate_recall_at_50", "candidate_recall@50")
    cand_c150 = _extract_metric(candidate, "candidate_recall_at_150", "candidate_recall@150")

    cand_dd_r5 = None
    if "doc_disjoint_recall_at_5" in candidate:
        cand_dd_r5 = float(candidate["doc_disjoint_recall_at_5"])
    elif "document_disjoint_recall_at_5" in candidate:
        cand_dd_r5 = float(candidate["document_disjoint_recall_at_5"])
    elif "document_disjoint" in candidate and isinstance(candidate["document_disjoint"], Mapping):
        cand_dd_r5 = _extract_metric(candidate["document_disjoint"], "recall_at_5", "doc_disjoint_recall_at_5")


    if cand_r5 is None or base_r5 is None:
        reasons.append("REJECT: Missing Recall@5 metric in candidate or baseline.")
        return False, {}, reasons

    cand_p5 = cand_p5 if cand_p5 is not None else 0.0
    base_p5 = base_p5 if base_p5 is not None else 0.0

    r5_delta = cand_r5 - base_r5
    p5_delta = cand_p5 - base_p5
    deltas["recall_at_5_delta"] = round(r5_delta, 6)
    deltas["precision_at_5_delta"] = round(p5_delta, 6)
    deltas["candidate_recall_at_5"] = cand_r5
    deltas["baseline_recall_at_5"] = base_r5

    # 4. Doc-disjoint robustness metric check
    if require_doc_disjoint:
        if cand_dd_r5 is None:
            reasons.append("REJECT: Missing doc-disjoint / document-disjoint robustness evaluation in candidate.")
        elif base_dd_r5 is not None:
            dd_delta = cand_dd_r5 - base_dd_r5
            deltas["doc_disjoint_recall_at_5_delta"] = round(dd_delta, 6)
            if dd_delta < -max_dd_reg:
                reasons.append(
                    f"REJECT: Doc-disjoint Recall@5 regressed by {-dd_delta:.4f} (tolerance: {max_dd_reg:.4f})."
                )

    # 5. Candidate Recall@50 / Recall@150 regression checks
    if cand_c50 is not None and base_c50 is not None:
        c50_delta = cand_c50 - base_c50
        deltas["candidate_recall_at_50_delta"] = round(c50_delta, 6)
        if c50_delta < -max_cand_reg:
            reasons.append(
                f"REJECT: Candidate Recall@50 regressed by {-c50_delta:.4f} (tolerance: {max_cand_reg:.4f})."
            )

    if cand_c150 is not None and base_c150 is not None:
        c150_delta = cand_c150 - base_c150
        deltas["candidate_recall_at_150_delta"] = round(c150_delta, 6)
        if c150_delta < -max_cand_reg:
            reasons.append(
                f"REJECT: Candidate Recall@150 regressed by {-c150_delta:.4f} (tolerance: {max_cand_reg:.4f})."
            )

    # 6. Parameter count budget
    cand_params = _extract_metric(candidate, "total_learned_parameters", "total_parameters", default=0)
    if cand_params is not None and cand_params > max_params:
        reasons.append(f"REJECT: Learned parameter budget exceeded: {cand_params:,} > {max_params:,} (<4B rule).")

    # 7. Recall@5-First Decision Rule
    if r5_delta > 1e-6:
        # Higher Recall@5
        pass
    elif abs(r5_delta) <= 1e-6:
        # Equal Recall@5 -> tie-break on Precision@5
        if p5_delta > 1e-6:
            pass
        else:
            reasons.append(
                f"REJECT: Recall@5 is equal ({cand_r5:.4f}) and Precision@5 did not improve (delta: {p5_delta:+.4f})."
            )
    else:
        # Lower Recall@5
        reasons.append(f"REJECT: Overall OOF Recall@5 regressed by {-r5_delta:.4f} ({cand_r5:.4f} vs {base_r5:.4f}).")

    is_promoted = len(reasons) == 0
    return is_promoted, deltas, reasons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate LegalIR run for production score promotion."
    )
    parser.add_argument("--candidate", type=Path, required=True, help="Path to candidate report (JSON)")
    parser.add_argument("--guard", type=Path, default=DEFAULT_GUARD_PATH, help="Path to production_score_guard.json")
    args = parser.parse_args()

    if not args.candidate.exists():
        print(f"[-] Candidate report not found: {args.candidate}", file=sys.stderr)
        return 1
    if not args.guard.exists():
        print(f"[-] Guard config not found: {args.guard}", file=sys.stderr)
        return 1

    candidate_data = json.loads(args.candidate.read_text(encoding="utf-8"))
    guard_data = json.loads(args.guard.read_text(encoding="utf-8"))

    print("=================================================================")
    print("LegalIR Production Score Promotion Gate")
    print(f"  • Candidate : {args.candidate}")
    print(f"  • Baseline  : {guard_data.get('baseline_label', 'default')}")
    print("=================================================================")

    is_promoted, deltas, reasons = evaluate_score_promotion(candidate_data, guard_data)

    print("\nDELTA METRICS:")
    for k, v in deltas.items():
        print(f"  • {k:35s}: {v}")

    print("\n=================================================================")
    if is_promoted:
        print("[+] PROMOTION APPROVED: Candidate meets all Recall@5-first guardrails.")
        return 0
    else:
        print("[-] PROMOTION REJECTED:")
        for r in reasons:
            print(f"    - {r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
