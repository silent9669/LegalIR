#!/usr/bin/env python3
"""Prepared (NOT executed) GPU pilot plan + budget estimate for fresh accounts.

No cloud call, no secret, no GPU is touched here. This script only prints
pilot commands and derives a cost/time envelope from the prior run report
(/Users/phucdang/Downloads/report.md: fold0 4406.5s R@5 92.30%, fold1
4267.0s R@5 91.48% on A100-SXM4-40GB) WITHOUT claiming those numbers benchmark
the new code. Treat them as the hypothesis to beat, not evidence.

IMPORTANT: the pipeline has NO single-fold production mode (FULL always runs
5 folds + doc-disjoint + final + private ensemble; there is no
LEGALIR_NUM_FOLDS-style limiter, deliberately, so a pilot can never silently
become a partial production run). Any "pilot" below is therefore a TIME-BOXED
FULL plumbing check that WILL be interrupted mid-run: use it only to verify
dispatch → checkout → HF preflight → warm-cache attach → first fold mining,
never as timing or Recall evidence. Do NOT run it as if it were a single-fold
pilot, and do NOT treat partial folds as pilot results.

Time-boxed plumbing pilot (requires explicit approval before running; billed):
  1) CPU warm-only (cheap, no GPU):
       python scripts/modal/run_full.py --warm-only --hf-repo OWNER/REPO
  2) Verify warm manifest + dry-run (no GPU, fail-closed on default repo),
     plus the read-only repo check (no creation, no upload):
       python scripts/modal/run_full.py --warm --private --push-config --detach \\
           --hf-repo OWNER/REPO --dry-run
       HF_TOKEN_WRITE=hf_... python scripts/check_hf_repo.py --repo OWNER/REPO
  3) Time-boxed FULL plumbing check with an explicit DURATION cap (example
     3h via MODAL_TIMEOUT_SECONDS). This caps job DURATION, not spend:
     retries, re-runs, image builds and Volume operations bill extra, and
     nothing on the platform enforces a money ceiling — record the approved
     budget, monitor continuously, and stop explicitly:
       MODAL_TIMEOUT_SECONDS=10800 python scripts/modal/run_full.py \\
           --private --push-config --detach --warm --hf-repo OWNER/REPO
     then monitor `modal app logs <APP_ID>` every 15 min; stop with
     `modal app stop <APP_ID> --yes`. Never auto-relaunch.

Local checks CANNOT certify the new account: Modal profile/quota/GPU SKU,
secret values, HF write access, repo visibility and real cost must be verified
LIVE by the teammate (checklist in the estimate output below). A green local
suite proves code correctness only.

Budget math below uses the report's per-fold wall times as the *baseline
hypothesis* and a configurable $/GPU-hour rate (Modal pricing varies by
account/region; fill in the real rate before dispatch).
"""

from __future__ import annotations

import argparse
import json
import sys

# Prior-run hypothesis only (A100-40GB, old code, 2 folds observed).
FOLD_SECONDS_BASELINE = (4406.5 + 4267.0) / 2.0  # ~4336.75s per fold
FOLDS_FULL = 5
OVERHEAD_FACTOR = 1.25  # disjoint + final + private ensemble + upload buffer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-dollars-per-hour", type=float, default=2.0,
                    help="Billed A100 $/hr for YOUR account/region (default placeholder 2.0).")
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args(argv)

    rate = float(args.gpu_dollars_per_hour)
    if rate <= 0:
        print("[!] --gpu-dollars-per-hour must be positive.", file=sys.stderr)
        return 2
    per_fold_hr = FOLD_SECONDS_BASELINE / 3600.0
    full_hr = per_fold_hr * FOLDS_FULL * OVERHEAD_FACTOR
    plumbing_cap_hr = 3.0
    estimate = {
        "note": "Estimate from PRIOR-run hypothesis only; new code has NO GPU measurement yet (CHƯA ĐO).",
        "single_fold_mode": False,
        "single_fold_note": (
            "No single-fold production mode exists; FULL always runs 5 folds + "
            "doc-disjoint + final + private ensemble. The capped command below is a "
            "plumbing check that will be interrupted, not a single-fold pilot."
        ),
        "baseline_hypothesis": {
            "source": "/Users/phucdang/Downloads/report.md folds 0/1 on A100-SXM4-40GB",
            "per_fold_seconds": round(FOLD_SECONDS_BASELINE, 1),
            "per_fold_recall@5": ["92.30%", "91.48%"],
            "folds_observed": 2,
            "full_5fold_observed": False,
        },
        "full_run_envelope": {
            "folds": FOLDS_FULL,
            "overhead_factor": OVERHEAD_FACTOR,
            "estimated_gpu_hours": round(full_hr, 2),
            "estimated_cost_usd_at_rate": round(full_hr * rate, 2),
            "rate_usd_per_hour": rate,
        },
        "timeboxed_plumbing_check": {
            "timeout_seconds": int(plumbing_cap_hr * 3600),
            "timeout_derived_cost_usd_at_rate": round(plumbing_cap_hr * rate, 2),
            "timeout_is_not_a_spend_cap": (
                "MODAL_TIMEOUT_SECONDS caps job duration only. Retries, re-runs, "
                "image builds and Volume ops bill extra; no platform money ceiling "
                "is set by this command. Approve a money budget and stop explicitly."
            ),
            "expected_outcome": "interrupted mid-run (plumbing signals only)",
        },
        "live_verify_checklist_new_account": [
            "modal profile active = YOUR account; GPU quota/SKU/region confirmed; approved money budget + stop procedure recorded",
            "dashboard secrets kaggle-secret + huggingface-secret belong to YOUR account (values never printed)",
            "scripts/check_hf_repo.py --repo OWNER/REPO exits 0 (exists + private + write); else BLOCKED",
            "warm manifest source_sha / revisions / dataset fingerprint match the dispatch SHA",
            "monitor modal app logs every 15 min; stop with modal app stop <id> --yes; confirm termination",
        ],
        "plumbing_commands": [
            "python scripts/modal/run_full.py --warm-only --hf-repo OWNER/REPO",
            "python scripts/modal/run_full.py --warm --private --push-config --detach --hf-repo OWNER/REPO --dry-run",
            "MODAL_TIMEOUT_SECONDS=10800 python scripts/modal/run_full.py --private --push-config --detach --warm --hf-repo OWNER/REPO",
        ],
        "go_criteria": [
            "Remote logs show 'HF repo: OWNER/REPO (source=flag)' on warm AND training jobs.",
            "attempt warm_cache_summary.json records warm_source_sha == training_sha "
            "(fields: warm_source_sha, warm_requested_label, warm_hf_repo, training_sha); "
            "models_attached/dataset_reused recorded as hit or fallback-download (both valid).",
            "Fold report shows retrieval_seconds/rerank_seconds/post_rerank_seconds + reranker_stage_timings + evidence_qinfo_cache.",
            "Recall@5 held at baseline recipe (r=64/listwise/2 epochs/depth200); no rerank_k/max_length/ensemble change for speed.",
        ],
        "not_run": True,
    }
    print(json.dumps(estimate, indent=2))
    if args.output_json:
        from pathlib import Path

        Path(args.output_json).write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
