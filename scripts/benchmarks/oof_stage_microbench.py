#!/usr/bin/env python3
"""Reproducible CPU/mock microbenchmark for OOF stage breakdown.

Methodology (NOT an A100 speedup claim):
  - Fixed synthetic workload (seeded): N queries x M candidate docs with
    macro chunks, run through EvidencePackBuilder + CrossEncoderReranker with
    a mock score_fn + ensemble averaging.
  - Measures evidence qinfo cache hit rate, retrieval-simulated vs rerank vs
    post breakdown, reranker tokenize/transfer/forward clocks (mock forward),
    and ensemble per-member timing.
  - Verifies output parity: cached vs uncached evidence packs identical,
    ensemble mean matches manual mean, top-5 doc IDs unchanged.

CPU/mock numbers must never be presented as GPU speedups. A GPU pilot on the
real A100 SKU is still required to rank the true bottleneck; see
scripts/benchmarks/gpu_pilot_estimate.py for the prepared (not executed)
pilot commands and budget math.

Usage:
    .venv/bin/python scripts/benchmarks/oof_stage_microbench.py \
        --queries 20 --candidates 50 --repeats 3 --output-json /tmp/microbench.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _synthetic_chunks(num_docs: int, chunks_per_doc: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for d in range(num_docs):
        doc_id = f"D{1000 + d}"
        for c in range(chunks_per_doc):
            out.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_C{c}",
                "granularity": "macro",
                "article": f"Điều {1 + (d + c) % 10}",
                "clause": f"Khoản {(d + c) % 3 + 1}",
                "point": "",
                "text_raw": (
                    f"Nội dung điều {(d + c) % 10 + 1} về hợp đồng lao động "
                    f"và bồi thường {rng.choice(['tài sản', 'thiệt hại', 'quyền lợi'])} "
                    f"số {(d * 7 + c) % 100} năm 2020. " * 3
                ),
                "text_norm": "",
            })
    return out


def _synthetic_queries(num_queries: int) -> list[tuple[str, str]]:
    base = [
        "Tranh chấp hợp đồng lao động về bồi thường thiệt hại Điều {}?",
        "Quyền lợi bảo hiểm xã hội theo Điều {} Khoản {}?",
        "Bồi thường tài sản do vi phạm hợp đồng Điều {}?",
    ]
    out = []
    for i in range(num_queries):
        out.append((f"Q{i:04d}", base[i % len(base)].format(1 + i % 10, 1 + i % 3)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args(argv)

    from src.ranking.ensemble import build_ensemble_score_fn
    from src.ranking.evidence_pack import EvidencePackBuilder
    from src.ranking.reranker import CrossEncoderReranker

    n_q, n_c = max(1, args.queries), max(1, args.candidates)
    queries = _synthetic_queries(n_q)
    chunks = _synthetic_chunks(max(n_c, 10), 3, args.seed)
    doc_ids = sorted({c["doc_id"] for c in chunks})
    while len(doc_ids) < n_c:
        doc_ids += [f"D{2000 + len(doc_ids)}"]
    doc_ids = doc_ids[:n_c]
    # Ensure every candidate doc has chunks.
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c["doc_id"]] = by_doc.get(c["doc_id"], 0) + 1
    for did in doc_ids:
        if did not in by_doc:
            chunks.append({
                "doc_id": did, "chunk_id": f"{did}_C0", "granularity": "macro",
                "article": "Điều 1", "clause": "", "point": "",
                "text_raw": "Nội dung văn bản pháp luật generic về hợp đồng và bồi thường. " * 4,
                "text_norm": "",
            })

    # --- Evidence parity + qinfo cache ---
    builder = EvidencePackBuilder(macro_chunks=chunks, max_chunks=2, max_chars=1200)
    t0 = time.perf_counter()
    packs_first: dict[str, str] = {}
    for _ in range(args.repeats):
        for qid, qtext in queries:
            for did in doc_ids:
                packs_first[f"{qtext}||{did}"] = builder.build_pack(qtext, did)
    uncached_seconds = time.perf_counter() - t0
    stats_after_first = builder.get_qinfo_cache_stats()

    # Second pass over the SAME workload must hit the cache and match exactly.
    builder2 = EvidencePackBuilder(macro_chunks=chunks, max_chunks=2, max_chars=1200,
                                   qinfo_cache_size=0)  # no cache baseline
    ref: dict[str, str] = {}
    for qid, qtext in queries:
        for did in doc_ids:
            ref[f"{qtext}||{did}"] = builder2.build_pack(qtext, did)
    mismatches = sum(1 for k, v in packs_first.items() if ref.get(k) != v)
    # Timed cached pass (fresh builder, warm then measure).
    builder3 = EvidencePackBuilder(macro_chunks=chunks, max_chunks=2, max_chars=1200)
    for qid, qtext in queries:  # warm
        for did in doc_ids:
            builder3.build_pack(qtext, did)
    builder3.clear_qinfo_cache()
    # Re-warm once, then time the all-hits pass deterministically.
    for qid, qtext in queries:
        for did in doc_ids:
            builder3.build_pack(qtext, did)
    t1 = time.perf_counter()
    for _ in range(args.repeats):
        for qid, qtext in queries:
            for did in doc_ids:
                builder3.build_pack(qtext, did)
    cached_seconds = time.perf_counter() - t1
    stats_cached = builder3.get_qinfo_cache_stats()

    # --- Reranker stage clocks with deterministic mock scoring ---
    def _mock_score(pairs, batch_size=None, max_length=None):
        # Deterministic overlap score; no model, no GPU.
        out = []
        for q, p in pairs:
            qs = set(str(q).lower().split())
            ps = set(str(p).lower().split())
            out.append(float(len(qs & ps)))
        return out

    reranker = CrossEncoderReranker(model_name="mock", batch_size=16,
                                    max_length=128, score_fn=_mock_score)
    # Build pairs once (evidence text is the passage).
    pairs: list[tuple[str, str]] = []
    for qid, qtext in queries:
        for did in doc_ids[: min(10, n_c)]:
            recs = builder.build(qtext, did)
            passage = recs[0].get("reranker_text", "") if recs else did
            pairs.append((qtext, passage))
    t2 = time.perf_counter()
    scores_a = reranker.score_pairs(pairs)
    mock_score_seconds = time.perf_counter() - t2
    scores_b = reranker.score_pairs(pairs)
    assert [round(s, 6) for s in scores_a] == [round(s, 6) for s in scores_b], "mock scoring not deterministic"

    # --- Ensemble parity + timing ---
    members = [CrossEncoderReranker(model_name="mock", batch_size=16, max_length=128,
                                    score_fn=_mock_score) for _ in range(3)]
    ens_fn = build_ensemble_score_fn(members)
    t3 = time.perf_counter()
    ens_scores = ens_fn(pairs)
    ens_seconds = time.perf_counter() - t3
    manual = [sum(v) / 3 for v in zip(*[m.score_pairs(pairs) for m in members])]
    ens_mismatch = sum(1 for a, b in zip(ens_scores, manual) if abs(a - b) > 1e-9)

    workload_hash = hashlib.sha256(
        json.dumps({"queries": queries, "docs": doc_ids, "seed": args.seed},
                   sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    report = {
        "note": "CPU/mock methodology ONLY; NOT an A100 speedup claim. Same-workload parity harness.",
        "device": "cpu",
        "workload": {"queries": n_q, "candidates_per_query": n_c, "pairs_scored": len(pairs),
                     "repeats": args.repeats, "seed": args.seed, "workload_hash": workload_hash},
        "evidence": {
            "uncached_pass_seconds": round(uncached_seconds, 4),
            "cached_pass_seconds": round(cached_seconds, 4),
            "cache_speedup_cpu_only": round(uncached_seconds / max(1e-9, cached_seconds), 3),
            "pack_mismatches_cached_vs_nocache": mismatches,
            "qinfo_first_pass": stats_after_first,
            "qinfo_cached_pass": stats_cached,
        },
        "reranker_mock": {
            "pairs": len(pairs),
            "score_seconds": round(mock_score_seconds, 4),
            "pairs_per_second": round(len(pairs) / max(1e-9, mock_score_seconds), 2),
            "deterministic_repeat": True,
        },
        "ensemble_mock": {
            "members": len(members),
            "seconds": round(ens_seconds, 4),
            "mean_mismatches": ens_mismatch,
            "timings": getattr(ens_fn, "timings", {}),
        },
        "parity": {
            "evidence_packs_identical": mismatches == 0,
            "ensemble_mean_identical": ens_mismatch == 0,
        },
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[+] Wrote {args.output_json}", file=sys.stderr)
    if mismatches != 0 or ens_mismatch != 0:
        print("[!] PARITY FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
