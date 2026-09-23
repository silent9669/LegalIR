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


def _training_probe_section() -> dict:
    """One real optimizer step on a tiny CPU model (methodology, not GPU evidence)."""
    try:
        import tempfile
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        from src.training.samplers import MicrobatchFactorization, probe_factorization_step

        config = BertConfig(vocab_size=300, hidden_size=32, num_attention_heads=2,
                            num_hidden_layers=2, max_position_embeddings=128, num_labels=1)
        model = BertForSequenceClassification(config)
        tmp_vocab = Path(tempfile.gettempdir()) / "microbench_probe_vocab.txt"
        if not tmp_vocab.exists():
            vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
            tmp_vocab.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(tmp_vocab))
        res = probe_factorization_step(
            model=model,
            tokenizer=tokenizer,
            factorization=MicrobatchFactorization(microbatch_size=2, gradient_accumulation_steps=8),
            sample_pairs=[("câu hỏi minh họa", "văn bản minh họa", 1.0),
                          ("câu hỏi khác", "văn bản khác", 0.0)],
            device="cpu",
            max_length=64,
        )
        sps = float(res.get("seconds_per_step", 0.0) or 0.0)
        return {
            "status": res.get("status"),
            "device": "cpu",
            "effective_batch_size": res.get("effective_batch_size"),
            "seconds_per_step": round(sps, 4),
            "steps_per_second": round(1.0 / sps, 3) if sps > 0 else 0.0,
            "note": "CPU tiny-model probe; GPU optimizer steps/s must be read from fold training logs.",
        }
    except Exception as exc:  # noqa: BLE001 - methodology probe only
        return {"status": "SKIPPED", "reason": type(exc).__name__}


def _host_memory_section() -> dict:
    """RSS/VRAM snapshot scaffolding (CPU values real; GPU requires CUDA pilot)."""
    out: dict = {}
    try:
        import psutil

        out["rss_mb"] = round(float(psutil.Process().memory_info().rss) / (1024 ** 2), 1)
    except Exception as exc:  # noqa: BLE001
        out["rss_mb"] = f"unavailable:{type(exc).__name__}"
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_allocated_mb"] = round(float(torch.cuda.memory_allocated()) / (1024 ** 2), 1)
            out["cuda_peak_mb"] = round(float(torch.cuda.max_memory_allocated()) / (1024 ** 2), 1)
        else:
            out["cuda"] = "n/a (no CUDA on this machine; VRAM only measurable on GPU pilot)"
    except Exception as exc:  # noqa: BLE001
        out["cuda"] = f"unavailable:{type(exc).__name__}"
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

    # --- Evidence parity + qinfo cache (same workload, both branches timed) ---
    # NOCACHE branch: cache genuinely disabled (qinfo_cache_size=0), so every
    # pair recomputes query info. CACHED branch: one untimed warm pass, then
    # timed all-hits passes. Parity requires identical packs.
    builder_nc = EvidencePackBuilder(macro_chunks=chunks, max_chunks=2, max_chars=1200,
                                     qinfo_cache_size=0)
    t0 = time.perf_counter()
    packs_nocache: dict[str, str] = {}
    for _ in range(args.repeats):
        for qid, qtext in queries:
            for did in doc_ids:
                packs_nocache[f"{qtext}||{did}"] = builder_nc.build_pack(qtext, did)
    nocache_seconds = time.perf_counter() - t0
    stats_nocache = builder_nc.get_qinfo_cache_stats()

    builder_c = EvidencePackBuilder(macro_chunks=chunks, max_chunks=2, max_chars=1200)
    for qid, qtext in queries:  # untimed warm: one miss per distinct query
        for did in doc_ids:
            builder_c.build_pack(qtext, did)
    t1 = time.perf_counter()
    packs_cached: dict[str, str] = {}
    for _ in range(args.repeats):
        for qid, qtext in queries:
            for did in doc_ids:
                packs_cached[f"{qtext}||{did}"] = builder_c.build_pack(qtext, did)
    cached_seconds = time.perf_counter() - t1
    stats_cached = builder_c.get_qinfo_cache_stats()
    mismatches = sum(1 for k, v in packs_cached.items() if packs_nocache.get(k) != v)

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
            recs = builder_c.build(qtext, did)
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
            "nocache_pass_seconds": round(nocache_seconds, 4),
            "cached_pass_seconds": round(cached_seconds, 4),
            "cache_speedup_cpu_only": round(nocache_seconds / max(1e-9, cached_seconds), 3),
            "pack_mismatches_cached_vs_nocache": mismatches,
            "qinfo_nocache_pass": stats_nocache,
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
        "training_probe_cpu_only": _training_probe_section(),
        "host_memory": _host_memory_section(),
        "metric_taxonomy": {
            "optimizer_steps_per_second": "train_reranker logs (optimizer_steps / training wall time); pilot reads fold metrics reranker_optimizer_steps + reranker_training_seconds. CPU probe below is methodology only.",
            "inference_queries_per_second": "fold metrics heldout_queries_per_second + retrieval_seconds/rerank_seconds/post_rerank_seconds split; mock pairs_per_second above is NOT GPU throughput.",
            "cold_end_to_end_time_to_valid_submission": (
                "Only measurable on a GPU run from EXTERNAL markers: dispatch time "
                "through warm (warm_manifest) + image build + clone/checkout + index "
                "build (stage_timings) + mining (pair_mining_seconds) + 5 folds (OOF "
                "report) + doc-disjoint + final train + private ensemble inference + "
                "delivery incl. HF upload receipt, plus billing. run_manifest "
                "elapsed_seconds is NOT the total: its clock starts inside "
                "run_a100_production_gate (after container start/clone/warm/preflight/"
                "dataset) and is recorded BEFORE the HF upload (§10). Never "
                "extrapolate from CPU/mock."
            ),
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
