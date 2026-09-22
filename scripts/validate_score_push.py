#!/usr/bin/env python3
"""Score-push coherence validator: fail fast on any max-score misconfiguration.

Run before every A100 dispatch:
    .venv/bin/python scripts/validate_score_push.py

Checks (all must pass):
 1. Reranker configs (base + v3 push): listwise loss, LoRA rank/alpha
    consistency, inference_batch>=64, rerank_k==candidate_k==200,
    max_length==512, max_negatives_per_group<=12, coverage_epochs>=1.
 2. Runtime configs (pipeline + legalir_v2): reranker max_length==512,
    evidence max_chunks==3 (train/infer agreement).
 3. Fusion: default w_rerank>=2.0 (trained discriminator dominates).
 4. Miner: near_miss band present in default limits + priority order.
 5. Ensemble: score averaging math + member resolution on a fixture tree.
 6. Parameter budget: LoRA estimate for r=64 stays < 4B with base models.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}", flush=True)
    if not cond:
        FAILURES.append(msg)


def _load(name: str) -> dict:
    p = REPO_ROOT / name
    if not p.is_file():
        check(False, f"config exists: {name}")
        return {}
    check(True, f"config exists: {name}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def main() -> int:
    base = _load("configs/experiments/reranker_lora.yaml")
    push = _load("configs/experiments/reranker_lora_v3_push.yaml")
    pipe = _load("configs/pipeline.yaml")
    algo = _load("configs/algorithm/legalir_v2.yaml")

    for cfg_name, cfg, min_rank in (("base", base, 32), ("v3-push", push, 64)):
        loss = str(cfg.get("loss_type", "")).lower()
        check(loss in ("listwise", "listwise_ce"), f"{cfg_name}: listwise loss (got {loss!r})")
        lora = cfg.get("lora", {}) or {}
        r = int(lora.get("r", cfg.get("lora_r", 0)))
        alpha = int(lora.get("lora_alpha", cfg.get("lora_alpha", 0)))
        check(r >= min_rank, f"{cfg_name}: LoRA r={r} >= {min_rank}")
        check(alpha == 2 * r, f"{cfg_name}: alpha==2r ({alpha}=={2 * r})")
        check(int(cfg.get("inference_batch_size", 0)) >= 64, f"{cfg_name}: inference_batch>=64")
        check(int(cfg.get("rerank_k", 0)) == 200, f"{cfg_name}: rerank_k==200 (candidate_k)")
        check(int(cfg.get("max_length", 0)) == 512, f"{cfg_name}: max_length==512")
        check(int(cfg.get("max_negatives_per_group", 0)) <= 12, f"{cfg_name}: group negs<=12 mined")
        check(int(cfg.get("coverage_epochs", 0)) >= 1, f"{cfg_name}: coverage_epochs>=1")
        check(cfg.get("pretrained_lora_path") in (None, "", "null"),
              f"{cfg_name}: cold start (no warm rank-lock)")

    for cfg_name, cfg in (("pipeline", pipe), ("legalir_v2", algo)):
        rr = (cfg.get("ranking", {}) or {}).get("reranker", {}) or {}
        ev = (cfg.get("ranking", {}) or {}).get("evidence", {}) or {}
        check(int(rr.get("max_length", 0)) == 512, f"{cfg_name}: reranker max_length==512")
        check(int(ev.get("max_chunks_per_doc", 0)) == 3, f"{cfg_name}: evidence chunks==3")

    from src.ranking.fusion import ReciprocalRankFusion
    f = ReciprocalRankFusion()
    check(float(f.weights.get("rerank", 0)) >= 2.0, f"fusion w_rerank>=2.0 (got {f.weights.get('rerank')})")

    from src.training.hard_negative_miner import HardNegativeMiner
    m = HardNegativeMiner()
    recs = m.mine_multi_band_negatives(
        query_id="q",
        candidates_by_source={
            "exact": [{"doc_id": "e1", "score": 1.0, "rank": 1}],
            "hybrid": [{"doc_id": f"h{i}", "score": 1.0 - i * 0.01, "rank": i + 1} for i in range(30)],
            "near_miss": [{"doc_id": f"n{i}", "score": 0.9, "rank": i + 3} for i in range(13)],
            "dense": [{"doc_id": f"s{i}", "score": 0.8, "rank": i + 1} for i in range(5)],
            "memory": [{"doc_id": f"m{i}", "score": 0.7, "rank": i + 1} for i in range(3)],
            "bm25": [{"doc_id": f"b{i}", "score": 5.0 - i * 0.1, "rank": i + 1} for i in range(30)],
            "medium_neg": [{"doc_id": f"d{i}", "score": 0.1, "rank": i + 21} for i in range(60)],
        },
        gold_doc_ids=["gold"],
        max_total=12,
    )
    srcs = [r["negative_source"] for r in recs]
    check(len(recs) == 12, f"miner fills 12 negatives (got {len(recs)})")
    check("near_miss" in srcs and "hybrid" in srcs, f"miner prioritizes boundary bands ({sorted(set(srcs))})")
    check("dense" in srcs and "memory" in srcs, f"miner keeps error-mode diversity ({sorted(set(srcs))})")

    from src.ranking.ensemble import build_ensemble_score_fn, resolve_ensemble_members

    class _Stub:
        def __init__(self, scores):
            self._s = scores

        def score_pairs(self, pairs, batch_size=None, max_length=None):
            assert len(pairs) == len(self._s)
            return list(self._s)

    fn = build_ensemble_score_fn([_Stub([1.0, 3.0]), _Stub([3.0, 1.0])])
    check(fn([("a", "b"), ("c", "d")]) == [2.0, 2.0], "ensemble averages member scores")
    try:
        build_ensemble_score_fn([])
        check(False, "ensemble rejects empty members")
    except ValueError:
        check(True, "ensemble rejects empty members")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "reranker_final").mkdir()
        (root / "reranker_final" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (root / "cv" / "fold_0" / "reranker_adapter").mkdir(parents=True)
        (root / "cv" / "fold_0" / "reranker_adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
        got = resolve_ensemble_members(root / "reranker_final", oof_cv_dir=root / "cv", num_folds=5)
        check(got == [str(root / "reranker_final"), str(root / "cv" / "fold_0" / "reranker_adapter")],
              f"ensemble resolves [final, fold_0] ({got})")

    # LoRA budget estimate: 4 target modules x 24 layers x 2*r*1024 (hidden) + base 702.75M.
    for r in (32, 64):
        est = 702_754_049 + 4 * 24 * 2 * r * 1024
        check(est < 4_000_000_000, f"LoRA r={r} budget {est/1e9:.3f}B < 4B")

    print("=" * 70)
    if FAILURES:
        print(f"VALIDATE_SCORE_PUSH: {len(FAILURES)} FAILURES", flush=True)
        return 1
    print("VALIDATE_SCORE_PUSH: ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
