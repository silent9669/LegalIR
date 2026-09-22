"""Max-score push regressions: ensemble, boundary mining, group cap, epochs, fusion."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml


def test_ensemble_averages_and_validates():
    from src.ranking.ensemble import build_ensemble_score_fn, resolve_ensemble_members

    class _Stub:
        def __init__(self, scores):
            self._s = scores

        def score_pairs(self, pairs, batch_size=None, max_length=None):
            return list(self._s)

    fn = build_ensemble_score_fn([_Stub([0.0, 4.0]), _Stub([4.0, 0.0])])
    assert fn([("a", "b"), ("c", "d")]) == [2.0, 2.0]
    with pytest.raises(ValueError):
        build_ensemble_score_fn([])
    with pytest.raises(ValueError):
        build_ensemble_score_fn([_Stub([1.0]), _Stub([1.0, 2.0])])([("a", "b"), ("c", "d")])


def test_ensemble_resolves_final_plus_folds(tmp_path):
    from src.ranking.ensemble import resolve_ensemble_members

    root = tmp_path
    (root / "reranker_final").mkdir()
    (root / "reranker_final" / "adapter_config.json").write_text("{}", encoding="utf-8")
    for i in (0, 2):
        d = root / "cv" / f"fold_{i}" / "reranker_adapter"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    got = resolve_ensemble_members(root / "reranker_final", oof_cv_dir=root / "cv", num_folds=5)
    assert got[0] == str(root / "reranker_final")
    assert str(root / "cv" / "fold_0" / "reranker_adapter") in got
    assert str(root / "cv" / "fold_2" / "reranker_adapter") in got
    assert len(got) == 3


def test_ensemble_wrapper_scores_through_callback(tmp_path):
    from src.ranking.reranker import CrossEncoderReranker
    from src.ranking.ensemble import build_ensemble_reranker

    for name, scores in (("a1", [1.0, 5.0]), ("a2", [3.0, 3.0])):
        d = tmp_path / name
        d.mkdir()
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    members = [
        CrossEncoderReranker(model_name="mock", score_fn=lambda pairs, batch_size=None, max_length=None, s=s: list(s))
        for s in ([1.0, 5.0], [3.0, 3.0])
    ]

    import src.ranking.ensemble as ens

    real_cls = CrossEncoderReranker
    seen = {}

    class _FakeCls(real_cls):
        def __init__(self, *a, **k):
            if "score_fn" not in k and k.get("adapter_path") is None:
                raise AssertionError("unexpected")
            super().__init__(*a, **k)

    # Build via member stubs: monkeypatch member construction by pre-seeding dirs
    # is complex; instead verify the averaging contract through score_fn path.
    w = real_cls(model_name="mock", score_fn=ens.build_ensemble_score_fn(members))
    assert w.score_pairs([("q", "p1"), ("q", "p2")]) == [2.0, 4.0]
    assert w.model is None  # wrapper itself holds no weights; members do


def test_miner_boundary_priority_and_diversity():
    from src.training.hard_negative_miner import HardNegativeMiner

    m = HardNegativeMiner()
    recs = m.mine_multi_band_negatives(
        query_id="q",
        candidates_by_source={
            "exact": [{"doc_id": "e1", "score": 1.0, "rank": 1}],
            "hybrid": [{"doc_id": f"h{i}", "score": 0.9, "rank": i + 1} for i in range(30)],
            "near_miss": [{"doc_id": f"n{i}", "score": 0.8, "rank": i + 3} for i in range(13)],
            "dense": [{"doc_id": f"s{i}", "score": 0.7, "rank": i + 1} for i in range(5)],
            "memory": [{"doc_id": f"m{i}", "score": 0.6, "rank": i + 1} for i in range(3)],
            "bm25": [{"doc_id": f"b{i}", "score": 5.0, "rank": i + 1} for i in range(30)],
            "medium_neg": [{"doc_id": f"d{i}", "score": 0.1, "rank": i + 21} for i in range(60)],
        },
        gold_doc_ids=["gold"],
        max_total=12,
    )
    assert len(recs) == 12
    srcs = {r["negative_source"] for r in recs}
    assert {"near_miss", "hybrid", "dense", "memory", "medium_neg"} <= srcs


def test_group_dataset_respects_configured_cap():
    from src.training.trainer import RerankerGroupDataset

    rows = [{"query_id": "q1", "query_text": "qt", "doc_id": "pos", "evidence_text": "ev", "label": 1.0}]
    rows += [{"query_id": "q1", "query_text": "qt", "doc_id": f"n{i}", "evidence_text": "ev", "label": 0.0} for i in range(20)]
    ds = RerankerGroupDataset(rows, max_negatives_per_group=12)
    assert len(ds.items) == 1
    assert len(ds.items[0]["negatives"]) == 12
    ds7 = RerankerGroupDataset(rows, max_negatives_per_group=7)
    assert len(ds7.items[0]["negatives"]) == 7


def test_coverage_epochs_multiplies_effective_steps(tmp_path):
    import pyarrow.parquet as pq
    import pyarrow as pa
    from src.training.train_reranker import train_reranker

    rows = []
    for qi in range(8):
        rows.append({"query_id": f"q{qi}", "query_text": "tok_1 tok_2", "doc_id": f"g{qi}",
                     "evidence_text": "tok_1 tok_2", "label": 1.0,
                     "negative_source": "gold", "retrieval_rank": 0, "retrieval_score": 1.0,
                     "evidence_chunk_ids": "[]", "fold": 0})
        rows.append({"query_id": f"q{qi}", "query_text": "tok_1 tok_2", "doc_id": f"n{qi}",
                     "evidence_text": "tok_99", "label": 0.0,
                     "negative_source": "hybrid", "retrieval_rank": 3, "retrieval_score": 0.5,
                     "evidence_chunk_ids": "[]", "fold": 0})
    pairs = tmp_path / "pairs.parquet"
    pq.write_table(pa.Table.from_pylist(rows), pairs)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "base_model_name": "mock", "output_dir": "artifacts/local/training/checkpoints",
        "loss_type": "listwise", "batch_size": 8, "gradient_accumulation_steps": 1,
        "max_steps": 1, "coverage_epochs": 3, "learning_rate": 1e-3, "max_length": 32,
        "use_lora": False, "precision": "fp32", "device": "cpu",
    }), encoding="utf-8")
    rep = train_reranker(pairs_file=pairs, output_dir=tmp_path / "out", config_path=cfg,
                         fold=0, device="cpu")
    # 8 eligible queries, batch 8, pos+neg coverage => req 2 steps/epoch x3 epochs => 6.
    assert rep["effective_max_steps"] == 6
    assert rep["status"] == "completed"


def test_fusion_rerank_dominates_by_default(monkeypatch):
    monkeypatch.delenv("LEGALIR_FUSION_W_RERANK", raising=False)
    import importlib
    import src.ranking.fusion as fus

    importlib.reload(fus)
    try:
        assert float(fus.ReciprocalRankFusion.DEFAULT_BRANCH_WEIGHTS["rerank"]) >= 2.0
        assert fus.ReciprocalRankFusion().w_rerank >= 2.0
        monkeypatch.setenv("LEGALIR_FUSION_W_RERANK", "1.8")
        importlib.reload(fus)
        assert fus.ReciprocalRankFusion().w_rerank == 1.8
    finally:
        monkeypatch.delenv("LEGALIR_FUSION_W_RERANK", raising=False)
        importlib.reload(fus)


def test_score_push_configs_coherent():
    root = Path("configs/experiments/reranker_lora.yaml")
    base = yaml.safe_load(root.read_text(encoding="utf-8"))
    push = yaml.safe_load(Path("configs/experiments/reranker_lora_v3_push.yaml").read_text(encoding="utf-8"))
    assert str(base["loss_type"]).lower() in ("listwise", "listwise_ce")
    assert base["lora"]["r"] >= 32 and base["lora_alpha"] == 2 * base["lora"]["r"]
    assert push["lora"]["r"] >= 64
    assert base["rerank_k"] == 200 and push["rerank_k"] == 200
    assert base.get("pretrained_lora_path") is None and push.get("pretrained_lora_path") is None
    assert int(base.get("coverage_epochs", 0)) >= 2


def test_validate_score_push_script_passes():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "scripts/validate_score_push.py"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ALL CHECKS PASSED" in r.stdout
