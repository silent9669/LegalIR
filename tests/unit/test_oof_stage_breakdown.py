"""Held-out inference breakdown: retrieval vs rerank vs post, no private text."""
from __future__ import annotations

from src.pipeline.oof_runner import OOFRunner
from src.ranking.reranker import CrossEncoderReranker


def _mock_score(pairs, batch_size=None, max_length=None):
    return [1.0 for _ in pairs]


def test_run_fold_records_stage_breakdown(tmp_path, monkeypatch):
    from src.pipeline import oof_runner as mod

    runner = OOFRunner(
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "idx",
        output_dir=tmp_path / "cv",
        num_folds=1,
        candidate_k=5,
        rerank_k=5,
        use_reranker=True,
        reranker_model="mock",
        smoke=True,
        smoke_sample_size=10,
    )
    runner.queries_map = {
        "q1": "hợp đồng lao động Điều 61",
        "q2": "bảo hiểm xã hội Điều 62",
        "q3": "bồi thường thiệt hại",
    }
    runner.qrels_map = {"q1": ["d1"], "q2": ["d2"], "q3": ["d1", "d3"]}
    runner.train_query_embeddings = {}
    # No dense/bm25 indexes: memory falls back to lexical, hybrid fuse is stubbed.
    runner.exact = None
    runner.bm25 = None
    runner.bm25_pyvi = None
    runner.dense = None
    from src.ranking.evidence_pack import EvidencePackBuilder

    runner.evidence_builder = EvidencePackBuilder(
        macro_chunks=[
            {"chunk_id": "d1_C0", "doc_id": "d1", "granularity": "macro",
             "article": "Điều 61", "text_raw": "hợp đồng lao động bồi thường", "text_norm": ""},
            {"chunk_id": "d1_C1", "doc_id": "d1", "granularity": "macro",
             "article": "Điều 62", "text_raw": "quyền lợi bảo hiểm thời hạn", "text_norm": ""},
            {"chunk_id": "d2_C0", "doc_id": "d2", "granularity": "macro",
             "article": "Điều 62", "text_raw": "bảo hiểm xã hội quyền lợi", "text_norm": ""},
            {"chunk_id": "d2_C1", "doc_id": "d2", "granularity": "macro",
             "article": "Điều 63", "text_raw": "thời hạn giải quyết khiếu nại", "text_norm": ""},
            {"chunk_id": "d3_C0", "doc_id": "d3", "granularity": "macro",
             "article": "", "text_raw": "bồi thường thiệt hại tài sản", "text_norm": ""},
            {"chunk_id": "d3_C1", "doc_id": "d3", "granularity": "macro",
             "article": "", "text_raw": "tài sản thiệt hại hợp đồng", "text_norm": ""},
        ]
    )
    reranker = CrossEncoderReranker(model_name="mock", batch_size=4, max_length=64,
                                    score_fn=_mock_score)

    def _fake_search(self, query, top_k=5, exclude_qid=None, q_emb=None, branch_candidates=None):
        # Deterministic canned pool; never logs query text.
        return [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}][:top_k]

    monkeypatch.setattr(mod.HybridSearchEngine, "search_candidates", _fake_search)

    fold_info = {"train_query_ids": ["q3"], "val_query_ids": ["q1", "q2"]}
    preds, cands, feats, metrics, runtimes = runner.run_fold(0, fold_info, reranker=reranker)

    assert set(preds) == {"q1", "q2"}
    for key in ("retrieval_seconds", "rerank_seconds", "post_rerank_seconds",
                "rerank_query_windows", "rerank_candidate_docs", "elapsed_seconds"):
        assert key in metrics, f"missing {key}"
    assert metrics["retrieval_seconds"] >= 0.0
    assert metrics["rerank_seconds"] >= 0.0
    assert metrics["post_rerank_seconds"] >= 0.0
    # Breakdown must not exceed the wall total by more than measurement slack.
    parts = metrics["retrieval_seconds"] + metrics["rerank_seconds"] + metrics["post_rerank_seconds"]
    assert parts <= metrics["elapsed_seconds"] + 5.0
    assert metrics["rerank_query_windows"] == 2
    assert metrics["rerank_candidate_docs"] >= 2
    # Evidence cache stats are label-free counters only.
    assert metrics["evidence_qinfo_cache"]["misses"] >= 1
    # Per-query runtimes still recorded; no query text in metrics values.
    assert set(runtimes) == {"q1", "q2"}
    blob = str(metrics)
    assert "hợp đồng" not in blob and "bảo hiểm" not in blob
