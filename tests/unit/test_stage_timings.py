"""Stage-clock instrumentation: durations/counts only, never private text."""
from __future__ import annotations

from src.ranking.ensemble import build_ensemble_score_fn
from src.ranking.reranker import CrossEncoderReranker


def _mock_score(pairs, batch_size=None, max_length=None):
    return [float(len(str(q)) + len(str(p))) % 7 for q, p in pairs]


def test_reranker_stage_timings_keys_and_reset():
    r = CrossEncoderReranker(model_name="mock", batch_size=4, max_length=32, score_fn=_mock_score)
    t = r.get_stage_timings()
    assert set(t) == {"load_seconds", "tokenize_seconds", "transfer_seconds",
                      "forward_seconds", "batches", "pairs", "oom_events",
                      "min_successful_batch_size"}
    # score_fn path bypasses tokenizer clocks but stays deterministic.
    s1 = r.score_pairs([("q1", "p1"), ("q2", "p2")])
    s2 = r.score_pairs([("q1", "p1"), ("q2", "p2")])
    assert s1 == s2
    r.reset_stage_timings()
    t2 = r.get_stage_timings()
    assert t2["batches"] == 0 and t2["pairs"] == 0


def test_reranker_real_tokenize_path_accumulates_clocks():
    import tempfile
    from pathlib import Path

    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    cfg = BertConfig(vocab_size=300, hidden_size=16, num_attention_heads=2,
                     num_hidden_layers=1, intermediate_size=32,
                     max_position_embeddings=64, num_labels=1)
    model = BertForSequenceClassification(cfg)
    vocab = Path(tempfile.gettempdir()) / "stage_timing_vocab.txt"
    if not vocab.exists():
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                                   + [f"tok_{i}" for i in range(295)]) + "\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vocab))
    r = CrossEncoderReranker(model_name="mock", batch_size=2, max_length=32, device="cpu")
    r.tokenizer = tok
    r.model = model
    r.device = "cpu"
    # Bypass score_fn so the timed tokenizer+forward path executes on CPU.
    r.score_fn = None
    scores = r.score_pairs([("hello world", "evidence one two"), ("foo bar", "evidence three")])
    assert len(scores) == 2
    t = r.get_stage_timings()
    assert t["batches"] >= 1 and t["pairs"] == 2
    assert t["tokenize_seconds"] >= 0.0 and t["forward_seconds"] >= 0.0


def test_ensemble_timings_and_mean_parity():
    members = [CrossEncoderReranker(model_name="mock", score_fn=_mock_score) for _ in range(3)]
    fn = build_ensemble_score_fn(members)
    pairs = [("q", "p one"), ("q", "p two three")]
    got = fn(pairs)
    manual = [sum(v) / 3 for v in zip(*[_mock_score(pairs) for _ in range(3)])]
    assert got == manual
    timings = fn.timings
    assert timings["calls"] == 1 and timings["pairs_scored"] == 2
    assert len(timings["per_member_seconds"]) == 3
    assert timings["total_seconds"] >= 0.0
