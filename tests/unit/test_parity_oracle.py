"""Tests for parity tolerances (P1) and oracle@K + leakage guards (P2)."""
from __future__ import annotations

import pytest

from src.evaluation.oracle_recall import assert_disjoint_fit_eval, oracle_recall_at_k
from src.evaluation.parity_tolerances import (
    assert_candidate_order_equal,
    assert_logits_close,
    assert_token_ids_equal,
    assert_top5_equal,
)


def test_oracle_separates_retrieval_from_ranking_errors():
    pools = {
        "q_hit": ["d1", "d9"],
        "q_deep": ["d0", "d0", "d2"],
        "q_miss": ["d7", "d8"],
    }
    qrels = {"q_hit": ["d1"], "q_deep": ["d2"], "q_miss": ["d3"], "q_empty": []}
    rep = oracle_recall_at_k(pools, qrels, ks=(1, 2, 3))
    assert rep["oracle_recall"][1] == pytest.approx(1 / 3)
    assert rep["oracle_recall"][2] == pytest.approx(1 / 3)
    assert rep["oracle_recall"][3] == pytest.approx(2 / 3)
    assert rep["retrieval_misses_at_max_k"] == ["q_miss"]
    assert rep["no_gold_queries"] == ["q_empty"]
    assert rep["evaluated_queries"] == 3


def test_oracle_rejects_empty_ks():
    with pytest.raises(ValueError):
        oracle_recall_at_k({}, {}, ks=())


def test_disjoint_guard_blocks_final_adapter_on_train_queries():
    train_qids = [f"q{i}" for i in range(7000)]
    with pytest.raises(AssertionError, match="Leakage"):
        assert_disjoint_fit_eval(train_qids, ["q0", "q1"], context="final-adapter")
    # Disjoint eval passes silently.
    assert_disjoint_fit_eval(["q0"], ["q1"], context="fold-adapter") is None


def test_disjoint_guard_blocks_fold_adapter_on_own_queries():
    with pytest.raises(AssertionError, match="Leakage"):
        assert_disjoint_fit_eval(["q1", "q2"], ["q2", "q3"], context="fold-0-adapter")


def test_logits_parity_exact_and_bf16_modes():
    assert_logits_close([1.0, 2.0], [1.0, 2.0], exact_on_cpu=True)["mismatches"] == 0
    # Tiny dtype-strength noise passes the BF16 tolerance.
    assert_logits_close([1.0], [1.005], exact_on_cpu=False)["mismatches"] == 0
    with pytest.raises(AssertionError):
        assert_logits_close([1.0], [1.005], exact_on_cpu=True)
    with pytest.raises(AssertionError):
        assert_logits_close([1.0], [1.5], exact_on_cpu=False)
    with pytest.raises(AssertionError, match="lengths"):
        assert_logits_close([1.0], [1.0, 2.0])


def test_top5_and_candidate_order_exact():
    a = {"q1": ["d1", "d2", "d3", "d4", "d5"]}
    b = {"q1": ["d1", "d2", "d3", "d4", "d5"]}
    assert_top5_equal(a, b)["mismatched_queries"] == 0
    assert_candidate_order_equal(a, b)["mismatched_queries"] == 0
    with pytest.raises(AssertionError):
        assert_top5_equal(a, {"q1": ["d1", "d2", "d3", "d4", "dX"]})
    with pytest.raises(AssertionError):
        assert_candidate_order_equal(a, {"q1": ["d2", "d1", "d3", "d4", "d5"]})
    with pytest.raises(AssertionError, match="coverage"):
        assert_top5_equal(a, {})


def test_token_ids_exact():
    assert_token_ids_equal([[1, 2, 3]], [[1, 2, 3]])
    with pytest.raises(AssertionError):
        assert_token_ids_equal([[1, 2, 3]], [[1, 2, 4]])
