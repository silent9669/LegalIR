import pytest
from src.evidence.pair_materializer import PairMaterializer, build_duplicate_closure


def test_build_duplicate_closure():
    dup_groups = [
        {"doc_ids": ["docA", "docB"]},
        {"doc_ids": ["docC", "docD", "docE"]},
    ]
    closure = build_duplicate_closure(dup_groups)
    assert closure["docA"] == {"docA", "docB"}
    assert closure["docB"] == {"docA", "docB"}
    assert closure["docC"] == {"docC", "docD", "docE"}
    assert "docZ" not in closure


def test_pair_materializer_leakage_assertion():
    train_qids = {"q1", "q2"}
    val_qids = {"q3", "q4"}

    pm = PairMaterializer(
        train_qids=train_qids,
        val_qids=val_qids,
        qrels={"q1": ["docA"], "q2": ["docB"]},
        duplicate_groups=[{"doc_ids": ["docA", "docA_dup"]}],
    )

    # Validating query IDs: train_qid is allowed
    pm.assert_fold_isolation("q1")

    # Validation query ID in train pairs must raise ValueError
    with pytest.raises(ValueError, match="Validation leakage detected"):
        pm.assert_fold_isolation("q3")

    # Negative checking: docA_dup must be blacklisted for q1 because docA is gold
    assert pm.is_negative_allowed(qid="q1", neg_doc_id="docA_dup") is False
    assert pm.is_negative_allowed(qid="q1", neg_doc_id="docA") is False
    assert pm.is_negative_allowed(qid="q1", neg_doc_id="docX") is True
