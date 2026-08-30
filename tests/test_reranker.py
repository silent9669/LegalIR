import pytest
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.reranker import CrossEncoderReranker

def test_evidence_pack_builder():
    macro_chunks = [
        {
            "chunk_id": "doc1_macro_01",
            "doc_id": "doc1",
            "article": "Điều 15. Thời hạn cấp phép",
            "text_norm": "[VĂN BẢN]: Thông tư 12/2020\n[ĐIỀU KHOẢN]: Điều 15. Thời hạn cấp phép\n[NỘI DUNG]: Thời hạn cấp phép là 10 ngày làm việc."
        },
        {
            "chunk_id": "doc1_macro_02",
            "doc_id": "doc1",
            "article": "Điều 16. Lệ phí",
            "text_norm": "[VĂN BẢN]: Thông tư 12/2020\n[ĐIỀU KHOẢN]: Điều 16. Lệ phí\n[NỘI DUNG]: Mức thu lệ phí là 100.000 đồng."
        }
    ]
    builder = EvidencePackBuilder(macro_chunks)

    # Build evidence pack for doc1
    pack = builder.build_evidence("Thời hạn cấp phép là bao nhiêu ngày?", "doc1")
    assert pack is not None
    assert "Thời hạn cấp phép là 10 ngày làm việc" in pack["evidence_text"]
    assert pack["chunk_id"] == "doc1_macro_01"


def test_reranker_orders_scored_tuple_candidates_by_cross_encoder_score():
    chunks = [
        {"chunk_id": "low-c", "doc_id": "low", "text_norm": "low evidence"},
        {"chunk_id": "high-c", "doc_id": "high", "text_norm": "high evidence"},
    ]
    builder = EvidencePackBuilder(macro_chunks=chunks)

    def score_pairs(pairs, batch_size=16, max_length=512):
        return [10.0 if "high evidence" in passage else 1.0 for _, passage in pairs]

    reranker = CrossEncoderReranker(model_name="mock", score_fn=score_pairs)
    ranked = reranker.rerank(
        "Which evidence applies?",
        [("low", 0.0), ("high", 0.0)],
        builder,
        top_k=2,
    )

    assert [candidate["doc_id"] for candidate in ranked] == ["high", "low"]
