import pytest
from src.ranking.evidence_pack import EvidencePackBuilder

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
