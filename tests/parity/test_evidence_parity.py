import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from src.evidence.macro_store import MacroEvidenceStore
from src.evidence.selector import LazyPositiveLocalizer, LazyEvidencePackBuilder


@pytest.fixture
def sample_macro_store(tmp_path):
    p = tmp_path / "chunks.parquet"
    table = pa.Table.from_pydict({
        "doc_id": ["doc1", "doc1", "doc2"],
        "chunk_id": ["c1", "c2", "c3"],
        "chunk_type": ["macro", "macro", "macro"],
        "text": [
            "Điều 15. Quy định về xử phạt vi phạm hành chính trong lĩnh vực thuế.",
            "Điều 16. Các trường hợp miễn trừ xử phạt vi phạm hành chính.",
            "Điều 5. Quyền và nghĩa vụ của người nộp thuế.",
        ],
        "chunk_index": [0, 1, 0],
    })
    pq.write_table(table, str(p))
    return MacroEvidenceStore(str(p))


def test_lazy_positive_localizer(sample_macro_store):
    localizer = LazyPositiveLocalizer(sample_macro_store)
    # Query referencing Điều 16
    q = "Trường hợp nào được miễn trừ xử phạt theo Điều 16?"
    best_chunk = localizer.localize(q, "doc1")
    assert best_chunk is not None
    chunk_id = best_chunk["chunk_id"] if isinstance(best_chunk, dict) else best_chunk.chunk_id
    assert chunk_id == "c2"


def test_lazy_evidence_pack_builder(sample_macro_store):
    builder = LazyEvidencePackBuilder(sample_macro_store, max_chunks=1)
    q = "Mức xử phạt theo điều 15"
    pack = builder.build_pack(q, "doc1")
    assert "[DOCUMENT]" in pack
    assert "[EVIDENCE 1]" in pack
    assert "Điều 15" in pack
