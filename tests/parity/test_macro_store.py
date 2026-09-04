import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from src.evidence.macro_store import MacroChunk, PreprocessedDoc, MacroEvidenceStore


@pytest.fixture
def mock_chunks_parquet(tmp_path):
    p = tmp_path / "chunks.parquet"
    table = pa.Table.from_pydict({
        "doc_id": ["d1", "d1", "d2", "d3"],
        "chunk_id": ["c1", "c2", "c3", "c4"],
        "chunk_type": ["macro", "macro", "macro", "micro"],
        "text": [
            "Article 1: General provisions.",
            "Article 2: Specific regulations.",
            "Article 1: Scope of application.",
            "Micro chunk to ignore",
        ],
        "chunk_index": [0, 1, 0, 0],
    })
    pq.write_table(table, str(p))
    return str(p)


def test_macro_evidence_store_basic(mock_chunks_parquet):
    store = MacroEvidenceStore(
        mock_chunks_parquet,
        max_cache_bytes=1024 * 1024,
        max_cached_docs=2,
    )

    # d1 has 2 macro chunks
    chunks_d1 = store.get_doc_chunks("d1")
    assert len(chunks_d1) == 2
    assert chunks_d1[0].chunk_id == "c1"
    assert chunks_d1[1].chunk_id == "c2"

    # d2 has 1 macro chunk
    chunks_d2 = store.get_doc_chunks("d2")
    assert len(chunks_d2) == 1
    assert chunks_d2[0].chunk_id == "c3"

    # d3 had only micro chunk, so 0 macro chunks
    chunks_d3 = store.get_doc_chunks("d3")
    assert len(chunks_d3) == 0

    assert store.cache_bytes() > 0


def test_macro_evidence_store_preprocessed_doc(mock_chunks_parquet):
    store = MacroEvidenceStore(mock_chunks_parquet)
    prep_d1 = store.get_preprocessed_doc("d1")
    assert prep_d1.doc_id == "d1"
    assert len(prep_d1.chunks) == 2
    assert "General provisions" in prep_d1.full_text


def test_macro_evidence_store_lru_eviction(mock_chunks_parquet):
    # Store with capacity 2 docs
    store = MacroEvidenceStore(
        mock_chunks_parquet,
        max_cache_bytes=100 * 1024 * 1024,
        max_cached_docs=2,
    )

    store.get_doc_chunks("d1")
    assert "d1" in store._cache
    store.get_doc_chunks("d2")
    assert "d2" in store._cache
    assert len(store._cache) == 2

    # Accessing d3 should evict d1 (least recently used)
    store.get_doc_chunks("d3")
    assert "d1" not in store._cache
    assert "d2" in store._cache
    assert "d3" in store._cache

    store.clear_cache()
    assert len(store._cache) == 0
    assert store.cache_bytes() == 0
