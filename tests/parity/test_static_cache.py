import pytest
import pyarrow.parquet as pq
from pathlib import Path
from src.retrieval.static_cache import (
    StaticCandidateRecord,
    StaticCacheWriter,
    StaticCacheReader,
    STATIC_CACHE_SCHEMA,
)


def test_static_cache_writer_and_reader(tmp_path):
    out_file = tmp_path / "test_cache.parquet"
    writer = StaticCacheWriter(str(out_file), batch_size=2)

    records = [
        StaticCandidateRecord(
            query_id="q1",
            branch="bm25_legal",
            rank=1,
            doc_id="doc_100",
            score=12.5,
            best_chunk_id="chunk_1",
        ),
        StaticCandidateRecord(
            query_id="q1",
            branch="dense",
            rank=1,
            doc_id="doc_200",
            score=0.89,
            best_chunk_id="chunk_2",
        ),
        StaticCandidateRecord(
            query_id="q2",
            branch="exact",
            rank=1,
            doc_id="doc_300",
            score=1.0,
            best_chunk_id=None,
        ),
    ]

    for r in records:
        writer.write_record(r)
    writer.close()

    assert out_file.is_file()

    # Read back
    reader = StaticCacheReader(str(out_file))
    q1_cands = reader.get_query_candidates("q1")
    assert len(q1_cands) == 2
    assert q1_cands[0].doc_id == "doc_100"
    assert q1_cands[0].branch == "bm25_legal"
    assert q1_cands[0].score == pytest.approx(12.5)

    q2_cands = reader.get_query_candidates("q2")
    assert len(q2_cands) == 1
    assert q2_cands[0].doc_id == "doc_300"
    assert q2_cands[0].best_chunk_id is None

    # Check all query IDs
    qids = reader.get_query_ids()
    assert set(qids) == {"q1", "q2"}


def test_static_cache_writer_no_qrels_accepted():
    writer = StaticCacheWriter("dummy.parquet")
    # Verify no attribute or method accepts qrels
    assert not hasattr(writer, "qrels")
    assert not hasattr(writer, "gold_docs")
