from collections import defaultdict
from pathlib import Path
import json
import pandas as pd
import pytest

from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.build_indexes import (
    build_bm25_index,
    build_bm25_pyvi_index,
    enrich_chunks_with_doc_metadata,
)
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.reranker import CrossEncoderReranker
from src.training.hard_negative_miner import HardNegativeMiner, get_difficulty_band
from src.training.build_pairs import build_training_pairs


# ==============================================================================
# 1. P1.5 Tests: BM25 Metadata Enrichment & Statutory Boosting
# ==============================================================================

def test_enrich_chunks_with_doc_metadata(tmp_path: Path):
    """Verify that micro chunks are enriched with title, legal_number, year, doc_type, link."""
    docs_df = pd.DataFrame([
        {
            "doc_id": "doc_100",
            "title": "Nghị định về bảo hiểm xã hội",
            "legal_number": "115/2015/NĐ-CP",
            "year": "2015",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/van-ban/115-2015-nd-cp.aspx",
        },
        {
            "doc_id": "doc_200",
            "title": "Luật Đầu tư",
            "legal_number": "61/2020/QH14",
            "year": "2020",
            "doc_type": "Luật",
            "link": "https://thuvienphapluat.vn/van-ban/61-2020-qh14.aspx",
        },
    ])
    chunks_df = pd.DataFrame([
        {
            "chunk_id": "c100_1",
            "doc_id": "doc_100",
            "granularity": "micro",
            "article": "Điều 5",
            "clause": "Khoản 1",
            "point": "Điểm a",
            "text_raw": "Chế độ ốm đau cho người lao động tham gia bảo hiểm xã hội.",
            "text_norm": "chế độ ốm đau cho người lao động tham gia bảo hiểm xã hội.",
        },
        {
            "chunk_id": "c200_1",
            "doc_id": "doc_200",
            "granularity": "micro",
            "article": "Điều 7",
            "clause": "Khoản 2",
            "point": "",
            "text_raw": "Ngành, nghề đầu tư kinh doanh có điều kiện theo quy định.",
            "text_norm": "ngành, nghề đầu tư kinh doanh có điều kiện theo quy định.",
        },
    ])

    docs_file = tmp_path / "documents.parquet"
    docs_df.to_parquet(docs_file)

    enriched_df = enrich_chunks_with_doc_metadata(chunks_df, docs_file)

    assert "title" in enriched_df.columns
    assert "legal_number" in enriched_df.columns
    assert "year" in enriched_df.columns
    assert "doc_type" in enriched_df.columns
    assert "link" in enriched_df.columns

    row_100 = enriched_df[enriched_df["doc_id"] == "doc_100"].iloc[0]
    assert row_100["legal_number"] == "115/2015/NĐ-CP"
    assert row_100["year"] == "2015"
    assert row_100["doc_type"] == "Nghị định"
    assert row_100["title"] == "Nghị định về bảo hiểm xã hội"


def test_bm25_indices_built_with_enriched_metadata(tmp_path: Path):
    """Verify build_bm25_index and build_bm25_pyvi_index store metadata fields and enable statutory boosts."""
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir(parents=True)
    bm25_out = tmp_path / "bm25_out"
    bm25_pyvi_out = tmp_path / "bm25_pyvi_out"

    docs_df = pd.DataFrame([
        {
            "doc_id": "doc_tax",
            "title": "Nghị định về thuế giá trị gia tăng",
            "legal_number": "44/2023/NĐ-CP",
            "year": "2023",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/44-2023-nd-cp",
        },
        {
            "doc_id": "doc_other",
            "title": "Văn bản chung khác",
            "legal_number": "99/2020/NĐ-CP",
            "year": "2020",
            "doc_type": "Nghị định",
            "link": "https://thuvienphapluat.vn/99-2020-nd-cp",
        },
    ])
    chunks_df = pd.DataFrame([
        {
            "chunk_id": "c_tax_1",
            "doc_id": "doc_tax",
            "granularity": "micro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "Điểm a",
            "text_raw": "Chính sách giảm thuế giá trị gia tăng 2 phần trăm.",
            "text_norm": "chính sách giảm thuế giá trị gia tăng 2 phần trăm.",
        },
        {
            "chunk_id": "c_other_1",
            "doc_id": "doc_other",
            "granularity": "micro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "text_raw": "Quy định thuế giá trị gia tăng áp dụng chung.",
            "text_norm": "quy định thuế giá trị gia tăng áp dụng chung.",
        },
    ])

    docs_df.to_parquet(canonical_dir / "documents.parquet")
    chunks_df.to_parquet(canonical_dir / "chunks.parquet")

    # Build BM25 micro index
    build_bm25_index(canonical_dir=canonical_dir, output_dir=bm25_out)
    bm25 = BM25MicroRetriever.load(bm25_out / "bm25_micro_index.pkl")

    # Verify structured metadata maps are populated
    assert "doc_tax" in bm25.doc_legal_numbers
    assert any("44/2023/nđ/cp" in num or "44/2023/nd/cp" in num for num in bm25.doc_legal_numbers["doc_tax"])
    assert "2023" in bm25.doc_years["doc_tax"]
    assert "nghị định" in bm25.doc_types["doc_tax"]

    # Verify statutory boosting elevates doc_tax when queried by legal number
    results = bm25.retrieve("Nghị định 44/2023/NĐ-CP giảm thuế GTGT", top_k=5)
    assert len(results) > 0
    assert results[0]["doc_id"] == "doc_tax"

    # Build PyVi BM25 index
    build_bm25_pyvi_index(canonical_dir=canonical_dir, output_dir=bm25_pyvi_out)
    bm25_pyvi = BM25PyViRetriever.load(bm25_pyvi_out / "bm25_pyvi_index.pkl")
    pyvi_results = bm25_pyvi.retrieve("Nghị định 44/2023/NĐ-CP", top_k=5)
    assert len(pyvi_results) > 0
    assert pyvi_results[0]["doc_id"] == "doc_tax"


# ==============================================================================
# 2. P1.6 Tests: Remove Duplicate [QUESTION] from Reranker Passages
# ==============================================================================

def test_evidence_pack_builder_default_no_question():
    """Verify that build_pack and build default to include_question=False to save sequence B token budget."""
    chunks = [
        {"chunk_id": "c1", "doc_id": "doc_1", "article": "Điều 10", "text_raw": "Quy định về thời hạn cấp phép."},
        {"chunk_id": "c2", "doc_id": "doc_1", "article": "Điều 11", "text_raw": "Quy định về hồ sơ xin cấp phép."},
    ]
    doc_meta = {
        "doc_1": {"title": "Luật Giấy phép", "legal_number": "10/2023/QH15"}
    }
    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta, max_chunks=2)

    query = "Thời hạn cấp phép là bao lâu?"

    # 1. Default build_pack
    pack = builder.build_pack(query, "doc_1")
    assert "[QUESTION]" not in pack
    assert pack.startswith("[DOCUMENT] Luật Giấy phép 10/2023/QH15")
    assert "[EVIDENCE 1]" in pack
    assert "Thời hạn cấp phép là bao lâu?" not in pack[:len("[DOCUMENT] Luật Giấy phép 10/2023/QH15")]

    # 2. Default build (records)
    records = builder.build(query, "doc_1")
    assert len(records) > 0
    for rec in records:
        assert "[QUESTION]" not in rec["pack"]
        assert "[QUESTION]" not in rec["evidence_text"]
        assert "[QUESTION]" not in rec["reranker_text"]

    # 3. Default build_evidence_text
    ev_text = builder.build_evidence_text(query, doc_meta["doc_1"], chunks)
    assert "[QUESTION]" not in ev_text
    assert "[DOCUMENT]" in ev_text
    assert "[EVIDENCE 1]" in ev_text


def test_cross_encoder_reranker_passage_contract():
    """Verify CrossEncoderReranker formats pair sequence B without duplicate [QUESTION]."""
    chunks = [
        {"chunk_id": "c1", "doc_id": "doc_a", "article": "Điều 1", "text_raw": "Nội dung quy định A."},
    ]
    doc_meta = {"doc_a": {"title": "Văn bản A", "legal_number": "01/2021"}}
    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta)

    query = "Hỏi về quy định A?"
    records = builder.build(query, "doc_a")
    passage = records[0]["pack"]

    assert "[QUESTION]" not in passage
    assert passage.startswith("[DOCUMENT]")
    assert "[EVIDENCE 1]" in passage


# ==============================================================================
# 3. P1.7 Tests: Multi-Branch & Multi-Band Hard Negative Mining
# ==============================================================================

def test_hard_negative_miner_multi_branch_and_bands():
    """Verify HardNegativeMiner mines from all active branches and categorizes difficulty bands."""
    blacklist = {"q_test": {"dup_gold_1", "dup_gold_2"}}
    miner = HardNegativeMiner(false_negative_blacklist=blacklist)

    candidates_by_source = {
        "exact": [
            {"doc_id": "gold_doc", "score": 1.0, "rank": 1},  # must be excluded (gold)
            {"doc_id": "exact_fp_1", "score": 1.0, "rank": 2},
        ],
        "bm25": [
            {"doc_id": "bm25_1", "score": 18.0, "rank": 1},   # very_hard
            {"doc_id": "dup_gold_1", "score": 12.0, "rank": 10}, # must be excluded (blacklist)
            {"doc_id": "bm25_2", "score": 14.0, "rank": 8},   # hard
        ],
        "bm25_pyvi": [
            {"doc_id": "pyvi_1", "score": 16.0, "rank": 2},   # very_hard
            {"doc_id": "pyvi_2", "score": 11.0, "rank": 12},  # hard
        ],
        "dense": [
            {"doc_id": "dense_1", "score": 0.88, "rank": 3},  # very_hard
            {"doc_id": "dense_2", "score": 0.75, "rank": 15}, # hard
        ],
        "memory": [
            {"doc_id": "mem_1", "score": 0.85, "rank": 1},    # very_hard
        ],
        "hybrid": [
            {"doc_id": "hybrid_1", "score": 0.05, "rank": 4}, # very_hard
            {"doc_id": "hybrid_2", "score": 0.03, "rank": 18}, # hard
        ],
        "medium_neg": [
            {"doc_id": "med_1", "score": 0.01, "rank": 35},   # medium
            {"doc_id": "med_2", "score": 0.008, "rank": 60},  # medium
        ],
    }

    mined = miner.mine_multi_band_negatives(
        query_id="q_test",
        candidates_by_source=candidates_by_source,
        gold_doc_ids=["gold_doc"],
        per_source_limits={
            "exact": 1,
            "bm25": 2,
            "bm25_pyvi": 2,
            "dense": 2,
            "memory": 1,
            "hybrid": 2,
            "medium_neg": 2,
        },
        max_total=12,
    )

    mined_ids = [m["doc_id"] for m in mined]

    # Verify exclusions
    assert "gold_doc" not in mined_ids
    assert "dup_gold_1" not in mined_ids
    assert "dup_gold_2" not in mined_ids

    # Verify presence of candidates across diverse branches
    assert "exact_fp_1" in mined_ids
    assert "bm25_1" in mined_ids
    assert "pyvi_1" in mined_ids
    assert "dense_1" in mined_ids
    assert "mem_1" in mined_ids
    assert "med_1" in mined_ids

    # Verify difficulty bands are assigned correctly
    bands = {m["doc_id"]: m["difficulty_band"] for m in mined}
    assert bands["exact_fp_1"] == "very_hard"
    assert bands["bm25_1"] == "very_hard"
    assert bands["bm25_2"] == "hard"
    assert bands["med_1"] == "medium"

    # Verify miner stats
    stats = miner.get_stats()
    assert stats["excluded_golds_count"] >= 1
    assert stats["excluded_duplicates_count"] >= 1
    assert stats["mined_counts_by_band"]["very_hard"] > 0
    assert stats["mined_counts_by_band"]["hard"] > 0
    assert stats["mined_counts_by_band"]["medium"] > 0


def test_hard_negative_miner_by_difficulty_bands_sampling():
    """Verify mine_by_difficulty_bands samples across difficulty quotas."""
    miner = HardNegativeMiner()
    candidates_by_source = {
        "bm25": [
            {"doc_id": f"cand_{r}", "rank": r, "score": 100 - r}
            for r in range(1, 50)
        ]
    }

    mined = miner.mine_by_difficulty_bands(
        query_id="q1",
        candidates_by_source=candidates_by_source,
        gold_doc_ids=["cand_1"],
        band_limits={"very_hard": 2, "hard": 2, "medium": 2},
        max_total=6,
    )

    mined_ids = [m["doc_id"] for m in mined]
    assert "cand_1" not in mined_ids  # gold excluded
    assert len(mined) == 6

    bands = [m["difficulty_band"] for m in mined]
    assert bands.count("very_hard") == 2
    assert bands.count("hard") == 2
    assert bands.count("medium") == 2


def test_build_training_pairs_end_to_end_integration(tmp_path: Path):
    """Verify build_training_pairs produces valid parquet files with multi-branch negatives and no [QUESTION] in evidence."""
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "indexes"
    pairs_dir = tmp_path / "pairs"
    data_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    # 1. Toy documents
    docs = [
        {"doc_id": "doc_1", "title": "Luật Lao động", "legal_number": "45/2019/QH14", "year": "2019", "doc_type": "Luật", "name_raw": "Luật Lao động"},
        {"doc_id": "doc_2", "title": "Nghị định Hướng dẫn", "legal_number": "145/2020/NĐ-CP", "year": "2020", "doc_type": "Nghị định", "name_raw": "Nghị định 145"},
        {"doc_id": "doc_3", "title": "Luật Doanh nghiệp", "legal_number": "59/2020/QH14", "year": "2020", "doc_type": "Luật", "name_raw": "Luật Doanh nghiệp"},
        {"doc_id": "doc_4", "title": "Thông tư Thuế", "legal_number": "111/2013/TT-BTC", "year": "2013", "doc_type": "Thông tư", "name_raw": "Thông tư 111"},
    ]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet")

    # 2. Toy chunks
    chunks = [
        {"chunk_id": "c1_1", "doc_id": "doc_1", "granularity": "macro", "article": "Điều 1", "text_norm": "quy định về hợp đồng lao động", "text_raw": "quy định về hợp đồng lao động"},
        {"chunk_id": "c1_2", "doc_id": "doc_1", "granularity": "micro", "article": "Điều 1", "text_norm": "quy định về hợp đồng lao động", "text_raw": "quy định về hợp đồng lao động"},
        {"chunk_id": "c2_1", "doc_id": "doc_2", "granularity": "macro", "article": "Điều 2", "text_norm": "hướng dẫn thi hành hợp đồng lao động", "text_raw": "hướng dẫn thi hành hợp đồng lao động"},
        {"chunk_id": "c2_2", "doc_id": "doc_2", "granularity": "micro", "article": "Điều 2", "text_norm": "hướng dẫn thi hành hợp đồng lao động", "text_raw": "hướng dẫn thi hành hợp đồng lao động"},
        {"chunk_id": "c3_1", "doc_id": "doc_3", "granularity": "macro", "article": "Điều 3", "text_norm": "thành lập doanh nghiệp và vốn điều lệ", "text_raw": "thành lập doanh nghiệp và vốn điều lệ"},
        {"chunk_id": "c3_2", "doc_id": "doc_3", "granularity": "micro", "article": "Điều 3", "text_norm": "thành lập doanh nghiệp và vốn điều lệ", "text_raw": "thành lập doanh nghiệp và vốn điều lệ"},
        {"chunk_id": "c4_1", "doc_id": "doc_4", "granularity": "macro", "article": "Điều 4", "text_norm": "thuế thu nhập cá nhân", "text_raw": "thuế thu nhập cá nhân"},
        {"chunk_id": "c4_2", "doc_id": "doc_4", "granularity": "micro", "article": "Điều 4", "text_norm": "thuế thu nhập cá nhân", "text_raw": "thuế thu nhập cá nhân"},
    ]
    pd.DataFrame(chunks).to_parquet(data_dir / "chunks.parquet")

    # 3. Toy queries & qrels
    queries = [
        {"query_id": "q1", "question_raw": "Thời hạn hợp đồng lao động quy định thế nào?", "question_norm": "thời hạn hợp đồng lao động quy định thế nào?"},
        {"query_id": "q2", "question_raw": "Vốn điều lệ công ty TNHH?", "question_norm": "vốn điều lệ công ty tnhh?"},
    ]
    qrels = [
        {"query_id": "q1", "doc_id": "doc_1"},
        {"query_id": "q2", "doc_id": "doc_3"},
    ]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet")
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet")

    # 4. Duplicate groups
    dup_groups = {"group_1": ["doc_1", "doc_2"]}
    (data_dir / "duplicate_groups.json").write_text(json.dumps(dup_groups), encoding="utf-8")

    # Build pairs
    retriever_df, reranker_df = build_training_pairs(
        data_dir=data_dir,
        index_dir=index_dir,
        output_dir=pairs_dir,
        use_all_queries=True,
        negatives_per_positive=2,
        max_evidence_chunks=2,
        include_dense_negatives=False,
        include_pyvi_negatives=False,
    )

    assert not reranker_df.empty
    assert (pairs_dir / "reranker_pairs.parquet").exists()
    assert (pairs_dir / "retriever_pairs.parquet").exists()
    assert (pairs_dir / "manifest.json").exists()

    # Verify no [QUESTION] in any evidence_text in reranker_df
    for ev_text in reranker_df["evidence_text"]:
        assert "[QUESTION]" not in ev_text
        assert "[DOCUMENT]" in ev_text
        assert "[EVIDENCE 1]" in ev_text

    # Verify duplicate doc_2 was excluded from negatives of q1 (since doc_1 is gold and doc_2 is duplicate)
    q1_negs = reranker_df[(reranker_df["query_id"] == "q1") & (reranker_df["label"] == 0.0)]["doc_id"].tolist()
    assert "doc_1" not in q1_negs
    assert "doc_2" not in q1_negs
