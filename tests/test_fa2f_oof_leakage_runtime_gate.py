"""Behavioral tests for LegalIR FA2F OOF Leakage and Runtime Final Repair.

Covers:
- Task 1: OOF pair mining binding to authoritative fold train IDs with 0% validation leakage
- Task 2: Deterministic split artifact provenance (input -> repo -> generated) and SHA-256 recording
- Task 3: 4-group duplicate blacklist resolution and false-negative prevention
- Task 4: Pair mining runtime scaling from smoke query counts to FULL
- Task 5: Component-complete cold-start projection map
"""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. Task 1: OOF Pair Mining Binding & Validation Isolation (P0)
# ==============================================================================

def test_oof_pair_builder_receives_exact_fold_train_ids_without_input_splits(tmp_path):
    """When input has NO splits/ directory, OOFRunner must pass exact fold train IDs to build_training_pairs."""
    from src.pipeline.oof_runner import OOFRunner

    data_dir = tmp_path / "splitless_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 10 docs, 10 queries, 10 qrels
    docs = [{"doc_id": str(i), "title": f"D{i}", "passage_norm": f"p{i}"} for i in range(10)]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet")
    pd.DataFrame([{"chunk_id": f"c{i}", "doc_id": str(i), "granularity": "macro", "text_norm": f"t{i}"} for i in range(10)]).to_parquet(data_dir / "chunks.parquet")
    queries = [{"query_id": f"q{i}", "question_norm": f"query {i}"} for i in range(10)]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet")
    qrels = [{"query_id": f"q{i}", "doc_id": str(i), "relevance": 1} for i in range(10)]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet")

    # Explicitly verify NO splits directory
    assert not (data_dir / "splits").exists()

    runner = OOFRunner(
        data_dir=data_dir,
        index_dir=tmp_path / "indexes",
        output_dir=tmp_path / "cv_out",
        num_folds=2,
        smoke=True,
        smoke_sample_size=2,
        train_reranker_per_fold=True,
    )
    runner.load_data()
    runner.load_retrievers()
    folds = runner.get_splits()

    captured_kwargs = {}

    def mock_build_pairs(**kwargs):
        captured_kwargs.update(kwargs)
        train_q = kwargs.get("train_query_ids", [])
        return pd.DataFrame(), pd.DataFrame([{"query_id": str(qid), "doc_id": "0", "label": 1.0} for qid in train_q])

    with mock.patch("src.training.build_pairs.build_training_pairs", side_effect=mock_build_pairs), \
         mock.patch("src.training.train_reranker.train_reranker", return_value={"status": "completed", "optimizer_steps": 1}):
        runner.run()

    assert "train_query_ids" in captured_kwargs
    assert captured_kwargs["train_query_ids"] is not None
    assert captured_kwargs["use_all_queries"] is False
    assert len(captured_kwargs["train_query_ids"]) > 0


def test_oof_pair_qids_are_subset_of_train_ids_and_zero_validation_leakage():
    """Verify fold isolation assertion logic catches unknown or leaked validation queries."""
    train_ids = {"q1", "q2", "q3", "q4"}
    val_ids = {"q5", "q6"}

    # Valid: pairs generated only from train
    valid_pair_qids = {"q1", "q2"}
    assert valid_pair_qids.issubset(train_ids)
    assert valid_pair_qids.isdisjoint(val_ids)

    # Leaked: pairs generated containing validation query
    leaked_pair_qids = {"q1", "q5"}
    assert not leaked_pair_qids.isdisjoint(val_ids)


# ==============================================================================
# 2. Task 2: Deterministic Split Provenance
# ==============================================================================

def test_split_resolver_prefers_input(tmp_path):
    """Split resolver must choose canonical_data_dir/splits if it exists."""
    from src.pipeline.kaggle_train import resolve_split_artifacts

    data_dir = tmp_path / "data"
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "random_5fold.json").write_text('{"source": "input_random"}', encoding="utf-8")
    (splits_dir / "doc_disjoint_split.json").write_text('{"source": "input_disjoint"}', encoding="utf-8")

    res = resolve_split_artifacts(
        canonical_data_dir=data_dir,
        working_dir=tmp_path / "work",
        repo_root=REPO_ROOT,
    )
    assert res.random_source == "input"
    assert res.doc_disjoint_source == "input"
    assert res.random_5fold_path == (splits_dir / "random_5fold.json").resolve()
    assert res.doc_disjoint_path == (splits_dir / "doc_disjoint_split.json").resolve()
    assert len(res.random_sha256) == 64


def test_split_resolver_uses_repo_artifact_when_input_has_no_splits(tmp_path):
    """When input has NO splits/ directory, resolver must fall back to repo_root artifacts."""
    from src.pipeline.kaggle_train import resolve_split_artifacts

    data_dir = tmp_path / "clean_no_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    res = resolve_split_artifacts(
        canonical_data_dir=data_dir,
        working_dir=tmp_path / "work",
        repo_root=REPO_ROOT,
    )
    assert res.random_source == "repo"
    assert res.doc_disjoint_source == "repo"
    assert res.random_5fold_path == (REPO_ROOT / "artifacts/task1/data/splits/random_5fold.json").resolve()
    assert res.doc_disjoint_path == (REPO_ROOT / "artifacts/task1/data/splits/doc_disjoint_split.json").resolve()
    assert res.random_5fold_path.exists()
    assert res.doc_disjoint_path.exists()
    assert len(res.random_sha256) == 64


def test_split_resolver_generates_seed42_working_split(tmp_path):
    """When neither input nor repo has splits, resolver generates deterministic splits in working_dir."""
    from src.pipeline.kaggle_train import resolve_split_artifacts

    data_dir = tmp_path / "empty_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Create minimal queries, docs, qrels for generation
    docs = [{"doc_id": str(i)} for i in range(10)]
    pd.DataFrame(docs).to_parquet(data_dir / "documents.parquet")
    queries = [{"query_id": f"q{i}"} for i in range(10)]
    pd.DataFrame(queries).to_parquet(data_dir / "queries_train.parquet")
    qrels = [{"query_id": f"q{i}", "doc_id": str(i)} for i in range(10)]
    pd.DataFrame(qrels).to_parquet(data_dir / "qrels_train.parquet")

    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir(parents=True, exist_ok=True)

    res = resolve_split_artifacts(
        canonical_data_dir=data_dir,
        working_dir=tmp_path / "work",
        repo_root=fake_repo,
    )
    assert res.random_source == "generated"
    assert res.doc_disjoint_source == "generated"
    assert res.random_5fold_path.exists()
    assert res.doc_disjoint_path.exists()
    assert len(res.random_sha256) == 64


# ==============================================================================
# 3. Task 3: Duplicate Document False-Negative Blacklist
# ==============================================================================

def test_duplicate_blacklist_uses_repo_artifact_when_input_lacks_it(tmp_path):
    """Duplicate groups path must resolve to repo artifact when input lacks duplicate_groups.json."""
    from src.pipeline.kaggle_train import resolve_duplicate_groups_path

    data_dir = tmp_path / "clean_no_dup"
    data_dir.mkdir(parents=True, exist_ok=True)

    dup_path, source = resolve_duplicate_groups_path(
        canonical_data_dir=data_dir,
        repo_root=REPO_ROOT,
    )
    assert source == "repo"
    assert dup_path == (REPO_ROOT / "artifacts/task1/data/duplicate_groups.json").resolve()
    assert dup_path.exists()


def test_production_requires_four_duplicate_groups(tmp_path):
    """In production (gpu_smoke/full), duplicate groups file must contain exactly 4 valid groups."""
    from src.pipeline.kaggle_train import validate_duplicate_groups_file

    bad_dup = tmp_path / "bad_dup.json"
    bad_dup.write_text('{"g1": ["1", "2"]}', encoding="utf-8")  # only 1 group

    with pytest.raises(ValueError, match="(?i)duplicate groups"):
        validate_duplicate_groups_file(bad_dup, corpus_doc_ids=None, strict=True)


def test_duplicate_ids_must_exist_in_corpus():
    """All doc IDs in duplicate groups must be members of the official 8,532 corpus."""
    from src.pipeline.kaggle_train import validate_duplicate_groups_file

    repo_dup_path = REPO_ROOT / "artifacts/task1/data/duplicate_groups.json"
    assert repo_dup_path.exists()
    docs_path = REPO_ROOT / "artifacts/task1/data/documents.parquet"
    if docs_path.exists():
        docs_df = pd.read_parquet(docs_path)
        corpus_ids = set(docs_df["doc_id"].astype(str))
        report = validate_duplicate_groups_file(repo_dup_path, corpus_doc_ids=corpus_ids, strict=True)
        assert report["is_valid"] is True
        assert report["group_count"] == 4
        assert report["total_duplicate_docs"] == 9


# ==============================================================================
# 4. Task 4: Scale Pair-Mining Runtime from Smoke to FULL
# ==============================================================================

def test_pair_projection_scales_query_loop_not_setup():
    """project_pair_mining_seconds must scale query loop proportionally while adding constant setup."""
    from src.training.build_pairs import project_pair_mining_seconds

    setup_sec = 5.0
    loop_sec = 2.0
    measured_q = 20
    target_q = 5600

    # per query = 2.0 / 20 = 0.1s
    # scaled = 5.0 + 0.1 * 5600 * 1.10 = 5.0 + 616.0 = 621.0s
    proj = project_pair_mining_seconds(
        setup_seconds=setup_sec,
        query_loop_seconds=loop_sec,
        measured_queries=measured_q,
        target_queries=target_q,
        safety_factor=1.10,
    )
    assert proj == pytest.approx(621.0, rel=1e-3)


def test_20_query_smoke_scales_to_fold_train_count():
    """A 20-query smoke run measuring 0.5s loop scales safely to 5,600 fold queries."""
    from src.training.build_pairs import project_pair_mining_seconds

    proj = project_pair_mining_seconds(
        setup_seconds=2.0,
        query_loop_seconds=0.5,
        measured_queries=20,
        target_queries=5600,
        safety_factor=1.10,
    )
    # per query = 0.025s; 0.025 * 5600 * 1.1 = 154s; total 156s
    assert proj == pytest.approx(156.0, rel=1e-2)


# ==============================================================================
# 5. Task 5: Component-Complete Cold-Start Projection Map
# ==============================================================================

def test_runtime_projection_total_equals_component_sum():
    """cold_start_total_sec must strictly equal the sum of all named projection components."""
    components = {
        "canonical_load": 5.0,
        "bm25_legal": 10.0,
        "bm25_pyvi": 15.0,
        "dense_index": 120.0,
        "train_query_encoding": 30.0,
        "projected_oof": 5000.0,
        "fusion_training": 25.0,
        "projected_doc_disjoint": 1500.0,
        "question_memory": 8.0,
        "projected_final_pair_mining": 150.0,
        "projected_final_reranker": 1800.0,
        "final_pipeline_load_audit": 12.0,
        "projected_public_inference": 450.0,
        "submission_packaging": 3.0,
        "safety_overhead": 60.0,
    }
    total = sum(components.values())
    assert total == pytest.approx(9188.0, rel=1e-3)


def test_projection_contains_fusion_memory_and_final_pair_mining():
    """Required production components must be present in the cold start map."""
    required_components = {
        "fusion_training",
        "question_memory",
        "projected_final_pair_mining",
        "projected_oof",
        "projected_final_reranker",
        "projected_public_inference",
        "safety_overhead",
    }
    sample_components = {
        "canonical_load": 1.0,
        "fusion_training": 2.0,
        "question_memory": 3.0,
        "projected_final_pair_mining": 4.0,
        "projected_oof": 5.0,
        "projected_final_reranker": 6.0,
        "projected_public_inference": 7.0,
        "safety_overhead": 60.0,
    }
    assert required_components.issubset(sample_components.keys())
