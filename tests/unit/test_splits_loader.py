import pytest
from pathlib import Path
from src.data.splits import load_5fold_splits, load_doc_disjoint_split, FoldSplit, DocDisjointSplit


def test_load_5fold_splits_canonical():
    dataset_p = Path("data/task1_canonical_v2")
    if not dataset_p.is_dir():
        pytest.skip("Canonical dataset directory not present.")

    splits = load_5fold_splits(dataset_p)
    assert len(splits) == 5
    for idx, fs in enumerate(splits):
        assert fs.fold_id == idx
        assert len(fs.train_qids) == 5600
        assert len(fs.val_qids) == 1400
        assert fs.train_qids.isdisjoint(fs.val_qids)


def test_load_doc_disjoint_split_canonical():
    dataset_p = Path("data/task1_canonical_v2")
    if not dataset_p.is_dir():
        pytest.skip("Canonical dataset directory not present.")

    split = load_doc_disjoint_split(dataset_p)
    assert len(split.train_qids) == 5600
    assert len(split.val_qids) == 1400
    assert split.train_qids.isdisjoint(split.val_qids)
