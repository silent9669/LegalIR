"""Shared index cache identity: match reuses, drift rebuilds, never silent."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.kaggle_train import (
    INDEX_CACHE_IDENTITY_FILE,
    _index_identity_mismatch,
    _read_index_cache_identity,
    build_index_cache_identity,
    persist_index_cache,
    warm_index_cache,
)


def _shared(monkeypatch, tmp_path) -> Path:
    shared = tmp_path / "shared-indexes"
    shared.mkdir()
    monkeypatch.setenv("LEGALIR_INDEX_CACHE_DIR", str(shared))
    return shared


def test_identity_build_has_pinned_revisions():
    ident = build_index_cache_identity()
    assert ident["schema"] == "v1"
    assert ident["dense_model"] == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    assert ident["dense_revision"]
    assert ident["reranker_revision"]
    assert ident["dataset_manifest_sha256"] is None


def test_identity_reads_dataset_manifest(tmp_path):
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"manifest_sha256": "abc123"}), encoding="utf-8")
    assert build_index_cache_identity(tmp_path)["dataset_manifest_sha256"] == "abc123"


def test_mismatch_detects_changed_keys():
    base = build_index_cache_identity()
    assert _index_identity_mismatch(dict(base), dict(base)) == []
    other = dict(base, dense_revision="STALE")
    assert _index_identity_mismatch(other, base) == ["dense_revision"]
    assert _read_index_cache_identity("/nonexistent-dir-xyz") is None


def test_warm_copies_on_match_and_skips_on_drift(tmp_path, monkeypatch, capsys):
    shared = _shared(monkeypatch, tmp_path)
    (shared / "bm25").mkdir()
    (shared / "bm25" / "index.pkl").write_bytes(b"x")
    ident = build_index_cache_identity()
    (shared / INDEX_CACHE_IDENTITY_FILE).write_text(json.dumps(ident), encoding="utf-8")

    target = tmp_path / "attempt" / "indexes"
    assert warm_index_cache(target, dict(ident)) is True
    assert (target / "bm25" / "index.pkl").is_file()

    target2 = tmp_path / "attempt2" / "indexes"
    stale = dict(ident, dataset_manifest_sha256="different")
    assert warm_index_cache(target2, stale) is False
    assert not (target2 / "bm25").exists()
    assert "drift" in capsys.readouterr().out


def test_warm_rejects_copy_without_identity_file_when_expected_given(tmp_path, monkeypatch, capsys):
    shared = _shared(monkeypatch, tmp_path)
    (shared / "bm25").mkdir()
    (shared / "bm25" / "index.pkl").write_bytes(b"x")
    target = tmp_path / "attempt" / "indexes"
    assert warm_index_cache(target, build_index_cache_identity()) is False
    assert not (target / "bm25").exists()
    assert "no identity file" in capsys.readouterr().out


def test_warm_without_identity_file_rejects_even_when_expected_omitted(tmp_path, monkeypatch):
    shared = _shared(monkeypatch, tmp_path)
    (shared / "bm25").mkdir()
    (shared / "bm25" / "index.pkl").write_bytes(b"x")
    target = tmp_path / "attempt" / "indexes"
    assert warm_index_cache(target) is False
    assert not (target / "bm25").exists()


def test_persist_writes_identity_and_replaces_stale_files(tmp_path, monkeypatch):
    shared = _shared(monkeypatch, tmp_path)
    (shared / "bm25").mkdir()
    (shared / "bm25" / "index.pkl").write_bytes(b"old_stale")
    src = tmp_path / "attempt" / "indexes"
    (src / "bm25").mkdir(parents=True)
    (src / "bm25" / "index.pkl").write_bytes(b"fresh_content")
    persist_index_cache(src, build_index_cache_identity())
    stored = _read_index_cache_identity(shared)
    assert stored is not None and stored["schema"] == "v1"
    assert (shared / "bm25" / "index.pkl").read_bytes() == b"fresh_content"


def test_persist_purges_stale_absent_entries_from_shared(tmp_path, monkeypatch):
    shared = _shared(monkeypatch, tmp_path)
    # Pre-populate shared with stale bm25_pyvi
    (shared / "bm25_pyvi").mkdir()
    (shared / "bm25_pyvi" / "old.idx").write_bytes(b"stale_pyvi")
    # Fresh attempt only has bm25
    src = tmp_path / "attempt" / "indexes"
    (src / "bm25").mkdir(parents=True)
    (src / "bm25" / "index.pkl").write_bytes(b"fresh_bm25")
    ident = build_index_cache_identity()
    persist_index_cache(src, ident)

    # Shared cache must NOT retain stale bm25_pyvi
    assert not (shared / "bm25_pyvi").exists()

    # Next attempt warms from shared: only bm25 is copied
    next_attempt = tmp_path / "next_attempt" / "indexes"
    assert warm_index_cache(next_attempt, ident) is True
    assert (next_attempt / "bm25" / "index.pkl").read_bytes() == b"fresh_bm25"
    assert not (next_attempt / "bm25_pyvi").exists()
