# LegalIR Task 1 Resumable Artifact Factory & Kaggle Final Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh, modular, resumable Validation / Artifact Factory and minimal Kaggle Final Trainer for DSC 2026 Task 1 LegalIR that inherits the canonical v2 dataset unchanged, guarantees zero validation leakage, prevents host-RAM exhaustion, and strictly respects the <4B parameter budget and Recall@5 ranking semantics.

**Architecture:** Static label-free retrieval (Legal BM25, PyVi BM25, DEk21 Dense, Exact Matcher) is computed once for all 8,000 queries (7k train + 1k public) and cached to normalized Parquet without qrels. Heavy retrieval models are unloaded. Evidence is served lazily via an Arrow-backed `MacroEvidenceStore` with bounded LRU cache (<=512 MB). 5-fold OOF and document-disjoint validation run as process-isolated resumable jobs. An immutable production bundle is verified and packaged. Kaggle executes only final all-7k BGE LoRA training and public reranking.

**Tech Stack:** Python 3.12, PyArrow, Pandas, NumPy, PyTorch, Transformers, PEFT, FAISS, LightGBM, pytest, GitHub Actions, Google Colab T4, Kaggle T4×2.

**Spec:** `docs/superpowers/specs/2026-09-04-task1-resumable-factory-pipeline-design.md` and authoritative documents in `docs/new_spec_legalir/`.

## Global Constraints

- Task 1 canonical v2 dataset only: 8,532 documents, 1,153,876 chunks (934,416 micro, 219,460 macro), 7,000 train queries, 7,637 qrels, 1,000 public queries, 4 duplicate groups.
- Canonical dataset files in `data/task1_canonical_v2/` are strictly read-only.
- Total system learned parameters must strictly remain < 4 Billion.
- Recall@5 is primary evaluation metric; Precision@5 is secondary tie-break. Maximum 5 predicted document IDs per query.
- Question Memory must be strictly fold-local (`pair_qids ⊆ train_ids`, `pair_qids ∩ val_ids = ∅`).
- Duplicate closure blacklist enforced for negative sampling across all training pairs.
- Effective reranker training batch size is fixed to 16.
- Kaggle final execution must never run BM25/PyVi/Dense indexing, 5-fold OOF, doc-disjoint validation, or hyperparameter search.
- Two-commit release protocol enforced: Commit A (runtime code), Colab T4 PASS, Commit B (release pins & notebooks), CI green.

---

### Task 1: Canonical Source Adapter, Hashing, and Manifests

**Files:**
- Create: `src/core/hashing.py`
- Create: `src/core/manifests.py`
- Create: `src/data/canonical.py`
- Test: `tests/unit/test_canonical.py`

**Interfaces:**
- Consumes: Raw filesystem files in `data/task1_canonical_v2/`.
- Produces:
  - `sha256_file(path: str) -> str`
  - `sha256_directory(path: str) -> str`
  - `sha256_dataframe(df: pd.DataFrame) -> str`
  - `CanonicalDatasetIdentity` dataclass
  - `verify_canonical_dataset(dataset_dir: str) -> CanonicalDatasetIdentity`
  - `Manifest`, `PreflightManifest`, `JobManifest`, `BundleManifest` dataclasses with `.to_json()` / `.from_json()`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_canonical.py
import pytest
from src.core.hashing import sha256_file
from src.core.manifests import PreflightManifest
from src.data.canonical import CanonicalDatasetIdentity, verify_canonical_dataset

def test_hashing_basic(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("legalir", encoding="utf-8")
    expected = "8c679a1f28b7e2a9e0f6396dc79b4d5386f68c74070a25925e0129a08e6baee0"
    assert sha256_file(str(f)) == expected

def test_canonical_identity_dataclass():
    ident = CanonicalDatasetIdentity(
        dataset_name="task1_canonical",
        version="v2",
        schema_version="hierarchical_micro_macro_v2",
        num_docs=8532,
        num_chunks=1153876,
        num_micro=934416,
        num_macro=219460,
        num_train_queries=7000,
        num_qrels=7637,
        num_public_queries=1000,
        num_duplicate_groups=4
    )
    assert ident.num_docs == 8532
    assert ident.num_train_queries == 7000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_canonical.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core'`

- [ ] **Step 3: Write minimal implementation**

Implement `src/core/hashing.py`, `src/core/manifests.py`, and `src/data/canonical.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_canonical.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/hashing.py src/core/manifests.py src/data/canonical.py tests/unit/test_canonical.py
git commit -m "feat(core): implement canonical source adapter, hashing, and manifests"
```

---

### Task 2: Host Memory Telemetry and RAM Guards

**Files:**
- Create: `src/core/memory.py`
- Test: `tests/memory/test_memory_guard.py`

**Interfaces:**
- Consumes: OS memory metrics via `psutil` / `/proc/meminfo`, PyTorch CUDA memory.
- Produces:
  - `MemorySnapshot` dataclass
  - `take_memory_snapshot() -> MemorySnapshot`
  - `release_memory()` (runs `gc.collect()`, `torch.cuda.empty_cache()`, and `ctypes` `malloc_trim` on Linux)
  - `check_memory_guard(min_available_gb: float = 3.0, max_rss_fraction: float = 0.70) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_memory_guard.py
import pytest
from src.core.memory import take_memory_snapshot, release_memory, check_memory_guard

def test_memory_snapshot():
    snap = take_memory_snapshot()
    assert snap.rss_bytes > 0
    assert snap.system_total_bytes > 0
    assert snap.system_available_bytes > 0

def test_memory_guard_passes_normal():
    check_memory_guard(min_available_gb=0.1, max_rss_fraction=0.99)

def test_memory_guard_raises_on_exhaustion(monkeypatch):
    from src.core import memory
    fake_snap = memory.MemorySnapshot(
        rss_bytes=100 * 1024**3,
        system_total_bytes=16 * 1024**3,
        system_available_bytes=500 * 1024**2, # 500MB
        system_used_bytes=15 * 1024**3,
        gpu_allocated_bytes=0,
        gpu_reserved_bytes=0
    )
    monkeypatch.setattr(memory, "take_memory_snapshot", lambda: fake_snap)
    with pytest.raises(MemoryError):
        check_memory_guard(min_available_gb=3.0, max_rss_fraction=0.70)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/memory/test_memory_guard.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/core/memory.py` with cross-platform memory stats and fallback handling.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/memory/test_memory_guard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/memory.py tests/memory/test_memory_guard.py
git commit -m "feat(core): implement host memory telemetry and RAM guard"
```

---

### Task 3: Static Candidate Cache and Branch Aggregation

**Files:**
- Create: `src/retrieval/static_cache.py`
- Create: `scripts/build_static_cache.py`
- Test: `tests/parity/test_static_cache.py`

**Interfaces:**
- Consumes: Branch search outputs (Legal BM25, PyVi BM25, DEk21 Dense, Exact Matcher).
- Produces:
  - `StaticCandidateRecord` / normalized schema: `(query_id, branch, rank, doc_id, score, best_chunk_id, second_score, mean_score, extra_json)`
  - `StaticCacheWriter(output_path: str)` streaming Parquet writer in batches of 2,000–5,000 rows.
  - `StaticCacheReader(cache_path: str)` for retrieving query candidates and branch scores without raw indexes.
  - No qrels accepted or accessed anywhere in `StaticCacheWriter`.

- [ ] **Step 1: Write the failing test**

```python
# tests/parity/test_static_cache.py
import pytest
import pyarrow.parquet as pq
from src.retrieval/static_cache import StaticCacheWriter, StaticCacheReader, StaticCandidateRecord

def test_static_cache_roundtrip(tmp_path):
    cache_file = str(tmp_path / "static_candidates.parquet")
    writer = StaticCacheWriter(cache_file, batch_size=2)
    records = [
        StaticCandidateRecord(query_id="q1", branch="bm25_legal", rank=1, doc_id="d1", score=10.5),
        StaticCandidateRecord(query_id="q1", branch="dense", rank=1, doc_id="d2", score=0.85),
        StaticCandidateRecord(query_id="q2", branch="exact", rank=1, doc_id="d3", score=1.0),
    ]
    for r in records:
        writer.add_record(r)
    writer.close()

    reader = StaticCacheReader(cache_file)
    q1_cands = reader.get_query_candidates("q1")
    assert len(q1_cands) == 2
    assert q1_cands[0].doc_id == "d1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/parity/test_static_cache.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/retrieval/static_cache.py` and `scripts/build_static_cache.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/parity/test_static_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/static_cache.py scripts/build_static_cache.py tests/parity/test_static_cache.py
git commit -m "feat(retrieval): implement static candidate cache writer and reader"
```

---

### Task 4: Dense Lifecycle Management and Memory Unload

**Files:**
- Modify/Create: `src/retrieval/dense.py`
- Test: `tests/parity/test_dense_lifecycle.py`

**Interfaces:**
- Consumes: DEk21 pretrained model, FAISS index.
- Produces:
  - `DenseRetriever.unload()` method releasing model, tokenizer, and CUDA buffers.
  - `DenseRetriever.drop_corpus_matrix()` releasing float32 numpy embedding matrix once FAISS index is built.
  - Metadata preservation (`doc_ids`, `chunk_ids`, embedding counts) even after matrix release.

- [ ] **Step 1: Write the failing test**

```python
# tests/parity/test_dense_lifecycle.py
import pytest
import numpy as np
from src.retrieval.dense import DenseIndexManager

def test_dense_matrix_drop_retains_metadata():
    mgr = DenseIndexManager()
    dummy_matrix = np.random.randn(10, 64).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(10)]
    mgr.load_embeddings(dummy_matrix, doc_ids)
    assert mgr.has_matrix() is True
    mgr.build_faiss()
    mgr.drop_corpus_matrix()
    assert mgr.has_matrix() is False
    assert mgr.num_docs == 10
    assert mgr.get_doc_id(0) == "doc_0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/parity/test_dense_lifecycle.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `DenseIndexManager` in `src/retrieval/dense.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/parity/test_dense_lifecycle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/dense.py tests/parity/test_dense_lifecycle.py
git commit -m "feat(retrieval): implement dense lifecycle management and matrix drop"
```

---

### Task 5: Lazy Arrow-Backed MacroEvidenceStore with Bounded LRU

**Files:**
- Create: `src/evidence/macro_store.py`
- Test: `tests/parity/test_macro_store.py`

**Interfaces:**
- Consumes: Canonical `chunks.parquet` (macro chunks only) via PyArrow dataset / memory mapping.
- Produces:
  - `MacroChunk` dataclass
  - `MacroEvidenceStore(chunks_path: str, max_cache_bytes: int = 512 * 1024 * 1024, max_cached_docs: int = 512)`
  - `get_doc_chunks(doc_id: str) -> list[MacroChunk]`
  - `get_preprocessed_doc(doc_id: str) -> PreprocessedDoc`
  - `cache_bytes() -> int`
  - `clear_cache() -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/parity/test_macro_store.py
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from src.evidence.macro_store import MacroEvidenceStore

def test_macro_store_lazy_loading(tmp_path):
    p = tmp_path / "chunks.parquet"
    table = pa.Table.from_pydict({
        "doc_id": ["d1", "d1", "d2"],
        "chunk_id": ["c1", "c2", "c3"],
        "chunk_type": ["macro", "macro", "macro"],
        "text": ["Article 1 text", "Article 2 text", "Doc 2 text"]
    })
    pq.write_table(table, str(p))

    store = MacroEvidenceStore(str(p), max_cache_bytes=1024*1024, max_cached_docs=2)
    chunks_d1 = store.get_doc_chunks("d1")
    assert len(chunks_d1) == 2
    assert chunks_d1[0].chunk_id == "c1"
    assert store.cache_bytes() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/parity/test_macro_store.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/evidence/macro_store.py` with PyArrow table indexing and `collections.OrderedDict` LRU cache.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/parity/test_macro_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evidence/macro_store.py tests/parity/test_macro_store.py
git commit -m "feat(evidence): implement lazy Arrow-backed MacroEvidenceStore with bounded LRU"
```

---

### Task 6: Lazy Positive Localization and Evidence Pack Builder

**Files:**
- Create: `src/evidence/selector.py`
- Test: `tests/parity/test_evidence_parity.py`

**Interfaces:**
- Consumes: `MacroEvidenceStore`, query string, candidate document ID.
- Produces:
  - `LazyPositiveLocalizer(evidence_store: MacroEvidenceStore)`
  - `localize_positive(query_text: str, gold_doc_id: str) -> tuple[str, str]` (best_chunk_id, chunk_text)
  - `LazyEvidencePackBuilder(evidence_store: MacroEvidenceStore, max_tokens: int = 400)`
  - `build_evidence(query_text: str, doc_id: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/parity/test_evidence_parity.py
import pytest
from src.evidence.macro_store import MacroEvidenceStore
from src.evidence.selector import LazyPositiveLocalizer, LazyEvidencePackBuilder

def test_selector_with_store(tmp_path):
    # Setup minimal macro store
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/parity/test_evidence_parity.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/evidence/selector.py` matching legacy scoring semantics faithfully without full-corpus memory overhead.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/parity/test_evidence_parity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evidence/selector.py tests/parity/test_evidence_parity.py
git commit -m "feat(evidence): implement lazy positive localizer and evidence pack builder"
```

---

### Task 7: Cached Pair Materializer with Duplicate Blacklist & Leakage Guard

**Files:**
- Create: `src/evidence/pair_materializer.py`
- Create: `scripts/build_fold_pairs.py`
- Test: `tests/leakage/test_pair_materializer.py`

**Interfaces:**
- Consumes: `StaticCacheReader`, fold-local split IDs (`train_qids`, `val_qids`), duplicate groups, `MacroEvidenceStore`.
- Produces:
  - `PairMaterializer` writing `train_pairs.parquet` and `validation_candidates.parquet`.
  - Enforces:
    1. `pair_qids ⊆ train_qids`
    2. `pair_qids ∩ val_qids = ∅`
    3. Gold positives (and any duplicate-equivalent doc) strictly forbidden from negative candidates.

- [ ] **Step 1: Write the failing test**

```python
# tests/leakage/test_pair_materializer.py
import pytest
from src.evidence.pair_materializer import PairMaterializer

def test_leakage_assertion_on_val_qid():
    pm = PairMaterializer(train_qids={"q1"}, val_qids={"q2"}, duplicate_groups=[])
    with pytest.raises(ValueError, match="Validation leakage detected"):
        pm.validate_query_id("q2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/leakage/test_pair_materializer.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/evidence/pair_materializer.py` and `scripts/build_fold_pairs.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/leakage/test_pair_materializer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evidence/pair_materializer.py scripts/build_fold_pairs.py tests/leakage/test_pair_materializer.py
git commit -m "feat(evidence): implement pair materializer with strict leakage and duplicate guards"
```

---

### Task 8: Resumable Sharded Fold Job

**Files:**
- Create: `src/validation/fold_job.py`
- Create: `scripts/run_fold.py`
- Test: `tests/integration/test_fold_job.py`

**Interfaces:**
- Consumes: Fold ID (0..4), fold pairs parquet, validation candidates parquet, BGE LoRA config.
- Produces:
  - Adapter directory `artifacts/factory/folds/fold_{N}/adapter/`
  - `oof_features.parquet`
  - `fold_metrics.json`
  - `job_manifest.json` (with status PASS and SHA-256 hashes of all outputs)
  - Resumes instantly if `job_manifest.json` is PASS and files match hashes.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_fold_job.py
import pytest
from src.validation.fold_job import FoldJobRunner

def test_fold_job_skip_if_already_completed(tmp_path):
    runner = FoldJobRunner(fold_id=0, work_dir=str(tmp_path))
    # mock existing valid manifest
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/integration/test_fold_job.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/validation/fold_job.py` and `scripts/run_fold.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/integration/test_fold_job.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/validation/fold_job.py scripts/run_fold.py tests/integration/test_fold_job.py
git commit -m "feat(validation): implement resumable sharded fold job"
```

---

### Task 9: Document-Disjoint Validation Job

**Files:**
- Create: `src/validation/doc_disjoint_job.py`
- Create: `scripts/run_doc_disjoint.py`
- Test: `tests/integration/test_doc_disjoint.py`

**Interfaces:**
- Consumes: Doc-disjoint train/val query and doc split definitions.
- Produces:
  - Document-disjoint OOF report `artifacts/factory/doc_disjoint/metrics.json`
  - Validates zero document overlap between train and test sets.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_doc_disjoint.py
import pytest
from src.validation.doc_disjoint_job import DocDisjointRunner

def test_doc_disjoint_contract():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/integration/test_doc_disjoint.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/validation/doc_disjoint_job.py` and `scripts/run_doc_disjoint.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/integration/test_doc_disjoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/validation/doc_disjoint_job.py scripts/run_doc_disjoint.py tests/integration/test_doc_disjoint.py
git commit -m "feat(validation): implement document-disjoint validation job"
```

---

### Task 10: OOF Aggregation, Fusion Training, and Production Lock

**Files:**
- Create: `src/validation/promotion.py`
- Create: `scripts/select_production_config.py`
- Test: `tests/unit/test_promotion.py`

**Interfaces:**
- Consumes: 5 fold `oof_features.parquet` files.
- Produces:
  - Aggregate Recall@1, Recall@3, Recall@5, Precision@5 metrics.
  - LightGBM / RRF fusion weights.
  - `production_lock.json` with immutable hashes and promotion verdict.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_promotion.py
import pytest
from src.validation.promotion import compare_score_promotion

def test_promotion_requires_recall5_improvement():
    baseline = {"recall@5": 0.850, "precision@5": 0.300}
    candidate_worse = {"recall@5": 0.849, "precision@5": 0.350}
    assert compare_score_promotion(candidate_worse, baseline) is False

    candidate_better = {"recall@5": 0.852, "precision@5": 0.290}
    assert compare_score_promotion(candidate_better, baseline) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_promotion.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/validation/promotion.py` and `scripts/select_production_config.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_promotion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/validation/promotion.py scripts/select_production_config.py tests/unit/test_promotion.py
git commit -m "feat(validation): implement OOF aggregation, fusion selection, and production lock"
```

---

### Task 11: Production Bundle Materializer and Verifier

**Files:**
- Create: `src/bundle/builder.py`
- Create: `src/bundle/verifier.py`
- Create: `scripts/build_production_bundle.py`
- Create: `scripts/verify_production_bundle.py`
- Test: `tests/release/test_bundle.py`

**Interfaces:**
- Consumes: All-7k final pairs, public candidates, public evidence, `production_lock.json`.
- Produces:
  - Directory `artifacts/bundle/production/` containing:
    - `final_training_pairs.parquet`
    - `public_candidates.parquet`
    - `public_evidence.parquet`
    - `production_lock.json`
    - `bundle_manifest.json` (SHA-256 for each item)
  - Strict verification function `verify_production_bundle(bundle_dir: str) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/release/test_bundle.py
import pytest
from src.bundle.builder import build_production_bundle
from src.bundle.verifier import verify_production_bundle

def test_bundle_build_and_verify(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/release/test_bundle.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/bundle/builder.py`, `src/bundle/verifier.py`, and CLI verification scripts.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/release/test_bundle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bundle/builder.py src/bundle/verifier.py scripts/build_production_bundle.py scripts/verify_production_bundle.py tests/release/test_bundle.py
git commit -m "feat(bundle): implement production bundle builder and cryptographic verifier"
```

---

### Task 12: Final Kaggle Trainer, Public Reranker, and Submission Packager

**Files:**
- Create: `src/production/final_train.py`
- Create: `src/production/public_rerank.py`
- Create: `src/production/submission.py`
- Create: `scripts/run_kaggle_final.py`
- Test: `tests/integration/test_final_production.py`

**Interfaces:**
- Consumes: Production bundle, canonical dataset, Hugging Face models.
- Produces:
  - Trained final BGE LoRA adapter (trained on all 7,000 queries, effective batch 16).
  - Reranked top-5 predictions for exact 1,000 public queries.
  - `submission.json` and `submission.zip` satisfying all competition constraints.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_final_production.py
import pytest
from src.production.submission import validate_submission_format

def test_submission_format_exact_1000():
    valid_sub = {f"q_{i}": [f"doc_{j}" for j in range(5)] for i in range(1000)}
    assert validate_submission_format(valid_sub, expected_qids=set(valid_sub.keys())) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/integration/test_final_production.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `src/production/final_train.py`, `src/production/public_rerank.py`, `src/production/submission.py`, and `scripts/run_kaggle_final.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/integration/test_final_production.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/production/final_train.py src/production/public_rerank.py src/production/submission.py scripts/run_kaggle_final.py tests/integration/test_final_production.py
git commit -m "feat(production): implement final Kaggle trainer, public reranker, and submission packager"
```

---

### Task 13: T4 Hardware Probe and Microbatch Factorization

**Files:**
- Create: `src/training/samplers.py`
- Create: `scripts/probe_t4_throughput.py`
- Test: `tests/unit/test_throughput_probe.py`

**Interfaces:**
- Consumes: PyTorch CUDA device specs.
- Produces:
  - Adaptive factorization selection preserving effective batch 16:
    - 8 microbatch x 2 accumulation
    - 4 microbatch x 4 accumulation
    - 2 microbatch x 8 accumulation
  - Records step latency and peak VRAM to select stable configuration.

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

```bash
git add src/training/samplers.py scripts/probe_t4_throughput.py tests/unit/test_throughput_probe.py
git commit -m "feat(training): implement T4 throughput probe and microbatch factorization"
```

---

### Task 14: Release Gates, Verifier, and CI Automation

**Files:**
- Create: `scripts/verify_release.py`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/release/test_release_gate.py`

**Interfaces:**
- Consumes: Git repository state, `compileall`, pytest suite, parameter audit, release approval.
- Produces: Complete pass/fail verdict with exit code 0 on pass, non-zero on failure.

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

```bash
git add scripts/verify_release.py .github/workflows/ci.yml tests/release/test_release_gate.py
git commit -m "ci(release): implement verify_release script and update GitHub CI workflow"
```

---

### Task 15: Notebook Generators and Release Pin Invariants

**Files:**
- Create: `scripts/generate_colab_smoke_notebook.py`
- Create: `scripts/generate_kaggle_notebook.py`
- Create: `notebooks/colab_t4_smoke.ipynb`
- Create: `notebooks/kaggle_final.ipynb`
- Test: `tests/release/test_notebook_invariants.py`

**Interfaces:**
- Consumes: Approved runtime git commit SHA, bundle path, dataset path.
- Produces:
  - Synchronized self-contained notebooks pinning exact approved Commit A.
  - Zero hardcoded API secrets.
  - Strict parity check between script contracts and notebook execution blocks.

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

```bash
git add scripts/generate_colab_smoke_notebook.py scripts/generate_kaggle_notebook.py notebooks/ tests/release/test_notebook_invariants.py
git commit -m "feat(release): generate Colab T4 smoke and Kaggle final production notebooks"
```
