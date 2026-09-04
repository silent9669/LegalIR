# LegalIR Task 1 Resumable Artifact Factory & Kaggle Final Trainer Design

## 1. Executive Summary & Goals

This design synthesizes the authoritative specification from `docs/new_spec_legalir/` into the canonical architecture for silent9669/LegalIR.
The legacy monolithic Kaggle FULL pipeline (which built indexes, trained 5 OOF folds, ran doc-disjoint validation, and performed final training and inference in a single session) exceeded Kaggle host RAM and lacked fault tolerance.

The new architecture splits the workload into two strictly decoupled systems:
1. **System A — Validation / Artifact Factory:**
   - Inherits Task 1 canonical v2 dataset unchanged (8,532 docs, 1,153,876 chunks, 7,000 train queries, 7,637 qrels, 1,000 public queries, 4 duplicate groups).
   - Computes static label-free retrieval (Legal BM25, PyVi BM25, DEk21 Dense, Exact Matcher) once across all 8,000 queries (7k train + 1k public) and caches candidates without qrels.
   - Releases heavy retrieval memory (BM25, PyVi, Dense transformers, FAISS/embeddings) before pair materialization.
   - Operates an Arrow-backed lazy `MacroEvidenceStore` with bounded LRU cache (<=512 MB, 256-512 docs) to replace full in-memory macro corpus preprocessing.
   - Keeps Question Memory strictly fold-local (train queries only; validation queries strictly forbidden).
   - Executes 5-fold OOF and document-disjoint validation as independent, resumable, isolated subprocess jobs.
   - Trains and evaluates fusion ranking based on OOF predictions, freezes `production_lock.json`.
   - Freezes an immutable, verified production bundle containing static cache, final 7k training pairs, public candidates, public evidence, and fusion weights.
2. **System B — Kaggle Final Train + Submit:**
   - Verifies exact approved runtime commit SHA, canonical dataset fingerprints, and production bundle manifest hashes.
   - Loads verified final training pairs and trains exactly one final all-7,000-query BGE LoRA reranker.
   - Verifies adapter integrity (param diff > 0, finite loss, PEFT reload check, <4B param budget).
   - Reranks pre-computed public candidates using public evidence (supporting 2x GPU batching on T4x2).
   - Applies approved locked fusion & top-5 selection.
   - Validates strict submission invariants (exact 1,000 public query keyset, 1-5 unique valid document IDs per query) and creates `submission.zip`.

---

## 2. Invariants & Constraints

- **Dataset Read-Only Invariant:** Canonical Parquets (`data/task1_canonical_v2/`) are read-only; no label modifications, no external data augmentation, no Task 2 mixing.
- **Scoring Invariant:** Official evaluation semantics remain Recall@5 as primary metric, Precision@5 as tie-break. Max 5 docs predicted per query.
- **Leakage Invariant:** Validation queries must never enter fold training pairs or fold-local Question Memory (`pair_qids ⊆ train_ids`, `pair_qids ∩ val_ids = ∅`). Duplicate closure prevents any negative document from matching any gold document.
- **Parameter Invariant:** Total system learned parameters must strictly remain < 4 Billion (monitored by parameter auditor).
- **RAM & Resource Guards:** Peak RSS <= 70% physical RAM; available RAM >= 3 GiB at stage boundaries. Subprocess isolation ensures clean OS memory reclamation upon fold exit.
- **Batch Size Invariant:** Effective reranker batch size must remain 16 (e.g. microbatch 8 with gradient accumulation 2, or 4 with accumulation 4, or 2 with accumulation 8 on T4).
- **Release Invariant:** 2-commit release protocol (Commit A = approved runtime; Colab T4 PASS; Commit B = release notebooks/bundle pins; CI green). Final Kaggle execution strictly pins Commit A.

---

## 3. System Architecture & Components

```text
CANONICAL TASK1 V2
        │
        ├─────────────────────────────┐
        ▼                             ▼
STATIC RETRIEVAL CACHE           SPLIT METADATA
Legal BM25 / PyVi / DEk21 /      5-fold CV
Exact Matcher                    Doc-disjoint
        │                             │
        ▼                             │
STATIC CANDIDATE PARQUET              │
(all 7k train + 1k public)            │
        │                             │
        ├─────────────────────────────┘
        ▼
FOLD PAIR MATERIALIZATION
Static cache + fold-local Question Memory + Lazy MacroEvidenceStore + Duplicate Blacklist
        │
        ▼
SHARDED VALIDATION JOBS (Process-Isolated)
Fold 0..4 OOF Jobs + Doc-Disjoint Job
        │
        ▼
OOF FEATURE PARQUETS & METRICS
        │
        ▼
FUSION & PRODUCTION LOCK
Ranker feature evaluation -> production_lock.json
        │
        ▼
PRODUCTION BUNDLE GENERATION & VERIFICATION
final_pairs.parquet + public_candidates.parquet + public_evidence.parquet + bundle_manifest.json
        │
        ▼
KAGGLE FINAL TRAINER
Verify Bundle -> Train Final BGE LoRA (all 7k) -> Rerank Public Candidates -> Top5 -> submission.zip
```

### 3.1. Core & Data Layer (`src/core/`, `src/data/`)
- `src/core/hashing.py`: SHA-256 calculation for files, directories, and dataframes.
- `src/core/manifests.py`: Dataclasses and JSON serialization for preflight, job, bundle, and release manifests.
- `src/core/memory.py`: Telemetry recording (RSS, system available/total, GPU VRAM) and proactive memory guards.
- `src/data/canonical.py`: Fast metadata-only inspection and full verification of the canonical v2 dataset.
- `src/data/splits.py`: Split loaders ensuring deterministic fold assignments and disjoint splits.
- `src/data/duplicate_groups.py`: Transitive closure over duplicate document groups.

### 3.2. Retrieval & Static Cache (`src/retrieval/`)
- `src/retrieval/static_cache.py`:
  - Static candidate streaming and normalized Parquet storage (`query_id`, `branch`, `rank`, `doc_id`, `score`, `best_chunk_id`, etc.).
  - Cache builder takes zero qrels.
  - Reader loads cached branch scores and candidates without loading the underlying indexes.
- `src/retrieval/bm25_legal.py`, `bm25_pyvi.py`, `dense.py`, `exact.py`:
  - Specialized retrieval branches with explicit `unload()` methods.
  - Dense retriever supports dropping the raw embedding matrix after FAISS index construction while retaining count metadata.
- `src/retrieval/question_memory.py`:
  - Fold-local training query memory index.

### 3.3. Evidence & Pair Materialization (`src/evidence/`)
- `src/evidence/macro_store.py`:
  - `MacroEvidenceStore` backed by PyArrow table with compact doc_id -> row_indices index.
  - Lazy per-document text extraction and cleaning with bounded LRU cache (capped at 512MB / 512 docs).
- `src/evidence/selector.py`:
  - `PositiveLocalizer` and `EvidencePackBuilder` querying `MacroEvidenceStore` lazily.
- `src/evidence/pair_materializer.py`:
  - Streams training pairs combining static candidate cache, fold-local question memory, lazy positive/negative evidence, and duplicate blacklist.

### 3.4. Validation Sharding & Fusion (`src/validation/`)
- `src/validation/fold_job.py`:
  - Standalone process executing a single fold: pair verification, BGE LoRA training, validation candidate reranking, OOF feature extraction, and metric calculation.
  - Resumable: skips execution if existing `job_manifest.json` verifies hashes and status is PASS.
- `src/validation/doc_disjoint_job.py`:
  - Standalone validation on the document-disjoint split.
- `src/validation/promotion.py`:
  - OOF feature aggregator and fusion model trainer.
  - Generates `production_lock.json` ensuring Recall@5 improves or ties with higher Precision@5.

### 3.5. Production Bundle & Final Runner (`src/bundle/`, `src/production/`)
- `src/bundle/builder.py` & `verifier.py`:
  - Assembles and cryptographically validates the immutable production bundle.
- `src/production/final_train.py`:
  - Trains final BGE LoRA on all 7,000 train queries using effective batch size 16.
- `src/production/public_rerank.py`:
  - Parallel or batched reranking of 1,000 public queries against pre-materialized evidence.
- `src/production/submission.py`:
  - Formats, validates keyset/bounds, and zips `submission.json`.

---

## 4. Execution & Release Gates

1. **Pre-flight & CI Gate:**
   - `python -m compileall -q src scripts`
   - `pytest -q` (Unit, parity, leakage, memory, release tests)
   - `python scripts/audit_parameters.py` (<4B params)
2. **Factory Validation Gate:**
   - All 5 OOF folds PASS + Doc-disjoint PASS.
   - Leakage counters zero.
   - `production_lock.json` generated.
   - Production bundle built and verified.
3. **Colab T4 Hardware Gate:**
   - Approved commit tested on Tesla T4 with real DEk21, FAISS, BGE LoRA training probe, adapter reload, and public reranking path.
4. **Kaggle Final Gate:**
   - Verification of Commit A and bundle fingerprints.
   - Final all-7k training + public rerank + packaging (< 9h budget).
