# Clean LegalIR High-Output Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cleanly restructure the LegalIR workspace to mirror Task 2 (`LegalQA - Public Test`), precompute DEk21 embeddings on MPS, run BGE cross-encoder reranking, and generate a high-recall submission.

**Architecture:** Standardized layout with `src/common/` (shared RAG core: normalize, legal_parser, bm25, dense_dek21, rrf, evidence, reranker) and `src/task1/` (memory, retrieve, rerank, selector, predict), driven by unified configs and scripts.

**Tech Stack:** Python 3.10+, PyTorch (MPS/CPU), Transformers, Sentence-Transformers, PyVi, Rank-BM25, PyArrow / Pandas, Scikit-learn.

**Spec:** `docs/superpowers/specs/2026-08-30-clean-legalir-optimization-design.md`

## Global Constraints

- Total learned parameters < 4.0B (~0.703B total for DEk21 + BGE Reranker v2 M3).
- Zero external legal corpus, synthetic data augmentation, or Task 2 data.
- Zero external API calls.
- Strict $1 \le \text{len}(\text{answer}) \le 5$ unique valid document IDs per query.
- Dual validation: 5-Fold CV and Document-Disjoint Split.

---

### Task 1: Clean Workspace Structure & Shared Common Modules

**Files:**
- Create: `src/common/__init__.py`
- Create: `src/common/normalize.py`
- Create: `src/common/legal_parser.py`
- Create: `src/common/rrf.py`
- Test: `tests/test_common_normalize.py`
- Test: `tests/test_common_legal_parser.py`

- [ ] **Step 1: Write tests for common normalize and legal parser**
- [ ] **Step 2: Implement `src/common/normalize.py`, `legal_parser.py`, `rrf.py`**
- [ ] **Step 3: Verify tests pass**
- [ ] **Step 4: Commit**

---

### Task 2: Fielded BM25 with Legal Entity Boosting

**Files:**
- Create: `src/common/bm25.py`
- Test: `tests/test_common_bm25.py`

- [ ] **Step 1: Write test for BM25 with legal number / article boosting**
- [ ] **Step 2: Implement `BM25Retriever` in `src/common/bm25.py`**
- [ ] **Step 3: Verify tests pass**
- [ ] **Step 4: Commit**

---

### Task 3: DEk21 Dense Macro Retriever Engine

**Files:**
- Create: `src/common/dense_dek21.py`
- Test: `tests/test_common_dense_and_rrf.py`

- [ ] **Step 1: Write test for DEk21Retriever with PyVi on MPS/CPU**
- [ ] **Step 2: Implement `DEk21Retriever` in `src/common/dense_dek21.py`**
- [ ] **Step 3: Verify tests pass**
- [ ] **Step 4: Commit**

---

### Task 4: Structured Evidence Packs & BGE Cross-Encoder Reranker

**Files:**
- Create: `src/common/evidence.py`
- Create: `src/common/reranker.py`
- Test: `tests/test_common_evidence.py`
- Test: `tests/test_common_reranker.py`

- [ ] **Step 1: Write tests for evidence pack builder and BGEReranker**
- [ ] **Step 2: Implement `EvidencePackBuilder` and `BGEReranker`**
- [ ] **Step 3: Verify tests pass**
- [ ] **Step 4: Commit**

---

### Task 5: Task 1 Modules & End-to-End LegalIR Pipeline

**Files:**
- Create: `src/task1/__init__.py`
- Create: `src/task1/memory.py`
- Create: `src/task1/retrieve.py`
- Create: `src/task1/rerank.py`
- Create: `src/task1/selector.py`
- Create: `src/task1/predict.py`
- Test: `tests/test_task1_end_to_end.py`

- [ ] **Step 1: Write test for Task 1 LegalIRPipeline end-to-end**
- [ ] **Step 2: Implement Task 1 memory, retrieve, rerank, selector, and predict modules**
- [ ] **Step 3: Verify tests pass**
- [ ] **Step 4: Commit**

---

### Task 6: Precompute Indexes & Run Dual Validation Benchmark

**Files:**
- Create: `scripts/01_build_dataset.py`
- Create: `scripts/02_build_indexes.py`
- Create: `scripts/03_run_benchmark.py`
- Test: `tests/test_benchmark_metrics.py`

- [ ] **Step 1: Implement dataset, indexing, and benchmark scripts**
- [ ] **Step 2: Execute benchmark on 5-fold CV & Document-Disjoint split**
- [ ] **Step 3: Verify benchmark reports**
- [ ] **Step 4: Commit**

---

### Task 7: Generate High-Score Submission & Verify Compliance

**Files:**
- Create: `scripts/04_predict_submission.py`
- Create: `scripts/audit_parameters.py`
- Output: `artifacts/task1/submissions/submission.json`, `submission.zip`

- [ ] **Step 1: Run full 4-branch pipeline with DEk21 + BGE Reranker on `public-official.json`**
- [ ] **Step 2: Validate all 1,000 query predictions and packaging invariants**
- [ ] **Step 3: Audit total system parameters (<4.0B)**
- [ ] **Step 4: Commit**
