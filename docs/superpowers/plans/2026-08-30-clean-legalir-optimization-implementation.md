# High-Output Clean LegalIR Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean and align the workspace structure matching Task 2 (LegalQA), precompute real DEk21 dense embeddings on MPS, run BGE cross-encoder reranking, and maximize Recall@5 on Codabench public test.

**Architecture:** Shared RAG core in `src/common/`, task-specific modules in `src/task1/`, 4-branch hybrid candidate retrieval (BM25 with legal entity boosting, DEk21 dense macro embeddings, question memory, exact matcher) fused via RRF ($k=60$) into candidate documents, reranked by `BAAI/bge-reranker-v2-m3` on structured multi-evidence packs, and selected into Top 5 unique document IDs.

**Tech Stack:** Python 3.10+, PyTorch (MPS/CPU), Hugging Face Transformers, Sentence-Transformers, PyVi, Rank-BM25 / BM25s, PyArrow / Pandas, Scikit-learn, PyTest.

**Spec:** `docs/superpowers/specs/2026-08-30-clean-legalir-optimization-design.md`

## Global Constraints

- Total learned parameters < 4.0B (~0.703B total for DEk21 + BGE Reranker).
- Strictly Task 1 official competition data only. Zero external legal corpus or external API calls.
- Every prediction answer list must have $1 \le \text{len}(\text{answer}) \le 5$ unique valid document IDs.
- Kaggle credential configured at `~/.kaggle/access_token`.
- All tests in `tests/` must pass cleanly.

---

### Task 1: Kaggle Token Configuration & Clean Config Setup

**Files:**
- Create: `~/.kaggle/access_token`
- Create: `configs/models.yaml`
- Create: `configs/task1.yaml`
- Modify: `configs/pipeline.yaml`
- Test: `tests/test_core_config.py`

- [ ] **Step 1: Configure Kaggle API Token**
Write token `KGAT_8ae9f76cd4bbb72f9ff6f0e5994f2344` to `~/.kaggle/access_token` with 600 permissions.

- [ ] **Step 2: Create `configs/models.yaml` and `configs/task1.yaml`**
Pin exact models (`CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` and `BAAI/bge-reranker-v2-m3`) and Task 1 parameters.

- [ ] **Step 3: Run test to verify config loading**
Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_core_config.py -v`

- [ ] **Step 4: Commit**
```bash
git add configs/ tests/test_core_config.py
git commit -m "feat(config): align configurations and models with Task 2 standards"
```

---

### Task 2: Shared Clean RAG Core in `src/common/`

**Files:**
- Create/Align: `src/common/normalize.py`
- Create/Align: `src/common/legal_parser.py`
- Create/Align: `src/common/bm25.py`
- Create/Align: `src/common/dense_dek21.py`
- Create/Align: `src/common/rrf.py`
- Create/Align: `src/common/evidence.py`
- Create/Align: `src/common/reranker.py`
- Test: `tests/test_common_normalize.py`, `tests/test_common_bm25.py`, `tests/test_common_dense_and_rrf.py`, `tests/test_common_reranker.py`

- [ ] **Step 1: Implement `src/common/` modules matching Task 2 architecture**
Implement text cleaning, legal signals extraction (`DOC_NUMBER_PATTERN`, `ARTICLE_PATTERN`), BM25 with legal entity boosting (+25.0 doc numbers, +12.0 articles), DEk21 retriever with PyVi and MPS device support, RRF fusion, evidence packs, and BGE cross-encoder.

- [ ] **Step 2: Run tests for `src/common/`**
Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_common_*.py -v`

- [ ] **Step 3: Commit**
```bash
git add src/common/ tests/test_common_*.py
git commit -m "feat(common): establish clean shared RAG core modules"
```

---

### Task 3: Task 1 Dedicated Pipeline in `src/task1/`

**Files:**
- Create: `src/task1/memory.py`
- Create: `src/task1/retrieve.py`
- Create: `src/task1/rerank.py`
- Create: `src/task1/selector.py`
- Create: `src/task1/predict.py`
- Test: `tests/test_task1_pipeline.py`

- [ ] **Step 1: Implement Task 1 retrieval, reranking, memory, and prediction pipeline**
Integrate Question Memory with fold-local fit, 4-branch hybrid candidate retrieval, BGE cross-encoder evidence reranking, Top-5 selection, and `LegalIRPipeline`.

- [ ] **Step 2: Run tests for `src/task1/`**
Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_task1_*.py -v`

- [ ] **Step 3: Commit**
```bash
git add src/task1/ tests/test_task1_*.py
git commit -m "feat(task1): implement clean Task 1 retrieval and ranking pipeline"
```

---

### Task 4: Scripts for Dataset & MPS Index Generation

**Files:**
- Create: `scripts/01_build_dataset.py`
- Create: `scripts/02_build_indexes.py`
- Output: `artifacts/task1/indexes/bm25/`, `artifacts/task1/indexes/dense_dek21/`, `artifacts/task1/indexes/question_memory/`

- [ ] **Step 1: Implement `01_build_dataset.py` and `02_build_indexes.py`**
Build canonical dataset and precompute real DEk21 embeddings on Apple Silicon MPS for all macro chunks, saving embeddings to `artifacts/task1/indexes/dense_dek21/`.

- [ ] **Step 2: Execute index build script on MPS**
Run: `PYTHONPATH=. ./.venv/bin/python scripts/02_build_indexes.py`
Verify: Real 768-dim embeddings generated and saved.

- [ ] **Step 3: Commit**
```bash
git add scripts/01_build_dataset.py scripts/02_build_indexes.py
git commit -m "feat(scripts): add dataset and MPS index generation scripts"
```

---

### Task 5: Dual Validation Benchmark Script

**Files:**
- Create: `scripts/03_run_benchmark.py`
- Create: `scripts/audit_parameters.py`
- Test: `tests/test_benchmark_metrics.py`

- [ ] **Step 1: Implement `03_run_benchmark.py` and parameter audit**
Run 5-Fold Cross-Validation and Document-Disjoint evaluation with candidate recall cutoffs (@20, @50, @100) and final Recall@5.

- [ ] **Step 2: Run benchmark smoke test**
Run: `PYTHONPATH=. ./.venv/bin/python scripts/03_run_benchmark.py --smoke`

- [ ] **Step 3: Commit**
```bash
git add scripts/03_run_benchmark.py scripts/audit_parameters.py
git commit -m "feat(eval): add dual validation benchmark and parameter audit scripts"
```

---

### Task 6: High-Output Submission Generation & Verification

**Files:**
- Create: `scripts/04_predict_submission.py`
- Output: `artifacts/task1/submissions/submission.json` and `submission.zip`, `submission.json` and `submission.zip`

- [ ] **Step 1: Run full 4-branch DEk21 + BM25 + Memory + BGE Reranker on `public-official.json`**
Run: `PYTHONPATH=. ./.venv/bin/python scripts/04_predict_submission.py`

- [ ] **Step 2: Validate submission compliance**
Verify: Exactly 1,000 queries, $1 \le \text{len}(\text{answer}) \le 5$, unique IDs, valid corpus IDs.

- [ ] **Step 3: Commit**
```bash
git add scripts/04_predict_submission.py submission.json submission.zip artifacts/task1/submissions/
git commit -m "feat(submission): generate high-recall submission with real DEk21 and BGE reranker"
```

---

### Task 7: Full Acceptance Suite & Workspace Cleanliness

**Files:**
- Test: `tests/`
- Documentation: `README.md`

- [ ] **Step 1: Run full test suite**
Run: `PYTHONPATH=. ./.venv/bin/pytest tests/ -v`

- [ ] **Step 2: Update README with reproduction steps**

- [ ] **Step 3: Commit**
```bash
git add README.md tests/
git commit -m "docs: finalize clean workspace structure and reproduction guide"
```
