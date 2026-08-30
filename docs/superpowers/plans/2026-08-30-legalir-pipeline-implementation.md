# LegalIR 4-Branch Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, validate, and package the high-recall 4-branch Vietnamese LegalIR retrieval and reranking system matching the Notion DSC 2026 Task 1 specification.

**Architecture:** A modular 4-branch candidate retrieval system (BM25 Micro, DEk21 Dense Macro with PyVi, Fold-Isolated Question Memory, Exact Matcher) fused via RRF into candidate document pools, reranked by `BAAI/bge-reranker-v2-m3` on structured macro evidence packs, and selected into Top 5 unique document IDs.

**Tech Stack:** Python 3.10+, PyTorch (MPS/CPU), Hugging Face Transformers, Sentence-Transformers, PyVi, Rank-BM25, PyArrow / Pandas, Scikit-learn, PyTest.

**Spec:** `docs/superpowers/specs/2026-08-30-legalir-pipeline-design.md`

## Global Constraints

- Total learned parameters < 4.0B (~0.703B total for DEk21 + BGE Reranker).
- Zero external legal corpus, synthetic data augmentation, or Task 2 data.
- Zero external API calls.
- Every prediction answer list must have $1 \le \text{len}(\text{answer}) \le 5$ unique valid document IDs.
- Dual validation: 5-Fold CV (seen documents + memory) and Document-Disjoint Split (unseen document generalization) using official Codabench scorer formulas.

---

### Task 1: Pipeline Configuration & Dependencies Update

**Files:**
- Modify: `configs/pipeline.yaml`
- Modify: `requirements.txt`
- Test: `tests/test_core_config.py`

**Interfaces:**
- Consumes: `src/core/config.py`
- Produces: Updated configurations specifying `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` (768-dim, PyVi) and `BAAI/bge-reranker-v2-m3`.

- [ ] **Step 1: Write the test for DEk21 configuration in `tests/test_core_config.py`**

```python
def test_pipeline_config_dek21():
    from src.core.config import load_pipeline_config
    cfg = load_pipeline_config()
    assert cfg.retrieval.dense_macro.model_name == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    assert cfg.retrieval.dense_macro.dimension == 768
    assert cfg.retrieval.dense_macro.use_pyvi is True
    assert cfg.ranking.reranker.model_name == "BAAI/bge-reranker-v2-m3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_core_config.py -v`
Expected: FAIL due to missing fields / model name mismatch.

- [ ] **Step 3: Update `configs/pipeline.yaml` and `requirements.txt`**

Ensure `pyvi` and `sentencepiece` are included in `requirements.txt` and `configs/pipeline.yaml` is updated with `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`, `dimension: 768`, and `use_pyvi: true`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_core_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/pipeline.yaml requirements.txt tests/test_core_config.py
git commit -m "feat(config): update pipeline configuration for DEk21 v2 and BGE reranker"
```

---

### Task 2: Dense DEk21 Macro Retriever Module

**Files:**
- Create/Modify: `src/retrieval/dense_macro.py`
- Test: `tests/test_dense_macro.py`

**Interfaces:**
- Consumes: `macro` chunks from `chunks.parquet`, `queries_train.parquet`, and PyVi segmentation.
- Produces: `DenseMacroRetriever` class with `.encode_corpus()`, `.encode_queries()`, `.search(query, top_k)` returning document scores.

- [ ] **Step 1: Write test for DEk21 dense retriever with PyVi**

```python
def test_dense_macro_retriever_initialization():
    from src.retrieval.dense_macro import DenseMacroRetriever
    retriever = DenseMacroRetriever(
        model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        dimension=768,
        use_pyvi=True
    )
    assert retriever.dimension == 768
    text = "Thời hạn cấp đăng ký xe máy là bao lâu?"
    norm = retriever.preprocess_text(text)
    assert len(norm) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dense_macro.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `DenseMacroRetriever` with PyVi and mean pooling**

Implement PyVi word segmentation (`pyvi.ViTokenizer.tokenize`), Hugging Face `AutoModel` / `AutoTokenizer` mean pooling with embedding normalization, batch inference, and FAISS / cosine similarity search with document score pooling.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dense_macro.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/dense_macro.py tests/test_dense_macro.py
git commit -m "feat(retrieval): implement DEk21 v2 dense macro retriever with PyVi"
```

---

### Task 3: Fold-Isolated Question Memory with Dual TF-IDF & DEk21 Embeddings

**Files:**
- Modify: `src/retrieval/question_memory.py`
- Test: `tests/test_dense_question_memory.py`
- Test: `tests/test_fold_isolation.py`

**Interfaces:**
- Consumes: `queries_train.parquet`, `qrels_train.parquet`, fold splits.
- Produces: `TrainQuestionMemory` with `fit(train_queries, qrels)` and `query(q_text, top_k, q_emb=None)` with strict fold isolation.

- [ ] **Step 1: Write test for fold-isolated Question Memory**

```python
def test_question_memory_dual_signal():
    from src.retrieval.question_memory import TrainQuestionMemory
    memory = TrainQuestionMemory(min_similarity=0.8)
    train_queries = {"q1": "thời hạn cấp đăng ký xe máy", "q2": "thủ tục đăng ký kinh doanh"}
    train_qrels = {"q1": ["doc_100"], "q2": ["doc_200"]}
    memory.fit(train_queries, train_qrels)
    
    hits = memory.search("thời hạn cấp đăng ký xe máy của người nước ngoài", top_k=5)
    assert any(h["doc_id"] == "doc_100" for h in hits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dense_question_memory.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Dual TF-IDF + DEk21 Question Memory**

Implement character n-gram TF-IDF vectorizer + cosine similarity combined with DEk21 embedding similarity, voting for gold document IDs with similarity-weighted scoring.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dense_question_memory.py tests/test_fold_isolation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/question_memory.py tests/test_dense_question_memory.py tests/test_fold_isolation.py
git commit -m "feat(retrieval): enhance question memory with dual TF-IDF and DEk21 embeddings"
```

---

### Task 4: 4-Branch Hybrid Search & RRF Candidate Fusion

**Files:**
- Modify: `src/retrieval/hybrid_search.py`
- Test: `tests/test_candidate_union.py`
- Test: `tests/test_retrieval_branches.py`

**Interfaces:**
- Consumes: BM25MicroEngine, DenseMacroRetriever, TrainQuestionMemory, ExactMatcher.
- Produces: `HybridSearchEngine.search(query, top_k_candidates=100, rrf_k=60)` returning ranked candidate document IDs.

- [ ] **Step 1: Write test for 4-branch hybrid search and RRF fusion**

```python
def test_hybrid_search_4_branches():
    from src.retrieval.hybrid_search import HybridSearchEngine
    # Verify that all 4 branches contribute to candidate generation
    ...
```

- [ ] **Step 2: Run test to verify it fails or needs update**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_candidate_union.py -v`

- [ ] **Step 3: Implement 4-branch candidate retrieval & RRF rank aggregation**

Combine BM25 Micro, Dense DEk21, Question Memory, and Exact Matcher. Implement RRF formula:
$RRF(\text{doc}) = \sum_{b} \frac{w_b}{60 + \text{rank}_b(\text{doc})}$.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_candidate_union.py tests/test_retrieval_branches.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/hybrid_search.py tests/test_candidate_union.py tests/test_retrieval_branches.py
git commit -m "feat(retrieval): unify 4-branch candidate search and RRF fusion"
```

---

### Task 5: Structured Evidence Pack Builder & Cross-Encoder Reranking

**Files:**
- Modify: `src/ranking/evidence_pack.py`
- Modify: `src/ranking/reranker.py`
- Modify: `src/ranking/selector.py`
- Test: `tests/test_evidence_pack.py`
- Test: `tests/test_reranker.py`
- Test: `tests/test_submission_compliance.py`

**Interfaces:**
- Consumes: Query text, candidate document IDs, `chunks.parquet`, `documents.parquet`.
- Produces: Evidence packs `[QUESTION] ... [DOCUMENT] ... [EVIDENCE 1] ...`, `CrossEncoderReranker.rerank()`, and `TopKSelector.select(top_k=5)`.

- [ ] **Step 1: Write test for evidence packs and reranker integration**

```python
def test_evidence_pack_and_reranker():
    from src.ranking.evidence_pack import EvidencePackBuilder
    from src.ranking.selector import TopKSelector
    builder = EvidencePackBuilder(max_chunks=2)
    # verify format [QUESTION] ... [DOCUMENT] ...
    selector = TopKSelector(max_k=5)
    result = selector.select([("doc_1", 10.0), ("doc_2", 8.0), ("doc_3", 7.0)])
    assert len(result) == 3
    assert result == ["doc_1", "doc_2", "doc_3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_reranker.py tests/test_submission_compliance.py -v`

- [ ] **Step 3: Implement evidence pack formatting & BGE reranker inference**

Format evidence strings, batch cross-encoder forward pass with `BAAI/bge-reranker-v2-m3`, rank candidates, and select top 1..5 unique valid document IDs.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_reranker.py tests/test_submission_compliance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ranking/evidence_pack.py src/ranking/reranker.py src/ranking/selector.py tests/test_reranker.py tests/test_submission_compliance.py
git commit -m "feat(ranking): implement structured evidence packs and BGE reranker selection"
```

---

### Task 6: Dual Validation Benchmark & Acceptance Suite

**Files:**
- Modify: `src/evaluation/benchmark.py`
- Modify: `src/evaluation/evaluator.py`
- Test: `tests/test_final_acceptance.py`
- Test: `tests/test_benchmark_metrics.py`

**Interfaces:**
- Consumes: Pipeline configuration, Canonical Dataset v2, 5-Fold splits, Document-Disjoint split.
- Produces: Detailed candidate recall (`Recall@20`, `Recall@50`, `Recall@100`) and final ranking metrics (`Recall@1`, `Recall@3`, `Recall@5`, `Precision@5`).

- [ ] **Step 1: Write test for benchmark runner and dual-validation reporting**

```python
def test_benchmark_dual_validation():
    from src.evaluation.evaluator import evaluate_predictions
    preds = {"q1": ["doc_1", "doc_2"]}
    golds = {"q1": ["doc_1"]}
    metrics = evaluate_predictions(preds, golds)
    assert "recall@5" in metrics
    assert "precision@5" in metrics
    assert metrics["recall@5"] == 1.0
    assert metrics["precision@5"] == 0.5
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_benchmark_metrics.py -v`

- [ ] **Step 3: Run full acceptance test suite**

Run: `PYTHONPATH=. ./.venv/bin/pytest tests/ -v`
Expected: All 56+ tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/evaluation/benchmark.py src/evaluation/evaluator.py tests/test_final_acceptance.py
git commit -m "feat(eval): finalize dual validation benchmark and acceptance suite"
```

---

### Task 7: End-to-End Pipeline & Submission Packaging

**Files:**
- Modify: `src/pipeline/run_all.py`
- Modify: `src/pipeline/predict.py`
- Modify: `scripts/04_predict_submission.sh`
- Output: `artifacts/task1/submissions/submission.json` & `submission.zip`

- [ ] **Step 1: Execute end-to-end pipeline on `public-official.json`**

Run: `PYTHONPATH=. ./.venv/bin/python -m src.pipeline.run_all --config configs/pipeline.yaml`
Verify: Generates valid predictions for all queries in `public-official.json`.

- [ ] **Step 2: Run submission validation checks**

Verify invariants:
- 100% of query keys match `public-official.json`.
- $1 \le \text{len}(\text{answer}) \le 5$ for all queries.
- Zero duplicate IDs per answer.
- All IDs exist in `documents.parquet`.
- `submission.zip` contains exactly `submission.json`.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/run_all.py src/pipeline/predict.py scripts/04_predict_submission.sh
git commit -m "feat(pipeline): generate and package compliant submission.zip"
```
