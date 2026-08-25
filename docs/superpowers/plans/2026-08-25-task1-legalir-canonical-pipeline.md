# Task 1 LegalIR Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an end-to-end, high-recall Legal Information Retrieval pipeline for DSC 2026 Task 1 that produces a valid, verified submission with maximum Codabench Recall@5 simulating the competition evaluation environment.

**Architecture:** A canonical dataset architecture with dual-granularity legal chunking (Micro for lexical BM25/exact match, Macro for dense semantic retrieval and evidence packs). A 4-branch hybrid candidate retrieval engine (Micro BM25 + Macro BGE-M3 + Train-Question Memory + Exact Metadata Matcher) feeds candidate pools into a Cross-Encoder reranker (BGE-reranker-v2-m3), combined with out-of-fold feature extraction and learned fusion (RRF / LightGBM) to select top-5 unique documents. Evaluated on both Random 5-fold CV and Document-Disjoint Validation Splits using exact Codabench scoring logic.

**Tech Stack:** Python 3.14, PyArrow, Pandas, Rank-BM25, Scikit-Learn, PyTorch, Hugging Face Transformers, FastParquet.

**Spec:** `docs/superpowers/specs/2026-08-25-task1-legalir-canonical-pipeline-design.md`

## Global Constraints
- Total neural parameters across active models must remain $< 4.0\text{B}$.
- Zero external APIs; zero external legal corpus; zero data/label transfer from Task 2.
- Submission output must strictly contain 1 to 5 unique document IDs per test query.
- Exactly 8,532 context documents must be indexed and available in the canonical dataset.

---

### Task 1: Canonical Dataset Builder & Invariant Verifier

**Files:**
- Create: `src/dataset/build_canonical.py`
- Create: `src/dataset/validator.py`
- Test: `tests/test_canonical_dataset.py`
- Outputs: `data/task1_canonical/v1/{manifest.json, audit_report.json, documents.parquet, chunks.parquet, queries_train.parquet, qrels_train.parquet}`

**Interfaces:**
- Consumes: `selected-contexts/` (8,532 JSON files), `train.json`, `public-official.json`
- Produces: `data/task1_canonical/v1/` Parquet files complying with Notion 02A schema.

- [ ] **Step 1: Write test for Canonical Dataset Validator & Builder**

Create `tests/test_canonical_dataset.py`:
```python
import os
import json
import pytest
import pandas as pd
from src.dataset.validator import validate_canonical_dataset

def test_canonical_dataset_invariants(tmp_path):
    # Create mock canonical files
    docs_df = pd.DataFrame([
        {
            "doc_id": "740",
            "name_raw": "Quyet-dinh-5868-QD-BYT",
            "title": "Quyết định 5868/QĐ-BYT 2018",
            "link": "https://example.com/740",
            "passage_raw": "Điều 1. Phạm vi\nĐiều 2. Đối tượng",
            "passage_norm": "điều 1. phạm vi\nđiều 2. đối tượng",
            "legal_number": "5868/QĐ-BYT",
            "year": "2018",
            "doc_type": "Quyết định",
            "is_empty": False
        }
    ])
    
    chunks_df = pd.DataFrame([
        {
            "chunk_id": "740_macro_001",
            "doc_id": "740",
            "granularity": "macro",
            "article": "Điều 1",
            "clause": None,
            "text_raw": "Điều 1. Phạm vi",
            "text_norm": "điều 1. phạm vi",
            "parent_chunk_id": None,
            "token_count": 10
        },
        {
            "chunk_id": "740_micro_001",
            "doc_id": "740",
            "granularity": "micro",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "text_raw": "Điều 1. Phạm vi",
            "text_norm": "điều 1. phạm vi",
            "parent_chunk_id": "740_macro_001",
            "token_count": 5
        }
    ])
    
    queries_df = pd.DataFrame([
        {
            "query_id": "101",
            "question_raw": "Quy định phạm vi là gì?",
            "question_norm": "quy định phạm vi là gì?",
            "gold_count": 1
        }
    ])
    
    qrels_df = pd.DataFrame([
        {
            "query_id": "101",
            "doc_id": "740",
            "relevance": 1
        }
    ])
    
    data_dir = tmp_path / "v1"
    data_dir.mkdir(parents=True)
    docs_df.to_parquet(data_dir / "documents.parquet")
    chunks_df.to_parquet(data_dir / "chunks.parquet")
    queries_df.to_parquet(data_dir / "queries_train.parquet")
    qrels_df.to_parquet(data_dir / "qrels_train.parquet")
    
    report = validate_canonical_dataset(str(data_dir))
    assert report["is_valid"] is True
    assert report["total_documents"] == 1
    assert report["total_chunks"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_canonical_dataset.py -v`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement Dataset Validator & Canonical Builder**

Create `src/dataset/validator.py`:
```python
import os
import json
import pandas as pd

def validate_canonical_dataset(canonical_dir: str) -> dict:
    docs_path = os.path.join(canonical_dir, "documents.parquet")
    chunks_path = os.path.join(canonical_dir, "chunks.parquet")
    queries_path = os.path.join(canonical_dir, "queries_train.parquet")
    qrels_path = os.path.join(canonical_dir, "qrels_train.parquet")

    assert os.path.exists(docs_path), f"Missing {docs_path}"
    assert os.path.exists(chunks_path), f"Missing {chunks_path}"
    assert os.path.exists(queries_path), f"Missing {queries_path}"
    assert os.path.exists(qrels_path), f"Missing {qrels_path}"

    docs_df = pd.read_parquet(docs_path)
    chunks_df = pd.read_parquet(chunks_path)
    queries_df = pd.read_parquet(queries_path)
    qrels_df = pd.read_parquet(qrels_path)

    doc_ids = set(docs_df["doc_id"].astype(str))
    chunk_doc_ids = set(chunks_df["doc_id"].astype(str))
    qrel_doc_ids = set(qrels_df["doc_id"].astype(str))

    errors = []
    
    # Invariant 1: Every chunk doc_id must exist in documents
    orphans = chunk_doc_ids - doc_ids
    if orphans:
        errors.append(f"Found {len(orphans)} chunk doc_ids not in documents.parquet")

    # Invariant 2: Every non-empty document must have at least one chunk
    non_empty_docs = set(docs_df[~docs_df["is_empty"]]["doc_id"].astype(str))
    docs_without_chunks = non_empty_docs - chunk_doc_ids
    if docs_without_chunks:
        errors.append(f"Found {len(docs_without_chunks)} non-empty documents with 0 chunks")

    # Invariant 3: Every qrel doc_id must exist in documents
    invalid_qrels = qrel_doc_ids - doc_ids
    if invalid_qrels:
        errors.append(f"Found {len(invalid_qrels)} qrel doc_ids not in documents.parquet")

    # Invariant 4: Micro chunks with parent_chunk_id must map to valid macro chunk_id
    macro_chunk_ids = set(chunks_df[chunks_df["granularity"] == "macro"]["chunk_id"])
    micro_chunks = chunks_df[chunks_df["granularity"] == "micro"]
    missing_parents = set(micro_chunks["parent_chunk_id"].dropna()) - macro_chunk_ids
    if missing_parents:
        errors.append(f"Found {len(missing_parents)} micro chunks referencing missing parent macro chunks")

    is_valid = len(errors) == 0
    report = {
        "is_valid": is_valid,
        "total_documents": len(docs_df),
        "total_chunks": len(chunks_df),
        "total_micro_chunks": int((chunks_df["granularity"] == "micro").sum()),
        "total_macro_chunks": int((chunks_df["granularity"] == "macro").sum()),
        "total_queries": len(queries_df),
        "total_qrels": len(qrels_df),
        "empty_documents_count": int(docs_df["is_empty"].sum()),
        "errors": errors
    }
    return report
```

Create `src/dataset/build_canonical.py` to parse all 8,532 documents into `documents.parquet`, `chunks.parquet`, `queries_train.parquet`, and `qrels_train.parquet`.

- [ ] **Step 4: Execute dataset builder & run validator**

Run:
```bash
.venv/bin/python src/dataset/build_canonical.py --output_dir data/task1_canonical/v1
.venv/bin/python -m pytest tests/test_canonical_dataset.py -v
```
Expected: PASS and `audit_report.json` generated with `is_valid: true` and 8,532 total documents.

- [ ] **Step 5: Commit**

```bash
git add src/dataset/ tests/test_canonical_dataset.py
git commit -m "feat(dataset): implement canonical dataset builder and invariant validator"
```

---

### Task 2: Dual Validation Protocol & Official Codabench Scorer

**Files:**
- Create: `src/evaluation/splits.py`
- Create: `src/evaluation/evaluator.py`
- Test: `tests/test_evaluation.py`
- Outputs: `data/task1_canonical/v1/splits/{random_5fold.json, doc_disjoint_split.json}`

**Interfaces:**
- Consumes: `data/task1_canonical/v1/queries_train.parquet`, `data/task1_canonical/v1/qrels_train.parquet`
- Produces: Evaluation reports containing Recall@1,3,5, Precision@5, Candidate Recall@20,50,100.

- [ ] **Step 1: Write test for Split Generators and Codabench Evaluation**

Create `tests/test_evaluation.py`:
```python
import pytest
from src.evaluation.evaluator import evaluate_predictions

def test_codabench_scoring_rules():
    # Case 1: Exact match with 1 gold
    y_true = {"q1": ["docA"]}
    y_pred = {"q1": {"answer": ["docA", "docB", "docC"]}}
    res = evaluate_predictions(y_pred, y_true)
    assert res["recall"] == 1.0
    assert pytest.approx(res["precision"], 0.01) == 1.0 / 3.0

    # Case 2: > 5 predictions gives 0.0
    y_pred_invalid = {"q1": {"answer": ["docA", "docB", "docC", "docD", "docE", "docF"]}}
    res_inv = evaluate_predictions(y_pred_invalid, y_true)
    assert res_inv["recall"] == 0.0
    assert res_inv["precision"] == 0.0

    # Case 3: Empty prediction gives 0.0
    y_pred_empty = {"q1": {"answer": []}}
    res_empty = evaluate_predictions(y_pred_empty, y_true)
    assert res_empty["recall"] == 0.0
    assert res_empty["precision"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_evaluation.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement Split Generator and Evaluator**

Create `src/evaluation/splits.py` to generate:
1. `random_5fold.json`: 5-fold stratified random query split.
2. `doc_disjoint_split.json`: Document-disjoint split ensuring no validation gold doc appears in the train partition.

Create `src/evaluation/evaluator.py` matching `Scoring-Program-Task-LegalIR/scoring.py` with multi-tier metrics.

- [ ] **Step 4: Run test and generate splits**

Run:
```bash
.venv/bin/python -m pytest tests/test_evaluation.py -v
.venv/bin/python src/evaluation/splits.py --canonical_dir data/task1_canonical/v1
```
Expected: PASS and split files created in `data/task1_canonical/v1/splits/`.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/ tests/test_evaluation.py
git commit -m "feat(eval): implement dual-split generator and codabench evaluator"
```

---

### Task 3: Multi-Branch Hybrid Candidate Retrieval Engine

**Files:**
- Create: `src/retrieval/bm25_micro.py`
- Create: `src/retrieval/dense_macro.py`
- Create: `src/retrieval/question_memory.py`
- Create: `src/retrieval/exact_matcher.py`
- Create: `src/retrieval/hybrid_search.py`
- Test: `tests/test_retrieval_branches.py`

**Interfaces:**
- Consumes: `data/task1_canonical/v1/{documents.parquet, chunks.parquet, queries_train.parquet, qrels_train.parquet}`
- Produces: Top-50 candidate documents with scores and rank features per query.

- [ ] **Step 1: Write tests for retrieval branches**

Create `tests/test_retrieval_branches.py` testing:
- BM25 indexing and scoring on micro chunks.
- Exact Matcher regex parsing of legal numbers and years.
- Question Memory nearest-neighbor retrieval with self-exclusion.
- Hybrid candidate union generator.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_retrieval_branches.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement all 4 retrieval branches and Candidate Union**

Implement:
- `src/retrieval/bm25_micro.py`: Tokenizer + inverted index with document-level score aggregation ($\max + 0.1 \times \text{mean}$).
- `src/retrieval/dense_macro.py`: Macro-chunk embeddings using `BAAI/bge-m3` with torch/numpy cosine similarity.
- `src/retrieval/question_memory.py`: TF-IDF word & char n-grams + dense similarity over `queries_train.parquet`.
- `src/retrieval/exact_matcher.py`: Regex extraction of legal numbers and metadata lookup.
- `src/retrieval/hybrid_search.py`: Candidate Union combining top candidates from all branches into a deduplicated candidate pool of $K=50$ documents per query.

- [ ] **Step 4: Run tests and verify Candidate Recall@50**

Run:
```bash
.venv/bin/python -m pytest tests/test_retrieval_branches.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/ tests/test_retrieval_branches.py
git commit -m "feat(retrieval): implement 4-branch hybrid candidate retrieval engine"
```

---

### Task 4: Weak Positive Localization & Hard Negative Mining

**Files:**
- Create: `src/training/positive_localizer.py`
- Create: `src/training/hard_negative_miner.py`
- Test: `tests/test_training_prep.py`
- Outputs: `data/task1_canonical/v1/reranker_train.parquet`, `data/task1_canonical/v1/retriever_train.parquet`

**Interfaces:**
- Consumes: Canonical dataset + Hybrid candidate pools
- Produces: High-quality training pairs with localized evidence chunks and mined hard negatives.

- [ ] **Step 1: Write test for Positive Localizer & Hard Negative Miner**

Create `tests/test_training_prep.py` verifying that:
- For a $(query, gold\_doc)$, the best macro chunk inside $gold\_doc$ is selected as positive evidence.
- Hard negatives exclude any gold document IDs and exclude near-duplicate query positives.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_training_prep.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement Positive Localizer & Hard Negative Miner**

Implement:
- `src/training/positive_localizer.py`: Scores query against candidate macro chunks of gold document using BM25 and dense similarity.
- `src/training/hard_negative_miner.py`: Mines non-gold top-ranked candidates from hybrid retrieval and constructs `(query, pos_evidence, neg_evidence)` pairs.

- [ ] **Step 4: Run test and generate training datasets**

Run:
```bash
.venv/bin/python -m pytest tests/test_training_prep.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/ tests/test_training_prep.py
git commit -m "feat(training): implement weak positive localizer and hard negative miner"
```

---

### Task 5: Cross-Encoder Evidence Reranking Engine

**Files:**
- Create: `src/ranking/evidence_pack.py`
- Create: `src/ranking/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: Query + Top candidate documents + Macro chunks
- Produces: Reranker scores (`reranker_best_score`, `reranker_second_score`, `best_macro_chunk_id`).

- [ ] **Step 1: Write test for Evidence Pack Formatter & Reranker**

Create `tests/test_reranker.py` verifying:
- Formatting of `[VĂN BẢN]`, `[ĐIỀU KHOẢN]`, `[NỘI DUNG]` evidence packs.
- Pair scoring with batch inference.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reranker.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement Evidence Pack & Cross-Encoder Reranker**

Implement:
- `src/ranking/evidence_pack.py`: Selects best macro chunk per document for the query to construct concise evidence packs.
- `src/ranking/reranker.py`: Wraps `BAAI/bge-reranker-v2-m3` to score $(query, evidence)$ pairs efficiently.

- [ ] **Step 4: Run test**

Run:
```bash
.venv/bin/python -m pytest tests/test_reranker.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ranking/ tests/test_reranker.py
git commit -m "feat(ranking): implement evidence pack formatter and cross-encoder reranker"
```

---

### Task 6: Out-of-Fold Feature Generation, Learned Fusion & Selection

**Files:**
- Create: `src/ranking/oof_features.py`
- Create: `src/ranking/fusion.py`
- Create: `src/ranking/selector.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: Multi-branch candidate outputs + Reranker scores
- Produces: Final deduplicated Top 5 document IDs per query.

- [ ] **Step 1: Write test for Feature Table & Selection**

Create `tests/test_fusion.py` verifying:
- Feature extraction across branches.
- RRF rank fusion formula.
- Strict selection of 1 to 5 unique document IDs.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement OOF Features, Fusion, and Selector**

Implement:
- `src/ranking/oof_features.py`: Extracts tabular features for each $(query, candidate\_doc)$.
- `src/ranking/fusion.py`: Implements RRF and LightGBM ranking models.
- `src/ranking/selector.py`: Deduplicates candidates and outputs valid Top-5 document IDs.

- [ ] **Step 4: Run test**

Run:
```bash
.venv/bin/python -m pytest tests/test_fusion.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ranking/ tests/test_fusion.py
git commit -m "feat(fusion): implement OOF feature extractor, learned fusion, and top-5 selector"
```

---

### Task 7: End-to-End Validation Benchmark & Submission Generator

**Files:**
- Create: `src/predict_submission.py`
- Create: `src/validate_all.py`
- Test: `tests/test_submission_compliance.py`
- Outputs: `submission.json`, `submission.zip`, validation metrics report

**Interfaces:**
- Consumes: `public-official.json`, full canonical dataset, trained indexes/models
- Produces: Validated `submission.json` and `submission.zip` matching Codabench specs.

- [ ] **Step 1: Write test for Submission Compliance & Codabench Evaluation**

Create `tests/test_submission_compliance.py` verifying:
- Exactly matches all query IDs in test file.
- Every answer is a list of 1 to 5 strings.
- All strings are valid document IDs from `documents.parquet`.
- No duplicates inside any answer list.

- [ ] **Step 2: Implement `src/validate_all.py` & `src/predict_submission.py`**

- `src/validate_all.py`: Runs full evaluation across Random 5-Fold CV and Document-Disjoint Split, computing exact Codabench Recall@5 and candidate diagnostics.
- `src/predict_submission.py`: Runs the complete multi-branch retrieval + reranking + fusion pipeline on `public-official.json` and generates `submission.json` and `submission.zip`.

- [ ] **Step 3: Run validation benchmark and generate submission**

Run:
```bash
.venv/bin/python src/validate_all.py
.venv/bin/python src/predict_submission.py --input_file public-official.json --output_file submission.json
.venv/bin/python -m pytest tests/test_submission_compliance.py -v
```
Expected: Validation metrics printed, all compliance checks PASS, `submission.zip` generated.

- [ ] **Step 4: Commit**

```bash
git add src/ tests/test_submission_compliance.py
git commit -m "feat(pipeline): complete end-to-end validation benchmark and submission generator"
```
