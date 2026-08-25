# UIT-DSC 2026 Task 1: Legal Information Retrieval (LegalIR)

High-Recall Vietnamese Legal Information Retrieval Pipeline for UIT Data Science Challenge 2026.

## 1. System Architecture

- **Canonical Dataset Package (`data/task1_canonical/v1/`)**:
  - `documents.parquet`: 8,532 official context documents (100% coverage, 0 dropped).
  - `chunks.parquet`: 885,084 dual-granularity chunks (683,800 micro chunks for BM25/Exact match and 201,284 macro chunks for Dense retrieval & Cross-Encoder evidence packs).
  - `queries_train.parquet` & `qrels_train.parquet`: 7,000 queries and 7,637 exploded relations.
  - `manifest.json` & `audit_report.json`: Dataset versioning and invariant verification.

- **Multi-Branch Hybrid Candidate Retrieval**:
  - **BM25 on Micro Chunks (`src/retrieval/bm25_micro.py`)**: Vectorized BM25 on clause-level micro chunks with document score aggregation ($\max + 0.1 \times \text{mean}$).
  - **Exact Identifier Matcher (`src/retrieval/exact_matcher.py`)**: Regex extraction of decree/circular/law numbers (e.g. `5868/QĐ-BYT`, `17/2022/TT-BGTVT`, `61/2020/QH14`) and titles.
  - **Train-Question Memory (`src/retrieval/question_memory.py`)**: Dual TF-IDF character n-gram + neural matching of past train questions with strict evaluation self-exclusion.
  - **Dense Retriever (`src/retrieval/dense_macro.py`)**: Macro-chunk embeddings using `BAAI/bge-m3`.

- **Cross-Encoder Reranking & Evidence Packs (`src/ranking/`)**:
  - `EvidencePackBuilder`: Constructs concise `[VĂN BẢN]`, `[ĐIỀU KHOẢN]`, `[NỘI DUNG]` context packs.
  - `CrossEncoderReranker`: `BAAI/bge-reranker-v2-m3` pair inference.

- **Learned Fusion & Top-5 Selection (`src/ranking/`)**:
  - Out-of-fold feature extraction across all branches.
  - Reciprocal Rank Fusion (RRF) and LightGBM ranking models.
  - `TopKSelector`: Strict selection of 1 to 5 unique document IDs.

---

## 2. Validation Benchmark (Codabench Simulation)

| Benchmark Protocol | Candidate Recall@50 | Codabench Recall@5 | Codabench Precision@5 | Recall@1 | Recall@3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Protocol 1: Random 5-Fold Cross Validation** (Seen Documents & Memory) | **95.76%** | **79.30%** | **16.91%** | **47.24%** | **69.98%** |
| **Protocol 2: Document-Disjoint Split** (Zero-Leakage Unseen Generalization) | **95.05%** | **72.79%** | **15.66%** | **39.43%** | **61.93%** |

---

## 3. Quickstart & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build Canonical Dataset (data/task1_canonical/v1/)
PYTHONPATH=. python3 src/dataset/build_canonical.py \
  --raw_contexts_dir selected-contexts \
  --train_json train.json \
  --output_dir data/task1_canonical/v1

# 3. Build Micro BM25 Index
PYTHONPATH=. python3 src/retrieval/build_indexes.py \
  --canonical_dir data/task1_canonical/v1 \
  --output_dir indexes

# 4. Run Full Dual Validation Benchmark
PYTHONPATH=. python3 src/validate_all.py

# 5. Run Test Suite
PYTHONPATH=. pytest -v

# 6. Generate Submission for Public Official Test
PYTHONPATH=. python3 src/predict_submission.py \
  --input_file public-official.json \
  --output_file submission.json \
  --output_zip submission.zip
```
