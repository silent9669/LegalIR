# UIT-DSC 2026 Task 1: Legal Information Retrieval (LegalIR)

High-Recall Vietnamese Legal Information Retrieval Pipeline for UIT Data Science Challenge 2026.

## 1. System Architecture & Shared Artifacts

- **Canonical Dataset Package (`artifacts/shared/canonical/v2/`)**:
  - `documents.parquet`: 8,532 official context documents (100% coverage, 0 dropped).
  - `chunks.parquet`: 1,153,876 dual-granularity chunks (934,416 micro chunks for BM25/Exact match and 219,460 macro chunks for Dense retrieval & Cross-Encoder evidence packs).
  - `queries_train.parquet` & `qrels_train.parquet`: 7,000 queries and 7,637 exploded relations.
  - `duplicate_groups.json` & `empty_context_ids.json`: Grouping of identical passages and metadata-only empty documents.
  - `splits/`: 5-fold cross-validation (`random_5fold.json`) and document-disjoint split (`doc_disjoint_split.json`).
  - `manifest.json` & `audit_report.json`: Invariant audit and checksum verification.

- **Multi-Branch Candidate Retrieval**:
  - **BM25 on Micro Chunks (`src/retrieval/bm25_micro.py`)**: Vectorized BM25 on clause-level micro chunks with document score aggregation ($\max + 0.1 \times \text{mean}$) and legal field weighting.
  - **Exact Identifier Matcher (`src/retrieval/exact_matcher.py`)**: Regex extraction of decree/circular/law numbers, titles, and legal metadata.
  - **Train-Question Memory (`src/retrieval/question_memory.py`)**: Fold-isolated TF-IDF character n-gram + neural memory with zero validation label leakage.
  - **Dense Retriever (`src/retrieval/dense_macro.py`)**: Macro-chunk embeddings using `BAAI/bge-m3`.

- **Cross-Encoder Reranking & Evidence Packs (`src/ranking/`)**:
  - `EvidencePackBuilder`: Constructs concise `[VĂN BẢN]`, `[ĐIỀU KHOẢN]`, `[NỘI DUNG]` context packs.
  - `CrossEncoderReranker`: `BAAI/bge-reranker-v2-m3` pair inference.

- **Fusion & Selection (`src/ranking/`)**:
  - Reciprocal Rank Fusion (RRF) and fold-safe LightGBM ranking models.
  - `TopKSelector`: Strict selection of 1 to 5 unique official document IDs.

---

## 2. Official Dual-Validation Benchmark Results

### Model Comparison under Identical Dataset & Split Checksums

| Model / Pipeline Configuration | Random 5-Fold Recall@5 | Random 5-Fold Prec@5 | Candidate Recall@50 | Doc-Disjoint Recall@5 | Doc-Disjoint Cand@50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strict Baseline** (`strict_baseline`) | **73.96% ± 0.98%** | 15.69% | 94.03% | **66.00%** | 93.08% |
| **Accepted Model: Fielded BM25 + Exact** (`final_model`) | **75.36% ± 1.17%** | **16.00%** | **94.42%** | **67.72%** | **93.76%** |

*All benchmarks are 100% leakage-free, cross-validated across all 5 folds with fold-isolated memory, and tested against the exact official Codabench scorer.*

---

## 3. Quickstart & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 2. Build Canonical Dataset v2 (from raw ZIP)
python -m src.dataset.build_canonical --config configs/pipeline.yaml

# 3. Build Micro BM25 Index
python -m src.retrieval.build_indexes --config configs/pipeline.yaml --bm25

# 4. Run Full Benchmark (5-fold + Doc-disjoint)
python -m src.evaluation.benchmark --config configs/pipeline.yaml

# 5. Generate Submission for Public Official Test
python -m src.pipeline.run_all --config configs/pipeline.yaml --offline

# 6. Run Test Suite
pytest tests/ -v
```
