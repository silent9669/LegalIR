# UIT-DSC 2026 Task 1: Legal Information Retrieval (LegalIR)

High-Recall Vietnamese Legal Information Retrieval Pipeline for UIT Data Science Challenge 2026.

---

## 1. System Architecture & Model Stack

- **Standardized Clean Workspace (`src/common/` & `src/task1/`)**:
  - `src/common/normalize.py`: Unicode NFC normalization, PyVi word segmentation, and legal signal extraction (decree/circular numbers, articles, clauses, years).
  - `src/common/legal_parser.py`: Hierarchical legal structure parser (Chương, Mục, Điều, Khoản, Điểm).
  - `src/common/bm25.py`: Fast `bm25s` micro-chunk indexing with legal entity boosting (+30.0 for exact decree numbers, +15.0 for articles, +8.0 for clauses).
  - `src/common/dense_dek21.py`: `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` (768-dim, PyVi segmentation) dense retriever on Apple Silicon `mps`/`cpu`.
  - `src/common/rrf.py`: Weighted Reciprocal Rank Fusion ($k=60$) across retrieval branches.
  - `src/common/evidence.py`: Structured multi-chunk evidence pack builder (`[QUESTION]`, `[DOCUMENT]`, `[EVIDENCE 1]`, `[EVIDENCE 2]`).
  - `src/common/reranker.py`: `BAAI/bge-reranker-v2-m3` cross-encoder scoring on `mps`/`cpu`.

- **Task 1 Pipeline Modules (`src/task1/`)**:
  - `src/task1/memory.py`: Fold-isolated Train-Question Memory (TF-IDF char n-grams + DEk21 train embeddings).
  - `src/task1/retrieve.py`: 4-Branch candidate search engine.
  - `src/task1/rerank.py`: Document-level evidence reranking.
  - `src/task1/selector.py`: Top-5 unique document ID selector with fallback guards.
  - `src/task1/predict.py`: End-to-end `LegalIRPipeline`.

---

## 2. Learned Parameter Compliance Audit (< 4.0B limit)

| Model Component | Hugging Face Repository | Parameter Count | Device |
| :--- | :--- | :---: | :---: |
| **Dense Retriever** | `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` | 134,998,272 (~0.135B) | `mps` / `cpu` |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | 567,755,777 (~0.568B) | `mps` / `cpu` |
| **Total System Parameters** | **LegalIR 4-Branch Stack** | **702,754,049 (~0.703B)** | **PASS (< 4.0B)** |

*Zero external legal corpus, zero synthetic data augmentation, zero Task 2 data, zero external API calls.*

---

## 3. Quickstart & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build Canonical Dataset from raw contexts & train.json
python scripts/01_build_dataset.py

# 3. Precompute BM25 & DEk21 Indexes on MPS
python scripts/02_build_indexes.py

# 4. Run Dual-Validation Benchmark (5-Fold CV & Document-Disjoint)
python scripts/03_run_benchmark.py

# 5. Generate High-Score Submission on Public Test Queries
python scripts/04_predict_submission.py

# 6. Verify Learned Parameters (<4.0B)
python scripts/audit_parameters.py

# 7. Run Full Test Suite
pytest tests/ -v
```
