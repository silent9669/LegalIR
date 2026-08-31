# UIT-DSC 2026 Task 1: Legal Information Retrieval (LegalIR)

High-Recall Vietnamese Legal Information Retrieval Pipeline for UIT Data Science Challenge 2026.

---

## 1. System Architecture & Model Stack

- **Standardized Clean Workspace (`src/common/` & `src/task1/`)**:
  - `src/common/normalize.py`: Unicode NFC normalization, PyVi word segmentation, and legal signal extraction (decree/circular numbers, articles, clauses, years).
  - `src/common/legal_parser.py`: Hierarchical legal structure parser (Chương, Mục, Điều, Khoản, Điểm).
  - `src/common/bm25.py`: Fast `bm25s` micro-chunk indexing with legal entity boosting (+30.0 for exact decree numbers, +15.0 for articles, +8.0 for clauses).
  - `src/common/dense_dek21.py`: `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` (768-dim, PyVi segmentation) dense retriever on Apple Silicon `mps`/`cpu` and CUDA `gpu` (FP16).
  - `src/common/rrf.py`: Weighted Reciprocal Rank Fusion ($k=60$) across retrieval branches.
  - `src/common/evidence.py`: Structured multi-chunk evidence pack builder (`[QUESTION]`, `[DOCUMENT]`, `[EVIDENCE 1]`, `[EVIDENCE 2]`).
  - `src/common/reranker.py`: `BAAI/bge-reranker-v2-m3` cross-encoder scoring on `mps`/`cpu` and CUDA `gpu` (FP16).

- **Task 1 Pipeline Modules (`src/task1/`)**:
  - `src/task1/memory.py`: Fold-isolated Train-Question Memory (TF-IDF char n-grams + DEk21 train embeddings).
  - `src/task1/retrieve.py`: 4-Branch candidate search engine (BM25 micro, DEk21 dense macro, Question memory, Exact legal matcher).
  - `src/task1/rerank.py`: Document-level evidence reranking.
  - `src/task1/selector.py`: Top-5 unique document ID selector with fallback guards.
  - `src/task1/predict.py`: End-to-end `LegalIRPipeline`.

---

## 2. Learned Parameter Compliance Audit (< 4.0B limit)

| Model Component | Hugging Face Repository | Parameter Count | Device |
| :--- | :--- | :---: | :---: |
| **Dense Retriever** | `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` | 134,998,272 (~0.135B) | `mps` / `cuda` (FP16) |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | 567,755,777 (~0.568B) | `mps` / `cuda` (FP16) |
| **Total System Parameters** | **LegalIR 4-Branch Stack** | **702,754,049 (~0.703B)** | **PASS (< 4.0B)** |

*Zero external legal corpus, zero synthetic data augmentation, zero Task 2 data, zero external API calls.*

---

## 3. Quickstart & Execution (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build Canonical Dataset from raw contexts & train.json
python scripts/01_build_dataset.py

# 3. Precompute BM25 & DEk21 Indexes
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

---

## 4. Kaggle GPU T4 Execution Guide

The standalone notebook for Task 1 is formatted to match Task 2 (`LegalQA training`):
- **Notebook Name**: **`LegalIR training`** (file: `kaggle_kernel_task1/legalir_training.ipynb`)
- **Dataset Name**: **`LegalIR`** (or `LegalIR Task 1 Clean Artifacts` / `LegalIR dataset`)

### Step 1: Kaggle Environment Setup
1. Open or create the Kaggle Notebook: **`LegalIR training`**.
2. Under **Input / Data**, attach your dataset: **`LegalIR`** (or `LegalIR Task 1 Clean Artifacts`).
3. Under **Notebook settings** (right sidebar):
   - **Accelerator**: Select **GPU T4** or **GPU T4 x2**.
   - **Internet**: Toggle **On** (required to install `bm25s`, `pyvi` and download model backbones).
   - **Secrets (Optional / Recommended)**: Add `HF_TOKEN` under **Add-ons -> Secrets** if you want high-bandwidth authenticated Hugging Face downloads.

### Step 2: Running the Notebook
- Either **Run All** cells interactively, or click **Save Version -> Save & Run All (Commit)** in the top right corner.
- The notebook will:
  1. Detect GPU T4 with FP16 acceleration.
  2. Ingest `documents.parquet`, `chunks.parquet`, `queries_train.parquet`, `qrels_train.parquet`, and `public-official.json`.
  3. Build BM25 Micro index & DEk21 Dense Macro index (automatically offloading DEk21 to CPU to free 100% of VRAM for the reranker).
  4. Run 5-Fold Cross-Validation.
  5. Run Public Test Inference using the 4-Branch hybrid search + BGE Reranker v2 M3 cross-encoder.
  6. Perform strict competition invariant validation (100% compliant).
  7. Export `submission.json` and `submission.zip` into `/kaggle/working/`.

---

## 5. How to Download `submission.json` and `submission.zip` After Run

Once the notebook finishes executing on Kaggle, you can download your submission package using any of the following 3 methods:

### Method A: Direct Download via Kaggle UI (Interactive Session)
1. In the open Kaggle notebook, look at the right-hand panel under **Output** (`/kaggle/working`).
2. If files are not immediately visible, click the **Refresh (🔄)** icon next to the Output folder.
3. You will see:
   - `submission.zip` (Official submission archive containing `submission.json`)
   - `submission.json` (Formatted predictions JSON)
   - `submission_manifest.json` (SHA-256 verification manifest)
4. Hover over `submission.zip` or `submission.json`, click the **three vertical dots (`⋮`)**, and select **Download**.

### Method B: Download from Saved Version (Background Commit Run)
1. Go to the notebook's main page on Kaggle (`https://www.kaggle.com/code/phucdangg/legalir-training`).
2. Click on the **Output** tab or click the **Version history** panel.
3. Select the latest completed version.
4. Scroll down to **Output Data** where `submission.zip` and `submission.json` are listed.
5. Click the **Download (⬇)** icon next to `submission.zip`.

### Method C: Download via Kaggle CLI (Local Terminal)
```bash
# Download output artifacts directly into artifacts/task1/submissions/kaggle/
./scripts/kaggle_run_task1.sh output
```
This will automatically download and verify `submission.json` and `submission.zip` into your local directory.

---

## 6. Execution Verification & Log Signals

When monitoring the notebook session logs, check for these confirmation markers:

- `✓ CUDA acceleration active with FP16 support on T4.`
- `Found canonical parquet data at: ...`
- `BM25s Indexing complete.`
- `✓ DEk21 model offloaded to CPU (VRAM fully reclaimed for Reranker).`
- `✓ Chunks dataframe freed from memory. System RAM optimized.`
- `Loading BGE Cross-Encoder Reranker BAAI/bge-reranker-v2-m3 on cuda (FP16=True)...`
- `>>> ALL SUBMISSION INVARIANTS VERIFIED (100% COMPLIANT) <<<`
- `Submission Package Created Successfully! (submission.zip)`
