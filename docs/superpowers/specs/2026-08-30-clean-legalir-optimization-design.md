# Technical Design Specification: High-Output Clean LegalIR Pipeline (Task 1)

- **Date**: 2026-08-30
- **Task**: UIT-DSC 2026 Task 1 — Legal Information Retrieval (LegalIR)
- **Goal**: Clean workspace architecture matching Task 2 (LegalQA) and maximize Recall@5 on Codabench.

---

## 1. Root Cause Analysis of Previous 27.7% Recall

The previous submission scored 27.7% Recall because:
1. **Dense Retrieval was completely un-indexed**: The pipeline fell back to an uninformative 256-dimension random hash fallback instead of real neural embeddings.
2. **Reranker was Disabled**: The submission script executed with `use_reranker: false`.
3. **BM25 Lexical Boosting was Missing**: Legal queries with exact decree/circular numbers (e.g. `5868/QĐ-BYT`, `44/2023/NĐ-CP`, `Điều 12`) were not boosted by legal entity regex matching.
4. **Question Memory was Not Integrated**: Near-duplicate train questions were not transferring their gold documents.

---

## 2. Target Clean Workspace Structure (Mirrored from Task 2)

```text
LegalIR - Public Test/
├── configs/
│   ├── models.yaml               # Pinned models: DEk21 v2 (768-dim) & BGE Reranker v2 M3
│   ├── task1.yaml                # Hyperparameters (K=100, RRF k=60, evidence packs, top-5)
│   └── pipeline.yaml             # Unified pipeline config
├── src/
│   ├── common/                   # Clean shared RAG core (matching Task 2)
│   │   ├── normalize.py          # Legal cleaning, PyVi tokenization, legal signals extraction
│   │   ├── legal_parser.py       # Hierarchy parser (Chương, Mục, Điều, Khoản, Điểm)
│   │   ├── bm25.py               # Fielded BM25 with legal entity boosting (law/decree codes)
│   │   ├── dense_dek21.py        # DEk21 embedding engine (PyVi, MPS/CPU, cosine search)
│   │   ├── rrf.py                # Reciprocal Rank Fusion
│   │   ├── evidence.py           # Structured multi-chunk evidence pack builder
│   │   └── reranker.py           # BGE-Reranker-v2-M3 cross-encoder batch inference
│   └── task1/                    # Task 1 specific pipeline
│       ├── memory.py             # Train-Question Memory (TF-IDF + DEk21 embeddings, fold-isolated)
│       ├── retrieve.py           # 4-Branch candidate generator (BM25 + DEk21 + Memory + Exact)
│       ├── rerank.py             # Cross-encoder reranker on evidence packs
│       ├── selector.py           # Top-5 unique document ID selector
│       └── predict.py            # End-to-end Task 1 LegalIR pipeline
├── artifacts/
│   └── task1/
│       ├── data/                 # documents.parquet, chunks.parquet, queries_train, qrels_train
│       ├── indexes/              # BM25 index, DEk21 dense vectors, Question Memory index
│       └── submissions/          # submission.json and submission.zip
├── scripts/
│   ├── 01_build_dataset.py       # Builds canonical dataset from raw contexts & train.json
│   ├── 02_build_indexes.py       # Precomputes BM25 & DEk21 embeddings on MPS
│   ├── 03_run_benchmark.py       # Dual validation (5-fold CV + Document-disjoint)
│   ├── 04_predict_submission.py  # Generates real DEk21 + BGE Reranker submission.zip
│   └── audit_parameters.py       # Validates < 4.0B parameter limit
├── tests/                        # Clean test suite for common + task1 modules
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 3. High-Output Retrieval & Reranking Architecture

1. **Boosted BM25 Micro**:
   - Scores micro chunks with legal entity boosting (+25.0 for exact document number `doc_numbers`, +12.0 for `Điều X`, +6.0 for `Khoản Y`).
   - Document pooling: $S_{\text{doc}} = \max(S_{\text{chunk}}) + 0.1 \times \text{mean}(S_{\text{chunk}})$.
2. **Dense Macro Retrieval (DEk21 v2)**:
   - Precomputes 768-dim normalized embeddings on PyVi-tokenized macro chunks using Apple Silicon `mps`.
   - Computes cosine similarity and retrieves Top-100 candidates per query.
3. **Train-Question Memory**:
   - Exact query match lookup: transfers gold doc IDs with maximum priority.
   - Dense + TF-IDF cosine similarity voting for near-match questions.
4. **Exact Legal Matcher**:
   - Deterministic regex extraction of legal numbers (e.g. `\d+/\d+/NĐ-CP`).
5. **Weighted RRF Fusion ($k=60$)**:
   - Fuses all 4 branches into Top-60 unique candidate documents.
6. **BGE Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`)**:
   - Builds 2-chunk structured evidence packs `[QUESTION] ... [DOCUMENT] ... [EVIDENCE 1] ... [EVIDENCE 2] ...`.
   - Performs batch inference on MPS and outputs final ranked documents.
7. **Top-5 Selector**:
   - Picks top 5 unique document IDs, guaranteeing $1 \le \text{len}(\text{answer}) \le 5$.

---

## 4. Verification and Submission Protocol

1. Precompute and verify DEk21 dense embeddings in `artifacts/task1/indexes/dense_dek21/`.
2. Run benchmark on training queries and verify Candidate Recall@50 > 95% and final Recall@5 > 80%.
3. Generate `artifacts/task1/submissions/submission.json` and package into `submission.zip`.
4. Validate compliance with 0 errors before submission.
