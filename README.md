# UIT-DSC 2026 Task 1: Legal Information Retrieval (LegalIR)

High-Recall Vietnamese Legal Information Retrieval Pipeline for UIT Data Science Challenge 2026.

## 1. System Architecture

- **Corpus Preprocessing & Legal-Aware Chunking (`src/legal_chunker.py`)**:
  - 100% document coverage across all 8,532 legal documents.
  - Recovers missing metadata from URL slugs (unquoting encoded characters).
  - Contextual chunk templates prepending Document Title, Legal Number, Publication Year, Chapter, and Article heading.
- **Multi-Branch Candidate Retrieval**:
  - **BM25 Inverted Index (`src/retrievers/bm25_retriever.py`)**: Multi-field weighted BM25 (Title $\times 3.0$, Article Header $\times 2.0$, Body $\times 1.0$) with document-level score aggregation ($\max + 0.1 \times \text{second\_best}$).
  - **Exact Identifier Matcher (`src/retrievers/exact_matcher.py`)**: Extracts decree/circular/law numbers (e.g. `58/2020/TT-BCA`, `43/2023/NĐ-CP`) from questions and boosts corresponding documents.
  - **Train-Question Memory (`src/retrievers/memory_retriever.py`)**: Indexes train queries via character 3–5 n-gram TF-IDF and transfers known positive document IDs with similarity-weighted votes.
  - **Dense & Semantic Retrieval (`src/retrievers/dense_retriever.py`)**: Pretrained `BAAI/bge-m3` embedding model.
  - **Cross-Encoder Reranker (`src/rerankers/reranker.py`)**: `BAAI/bge-reranker-v2-m3` for scoring query-chunk pairs.
- **Fusion & Selection (`src/predict.py`)**:
  - Reciprocal Rank Fusion (RRF) and candidate deduplication to select top 5 unique document IDs.

## 2. Quickstart & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build processed chunk corpus & metadata
PYTHONPATH=src python3 src/legal_chunker.py

# 3. Build BM25 index
PYTHONPATH=src python3 -c "from retrievers.bm25_retriever import BM25Retriever; BM25Retriever().build_index('data/processed_chunks.jsonl', 'data/bm25_index.pkl')"

# 4. Run local evaluation
PYTHONPATH=src python3 src/evaluate.py

# 5. Generate submission.json and submission.zip for public-official.json
PYTHONPATH=src python3 src/predict.py
```
