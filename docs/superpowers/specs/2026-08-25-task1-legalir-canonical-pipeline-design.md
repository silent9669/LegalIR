# Architectural Design Specification: DSC 2026 Task 1 LegalIR Pipeline

**Date:** 2026-08-25  
**Topic:** Task 1 — Legal Information Retrieval: End-to-End Canonical Pipeline & Codabench Evaluation Simulation  
**Status:** Approved for Implementation Planning  
**Target:** Maximizing Official Codabench Recall@5 on both Seen & Unseen Legal Documents under <4B System Parameter Budget.

---

## 1. System Goals & Official Constraints

### 1.1 Objective & Task Definition
Given a Vietnamese legal question, retrieve up to 5 unique legal `document_id`s from the official corpus of 8,532 context documents.

### 1.2 Evaluation & Codabench Metric Formula
The official Codabench scorer evaluates:
$$\text{Recall}_i = \frac{|\text{Gold}_i \cap \text{Pred}_i|}{|\text{Gold}_i|} \quad \text{if } 0 < |\text{Pred}_i| \le 5 \text{ else } 0$$
$$\text{Precision}_i = \frac{|\text{Gold}_i \cap \text{Pred}_i|}{|\text{Pred}_i|} \quad \text{if } 0 < |\text{Pred}_i| \le 5 \text{ else } 0$$

- **Primary Metric:** Macro Average $\text{Recall}$ (across all query evaluation sets).
- **Secondary Metric:** Macro Average $\text{Precision}$.
- **Strict Validity Rule:** If $|\text{Pred}_i| == 0$ or $|\text{Pred}_i| > 5$, both $\text{Recall}_i$ and $\text{Precision}_i$ are 0.0 for that query. Output must contain strictly 1 to 5 unique document IDs.

### 1.3 Competition System Constraints
- **Model Budget:** Total neural parameters across all active models in the pipeline $< 4.0\text{B}$. (Selected stack: BGE-M3 [~0.56B] + BGE-reranker-v2-m3 [~0.57B] = ~1.13B params $\ll 4.0\text{B}$).
- **External Data & API Constraint:** Strict zero external legal corpus, zero external APIs (no OpenAI/Claude/Gemini API calls in data, training, or inference). Open-weight models with permissive licenses only.
- **Task Separation:** Zero data or label transfer between Task 1 and Task 2.

---

## 2. Canonical Dataset Architecture (`data/task1_canonical/v1/`)

The pipeline operates on a unified, single source of truth structured into versioned Parquet tables:

```
data/task1_canonical/v1/
├── manifest.json                # Version, schema hash, normalization rules
├── audit_report.json            # Dataset validation report, empty/duplicate statistics
├── documents.parquet            # 1 row = 1 official legal document (8,532 rows)
├── chunks.parquet               # Dual-granularity hierarchical chunks (micro + macro)
├── queries_train.parquet        # Train queries (query_id, question_raw, question_norm, gold_count)
├── qrels_train.parquet          # Normalized query-to-document relations (exploded multi-positives)
└── splits/
    ├── random_5fold.json        # 5-fold cross-validation query splits
    └── doc_disjoint_split.json  # Unseen-document evaluation split (zero overlap of gold doc_ids)
```

### 2.1 Table Schemas

#### A. `documents.parquet`
- `doc_id` (string): Official document ID (e.g. `"740"`, `"280282"`).
- `title` (string): Normalized legal title extracted from name/passage (e.g. `"Quyết định 5868/QĐ-BYT 2018"`).
- `name_raw` (string): Raw filename/name from official context.
- `link` (string): Source link from context.
- `passage_raw` (string): Untouched original legal text.
- `passage_norm` (string): Unicode NFC, whitespace-normalized copy.
- `legal_number` (string): Extracted document number (e.g. `"5868/QĐ-BYT"`, `"17/2022/TT-BGTVT"`).
- `year` (string/int): Promulgation year (e.g. `2018`, `2022`).
- `doc_type` (string): Legal document type (e.g. `"Luật"`, `"Nghị định"`, `"Thông tư"`, `"Quyết định"`).
- `is_empty` (boolean): Flag for empty passages (audited, not silently dropped).

#### B. `chunks.parquet`
- `chunk_id` (string): Unique chunk identifier (e.g. `"740_macro_001"`, `"740_micro_002"`).
- `doc_id` (string): Reference to parent document ID.
- `granularity` (string): `"micro"` or `"macro"`.
- `article` (string): Heading of Điều/Mục (e.g. `"Điều 12. Trách nhiệm của..."`).
- `clause` (string/null): Khoản/Điểm identifier if micro chunk.
- `text_raw` (string): Raw text of the chunk.
- `text_norm` (string): Cleaned and contextualized searchable content.
- `parent_chunk_id` (string/null): For micro chunks, links to the encompassing macro chunk.
- `token_count` (int): Approximate token count.

#### C. `queries_train.parquet` & `qrels_train.parquet`
- `queries_train.parquet`: `query_id`, `question_raw`, `question_norm`, `gold_count`.
- `qrels_train.parquet`: `query_id`, `doc_id`, `relevance` (integer 1). Multi-positive queries from `train.json` are exploded into individual rows.

---

## 3. Dual-Granularity Hierarchical Legal Chunker

### 3.1 Parser Architecture
Legal passages follow Vietnamese legislative hierarchy: `Chương` $\rightarrow$ `Mục` $\rightarrow$ `Điều` $\rightarrow$ `Khoản` $\rightarrow$ `Điểm`.

1. **Macro Chunks (Dense & Reranker Evidence)**:
   - Target size: 400–800 tokens.
   - Encompasses the full `Điều` (Article) or a logical grouping of related `Khoản` along with legal document headers (`[VĂN BẢN]`, `[ĐIỀU KHOẢN]`).
   - Ensures that legal subject, condition, and obligation stay in the same contextual window.
2. **Micro Chunks (Lexical & Exact Match BM25)**:
   - Target size: 100–250 tokens.
   - Focuses on individual `Khoản` (Clause) or specific provisions.
   - Eliminates document length dilution in BM25 scoring and captures precise legal keywords, acronyms, numbers, and definitions.
3. **Fallback Handling**:
   - For unstructured texts lacking explicit `Điều/Khoản`, a sliding window of 600 characters with 150 characters overlap is applied while preserving document metadata.
   - Empty passages generate a single metadata-only chunk with `is_empty=True`.

---

## 4. Dual Validation Protocols (Simulating Real Codabench)

To avoid overfitting to seen training documents and guarantee high test generalization, two independent validation splits are generated:

### 4.1 Protocol 1: Random 5-Fold Cross Validation
- **Objective:** Evaluates retrieval accuracy on distribution with seen document anchors and assesses the strength of the Train-Question Memory branch.
- **Split:** Queries in `train.json` partitioned into 5 balanced folds (80% train / 20% val).

### 4.2 Protocol 2: Document-Disjoint Validation Split
- **Objective:** Evaluates out-of-distribution generalization to completely unseen legal documents.
- **Split:** Queries grouped by their gold `doc_id` clusters such that no gold document in the validation fold ever appears as a positive in the training fold.

### 4.3 Validation Metrics Suite
1. **Candidate Recall@K ($K \in \{20, 50, 100\}$)**: Measures the proportion of gold documents successfully captured in the candidate retrieval pool before reranking.
2. **Leaderboard Metrics**:
   - Official Codabench $\text{Recall}$ (Macro Average, strictly capped at Top 5).
   - Official Codabench $\text{Precision}$ (Macro Average).
   - $\text{Recall}@1$, $\text{Recall}@3$, $\text{Recall}@5$.
3. **Automated Error Taxonomy Logging**:
   - `MISSING_FROM_INDEX`: Gold chunk/doc not indexed.
   - `LEXICAL_PARAPHRASE`: BM25 missed due to phrasing, dense recovered.
   - `DOMAIN_CONFUSION`: Wrong legal domain retrieved due to shared generic entities.
   - `SAME_LAW_WRONG_ARTICLE`: Correct document found, but wrong provision ranked higher.
   - `RERANK_FAILURE`: Gold present in candidate pool (Rank 6–50) but missed in final Top 5.

---

## 5. Multi-Branch Hybrid Candidate Retrieval Engine

Every query is dispatched in parallel to four complementary retrieval branches:

```
                      Query
                        │
      ┌─────────────────┼─────────────────┬────────────────┐
      ▼                 ▼                 ▼                ▼
 Branch 1          Branch 2          Branch 3         Branch 4
   BM25              Dense            Question          Exact
(Micro Chunks)   (Macro Chunks)        Memory          Matcher
      │                 │                 │                │
      └─────────────────┼─────────────────┴────────────────┘
                        │
                        ▼
                 Candidate Union
            (Top 50 unique doc candidates)
```

1. **Branch 1: Micro-Chunk BM25 Inverted Index**
   - Indexes micro chunks with field boosting on title, legal number, and article keywords.
   - Aggregates chunk BM25 scores to document level using $\max(\text{score}) + 0.1 \times \text{mean}(\text{score})$.
2. **Branch 2: Macro-Chunk Dense Retriever (BGE-M3 / Vector Index)**
   - Encodes macro chunks with `BAAI/bge-m3` dense representation (normalized vectors).
   - Computes cosine similarity against query vector; aggregates to document level via top macro chunk score.
3. **Branch 3: Train-Question Memory**
   - Builds a dual index (TF-IDF character n-grams + dense embedding) over all training questions.
   - For a query, retrieves top-matching historic questions ($sim > 0.82$) and transfers their official gold document votes as high-confidence prior candidates. (In validation, self-query is strictly excluded).
4. **Branch 4: Exact Legal Identifier & Metadata Matcher**
   - Regex extraction of legal numbers (e.g. `\d+/\d+/(?:TT|NĐ|QĐ|CT|NQ)-[A-Z]+`), years, and law titles.
   - Performs exact keyword matching against `documents.parquet` metadata.
5. **Candidate Union**:
   - Fuses top results from all 4 branches into a candidate pool of $K=50$ unique documents per query.
   - Target: Candidate Recall@50 $> 94\%$.

---

## 6. Weak Positive Localization & Hard-Negative Mining

### 6.1 Weak Positive Localization
`train.json` provides $(query, gold\_doc\_id)$ without specific paragraph annotations.
- For each $(query, gold\_doc\_id)$, compute similarity of query against all macro chunks belonging to $gold\_doc\_id$ using BM25 + Dense similarity.
- Select the top-1 (or top-2) highest-scoring macro chunks as the localized **positive evidence pack**.

### 6.2 Hard Negative Mining
- For each training query, run the multi-branch retrieval engine over the corpus.
- Exclude all gold documents for this query.
- Guard against false negatives (do not mine documents that are positives for duplicate/near-identical train queries).
- Select top 5–15 highest-ranked non-gold candidates as **hard negatives** (e.g. same law wrong article, confusing legal domain).

### 6.3 Neural Fine-Tuning / Reranker Training
- Export `reranker_train.parquet`: `(query, positive_evidence, hard_negative_evidence)`.
- Train Cross-Encoder `BAAI/bge-reranker-v2-m3` using pairwise Margin / BCE ranking loss.

---

## 7. Out-Of-Fold Feature Generation & Learned Fusion

To combine heterogeneous ranking signals without feature leakage, out-of-fold candidate features are generated across 5 folds:

### 7.1 Feature Set per $(Query, Candidate\_Doc)$
1. `bm25_rank`, `bm25_score`
2. `dense_rank`, `dense_score`
3. `memory_rank`, `memory_similarity`, `memory_vote_count`
4. `exact_legal_num_match`, `exact_year_match`, `exact_doc_type_match`
5. `reranker_best_score`, `reranker_second_score`, `evidence_chunk_count`
6. `candidate_branch_overlap_count` (how many branches voted for this doc)

### 7.2 Fusion Model & Selection Policy
- **Baseline:** Reciprocal Rank Fusion (RRF) with component weights:
  $$\text{RRF}(d) = \sum_{b \in \text{Branches}} \frac{w_b}{k + \text{rank}_b(d)} + w_{\text{rerank}} \cdot \text{sigmoid}(\text{reranker\_score})$$
- **Learned Ranker:** LightGBM / LambdaMART trained on OOF candidate features.
- **Final Top-5 Selection:** Sort by final fused score $\rightarrow$ Deduplicate $\rightarrow$ Take exactly Top 5 document IDs.

---

## 8. Inference Pipeline & Submission Verification

```
Test Queries (public-official.json)
       │
       ▼
Normalize & Extract Queries
       │
       ▼
Multi-Branch Retrieval (BM25 + Dense + Memory + Exact)
       │
       ▼
Candidate Pool Union (Top 50 candidates/query)
       │
       ▼
Cross-Encoder Evidence Pack Scoring
       │
       ▼
Fusion Ranker & Top-5 Unique Document Selector
       │
       ▼
Automated Submission Validator
  ├── 100% Query Key Coverage
  ├── Strictly 1 <= len(answer) <= 5
  ├── Unique string document IDs
  └── All IDs verified in official corpus
       │
       ▼
submission.json & submission.zip
```

---

## 9. Invariants & Quality Assurance Checklist

- [x] Canonical dataset built exclusively from official files in `selected-contexts/` and `train.json`.
- [x] Zero external data, zero API dependencies.
- [x] All 8,532 context documents accounted for with 0 dropped documents.
- [x] Total neural parameters: BGE-M3 + BGE-reranker-v2-m3 $\approx 1.13\text{B} < 4.0\text{B}$.
- [x] Identical scoring function to official Codabench scorer.
- [x] Dual validation (Random 5-fold CV + Document-Disjoint Split) verified before any submission artifact is frozen.
