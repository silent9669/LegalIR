# LegalIR Task 1 — Kaggle High-Score Training & Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use a disciplined plan-execution workflow (for Superpowers-enabled agents, use `superpowers:subagent-driven-development` or `superpowers:executing-plans`) and implement this document task-by-task with verification checkpoints.

**Goal:** Refactor the LegalIR repository into a clean, single-source-of-truth Kaggle T4 x2 training/inference system that maximizes official Task 1 Recall while remaining fully competition-compliant and producing a rigorously validated `submission.zip`.

**Architecture:** Legal-aware dual lexical retrieval + dense retrieval + exact matching + fold-safe train-question memory feed a high-recall candidate union. Query-aware evidence localization and a genuinely fine-tuned cross-encoder reranker then feed OOF-validated fusion before deterministic top-5 selection and official-scorer-equivalent submission validation.

**Tech Stack:** Python, PyTorch, Transformers, PEFT/LoRA, Accelerate/DDP, bm25s, PyVi, pandas/pyarrow, scikit-learn, optional LightGBM/FAISS, Hugging Face local model weights, Kaggle T4 x2.

**Spec:** This file is intentionally a combined design specification + implementation plan because the requested deliverable is one self-contained Markdown instruction file for the coding agent.

## Global Constraints

- Official Task 1 Recall is the primary model-selection metric; Precision is secondary/tie-break.
- The total learned parameter count of the entire final system must be **strictly below 4,000,000,000**.
- Use **Task 1 organizer data only**; no Task 2 data, external corpus, synthetic augmentation, or external labeling/inference API.
- Final answers must contain **1-5 unique valid document IDs**; default to exactly 5 unless full OOF proves a smaller dynamic `k` preserves Recall.
- Kaggle target hardware is **GPU T4 x2**; use both GPUs intentionally where stable.
- `HF_TOKEN` must come from Kaggle Secrets and must never be printed or committed.
- Do not claim improvement without saved scorer-equivalent validation evidence.

---

> **For the coding agent:** Treat this document as the implementation contract. Do not merely suggest changes. Inspect the repository first, implement the required changes, add tests, refactor duplicated code, and leave the repository in a reproducible state that can be run end-to-end on Kaggle GPU **T4 x2**. Do not claim a score improvement that has not been measured with the official scoring semantics.

## Mission

Repository: `https://github.com/silent9669/LegalIR`

Observed audit baseline commit at the time this brief was written: `70ada3b45544e21e805e6e184d7e3e702327d4f5` on `main`. If `HEAD` differs, re-audit the new `HEAD` before editing and record the actual starting commit.

The goal is to turn the current LegalIR workspace into a **clean, single-source-of-truth, Kaggle-first training and inference pipeline** optimized for the highest possible UIT Data Science Challenge 2026 Task 1 score under the official constraints.

Primary success metric: **official mean Recall**, because the organizer uses Recall as the ranking metric and Precision only as the tie-breaker.

The finished Kaggle notebook must be capable of:

1. discovering the attached official LegalIR canonical data;
2. authenticating to Hugging Face via Kaggle Secret `HF_TOKEN` without printing the token;
3. building/loading all retrieval indexes efficiently;
4. constructing leakage-safe training pairs from Task 1 organizer data only;
5. **actually training** at least the reranker, not simulating training;
6. optionally fine-tuning a dense retriever if and only if validation proves it improves candidate recall;
7. running full leakage-safe OOF validation, not a 100-query sample presented as final CV;
8. selecting the best validated configuration;
9. retraining the selected trainable component(s) on all 7,000 Task 1 training questions;
10. running public-test inference;
11. validating `submission.json` against strict invariants and official scorer behavior;
12. creating `submission.zip` containing **only** `submission.json`;
13. exporting model weights/adapters, indexes/config manifests, CV metrics, parameter audit, runtime metadata, and the final submission under `/kaggle/working/`.

The coding agent should optimize correctness and Recall first, then runtime. Do not trade away Recall merely to make the notebook look faster.

---

# 1. Official Rules — Non-Negotiable

Use the repository copies / supplied source files as the source of truth:

- `rules.txt`
- `scoring.py`
- `DSC2026_Task1_LegalIR_Data_Overview.docx`
- `train.json`
- `public-official.json`
- `selected-contexts.zip`

The following requirements are mandatory.

## 1.1 Model parameter budget

The **total number of learned parameters of every model used by the Task 1 system must be strictly below 4,000,000,000 parameters**. This is the sum across embedding models, rerankers, and any other learned model used in the final pipeline.

LoRA, quantization, 8-bit/4-bit loading, pruning, and other memory optimizations do **not** reduce the model's parameter count for competition-rule purposes. A model whose architecture contains more than 4B parameters remains illegal even when quantized.

Create a hard-failing `parameter_audit.json` generator. The final Kaggle run must abort before submission packaging if the full final system is `>= 4_000_000_000` learned parameters.

## 1.2 Data restrictions

Allowed training/evaluation data:

- Task 1 `train.json`;
- Task 1 `selected-contexts.zip` / canonical derivatives of it;
- Task 1 `public-official.json` questions for inference only;
- organizer-provided Task 1 metadata and derived features from those files.

Forbidden:

- any external legal corpus;
- crawling or scraping legal websites;
- Task 2 data;
- synthetic data augmentation;
- pseudo-labels produced using an external API/model service;
- external API calls for inference, labeling, retrieval, or generation.

Pretrained open models are allowed under the organizer rule, provided the model/license is acceptable for the competition and the full final system remains under the total 4B parameter cap.

Internet may be used on Kaggle for package/model-weight downloads. It must not be used to acquire additional training documents or labels.

## 1.3 Submission scoring semantics

The official scorer behaves as follows:

```python
recall_q = len(set(gold) & set(pred)) / len(gold)
```

only when:

```python
0 < len(pred) <= 5
```

Otherwise that question receives Recall = 0 and Precision = 0.

Important implications:

- Never output more than 5 IDs.
- Never output an empty list.
- Never output duplicate IDs. Duplicates do not increase set overlap and can reduce Precision because the scorer divides by list length.
- The prediction key set must exactly equal the test query key set. Do not merely compare counts.
- Every predicted ID must be a valid official context ID.
- The final ranking objective is Recall first. Do not reduce `k` merely to improve Precision unless OOF validation proves **zero Recall regression** and a Precision gain.

Training-label distribution from the supplied `train.json`:

- 6,447 / 7,000 questions have 1 relevant document;
- 485 have 2;
- 53 have 3;
- 14 have 4;
- 1 has 5;
- total qrels = 7,637.

Therefore the default final submission policy should be **exactly top 5 unique valid documents per query**. A dynamic `k < 5` policy is allowed only after full OOF evidence that official Recall is unchanged or higher.

## 1.4 Empty and duplicate training passages

Organizer clarification states that empty/duplicate passages exist in training data, while public/private gold answers do not have that issue. Preserve the current duplicate/empty audit, and use duplicate groups to prevent false negatives during hard-negative mining.

Do not silently discard a document ID from the corpus if doing so could make a training qrel unresolved. Empty documents may require metadata-only fallback representation for training robustness, but they should not be artificially favored during public inference.

---

# 2. Current Repository Audit — Problems That Must Be Fixed

Before modifying code, reproduce and document these findings against current `HEAD`. If any item has already been fixed, mark it verified rather than reimplementing it.

## 2.1 Current benchmark reference

The accepted benchmark artifact currently reports approximately:

```text
Random 5-fold:
Recall@5             = 0.753576
Precision@5          = 0.159971
Candidate Recall@20  = 0.898279
Candidate Recall@50  = 0.944217
Candidate Recall@100 = 0.965229
Candidate Recall@150 = 0.973495

Document-disjoint:
Recall@5             = 0.677202
Precision@5          = 0.146429
Candidate Recall@50  = 0.937619
```

Treat these values as a historical reference only. Recompute a fresh baseline using the current `HEAD` and the official scorer semantics before making model-selection claims.

The gap between candidate recall and final Recall@5 is the main optimization opportunity. Candidate Recall@150 near 97.35% means a much better ranking stage can potentially recover substantial score without needing a completely new corpus retriever.

## 2.2 `train_reranker.py` currently does not perform real training

At the audited commit, `src/training/train_reranker.py` loads training pairs and writes a `training_manifest.json`, but does not instantiate an optimizer/loss, does not run backward passes, does not update model weights, and does not save a genuinely trained checkpoint.

This is a **critical defect**. Replace simulated training with real supervised reranker training.

Add a test proving that training changes trainable parameters/adapters. A manifest saying `status=completed` is not evidence of training.

## 2.3 Notebook/source drift

The Kaggle notebook currently embeds copies of retrieval/reranking classes instead of using repository modules. This has already caused source/notebook behavior to diverge.

Examples observed in the notebook generator:

- legal signals are extracted but BM25 source-style legal boosts are not actually applied;
- corpus/query BM25 tokenization is inconsistent;
- evidence map uses the first two macro chunks per document rather than query-relevant chunks;
- dense model is offloaded to CPU before many train/public query embeddings are computed;
- top-5 reranking is evaluated on only the first 100 validation queries per fold;
- reranker `max_length` is 256 despite repository config specifying 512;
- T4 x2 is not intentionally utilized;
- `torch` is reinstalled inside Kaggle, risking CUDA environment breakage;
- fallback IDs can be derived from an unordered set.

The notebook must become a thin orchestrator. Pipeline logic must live in tested Python modules.

## 2.4 Duplicate source architectures

The repository currently contains overlapping implementations under combinations of:

- `src/common/`
- `src/task1/`
- `src/retrieval/`
- `src/ranking/`
- `src/training/`
- `src/pipeline/`

Do not preserve two independent implementations of BM25, dense retrieval, evidence building, reranking, or candidate fusion.

The canonical implementation after refactor should use:

```text
src/core/
src/dataset/
src/models/
src/retrieval/
src/ranking/
src/training/
src/evaluation/
src/pipeline/
```

`src/common/` and `src/task1/` may temporarily contain compatibility imports during migration, but duplicated implementation logic must be removed. Delete compatibility wrappers only after all callers/tests are migrated.

---

# 3. Required Target Architecture

Implement and validate this architecture. Components may be removed from the final model only if an ablation proves they are useless or harmful.

```text
                    OFFICIAL TASK-1 DATA ONLY
                             |
                             v
                   Canonical Legal Dataset
                             |
                +------------+------------+
                |                         |
                v                         v
        legal-aware micro chunks    legal-aware macro chunks
                |                         |
     +----------+----------+              |
     |                     |              |
     v                     v              v
Raw/legal BM25       PyVi BM25       Dense Retriever(s)
     |                     |              |
     +----------+----------+--------------+
                |                         |
                +--------+--------+-------+
                         |        |
                         v        v
                  Exact matcher  Train-question memory
                         \        /
                          \      /
                           v    v
                 candidate union + features
                         TOP 150-200
                              |
                              v
                 query-aware intra-document
                    evidence localization
                      2-4 chunks/doc
                              |
                              v
                    trained reranker(s)
                     TOP 80-150 budget
                              |
                              v
                     OOF learned fusion
                (or validated weighted RRF)
                              |
                              v
                         TOP 5 UNIQUE
                              |
                              v
                 strict submission validator
                              |
                              v
                        submission.zip
```

The exact candidate/rerank cutoffs are hyperparameters to validate. Never hard-code 50 merely because the old notebook used 50.

---

# 4. Workspace Refactor — Single Source of Truth

The refactor must make Kaggle execution safer, not merely aesthetic.

## 4.1 Desired repository responsibilities

Use the existing top-level structure where possible:

```text
LegalIR/
├── README.md
├── requirements.txt
├── requirements-kaggle.txt
├── configs/
│   ├── pipeline.yaml
│   ├── kaggle.yaml
│   └── experiments/
│       ├── reranker_lora.yaml
│       ├── dense_finetune.yaml
│       └── fusion.yaml
├── legalir_training.ipynb              # canonical Kaggle notebook, thin orchestrator
├── scripts/
│   ├── 01_build_dataset.py
│   ├── 02_build_indexes.py
│   ├── 03_build_training_pairs.py
│   ├── 04_train_reranker.py
│   ├── 05_run_oof.py
│   ├── 06_train_final.py
│   ├── 07_predict_submission.py
│   ├── 08_validate_submission.py
│   └── audit_parameters.py
├── src/
│   ├── core/
│   ├── dataset/
│   ├── models/
│   ├── retrieval/
│   ├── ranking/
│   ├── training/
│   ├── evaluation/
│   └── pipeline/
├── tests/
└── artifacts/
    ├── task1/data/                      # canonical organizer-derived data; may be Git ignored if large
    └── local/                           # all generated indexes/checkpoints/runs; Git ignored
```

Do not create new parallel folders merely to preserve old code.

## 4.2 Notebook rule

`legalir_training.ipynb` must not contain independent copies of the core retrieval/ranking classes.

The notebook should only:

1. set environment/seeds;
2. read Kaggle Secret `HF_TOKEN`;
3. install missing non-CUDA-critical dependencies;
4. clone/checkout the repository into `/kaggle/working/LegalIR` when source modules are not already available;
5. locate the attached dataset;
6. load the YAML configuration;
7. call the tested training/evaluation/inference entrypoints;
8. display concise metrics and artifact paths.

Do **not** reinstall `torch`, CUDA, or NVIDIA runtime packages in the notebook unless a tested environment failure makes it absolutely necessary.

The notebook must print:

- source Git commit SHA;
- Python version;
- PyTorch version;
- CUDA version;
- GPU count;
- both GPU names/memory when T4 x2 is available;
- package versions relevant to reproduction;
- config hash;
- dataset manifest hash.

Never print `HF_TOKEN`.

## 4.3 Kaggle source bootstrap

If repository modules are absent in the Kaggle runtime, use a deterministic bootstrap similar to:

```text
/kaggle/working/LegalIR
```

Clone the public repository and record the resulting commit SHA. Do not download any external data during this step.

The final run manifest must contain the exact commit so the run can be reproduced later even if `main` changes.

---

# 5. Canonical Data and Preprocessing Requirements

Preserve the existing canonical dataset semantics unless an ablation proves a change helps.

Current manifest reference:

```text
total_documents     = 8,532
total_chunks        = 1,153,876
total_micro_chunks  = 934,416
total_macro_chunks  = 219,460
total_queries       = 7,000
total_qrels         = 7,637
duplicate_groups    = 4
empty_documents     = 20
```

## 5.1 Legal structure

Preserve legal hierarchy when possible:

- document title;
- legal number;
- document type;
- year;
- Chương;
- Mục;
- Điều;
- Khoản;
- Điểm.

Micro chunks should favor precise statutory retrieval. Macro chunks should preserve enough surrounding context for semantic retrieval/reranking.

## 5.2 Normalization

Create separate representations rather than one destructive normalization:

```text
text_raw             # original organizer text
text_nfc             # Unicode NFC + whitespace normalization
text_lexical         # legal punctuation/identifiers preserved
text_pyvi            # PyVi-segmented representation for Vietnamese lexical/dense models that benefit from it
```

Do not remove `/`, `-`, digits, article numbers, years, or legal suffixes from the legal-aware lexical representation.

Examples that must survive normalization:

```text
123/2020/NĐ-CP
15/2021/TT-BTC
Điều 15
Khoản 2
Điểm a
2023
```

---

# 6. Candidate Retrieval — Raise Recall Before Reranking

Candidate retrieval must be optimized using full OOF candidate recall at:

```text
@20, @50, @100, @150, @200
```

## 6.1 Dual lexical retrieval

Implement two deliberately different BM25 branches.

### Branch A — raw/legal BM25

Tokenize the corpus and query consistently using a legal-preserving tokenizer. Preserve identifiers and Vietnamese words. Add query-time exact legal boosts for:

- exact legal document number;
- exact `Điều` number;
- exact `Khoản` number;
- exact `Điểm` where reliable;
- year;
- document type/title match.

Do not compute `signals = extract_legal_signals(query)` and then ignore them.

### Branch B — PyVi BM25

Both the corpus and the query must use the same PyVi tokenization. Do not index whitespace tokens and query with underscore-joined PyVi tokens.

Use this branch for natural-language semantic overlap that raw legal tokenization may miss.

## 6.2 Exact legal matcher

Keep a deterministic exact branch. It should produce features, not blindly override ranking.

Required features include:

```text
exact_legal_number
exact_article
exact_clause
exact_point
exact_year
exact_doc_type
exact_title_overlap
exact_score
```

An exact document number match should be very strong, but the final weight must be validated OOF.

## 6.3 Dense retrieval

Keep the current Vietnamese DEk21 baseline as a required benchmark:

```text
CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2
```

Do not assume it is final best.

The agent may benchmark additional pretrained models if all of the following hold:

1. weights are downloadable and controllable locally;
2. license is compatible with the organizer's academic/non-commercial constraints;
3. no external training corpus is added by the team;
4. total final system parameter count remains <4B;
5. full OOF candidate recall improves enough to justify runtime.

High-value candidates to benchmark if feasible include multilingual embedding models such as:

```text
BAAI/bge-m3
Qwen/Qwen3-Embedding-0.6B
```

Do not adopt either merely because this brief names it. Measure it on this dataset.

### Dense encoding efficiency

Before offloading the dense model, batch-encode and cache:

- all macro chunks;
- all 7,000 training queries;
- all public test queries.

Never re-run a transformer forward pass for the same query text in each CV fold when the underlying pretrained dense encoder is unchanged.

Store embeddings as `float16` or `float32` according to measured recall/stability. Normalize once. Use matrix multiplication/FAISS efficiently.

If the dense encoder is fine-tuned per fold, fold-specific query/corpus embeddings must be regenerated only for that trained checkpoint.

## 6.4 Train-question memory

Train-question memory is allowed because it uses organizer Task 1 training data.

CV requirements:

- memory must be constructed from training-fold questions only;
- validation qrels must never enter that fold's memory;
- if searching a training question during training-pair generation, exclude its own `query_id`;
- dense query embeddings should be taken from the precomputed cache when possible.

Final public inference may build memory from all 7,000 training questions and their Task 1 qrels.

Validate thresholds such as:

```text
min_similarity: 0.78, 0.82, 0.86, 0.90
neighbors:      3, 5, 8, 10
```

Do not overfit these values on one fold.

## 6.5 Candidate union

Retrieve enough documents from each branch before fusion. Start with:

```text
raw BM25:        top 150-250
a PyVi BM25:     top 150-250
dense:           top 100-200
memory:          top 10-30
exact:           all confident matches, bounded
```

Union unique document IDs and generate a candidate feature record for each query/document pair.

Candidate union target should be evaluated at 150 and 200. Historical Candidate Recall@150 is ~0.9735. New work must not regress this without a clearly superior Recall@5 outcome.

---

# 7. Query-Aware Evidence Localization — Required Before Reranking

Never rerank a document using its first two macro chunks simply because they are first in source order.

For every candidate document, select query-relevant evidence from within the document.

Use a fast intra-document scoring combination based on available signals:

```text
micro BM25 score
macro dense similarity
article/clause/legal-number match
query-token overlap
retrieval branch best_chunk_id
```

Select 2-4 complementary evidence chunks per candidate. Prefer non-identical evidence and avoid wasting context on adjacent duplicates unless both are needed.

The evidence builder should create a deterministic pack such as:

```text
[DOCUMENT] <title> <legal_number>
[EVIDENCE 1] <article/clause + text>
[EVIDENCE 2] <article/clause + text>
...
```

The query itself should be passed as the cross-encoder's first input, so do not redundantly repeat long copies of the query inside every passage unless measured to help.

Test that a query mentioning `Điều 61` can select the chunk containing `Điều 61` from a long document rather than always selecting early articles.

---

# 8. Hard-Negative Mining — Build Training Data That Matches the Failure Mode

Hard-negative quality is more important than simply creating many random negatives.

For each training query:

1. include every gold document as a positive;
2. retrieve a candidate pool with **fold-safe retrieval**;
3. exclude all gold IDs;
4. exclude duplicate-equivalent IDs using `duplicate_groups.json` to avoid false negatives;
5. sample hard negatives across difficulty bands.

Recommended negative buckets:

```text
A. top lexical false positives        ranks 1-10
B. top dense false positives          ranks 1-20
C. exact/legal-confusable negatives   matching year/article/title/number fragments
D. memory false positives             high neighbor-vote documents
E. reranker false positives           after one initial reranker pass, if iterative mining is used
F. medium negatives                   ranks 20-100 for diversity
```

Start with roughly 6-12 negatives per positive, but tune this based on training stability and Kaggle runtime.

Do not make the dataset overwhelmingly easy by using random documents as the majority of negatives.

Save mined pairs to Parquet with provenance fields:

```text
query_id
query_text
doc_id
label
negative_source
retrieval_rank
retrieval_score
evidence_chunk_ids
evidence_text
fold
```

The training-pair manifest must record counts per negative source and number of excluded duplicate/false-negative cases.

---

# 9. Real Reranker Fine-Tuning — Highest Priority Training Change

The baseline reranker is:

```text
BAAI/bge-reranker-v2-m3
```

It must remain a benchmark. Fine-tune it on organizer Task 1 pairs.

## 9.1 Training method

Use PEFT/LoRA first because it is practical on T4 x2 and does not change the competition's underlying parameter-count interpretation.

Recommended starting configuration, subject to OOF tuning:

```yaml
max_length: 512
precision: fp16
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2.0e-5
weight_decay: 0.01
warmup_ratio: 0.05
epochs: 1-3
per_device_train_batch_size: 2-8
gradient_accumulation_steps: chosen for effective batch 32-64
gradient_checkpointing: true
max_grad_norm: 1.0
seed: 42
```

Do not blindly assume target module names. Inspect the actual model and record the matched LoRA modules. Fail if the intended target list matches zero modules.

Use dynamic padding. Do not truncate all evidence to 256 tokens unless a 256-vs-384-vs-512 ablation proves it is better.

## 9.2 Loss alignment

Implement at least one ranking-aligned objective and compare against simple binary classification.

Acceptable approaches:

- pointwise BCE / binary cross-entropy on positive and hard-negative pairs;
- pairwise margin/logistic ranking loss;
- listwise softmax over one positive + N negatives for each query.

Prefer listwise or pairwise if stable because the task is ranking, not calibrated binary classification.

Select the objective by OOF official Recall@5, not training loss.

## 9.3 Proof that training is real

The training code must save:

- adapter/checkpoint weights;
- optimizer/scheduler state if resumability is enabled;
- training log;
- best-step metric;
- config;
- base-model revision;
- trainable parameter count;
- pre/post hash or numeric norm check showing trainable weights changed.

A unit/integration test must verify that a tiny training run changes at least one trainable parameter and produces a reloadable checkpoint.

## 9.4 Optional reranker candidates

If Kaggle time permits, benchmark one additional strong reranker under the 4B total budget, for example:

```text
Qwen/Qwen3-Reranker-0.6B
```

Only adopt it if the model license is compatible and OOF Recall@5 improves. If both rerankers are kept in the final system, both parameter counts must be included in the total <4B audit.

Do not turn this into an uncontrolled model zoo. Stop adding models when OOF improvements are not material.

---

# 10. Optional Dense Retriever Fine-Tuning

This is secondary to reranker training.

Only implement dense fine-tuning after the repaired baseline + trained reranker is working end-to-end.

Use organizer-only query-positive pairs and hard negatives. Suitable objectives include contrastive InfoNCE / multiple-negative ranking loss.

For each fold, train only on the fold's training questions, regenerate that fold's dense index, and evaluate validation candidate recall.

Adopt a fine-tuned dense checkpoint only if it improves candidate recall and final Recall@5 enough to justify the extra training/indexing cost.

Do not spend the entire Kaggle session retraining a dense model for negligible candidate gain when ranking remains the dominant error source.

---

# 11. Fusion — Use OOF Features, Not Hand-Tuned Intuition Alone

Weighted RRF is a strong baseline and must remain available.

Also implement a learned fusion option using only organizer-derived OOF features. Preferred first choice: a small LightGBM LambdaRank/ranker or another compact tree/linear ranker.

Candidate features may include:

```text
raw_bm25_rank
raw_bm25_score
pyvi_bm25_rank
pyvi_bm25_score
dense_rank
dense_score
dense_second_score
dense_margin
memory_rank
memory_similarity
memory_vote_count
exact_score
exact_* booleans
source_count
rrf_score
reranker_score
reranker_second_score
reranker_margin
query_length
gold-prior-safe document frequency feature from training folds only
```

OOF discipline is mandatory:

- for validation fold `f`, train the fusion model using candidates/features/labels from other folds only;
- evaluate on fold `f`;
- no validation qrel may influence the fold's fusion model;
- after model selection, train the final fusion model on all OOF training features and use it for public inference.

Keep RRF as fallback. If learned fusion does not beat validated RRF, use RRF.

---

# 12. Full Validation Protocol — No Fake CV

The agent must provide two validation protocols.

## 12.1 Primary: random 5-fold, leakage-safe

For all 7,000 training questions:

- 5 folds;
- fold-isolated question memory;
- fold-isolated trainable models;
- fold-isolated learned fusion;
- every validation query scored with the full final pipeline;
- official Recall/Precision semantics.

Report per fold and aggregate:

```text
Recall@1
Recall@3
Recall@5       <-- primary model-selection metric
Precision@5
Candidate Recall@20
Candidate Recall@50
Candidate Recall@100
Candidate Recall@150
Candidate Recall@200
runtime/query
```

Do not call a 100-query-per-fold reranker sample "5-fold CV". A small sample is allowed only as `smoke_cv` and must be labeled as such.

## 12.2 Secondary: document-disjoint robustness split

Keep the document-disjoint validation as a robustness diagnostic. Do not optimize exclusively on it if the competition's public/private distribution is closer to the random seen-document setting, but reject changes that catastrophically harm it without a strong primary-CV gain.

## 12.3 Baseline first

Before training changes, run the repaired current baseline and save:

```text
artifacts/local/runs/<run_id>/baseline_metrics.json
```

Every subsequent accepted change must have an ablation row against this baseline.

---

# 13. Experiment/Ablation Order

Do not modify ten things at once and then guess which helped. Run experiments in this order, caching reusable artifacts.

## Stage A — correctness repair

```text
A0 current HEAD baseline
A1 consistent legal/raw BM25 + legal boosts
A2 PyVi BM25 second branch
A3 dense embedding precompute/cache
A4 query-aware evidence localization
A5 candidate cutoff sweep 50/100/150/200
```

Accept repairs that improve or preserve full OOF Recall@5 and improve pipeline correctness.

## Stage B — supervised ranking

```text
B1 real BGE reranker LoRA, BCE baseline
B2 pairwise/listwise reranker objective
B3 hard-negative mix sweep
B4 evidence chunk count 2 vs 3 vs 4
B5 max_length 256 vs 384 vs 512
B6 rerank budget 50 vs 80 vs 100 vs 120 vs 150
```

## Stage C — fusion

```text
C1 weighted RRF tuned on OOF
C2 learned OOF fusion
C3 calibrated combination of retrieval + reranker score
```

## Stage D — optional model upgrades

Only after A-C are stable:

```text
D1 alternative dense model
D2 dense fine-tuning
D3 alternative reranker / two-reranker ensemble
```

For expensive experiments, use a clearly labeled development subset first, then confirm any promising change on full CV before accepting it.

---

# 14. Acceptance Gates for Score Improvements

A coding agent must not declare "better" because training loss decreased or because one fold improved.

Use these gates.

## 14.1 Correctness gate

Mandatory:

- all tests pass;
- official scorer parity test passes;
- no leakage tests fail;
- parameter audit passes;
- submission validator passes;
- trained checkpoint reloads and produces deterministic inference;
- candidate IDs are unique and valid.

## 14.2 Retrieval gate

Target:

- Candidate Recall@150 >= historical ~0.9735, preferably higher;
- measure @200 to discover remaining retrievable headroom.

A small regression is permissible only if final full-CV Recall@5 improves materially.

## 14.3 Ranking gate

Primary criterion:

```text
mean full 5-fold Recall@5
```

Prefer changes with at least +0.3 percentage-point absolute full-CV improvement; for high-cost model additions, prefer +0.5 point or more unless runtime is negligible.

Do not overstate tiny differences smaller than fold variance.

## 14.4 Robustness gate

Track document-disjoint Recall@5. Flag any major regression in the final report and justify it quantitatively.

---

# 15. T4 x2 Utilization

The notebook must detect:

```python
torch.cuda.device_count()
```

and print each device.

The second T4 must not be silently wasted.

Preferred utilization strategy:

## During embedding/index phase

If the encoder supports safe multi-GPU batching, shard encoding across both GPUs. Otherwise use GPU 0 for dense encoding and prepare CPU lexical indexes concurrently only when this does not create memory contention.

## During reranker training

Use `accelerate`/DDP with two processes when stable. Launch training as a script from the notebook rather than trying to hand-roll notebook DataParallel logic.

If two-GPU training proves unstable, fall back safely to one T4 and record why; do not introduce fragile distributed logic merely to report 100% GPU utilization.

## During reranker inference

Shard query/candidate batches across GPU 0 and GPU 1 where practical. Merge results deterministically by query ID.

Do not let both processes independently load multi-gigabyte copies of unnecessary corpus data into system RAM.

---

# 16. Kaggle Runtime Engineering

Assume a finite Kaggle GPU session and leave safety margin.

## 16.1 Cache boundaries

Save phase outputs under:

```text
/kaggle/working/legalir_run/
```

Recommended structure:

```text
legalir_run/
├── cache/
│   ├── dense/
│   ├── bm25/
│   ├── query_embeddings/
│   └── training_pairs/
├── checkpoints/
│   ├── reranker/
│   └── dense/                # optional
├── cv/
│   ├── oof_predictions.parquet
│   ├── oof_features.parquet
│   └── cv_report.json
├── logs/
├── run_config.yaml
├── environment.json
├── parameter_audit.json
├── ablation_report.csv
├── submission.json
├── submission.zip
└── submission_manifest.json
```

Every expensive phase should check a manifest/hash before recomputing.

## 16.2 Memory

Avoid retaining the 1.15M-row chunk DataFrame in multiple Python-list copies.

Use:

- Parquet column projection;
- categorical/integer dtypes where safe;
- NumPy arrays for embeddings;
- memory mapping when beneficial;
- batch operations;
- explicit `del`, `gc.collect()`, `torch.cuda.empty_cache()` only at phase boundaries.

Do not repeatedly call `to_dict("records")` on the entire 1.15M chunk table if a columnar implementation can avoid it.

## 16.3 Reproducibility

Set seeds for:

```text
python random
numpy
torch CPU
torch CUDA
training sampler
LightGBM / fusion learner
```

Record all seeds and deterministic settings. Full bitwise CUDA determinism is not required if it causes unacceptable performance, but reruns should be statistically stable and the setting must be documented.

---

# 17. Dependency Policy

Create `requirements-kaggle.txt` after testing the actual Kaggle environment.

Rules:

- do not install/upgrade `torch` by default;
- do not install a different CUDA runtime;
- pin libraries whose API differences affect the pipeline;
- avoid packages not used by final code;
- record exact installed versions in `environment.json`.

Likely required packages include only the libraries actually retained, such as:

```text
transformers
peft
accelerate
bm25s
pyvi
pandas
pyarrow
scikit-learn
lightgbm           # only if learned fusion wins
faiss-cpu          # only if used
huggingface_hub
tqdm
```

Do not add a dependency because it was used in an abandoned experiment.

---

# 18. Hugging Face Secret Handling

The user has already added the Hugging Face token to Kaggle Add-ons/Secrets.

Use:

```text
HF_TOKEN
```

Requirements:

- retrieve through `kaggle_secrets.UserSecretsClient()` or existing environment variable;
- pass token to Hugging Face download calls only where needed;
- never print it;
- never write it into notebook output, config, Git, manifest, exception dump, or shell history;
- notebook must still work with public models if the token is absent, where rate limits permit.

---

# 19. Submission Generation and Validation — Must Be Exact

Create a dedicated module such as:

```text
src/evaluation/submission.py
```

with strict validation before ZIP creation.

## 19.1 Required validation

Given `public-official.json` and the final predictions:

```text
prediction_keys == public_query_keys
```

must be exact.

For every query:

```text
answer is a list
1 <= len(answer) <= 5
all IDs are strings
len(answer) == len(set(answer))
all IDs exist in official context ID set
```

Default final policy:

```text
len(answer) == 5
```

unless a validated dynamic-k policy won full OOF Recall first.

## 19.2 Official scorer parity

Add a test comparing the project's local evaluation function to the supplied official `scoring.py` logic on controlled examples including:

- perfect predictions;
- one correct out of five;
- empty answer;
- six answers -> zero score for that query;
- duplicate IDs;
- multiple gold documents;
- missing/wrong query key behavior.

Do not modify the official scorer to make project outputs look better.

## 19.3 ZIP structure

`submission.zip` must contain exactly:

```text
submission.json
```

at archive root. No directory prefix. No manifest inside the ZIP unless organizer instructions later explicitly allow it.

The manifest remains outside the ZIP in `/kaggle/working/legalir_run/`.

## 19.4 Manifest

Create `submission_manifest.json` with at least:

```text
git_commit
config_sha256
dataset_manifest_sha256
query_count
prediction_count
all_answers_valid
all_ids_valid
parameter_total
model_names_and_revisions
submission_json_sha256
submission_zip_sha256
created_utc
```

---

# 20. Tests That Must Exist

Preserve useful current tests and add focused tests for the repaired system.

At minimum add tests covering:

```text
1. legal identifier normalization is lossless
2. raw BM25 corpus/query tokenizer consistency
3. PyVi BM25 corpus/query tokenizer consistency
4. legal boosts actually change ranking in a controlled example
5. exact matcher handles NaN/null metadata
6. candidate union deduplicates IDs deterministically
7. query memory excludes validation/self query
8. duplicate-group blacklist prevents false negatives
9. evidence localization selects query-relevant article/chunk
10. reranker evidence pack stays within token budget
11. tiny reranker training changes trainable weights
12. trained reranker checkpoint reloads
13. OOF fold construction has no label leakage
14. learned fusion trains without validation-fold labels
15. parameter audit sums all final learned components
16. parameter audit rejects >=4B
17. official scorer parity
18. submission exact query-key equality
19. submission max-five rule
20. submission uniqueness/valid-ID rule
21. ZIP contains only submission.json at root
22. deterministic fallback order
23. notebook parses as valid nbformat
24. Kaggle config paths resolve safely
```

Add a fast smoke mode for CI/local development that does not download large models. Use mocks/tiny fixtures where appropriate. Do not replace full Kaggle validation with mocks.

---

# 21. Required Kaggle Notebook Flow

The final notebook should be easy to inspect and safe to `Run All`.

Recommended cells:

```text
0. Title, objective, rules summary
1. Environment + HF secret + GPU x2 detection
2. Clone/resolve repository and print commit
3. Install minimal Kaggle dependencies
4. Discover/validate LegalIR data
5. Load config + run parameter/model-license preflight
6. Build/load lexical and dense caches
7. Build fold-safe hard-negative training pairs
8. Train reranker fold checkpoints / run OOF
9. Aggregate full CV + ablation table
10. Select final config
11. Train final model on all 7,000 train queries
12. Build full train-question memory / final indexes
13. Public inference
14. Strict submission validation
15. Create submission.json + submission.zip + manifests
16. Final summary: score estimates, artifacts, hashes, runtime
```

For development, the notebook may expose:

```text
RUN_MODE = "smoke" | "full"
```

The notebook checked into `main` for the final run should default to `full` or make the required one-line change obvious.

Never silently skip training because a checkpoint is missing. Never silently fall back from a trained reranker to the pretrained reranker without a loud warning and manifest flag.

---

# 22. Model Selection and Parameter Budget Strategy

The current stack is comfortably below 4B, so use the remaining budget only where OOF evidence justifies it.

Baseline components to audit exactly at runtime:

```text
DEk21 embedding model         ~0.135B (verify exactly)
BGE reranker v2 m3            ~0.568B (verify exactly)
```

Potential additional models may fit under the budget, but the agent must calculate exact counts from model configs/weights, not from this approximate table.

A reasonable high-score search order is:

```text
1. DEk21 + trained BGE reranker
2. add BGE-M3 or Qwen3-Embedding-0.6B as a second dense branch if candidate recall improves
3. test Qwen3-Reranker-0.6B as alternative/ensemble only if full OOF Recall@5 improves
```

Do not use a 4B model plus any other learned component because the total-system rule would almost certainly be violated. Never use a >4B model even quantized.

---

# 23. Logging and Experiment Registry

Every experiment must have a unique `run_id` and save:

```text
run_id
parent_run_id
source_git_commit
config hash
model revisions
parameter total
fold metrics
mean/std metrics
candidate cutoffs
training runtime
inference runtime
peak GPU memory
overall wall time
accepted/rejected status
reason
```

Maintain `ablation_report.csv` with one row per experiment. Do not overwrite historical rows.

An accepted experiment should identify exactly which previous experiment it beat.

---

# 24. Do Not Do These Things

The coding agent must not:

- use Task 2 data;
- use external legal data;
- use synthetic augmentation;
- call ChatGPT/OpenAI/Gemini/Claude or any external API for labels/features;
- use a model whose final-system parameter sum reaches/exceeds 4B;
- assume quantization changes parameter-count legality;
- hard-code the user's HF token;
- reinstall PyTorch blindly on Kaggle;
- duplicate pipeline implementation inside the notebook;
- keep both old and new BM25/reranker classes as active sources of truth;
- treat a 100-query sample as final CV;
- use public test answers (they are null and unavailable);
- train on validation-fold qrels;
- use duplicate gold-equivalent docs as negatives;
- claim "training completed" without changed weights;
- claim "score improved" without full scorer-equivalent validation;
- optimize Precision at the expense of Recall without explicit evidence;
- output >5 IDs;
- create a ZIP with extra files or nested paths;
- silently swallow OOM/model-download/training failures and produce a fallback submission as if it were the intended model.

---

# 25. Implementation Order for the Coding Agent

Follow this sequence. Do not start with model experimentation before the code path is trustworthy.

## Phase 1 — Freeze and measure current state

- inspect repository tree and recent commits;
- run current test suite;
- run static notebook inspection;
- record current baseline metrics/artifacts if executable locally;
- inventory duplicate implementations and callers;
- produce a short `docs/current_pipeline_audit.md`.

## Phase 2 — Refactor to one canonical pipeline

- choose canonical modules under `src/retrieval`, `src/ranking`, etc.;
- migrate unique working behavior from `src/common` / `src/task1`;
- update scripts/imports/tests;
- make notebook a thin orchestrator;
- remove/deprecate duplicate logic only after tests pass.

## Phase 3 — Fix retrieval correctness/performance

- dual BM25 with consistent tokenization;
- legal boosts;
- dense precompute/caching;
- fold-safe memory;
- candidate feature records;
- full cutoff diagnostics.

## Phase 4 — Fix evidence localization

- query-aware chunk selection;
- deterministic evidence packs;
- token-budget controls;
- tests on long legal documents.

## Phase 5 — Implement real reranker training

- hard-negative generation;
- LoRA/PEFT trainer;
- real loss/backprop/checkpoints;
- two-GPU `accelerate` path;
- tiny training integration test.

## Phase 6 — Full OOF pipeline

- 5 fold end-to-end runs;
- official scorer parity;
- ablations;
- choose rerank budget/evidence length/training objective.

## Phase 7 — Learned fusion

- generate OOF features;
- fit fold-isolated fusion;
- compare against RRF;
- accept only if full Recall@5 wins.

## Phase 8 — Optional model upgrades

- only after stable A-C pipeline;
- benchmark alternate dense/reranker models one at a time;
- run parameter audit for every candidate final stack.

## Phase 9 — Final Kaggle training + public prediction

- train chosen reranker on all train data;
- optional final dense fine-tune if selected;
- build final indexes/memory;
- infer all public questions;
- validate;
- package;
- save artifacts/hashes.

## Phase 10 — Documentation and cleanup

Update `README.md` with exact Kaggle steps and expected outputs. Remove stale instructions that describe simulated training or obsolete notebook behavior.

---

# 26. Definition of Done

The task is complete only when all of the following are true:

```text
[ ] Repository has one active implementation for each pipeline responsibility.
[ ] legalir_training.ipynb is a thin orchestrator and can Run All on Kaggle.
[ ] HF_TOKEN is read securely from Kaggle Secrets.
[ ] Kaggle T4 x2 is detected and intentionally used where stable.
[ ] PyTorch is not unnecessarily reinstalled.
[ ] BM25 corpus/query tokenization is consistent.
[ ] Legal signal boosts are actually applied and tested.
[ ] Candidate Recall is measured through @200.
[ ] Query-aware evidence localization is implemented.
[ ] Hard negatives are fold-safe and duplicate-safe.
[ ] Reranker training performs real backprop and saves changed weights.
[ ] Full 5-fold OOF reranked Recall@5 is computed on every validation question.
[ ] Document-disjoint robustness score is reported.
[ ] Fusion is evaluated OOF, not trained on validation labels.
[ ] Final model configuration is selected by official Recall first.
[ ] Final learned-parameter sum is <4B and audited.
[ ] Public predictions contain the exact public query key set.
[ ] Each answer has 1-5 unique valid IDs; default final submission uses 5.
[ ] submission.zip contains only submission.json at archive root.
[ ] Local evaluator matches supplied scoring.py semantics.
[ ] All unit/integration tests pass.
[ ] CV report, ablation report, model checkpoints, config, environment, hashes and submission are exported.
[ ] README contains reproducible Kaggle instructions.
[ ] No score claim is made without a corresponding saved metrics artifact.
```

---

# 27. Final Report the Coding Agent Must Return

When implementation is finished, return a concise engineering report containing:

## Repository changes

```text
files created
files modified
files deleted/deprecated
new canonical architecture
```

## Verification

```text
pytest result
notebook validation result
official scorer parity result
parameter audit result
```

## Validation table

At minimum:

```text
experiment | cand@50 | cand@100 | cand@150 | cand@200 | recall@5 | precision@5 | doc-disjoint recall@5 | runtime
```

Clearly distinguish:

- locally measured values;
- Kaggle full-run values;
- historical baseline values.

If the coding agent cannot execute Kaggle GPUs itself, it must **not invent Kaggle metrics**. It should state exactly what remains to be run and ensure the notebook emits all required metrics when the user runs it.

## Final chosen stack

Report:

```text
dense model(s) + revisions
reranker model(s) + revisions
fine-tuning method
fusion method
candidate cutoff
rerank cutoff
evidence chunks per document
max token length
exact total parameter count
```

## Kaggle execution

Report the exact notebook path and the expected final files in `/kaggle/working/legalir_run/`.

The single most important user-facing artifact after a successful full Kaggle run is:

```text
/kaggle/working/legalir_run/submission.zip
```

Do not finish the task with only code changes. Finish with a pipeline that is measurable, reproducible, competition-valid, and ready for a full Kaggle T4 x2 run.
