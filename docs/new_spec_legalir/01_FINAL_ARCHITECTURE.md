# Final Architecture

## 1. Design decision

The permanent architecture is split into two independent systems.

### System A — Validation / Artifact Factory

Responsibilities:

- canonical dataset verification;
- one-time static retrieval;
- fold-safe candidate materialization;
- fold-safe pair/evidence materialization;
- sharded 5-fold OOF;
- document-disjoint validation;
- fusion selection;
- production-config selection;
- production-bundle generation.

This system is resumable. A failed shard restarts only that shard.

### System B — Kaggle Final Train + Submit

Responsibilities:

- verify the exact approved runtime;
- verify the canonical dataset;
- verify production-bundle fingerprints;
- train exactly one final all-7,000-query LoRA adapter;
- rerank public candidates;
- apply approved fusion/top-5 logic;
- validate submission;
- package submission.

It must not:

- rebuild full BM25/PyVi/Dense indexes;
- rerun five-fold OOF;
- rerun document-disjoint validation;
- select new hyperparameters;
- change protected scoring config.

## 2. Data flow

```text
CANONICAL TASK1 V2
        │
        ├───────────────┐
        │               │
        ▼               ▼
STATIC RETRIEVAL     SPLIT METADATA
Legal/PyVi/Dense/    random 5-fold
Exact                doc-disjoint
        │               │
        ▼               │
STATIC CANDIDATE CACHE  │
        │               │
        ├───────────────┘
        ▼
FOLD-SAFE CANDIDATES
static branches + fold-local Question Memory
        │
        ▼
LAZY EVIDENCE / HARD-NEGATIVE MATERIALIZATION
        │
        ▼
FOLD 0..4 JOBS + DOC-DISJOINT JOB
        │
        ▼
OOF FEATURES / METRICS
        │
        ▼
FUSION + CONFIG PROMOTION
        │
        ▼
PRODUCTION LOCK
        │
        ▼
FINAL PAIRS + PUBLIC CANDIDATES/EVIDENCE
        │
        ▼
IMMUTABLE PRODUCTION BUNDLE
        │
        ▼
KAGGLE FINAL:
ONE FINAL LORA → PUBLIC RERANK → TOP5 → ZIP
```

## 3. Why this is superior

### Memory

The old runtime duplicated:

- full Parquet frames;
- BM25;
- PyVi;
- Dense/FAISS;
- macro Python dictionaries;
- PositiveLocalizer state;
- EvidencePackBuilder state.

The new architecture builds heavy retrieval once, writes results to disk, then unloads heavy state.

### Runtime

The final Kaggle session no longer spends hours on:

- PyVi indexing;
- Dense corpus encoding;
- five OOF fold trainings;
- doc-disjoint training.

The final run is dominated by one all-query LoRA training and public inference.

### Reliability

Every long-running factory stage has:

- deterministic inputs;
- output manifest;
- SHA-256;
- completion marker;
- resumable shard ID.

### Score integrity

Validation remains full 5-fold OOF and doc-disjoint. It is not removed; it is moved out of the final production session.

## 4. Protected score semantics

These values are production-protected unless promoted by leakage-safe OOF evidence:

- retrieval branches;
- RRF logic;
- branch weights;
- candidate depth;
- rerank depth;
- evidence-selection formula;
- negative-mining source policy;
- duplicate blacklist;
- BGE model;
- LoRA rank/alpha/dropout;
- loss;
- learning rate;
- max sequence length;
- effective training batch;
- fusion feature schema;
- final top-5 selector.

## 5. Success definition

A release is production-ready only when:

1. static retrieval cache passes live-vs-cache parity;
2. evidence layer passes legacy-vs-lazy parity;
3. all 5 folds pass leakage checks;
4. doc-disjoint evaluation passes;
5. selected config is locked;
6. production bundle verifies;
7. GitHub CI is green;
8. Colab T4 final-training smoke passes;
9. Kaggle runtime projection is within budget;
10. final Kaggle notebook pins the exact approved runtime.
