# LegalIR Refresh — Start Here

## Purpose

This package is the authoritative fresh-start design for DSC 2026 Task 1 LegalIR.

It inherits the existing official Task 1 canonical dataset and the validated scoring ideas from the current repository, but replaces the monolithic "index + 5-fold OOF + doc-disjoint + final train + public inference in one Kaggle session" runtime with a resumable two-system architecture:

```text
VALIDATION / ARTIFACT FACTORY
        ↓
immutable production bundle
        ↓
KAGGLE FINAL TRAIN + SUBMIT
```

The objective is not only to stop the host-RAM failure. The objective is to make the pipeline:

- finish reliably on Kaggle T4×2;
- preserve leakage-safe Recall@5 evaluation;
- preserve or improve ranking quality;
- make expensive work resumable;
- make every artifact reproducible and hash-verified;
- prevent one failed fold from wasting hours of completed work.

## Current evidence inherited from the old system

The old runtime proved:

- canonical v2 dataset identity is valid;
- 8,532 documents;
- 1,153,876 chunks;
- 934,416 micro chunks;
- 219,460 macro chunks;
- 7,000 train queries;
- 7,637 qrels;
- 1,000 public queries;
- 4 duplicate-document groups;
- DEk21 works on Tesla T4;
- BGE reranker-v2-m3 + LoRA works on Tesla T4;
- FAISS works;
- the learned-parameter budget is safely below 4B;
- fold-isolation and public-count bugs were already repaired.

The failed FULL run also proved that the monolithic execution model is unsuitable: it reached Fold-0 pair setup after hours of indexing, duplicated large retrieval/evidence state, and the kernel died before fold training began.

## Final release workflow

```text
source implementation
    ↓
GitHub CI GREEN
    ↓
artifact-factory validation jobs
    ↓
full 5-fold OOF + doc-disjoint accepted
    ↓
production bundle frozen
    ↓
Colab T4 production-contract smoke
    ↓
release-only notebook pin
    ↓
GitHub CI GREEN
    ↓
manual Kaggle T4×2 FINAL
```

## Read order

1. `01_FINAL_ARCHITECTURE.md`
2. `02_WORKSPACE_FOLDER_STRUCTURE.md`
3. `03_DATASET_INHERITANCE_AND_PROVENANCE.md`
4. `04_ARTIFACT_FACTORY_PIPELINE.md`
5. `05_STATIC_RETRIEVAL_CACHE_AND_EVIDENCE.md`
6. `06_VALIDATION_SHARDING_OOF_DOC_DISJOINT.md`
7. `07_FINAL_KAGGLE_T4X2_PRODUCTION.md`
8. `08_MEMORY_RUNTIME_BUDGET.md`
9. `09_IMPLEMENTATION_PLAN.md`
10. `10_TESTING_CI_COLAB_RELEASE_GATES.md`
11. `11_SCORE_OPTIMIZATION_AND_PROMOTION.md`
12. `12_FAILURE_RECOVERY_RESUME.md`
13. `13_OPERATIONS_RUNBOOK.md`
14. `14_ACCEPTANCE_CHECKLIST.md`
15. `99_LEGACY_DECISIONS.md`

## Non-negotiable invariants

- Task 1 data only.
- No external augmentation.
- Learned parameters < 4B.
- Official scoring semantics unchanged.
- Maximum five predicted docs/query.
- Recall@5 is primary.
- Precision@5 is tie-break.
- Question Memory is fold-local.
- Validation queries never enter fold training pairs or memory.
- Final Kaggle notebook never reruns five-fold validation.
- Production bundle must be provenance-verified before use.
