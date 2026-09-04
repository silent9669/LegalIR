# Fresh Workspace and Folder Structure

## Recommended repository layout

```text
LegalIR/
├── README.md
├── pyproject.toml / requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   ├── production/
│   │   ├── retrieval.yaml
│   │   ├── reranker.yaml
│   │   ├── fusion.yaml
│   │   └── production_lock.yaml
│   ├── factory/
│   │   ├── cache_build.yaml
│   │   ├── validation.yaml
│   │   └── memory.yaml
│   └── smoke/
│       └── colab_t4.yaml
│
├── src/
│   ├── core/
│   │   ├── hashing.py
│   │   ├── memory.py
│   │   ├── manifests.py
│   │   └── provenance.py
│   │
│   ├── data/
│   │   ├── canonical.py
│   │   ├── splits.py
│   │   └── duplicate_groups.py
│   │
│   ├── retrieval/
│   │   ├── bm25_legal.py
│   │   ├── bm25_pyvi.py
│   │   ├── dense.py
│   │   ├── exact.py
│   │   ├── question_memory.py
│   │   ├── static_cache.py
│   │   └── fusion.py
│   │
│   ├── evidence/
│   │   ├── macro_store.py
│   │   ├── selector.py
│   │   └── pair_materializer.py
│   │
│   ├── training/
│   │   ├── reranker.py
│   │   ├── samplers.py
│   │   └── coverage.py
│   │
│   ├── validation/
│   │   ├── fold_job.py
│   │   ├── doc_disjoint_job.py
│   │   ├── metrics.py
│   │   └── promotion.py
│   │
│   ├── bundle/
│   │   ├── builder.py
│   │   └── verifier.py
│   │
│   └── production/
│       ├── final_train.py
│       ├── public_rerank.py
│       └── submission.py
│
├── scripts/
│   ├── verify_dataset.py
│   ├── build_static_cache.py
│   ├── build_fold_pairs.py
│   ├── run_fold.py
│   ├── run_doc_disjoint.py
│   ├── select_production_config.py
│   ├── build_production_bundle.py
│   ├── verify_production_bundle.py
│   ├── run_colab_t4_smoke.py
│   ├── run_kaggle_final.py
│   └── verify_release.py
│
├── tests/
│   ├── unit/
│   ├── parity/
│   ├── leakage/
│   ├── memory/
│   ├── integration/
│   └── release/
│
├── notebooks/
│   ├── colab_t4_smoke.ipynb
│   └── kaggle_final.ipynb
│
├── artifacts/
│   ├── factory/
│   │   ├── static_cache/
│   │   ├── evidence/
│   │   ├── folds/
│   │   ├── doc_disjoint/
│   │   └── fusion/
│   ├── bundle/
│   └── local/
│
└── docs/
    ├── architecture/
    ├── plans/
    └── runbooks/
```

## Data must stay outside source

The official canonical dataset is read-only:

```text
data/task1_canonical_v2/
```

or an external mounted path.

Never commit or rewrite the canonical Parquets as part of the refresh.

Derived data belongs under:

```text
artifacts/
```

or a separate Kaggle/Drive derived-cache dataset.

## Fresh-start Git strategy

Recommended:

```text
main
  old validated history retained

refresh/factory-v1
  clean implementation branch

release/factory-v1
  approved runtime + generated notebooks
```

Use a clean worktree when implementing the refresh.

Do not delete the old runtime until the new system passes parity and validation gates.
