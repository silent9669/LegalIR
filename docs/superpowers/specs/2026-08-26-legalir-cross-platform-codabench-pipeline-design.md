# LegalIR Cross-Platform Codabench Pipeline Design

**Status:** Approved for implementation planning  
**Date:** 2026-08-26  
**Primary objective:** Maximize official Task 1 Recall@5 using leakage-free, Codabench-equivalent validation while keeping shared data reproducible and model artifacts local to each experimenter.

## 1. Context

The current repository has a valid baseline data package and a runnable BM25 + exact matcher + character-TFIDF question-memory pipeline. It does not yet run the full pipeline described in the Notion Task 1 specification. Dense BGE-M3 retrieval, neural question memory, cross-encoder reranking, OOF feature generation, and LightGBM fusion are not connected to production validation or inference.

The existing benchmark is not an accepted model-selection result because question memory was built from all labeled queries, including validation queries, and the committed benchmark ran only one random fold. Existing scores must be labeled `legacy_leaky_baseline` and must not be compared as if they were strict cross-validation results.

The project is developed collaboratively across macOS and Windows. Shared datasets, schemas, split definitions, validators, tests, and accepted benchmark summaries belong on `main`. Model weights, embeddings, checkpoints, training caches, and full experiment runs remain local and are never committed.

## 2. Goals

1. Rebuild and validate a canonical dataset that follows the latest Notion requirements.
2. Reproduce the official Codabench scoring behavior and submission constraints locally.
3. Run all five random folds and a strict document-disjoint split without label leakage.
4. Implement the full retrieval and ranking path:
   - fielded micro-chunk BM25;
   - macro-chunk BGE-M3 dense retrieval;
   - fold-safe lexical and dense train-question memory;
   - exact legal identifier and metadata matching;
   - evidence-pack construction;
   - BGE reranker scoring;
   - OOF LightGBM fusion;
   - strict top-five selection.
5. Download and use pinned pretrained models locally on the Apple M3 Pro through MPS or CPU fallback.
6. Keep one authoritative copy of each shared dataset artifact and remove verified duplicates and orphan files.
7. Make shared code and evaluation portable between macOS and Windows.
8. Preserve full run provenance: code commit, configuration, dataset checksums, fold checksum, model revision, environment, predictions, and metrics.

## 3. Non-goals

1. Model weights, LoRA adapters, optimizer states, dense embeddings, or local caches will not be shared through Git.
2. Official BTC data will not be published to a public remote without explicit confirmation that redistribution is permitted.
3. External legal corpora, Task 2 data, synthetic augmentation, and online inference APIs are prohibited.
4. FAISS is not a required core dependency; the first dense implementation favors exact, portable search over platform-specific ANN acceleration.
5. Full neural fine-tuning is not required before a correct zero-shot dense and reranking baseline exists.
6. The old benchmark values are not acceptance targets because they were produced with validation leakage.

## 4. Constraints and fixed decisions

- Development and full model execution must work on an Apple M3 Pro with 36 GiB unified memory.
- Shared code must also run on Windows.
- The final submission contains one to five unique official document IDs per query. Recall@5 is the primary optimization target.
- The corpus must retain all 8,532 official documents, including empty and duplicate passages.
- Total neural parameters must remain below 4B.
- Pretrained model revisions are immutable:
  - `BAAI/bge-m3` at revision `5617a9f61b028005a4858fdac845db406aefb181`;
  - `BAAI/bge-reranker-v2-m3` at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- The minimum required model payload is approximately 4.6 GB. It is downloaded once into a local artifact cache.
- Python CLIs are canonical. Shell scripts are optional wrappers and cannot be the only supported entrypoints.
- Runtime paths use `pathlib.Path`; no absolute user paths or OS-specific separators are allowed.

## 5. Repository and artifact ownership

### 5.1 Shared artifacts

The authoritative shared layout is:

```text
artifacts/
└── shared/
    ├── raw/
    │   ├── selected-contexts.zip
    │   ├── train.json
    │   └── public-official.json
    ├── canonical/
    │   └── v2/
    │       ├── documents.parquet
    │       ├── chunks.parquet
    │       ├── queries_train.parquet
    │       ├── qrels_train.parquet
    │       ├── duplicate_groups.json
    │       ├── empty_context_ids.json
    │       ├── manifest.json
    │       ├── audit_report.json
    │       └── splits/
    │           ├── random_5fold.json
    │           └── doc_disjoint_split.json
    ├── manifests/
    │   ├── artifacts.sha256
    │   └── dataset_provenance.json
    ├── benchmarks/
    │   └── accepted/
    └── submissions/
        └── accepted/
```

`selected-contexts.zip` is the single authoritative raw context representation. The canonical builder reads the ZIP directly; an extracted `selected-contexts/` directory is not retained.

Large shared payloads may use Git LFS on `main` only when the remote is private or dataset redistribution has been explicitly approved. Until that approval exists, `main` tracks the directory contract, manifests, checksums, and code while the binary bundle is exchanged through private team storage. No implementation step may publish BTC files to a public remote implicitly.

### 5.2 Local artifacts

Every collaborator owns their local derived artifacts:

```text
artifacts/
└── local/
    ├── models/
    │   └── huggingface/
    ├── indexes/
    │   ├── bm25/
    │   ├── dense/
    │   └── question_memory/
    ├── training/
    │   ├── pairs/
    │   └── checkpoints/
    ├── cache/
    └── runs/
        └── <run_id>/
```

`artifacts/local/**` is ignored by Git. A model branch commits code, configuration, metric summaries, and ablation reports, but never weights, embeddings, checkpoints, caches, or optimizer state.

### 5.3 Run artifacts

Each run directory contains:

```text
artifacts/local/runs/<run_id>/
├── config.snapshot.yaml
├── manifest.json
├── metrics.json
├── predictions.json
├── candidate_metrics.json
├── error_analysis.jsonl
└── run.log
```

The manifest records:

- run ID and UTC timestamp;
- Git commit;
- operating system, architecture, Python, and package versions;
- device and dtype;
- model IDs, revisions, licenses, and parameter counts;
- raw and canonical dataset checksums;
- split checksum and fold ID;
- random seeds;
- index checksums;
- command and resolved output paths.

## 6. Branching and collaboration

`main` contains stable, model-independent contracts and accepted improvements. Experiment branches follow `exp/<owner>-<experiment>` naming, for example:

```text
exp/phuc-bge-reranker-lora
exp/teammate-a-dense-tuning
exp/teammate-b-lightgbm-fusion
fix/canonical-chunking-v2
```

Experiment branches may change model code, evidence formatting, loss, hyperparameters, local index settings, and fusion features. They may not silently change qrels, fold membership, official scoring, dataset version, or submission constraints.

An experiment can merge to `main` only when:

1. shared tests pass;
2. it uses the same canonical and fold checksums as its comparison baseline;
3. all five random folds run without memory or OOF leakage;
4. document-disjoint validation runs;
5. official scorer equivalence and submission-compliance tests pass;
6. candidate and final ablations are recorded;
7. no model or cache artifact is committed;
8. code supports both macOS and Windows paths and process startup.

## 7. Canonical data repair

The new dataset version is `canonical/v2`. Version `v1` is not overwritten because its chunk semantics differ; after `v2` is fully validated and its hashes are recorded, duplicate `v1` payloads are removed from the working tree.

### 7.1 Legal-aware parsing

The parser preserves:

```text
Chương → Mục → Điều → Khoản → Điểm
```

It records the nearest available hierarchy metadata on each chunk. Legal numbers, years, monetary values, percentages, article/clause/point identifiers, acronyms, authorities, provinces, titles, and links remain intact.

### 7.2 Chunking

- Macro chunks target 400–800 tokens.
- Micro chunks target 100–250 tokens.
- Oversized articles are split; a single article cannot produce an unbounded chunk.
- Unstructured long text uses token-aware 700–1,200-token macro fallback windows with 150-token overlap, following the latest Notion-derived requirement.
- Micro chunks are derived from macro chunks and retain `parent_chunk_id`.
- Short legal units remain intact when splitting them would destroy meaning; deviations are reported by the audit instead of silently ignored.
- Empty passages produce one metadata-only canonical record with `is_empty=true` and are not inserted as ordinary body text into BM25 or dense indexes.
- Duplicate normalized passages are grouped. Content is indexed once per duplicate group while every original document ID mapping is preserved.

### 7.3 Required validation invariants

The validator fails on:

- missing files, columns, or incompatible dtypes;
- document count other than 8,532;
- duplicate document IDs or chunk IDs;
- invalid granularity values;
- invalid micro-to-macro parent relationships, including cross-document parents;
- missing query or qrel references;
- duplicate qrels;
- relevance values other than 1;
- inconsistent `gold_count`;
- lost empty documents or duplicate mappings;
- invalid chunk size exceptions not documented in the audit;
- mismatched manifest, schema, source, or split checksums;
- index coverage inconsistent with the canonical package.

## 8. Configuration and reproducibility

The four declarative YAML files are replaced by one runtime-loaded configuration:

```text
configs/pipeline.yaml
```

It contains dataset, chunking, retrieval, memory, reranker, fusion, evaluation, and path sections. Every command accepts a config path and writes the fully resolved snapshot into its run directory.

Dependencies must include all direct runtime and test requirements. The repository declares a supported Python range and records the exact tested macOS arm64 environment. Model downloads use the pinned revisions and place files only under `artifacts/local/models/huggingface` so the Hugging Face default cache does not create a second model copy elsewhere.

The required command sequence is platform-neutral:

```text
python -m src.artifacts verify
python -m src.dataset build
python -m src.indexes build
python -m src.evaluation benchmark
python -m src.inference predict
python -m src.submission validate-and-package
```

A final orchestrator runs the same sequence from verified BTC inputs to `submission.zip` without external APIs. After the pretrained models are cached locally, it supports offline execution.

## 9. Codabench-equivalent validation

### 9.1 Official scorer equivalence

The official scoring implementation remains unmodified. The internal evaluator is tested against its `eval_retrieval` behavior on golden and generated cases. The compatibility suite covers:

- one to five predictions;
- empty predictions;
- more than five predictions;
- missing and extra query IDs;
- duplicate IDs;
- unknown corpus IDs;
- multiple gold documents;
- wrong JSON value types;
- exact macro averaging;
- precision denominators matching prediction length.

A strict pre-submission validator rejects malformed output before scoring. A packaging test confirms the ZIP contains exactly one root member named `submission.json` and that the bytes inside match the validated JSON.

### 9.2 Random five-fold protocol

All five folds run. For each fold:

- question memory is built only from that fold's training query IDs and qrels;
- any trained reranker or fusion model is fitted only on the training partition;
- validation queries and labels never enter memory, negative mining, feature fitting, calibration, or threshold selection;
- unsupervised corpus indexes may be shared because they use no labels;
- predictions and candidate sets are persisted for audit.

The report includes per-fold values, mean, standard deviation, and micro diagnostics.

### 9.3 Document-disjoint protocol

No validation gold document is a positive in the training partition. Memory is built only from document-disjoint training queries. Components, duplicate groups, and false-negative guards respect the split boundary.

The document-disjoint result is an untouched generalization gate, not the primary hyperparameter-tuning set.

### 9.4 Metrics

Every accepted run reports:

- Candidate Recall@20, @50, @100, and @150;
- Recall@1, @3, and @5;
- official macro Recall and Precision;
- prediction-count distribution;
- per-branch and union candidate recall;
- query buckets and error taxonomy;
- runtime and peak memory.

Error categories include missing candidate, lexical/paraphrase miss, same-law wrong article, domain confusion, temporal ambiguity, location mismatch, multi-document incomplete recall, reranker failure, and fusion failure.

## 10. Retrieval pipeline

### 10.1 Fielded BM25

Micro chunks are indexed with separate signals for legal number, title, article metadata, body, and URL slug. Initial field weights follow the Notion specification. Document aggregation uses best and second-best evidence scores; mean score remains available as a fusion feature. Tokenization and stable tie-breaking are deterministic.

The BM25 index is a rebuildable local cache. Its manifest binds it to canonical chunk checksum, tokenizer version, code commit, and package versions. A pickle may be used only as a trusted local cache; it is never treated as a portable shared artifact.

### 10.2 Exact legal matcher

Exact features include legal number, title, year, document type, article, clause, point, acronym, authority, and province/city signals. All high-confidence exact hits enter the candidate union. Exact matches remain separate features instead of overriding every other rank unconditionally.

### 10.3 Dense macro retrieval

`BAAI/bge-m3` encodes macro chunks and queries. Dense embeddings are normalized and stored locally as float16 NumPy arrays with a metadata manifest. Candidate scoring uses exact batched dot products and stable document aggregation. The 201k-scale macro corpus is small enough for an exact local implementation, avoids FAISS platform differences, and maximizes candidate recall.

Encoding uses MPS on the Mac with CPU fallback. Batch size and sequence length are measured, not hardcoded beyond safe defaults. The model's 8,192-token capacity is available, but the canonical chunk limits prevent pathological inputs.

### 10.4 Fold-safe question memory

Question memory has:

- exact normalized-question lookup;
- character TF-IDF similarity;
- BGE-M3 dense question similarity;
- vote counts and positive-document frequency;
- ambiguity and near-duplicate groups;
- fold-specific construction.

The initial lexical threshold is 0.82 and may be tuned only inside training folds. Self-query exclusion alone is not considered leakage protection.

### 10.5 Candidate union

Each branch returns ranked document candidates and branch-specific scores. The union retains up to 150 unique candidates for diagnostics. A cheap first-stage fusion selects the top 50 for cross-encoder reranking. Candidate Recall@20/50/100/150 determines whether the candidate stage is sufficient before reranker optimization begins.

## 11. Evidence and reranking

For each candidate document, the evidence builder selects the strongest macro evidence using lexical and dense chunk signals and formats:

```text
[VĂN BẢN]
[ĐIỀU KHOẢN]
[NỘI DUNG]
```

The reranker scores the best evidence chunks with `BAAI/bge-reranker-v2-m3`. Per-document features include:

- best reranker score;
- second-best score;
- best macro chunk ID;
- evidence chunk count;
- score margin;
- title/body overlap.

Zero-shot reranking is integrated and benchmarked before any fine-tuning. Local LoRA training is attempted only if candidate recall is high, zero-shot reranking is stable, and a leakage-free training-pair dataset exists.

## 12. Training-pair preparation

Positive localization combines lexical and dense chunk scores inside each gold document and may retain the top one or two macro chunks.

Hard negatives come from high-ranked non-gold candidates. The miner excludes:

- all gold documents for the query;
- positives from exact and near-duplicate query groups;
- duplicate-passage equivalents that would create false negatives.

It prioritizes difficult same-law, same-topic, wrong-article, wrong-procedure, temporal, and location confusions. Training pairs are local derived artifacts and are not committed.

## 13. OOF features and learned fusion

OOF generation is fold-aware. A fold's fusion model never trains on that fold's labels or reranker outputs fitted using those labels.

Features include:

- branch ranks and normalized scores;
- branch-presence and overlap count;
- BM25 best, second, and mean evidence scores;
- dense best and second evidence scores;
- exact-match booleans;
- memory lexical/dense similarity and vote counts;
- reranker best, second, margin, and evidence count;
- article/year/province/entity/title overlap;
- candidate source count and reciprocal-rank features.

RRF is the always-available baseline. LightGBM LambdaMART is accepted only if it improves leakage-free mean Recall@5 under the experiment gates. The final model is trained locally on all OOF-compatible training records and is not committed.

## 14. Performance selection gates

No component is called an improvement solely because one fold improves. An experiment is accepted when:

1. it uses identical dataset and split checksums to the baseline;
2. mean five-fold official Recall@5 improves;
3. at least three of five folds do not regress;
4. document-disjoint Recall@5 does not fall by more than 0.01 absolute unless the primary gain and error analysis justify the trade-off;
5. candidate recall remains adequate for the reranked pool;
6. submission compliance is unchanged;
7. runtime and memory fit the local-M3 constraint.

Ties are resolved by document-disjoint Recall@5, then Precision, then lower runtime and memory.

## 15. Cross-platform behavior

- `pathlib.Path` is mandatory for filesystem operations.
- The canonical command is `python -m ...`; `.sh` files are convenience wrappers only.
- Multiprocessing entrypoints use `if __name__ == "__main__"`.
- Windows defaults to `num_workers=0`; users may override it after testing.
- Local functions and lambdas are not passed to spawned workers.
- ZIP member paths use POSIX separators.
- Stable sorting uses descending score and ascending document ID as a deterministic tiebreaker.
- Run manifests record MPS, CUDA, or CPU and inference dtype because neural floating-point rankings can differ slightly across backends.
- Shared tests run without downloading models. Slow model and full-artifact suites are explicitly marked.

## 16. Test strategy

### Fast tests

- text normalization and hierarchy parsing;
- token-aware macro/micro chunk boundaries and overlap;
- empty and duplicate handling;
- full canonical schema and referential invariants;
- fielded BM25 and exact matcher behavior;
- fold-specific question-memory construction;
- official scorer equivalence;
- strict submission validation and ZIP byte consistency;
- stable tie-breaking and cross-platform paths;
- feature schema versioning.

### Integration tests

- raw ZIP to canonical package on a deterministic fixture;
- canonical package to BM25 and dense-index metadata;
- one-fold retrieval with no validation IDs in memory;
- zero-shot reranker wiring with a stubbed lightweight model;
- OOF feature generation with explicit leakage assertions;
- end-to-end prediction and official scoring on a fixture.

### Full artifact tests

- verify all BTC and canonical checksums;
- validate the 8,532-document package;
- verify index coverage;
- run actual model smoke tests from the pinned local cache;
- execute all five folds and document-disjoint validation;
- package and reopen the final submission ZIP.

## 17. Cleanup and migration

Cleanup occurs only after checksum verification and successful replacement artifacts exist.

The migration retains:

- one raw `selected-contexts.zip`;
- one canonical `v2` package;
- one accepted benchmark/submission copy;
- local derived indexes and runs only under `artifacts/local`.

After verification, it removes:

- extracted `selected-contexts/`;
- duplicate `data/task1_canonical/v1` and `artifacts/data` payloads;
- duplicate root and artifact BM25 pickle copies;
- stale root `chunks.parquet`;
- stale root `submission.zip` and duplicated root reports;
- generated `.playwright-mcp`, test caches, `__pycache__`, and OS metadata.

Unreachable Git objects containing old large blobs are not pruned automatically. Reflog expiration and aggressive garbage collection require a separate explicit confirmation because they remove recovery history.

## 18. Implementation phases

1. **Shared artifact contract and migration tooling**
   - Add manifests, verification, path resolution, and safe duplicate detection.
2. **Canonical v2 builder and validator**
   - Repair hierarchy parsing, chunking, duplicate/empty handling, and schema checks.
3. **Strict Codabench validation baseline**
   - Fix memory isolation, run five folds, run document-disjoint, and establish accepted baseline metrics.
4. **Model bootstrap and dense retrieval**
   - Download pinned BGE-M3 locally, encode macros, and integrate exact dense search.
5. **Dense question memory and exact-signal expansion**
   - Add fold-safe neural memory and missing legal metadata signals.
6. **Evidence packs and zero-shot reranking**
   - Integrate the pinned BGE reranker and record document evidence features.
7. **OOF features and LightGBM fusion**
   - Train and evaluate fold-safe fusion.
8. **Optional local fine-tuning**
   - Generate guarded training pairs and benchmark LoRA only after previous gates pass.
9. **Final ablations and submission**
   - Select the best strict configuration, regenerate full-train artifacts locally, validate, package, and record checksums.
10. **Verified cleanup**
    - Remove duplicate/orphan working-tree artifacts and leave the repository clean.

## 19. Definition of Done

The project is complete when:

1. a verified raw BTC artifact bundle and pinned local pretrained models can produce `submission.zip` with one standard platform-neutral command and no external API;
2. canonical data passes the complete validator and preserves all 8,532 official document IDs;
3. the accepted benchmark contains five leakage-free random folds and one document-disjoint run;
4. internal metrics match the official scorer on compatibility tests;
5. the active runtime includes the selected BM25, exact, memory, dense, reranker, and fusion components rather than unconnected scaffolds;
6. every accepted model change has ablation evidence under identical data and split checksums;
7. model weights and derived caches remain local;
8. shared source code works on macOS and Windows;
9. the final ZIP contains exactly the validated `submission.json` bytes;
10. no duplicate or orphan dataset/index/output files remain in the working tree, and `git status` is clean.
