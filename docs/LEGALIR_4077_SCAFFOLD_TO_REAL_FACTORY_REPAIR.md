# LegalIR 4077 — Scaffold-to-Real Factory / Production Repair

**Repository:** `silent9669/LegalIR`  
**Audited HEAD:** `4077d8383b520a9431fd6f34e0113ad7bc3ef3f6`

## Verdict

**DO NOT RUN COLAB OR KAGGLE WITH HEAD 4077 YET.**

The refreshed architecture is correct, but the implementation is mostly scaffolding. GitHub CI is green because it runs only the new modular suite (43 tests) and does not prove the real official-data factory, real fold jobs, real final LoRA training, real production fusion, or the generated notebooks.

Keep the approved architecture:

```text
Validation / Artifact Factory
        ↓
immutable verified production bundle
        ↓
Kaggle one-final-LoRA + public reranking
```

The task is to wire it to the proven legacy production modules and the actual canonical v2 schema.

## P0-1 — Canonical schema mismatch

Current `src/data/canonical.py` and `MacroEvidenceStore` expect `chunk_type` and `text`. The existing canonical pipeline uses `granularity`, `text_norm`, `text_raw`, plus structural fields. The clean dataset also may not contain `duplicate_groups.json` or split artifacts at the dataset root.

Fix one authoritative adapter that reads the actual Parquet schema and resolves duplicate/split artifacts using:

```text
dataset root
→ dataset root/splits
→ repo artifacts/task1/data
→ FAIL
```

Require exact official counts: 8,532 docs; 1,153,876 chunks; 934,416 micro; 219,460 macro; 7,000 train; 7,637 qrels; 1,000 public; 4 duplicate groups.

## P0-2 — Static cache builder is a stub

`scripts/build_static_cache.py` only verifies paths and prints targets. It does not load Legal BM25, PyVi, DEk21/FAISS or Exact; it does not encode queries; it writes no cache.

Implement using the proven legacy classes, not new ranking math. Cache label-free branch results for all 7k train + 1k public queries at the depth required to reproduce production candidate fusion. The builder must not accept qrels.

## P0-3 — Static cache reader is not memory-safe

Current reader calls `pq.read_table(...).to_pandas()` for the complete cache, then scans it for every query. At production scale this can be millions of rows.

Write cache rows sorted by query and persist a query→row-group/shard index. Read only relevant row groups/shards. Prohibit a full-cache Pandas materialization.

## P0-4 — MacroEvidenceStore must use real fields and stay lazy

Use `granularity == "macro"`; preserve `text_norm`, `text_raw`, chapter/section/article/clause/point. Prefer Arrow Dataset/row-group reads where practical. Keep the byte-bounded LRU. Do not duplicate large joined full-text strings unless required by legacy scoring.

## P0-5 — Current evidence tests are not parity tests

The current tests only validate hand-written samples. Compare new lazy logic directly against legacy `PositiveLocalizer` and `EvidencePackBuilder` on at least 100 real positive query/gold-doc pairs and 100 query/candidate-doc pairs. Require exact selected chunk IDs and evidence text, except explicitly documented normalization-only differences.

## P0-6 — PairMaterializer changes training semantics

Current materializer takes first allowed static rows and does not reproduce fold-local Question Memory or the old hard-negative source policy.

Refactor the proven old `build_training_pairs()` to accept injected:

```text
candidate_provider = cached static branches + fold-local Question Memory
evidence_provider  = lazy MacroEvidenceStore-backed parity implementation
```

Preserve RRF/branch weights, source limits, medium-negative bands, hybrid negatives, duplicate blacklist, negatives-per-positive and query-balanced coverage.

Hard assertions:

```text
pair_qids ⊆ train_qids
pair_qids ∩ val_qids = ∅
memory_qids ⊆ train_qids
memory_qids ∩ val_qids = ∅
```

## P0-7 — build_fold_pairs.py is a stub

It must load the authoritative fold split, queries/qrels, static cache, duplicate map, fold-local Question Memory and MacroEvidenceStore; call the real pair materializer; write pair Parquet + manifest; and fail if the artifact is empty or incomplete.

## P0-8 — FoldJobRunner is mock-only

Current non-mock path creates no metrics/predictions/features, then tries to hash them. Implement a real fold job:

1. verify pair artifact;
2. train BGE+LoRA through `train_reranker`;
3. fresh-reload adapter;
4. generate held-out candidates from static cache + fold-local memory;
5. build real evidence;
6. score approved `rerank_k`;
7. extract the exact old fusion feature schema;
8. write OOF features/predictions;
9. calculate official-equivalent metrics;
10. write hashes/manifest and PASS only after verification.

No constant fake metrics are allowed.

## P0-9 — Document-disjoint runner is also mock-only

Implement using the same production primitives and fixed official document-disjoint split. Produce Recall@1/3/5, Precision@5, and Candidate Recall@20/50/100/150.

## P0-10 — Final trainer falsely PASSes without training

Current production path returns only `{"status":"PASS"}`.

Call the proven `train_reranker()` on `final_training_pairs.parquet` with all eligible 7,000 queries, full-coverage enforcement, approved BGE/LoRA config, max length 512, effective batch 16, FP16 and gradient checkpointing.

PASS requires optimizer steps >0, finite loss, param_diff >0, positive/negative coverage, adapter SHA, fresh reload, active PEFT, finite inference and total learned params <4B.

## P0-11 — Public reranking is not production inference

Current code ignores the contents of `production_lock.json`, does not load the final adapter or public evidence/fusion artifact, and hard-codes simplified RRF + coefficient logic.

Production inference must use:

```text
public_candidates.parquet
public_evidence.parquet
final adapter
production_lock.json
frozen fusion artifact
```

Score approved rerank_k with the final adapter, reconstruct the exact OOF feature schema, apply frozen fusion/top-5 semantics, and emit 1..5 unique official doc IDs.

## P0-12 — Production bundle can PASS while incomplete

Builder currently adds files only if they happen to exist; verifier checks only listed files.

Define a schema-versioned mandatory set including final pairs, public candidates, public evidence, production lock, fusion artifact, static-cache provenance, validation summary and dataset provenance. Builder fails if any required file is missing. Verifier checks SHA, size, schema/row counts, runtime SHA, dataset fingerprint, config SHA, validation approvals and 5/5 fold + doc-disjoint manifests.

Never use placeholder defaults such as `a0efb25` or `canonical_v2_fingerprint`.

## P0-13 — Kaggle notebook pin is broken

The new generator pins old `a0efb25`, then tries to run `scripts/run_kaggle_final.py`, which belongs to the refreshed runtime. Pin the final notebook only after the completed refreshed runtime passes CI + Colab. Use the full 40-character Commit A SHA and fail closed on mismatch.

## P0-14 — Do not reinstall Kaggle's PyTorch/CUDA stack

Current notebook runs `pip install -r requirements.txt`, which can replace Kaggle's working GPU stack. Use a Kaggle-specific lightweight dependency preflight/install. Never reinstall torch unless explicitly required and verified.

## P0-15 — Colab notebook is broken

It pins old `a0efb25` and calls `run_colab_t4_smoke.py` without its required `--data-dir`, `--work-dir`, and `--target-sha` arguments. Generate a refreshed notebook that pins Commit A and passes all required arguments.

## P0-16 — Throughput probe is fake

Current probe only allocates an integer tensor. Test real BGE+PEFT forward/backward/optimizer steps on real query/evidence pairs at max_length=512, FP16 and gradient checkpointing.

Probe 8×2 → 4×4 → 2×8, always preserving effective batch 16. Persist finite loss, param_diff, OOM events, peak VRAM and seconds/optimizer-step.

## P0-17 — Green CI excludes the old regression suite

Latest CI reports 43 tests because `pytest.ini` and CI explicitly target only new subdirectories. Root `tests/test_*.py` suites are excluded.

Set:

```ini
testpaths = tests
```

and run:

```bash
pytest -q
```

Port/delete historical tests only with explicit justification. Preserve high-value historical gates: public1000, canonical schema, split provenance, OOF leakage, duplicates, query-balanced coverage, FAISS, Dense max-length clamp, PEFT audit, T4 topology, official scorer, notebook/runtime pinning.

## P1 — verify_release.py is still a scaffold gate

It currently proves syntax + 43 tests + parameter budget, not the production path. Add behavioral orchestration tests using mocks at the heavyweight model boundary while calling real factory/fold/final orchestration. CI must detect fake PASS paths.

## Implementation strategy

**Reuse proven code; do not build a parallel scoring system.** Keep old real implementations of retrievers, HybridSearchEngine, Question Memory, pair mining, train_reranker, CrossEncoderReranker, fusion/features, official metrics and top-5. Refactor ownership/caching/lifecycle around them.

### Commit A1 — canonical + static cache
Real schema/fallbacks, real static cache, bounded reader, live-vs-cache parity.

### Commit A2 — lazy evidence + pair materialization
Real schema, legacy evidence parity, old negative semantics with cache provider, fold-local memory, real fold-pair CLI.

### Commit A3 — validation shards
Real fold job, real doc-disjoint job, OOF features and official metrics.

### Commit A4 — strict bundle + final production
Mandatory bundle, real final LoRA, fresh reload, real public BGE/fusion inference, strict submission.

### Commit A5 — CI / notebook / GPU gate
Run all old + new tests, parameter audit, release verifier. Push and require GREEN CI, then run real Colab T4 on exact A5 SHA.

### Commit B — release only
Record T4 report and pin generated notebooks to A5. No source changes. Require GREEN CI.

## Mandatory new tests

```text
test_real_canonical_fixture_uses_granularity_and_text_norm
test_duplicate_groups_repo_fallback
test_static_cache_cli_writes_nonempty_train_and_public_cache
test_static_cache_reader_never_materializes_full_dataframe
test_static_cache_live_branch_parity
test_static_cache_fusion_parity
test_macro_store_real_schema
test_legacy_positive_localizer_parity_on_official_sample
test_legacy_evidence_pack_parity_on_official_sample
test_pair_builder_uses_fold_local_question_memory
test_pair_negative_source_policy_matches_legacy
test_pair_qids_subset_train
test_pair_qids_disjoint_val
test_duplicate_equivalent_gold_not_negative
test_fold_nonmock_calls_real_training_interface
test_fold_nonmock_generates_real_feature_schema
test_doc_disjoint_nonmock_calls_real_pipeline
test_no_constant_fake_metrics_in_production
test_final_train_nonmock_calls_train_reranker
test_final_train_rejects_zero_optimizer_steps
test_final_train_rejects_zero_param_diff
test_public_rerank_loads_final_adapter
test_public_rerank_reads_production_lock
test_public_rerank_uses_frozen_fusion
test_public_rerank_uses_public_evidence
test_bundle_missing_required_file_fails
test_bundle_runtime_dataset_config_hashes_are_real
test_t4_probe_performs_real_backward_step
test_effective_batch_always_16
test_kaggle_notebook_pin_contains_new_runtime_full_sha
test_kaggle_notebook_target_script_exists_at_pin
test_colab_notebook_passes_required_cli_args
test_all_legacy_regression_tests_are_collected
```

## Gate status at 4077

```text
GitHub CI                         GREEN
new modular tests                43 PASS
parameter budget                 PASS
real canonical compatibility     FAIL
real static cache build          NOT IMPLEMENTED
bounded static cache read        FAIL
legacy evidence parity           NOT PROVEN
real fold pair CLI               NOT IMPLEMENTED
fold-local memory fusion         MISSING
real FoldJobRunner               NOT IMPLEMENTED
real doc-disjoint runner         NOT IMPLEMENTED
real final adapter training      NOT IMPLEMENTED
real public BGE/fusion inference NOT IMPLEMENTED
strict production bundle         INCOMPLETE
real T4 throughput probe         NOT IMPLEMENTED
Colab notebook                   BROKEN
Kaggle notebook runtime pin      BROKEN
old regression suite in CI       EXCLUDED

READY FOR COLAB                  NO
READY FOR KAGGLE                 NO
```

## Required final agent report

```markdown
# LegalIR 4077 Scaffold-to-Production Repair Report

## Git
- audited head:
- repaired runtime SHA:
- release SHA:

## Dataset compatibility
- actual chunk schema:
- duplicate source:
- split source:
- canonical official test:

## Static cache
- real retrievers used:
- train/public queries:
- rows:
- peak RSS:
- reader architecture:
- live/cache parity:

## Evidence
- Arrow backend/LRU:
- positive parity:
- evidence parity:

## Pair materialization
- fold memory:
- source-policy parity:
- validation leakage:
- duplicate exclusions:

## Validation
- real fold path:
- real doc-disjoint path:
- feature schema parity:
- metrics real:

## Final production
- real train_reranker:
- optimizer steps:
- param_diff:
- adapter reload:
- public BGE inference:
- frozen fusion:
- strict top5:

## Bundle
- required files:
- runtime/dataset/config hashes:
- verifier:

## CI
- total tests:
- legacy tests collected:
- status:

## Colab
- target SHA/T4:
- real throughput probe:
- effective batch:
- result:

## Kaggle notebook
- pinned runtime:
- target script exists:
- no mock:
- no torch reinstall:

READY FOR COLAB: YES/NO
READY FOR KAGGLE FINAL: YES/NO
```
