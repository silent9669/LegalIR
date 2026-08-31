# LegalIR Task 1 — DCC007 Final Pre-GPU-Smoke Repair Contract

**Repository:** `silent9669/LegalIR`  
**Audited HEAD:** `dcc007563733d24863f9c0411a0329a743188736`  
**Previous audited HEAD:** `f2080e4881fbb54e41dff33836de014252719076`  
**Target:** Kaggle T4 ×2  
**Primary metric:** Recall@5  
**Secondary metric:** Precision@5  
**Decision:** **NOT READY FOR FULL KAGGLE RUN**

This commit resolves most of the F208 contract. Do **not** redesign the architecture. Fix only the remaining correctness, production-gating, runtime, and score-critical issues below.

Official dataset identity for this repair gate:

```text
canonical documents : 8,532
canonical train QIDs : 7,000
public-official QIDs : 999
canonical chunks     : 1,153,876
micro chunks         : 934,416
macro chunks         : 219,460
```

The final full run must remain Task1-only and under the competition `<4B learned parameters` limit.

---

# 0. Preserve working DCC007 fixes

Do not regress these:

- `gpu_smoke` requires CUDA and at least two GPUs.
- Standard mapping is now Dense=`cuda:0`, reranker=`cuda:1`.
- FULL/GPU_SMOKE require `public-official.json`.
- Dense corpus batch was lowered to 32.
- Dense inference uses CUDA FP16 autocast and adaptive OOM batch reduction.
- `faiss-cpu` is in requirements/notebook preflight.
- `CrossEncoderReranker.ensure_loaded()` exists.
- ExactMatcher accepts chunk statutory metadata and accent-folded title forms.
- OOF ExactMatcher now receives chunks.
- OOF fold reranker OOM counts are exported.
- final/public submission validation remains strict.
- AMP GradScaler and finite loss/gradient checks remain enabled.
- adapter SHA-256 verification remains enabled.
- both Kaggle notebooks remain byte-identical.
- notebook default remains `gpu_smoke`.

---

# 1. P0 — Fix ExactMatcher stale-variable cross-document contamination

`src/retrieval/exact_matcher.py` contains leftover statements after the chunk statutory indexing logic.

The problematic pattern is effectively:

```python
for c in chunk_records:
    did = ...
    ...
    self.doc_points[did].add(...)

    # WRONG — stale doc_id from prior document loop
    self.doc_clauses[doc_id].add(normalize_text(cl))

# WRONG — stale d/doc_id from prior document loop
pt = d.get("point")
...
self.doc_points[doc_id].add(...)
```

This can:

- attach a chunk's clause to the wrong document;
- inject false exact features;
- contaminate ranking/fusion signals;
- crash when chunks are supplied without document records.

## Required fix

The chunk indexing loop may write only through its current `did`.

Delete every stale reference to:

```text
d
doc_id
```

outside the document loop.

Desired structure:

```python
for c in chunk_records:
    did = str(c.get("doc_id", ""))
    if not did:
        continue
    # add article/clause/point only to did
```

## Mandatory tests

```text
test_exact_chunk_features_do_not_cross_contaminate_documents
test_exact_matcher_chunks_only_does_not_reference_stale_document
test_exact_article_clause_point_are_assigned_to_correct_doc
```

Use at least two documents with different article/clause/point values and assert complete isolation.

---

# 2. P0 — Current “query-balanced” training is not actually query-balanced at runtime

`balance_pairs_by_query()` reorders rows, but the DataLoader is created with:

```python
shuffle=True
```

This destroys that ordering.

There is a second issue: the reorder function emits up to:

```text
positive + negative
```

for one query before moving to the next query.

With 7,000 queries and a desired positive+negative exposure per query, this needs roughly:

```text
14,000 row exposures
```

Current FULL configuration:

```text
batch_size = 2
gradient_accumulation_steps = 8
max_steps = 500
```

gives roughly:

```text
8,000 row exposures
```

so it cannot guarantee full training-query coverage.

The new coverage fields only **measure** the result after training; they do not guarantee it.

## Required fix

Implement a real query-aware sampler or batch sampler.

Preferred contract:

1. Build query groups keyed by `query_id`.
2. Determine eligible queries with at least one positive and one usable negative.
3. First training cycle must schedule every eligible query exactly once.
4. A query training unit should contain at least:
   - one positive;
   - one strongest available hard negative.
5. Only after every query has been seen may repeated/extra negatives be sampled.
6. Seed must be deterministic (`42`).
7. The DataLoader must not independently reshuffle away the query schedule.

For pair/BCE mode either:

```text
use a custom Sampler/BatchSampler
```

or explicitly build the ordered rows and use:

```python
shuffle=False
```

A custom query sampler is preferred.

## FULL readiness invariant

After final training:

```text
eligible_training_queries
actual_unique_queries_seen
actual_query_coverage_pct
positive_queries_seen
queries_with_negative_seen
```

must be present.

Require:

```python
actual_query_coverage_pct >= 99.0
```

Prefer `100.0%`.

If some query has no valid pair, report it explicitly and compute coverage against both:

```text
all organizer train queries
eligible queries
```

Do not falsely write “trained on all 7,000 queries” unless actual coverage proves it.

## Step-count policy

Do not arbitrarily increase training steps.

Measure the sampler requirement and T4 throughput.

For an effective batch of 16:

```text
14,000 row exposures ≈ 875 optimizer steps
```

is a useful upper reference, but the actual implementation should derive the required bounded step count from the training-unit schedule.

## Mandatory tests

```text
test_query_sampler_survives_dataloader_iteration
test_query_sampler_covers_all_eligible_qids_before_repeat
test_final_training_hard_fails_below_required_query_coverage
test_query_sampler_is_deterministic
```

The existing test that only inspects `balance_pairs_by_query()` is not sufficient.

---

# 3. P0 — gpu_smoke still does not prove official dataset identity

The orchestrator validates canonical schema but does not enforce that the production smoke/full dataset is the actual organizer Task 1 dataset.

A structurally valid small dataset can therefore enter `gpu_smoke`.

Current canonical validator supports an optional expected document count, but the orchestrator does not use an official identity gate.

## Required fix

For both:

```text
gpu_smoke
full
```

require at minimum:

```python
len(df_docs) == 8532
len(df_queries) == 7000
len(public_data) == 999
```

Also validate the canonical manifest where available:

```text
dataset == task1_canonical
version/schema expected
total_documents == 8532
total_queries == 7000
```

Better: store/verify source fingerprints for:

```text
train.json
selected-contexts.zip
public-official.json
```

when the canonical package is built.

Do not require a hard-coded chunk count as the sole identity mechanism because chunk-generation changes may legitimately alter it. The document/query/source identity is the critical gate.

## Smoke CLI fix

`scripts/smoke_kaggle_pipeline.py` currently builds toy data when:

```text
--tiny
```

or when a usable data dir is absent.

Production modes must reject this.

Required behavior:

```text
--tiny --run-mode smoke       => allowed
--tiny --run-mode gpu_smoke   => hard fail
--tiny --run-mode full        => hard fail
gpu_smoke without real data   => hard fail
full without real data        => hard fail
```

If a toy GPU test is retained, name it differently and mark it explicitly:

```text
NON_PRODUCTION
NOT_A_READINESS_GATE
```

## Mandatory tests

```text
test_gpu_smoke_rejects_toy_dataset
test_full_rejects_toy_dataset
test_gpu_smoke_requires_8532_docs_7000_train_999_public
test_gpu_smoke_cli_rejects_tiny
```

---

# 4. P0 — PEFT force-load audit still swallows loading failures

`LegalIRPipeline.audit_parameters()` now tries:

```python
self.reranker.ensure_loaded()
```

but catches any exception and silently continues.

If loading the final PEFT adapter fails, the audit can fall back to auditing only the BGE model name.

That defeats the purpose of the final loaded-system parameter gate.

## Required fix

Make parameter auditing aware of strict production mode.

Recommended API:

```python
pipeline.audit_parameters(
    output_json=...,
    raise_on_violation=True,
    require_loaded_models=True,
)
```

When `require_loaded_models=True`:

```text
ensure_loaded failure -> raise
reranker.model is None -> raise
adapter requested but no active PEFT adapter -> raise
adapter parameter count <= 0 -> raise
```

FULL/GPU_SMOKE must use `require_loaded_models=True`.

The final report must prove:

```text
is_peft_lora = true
adapter_parameters > 0
total system parameters > base-only 702,754,049
total system parameters < 4,000,000,000
```

Keep the base-only static/preflight audit separately.

## Tighten PEFT detection

The parameter auditor currently treats a generic module with `base_model` as PEFT.

That is too broad because many Hugging Face models expose a `base_model` property.

Detect PEFT through strong signals such as:

```text
peft_config
PeftModel
LoRA parameter names
```

Do not classify a generic HF model as LoRA merely because `base_model` exists.

## Mandatory tests

```text
test_strict_final_audit_raises_if_adapter_force_load_fails
test_strict_final_audit_counts_real_lora_params
test_plain_hf_model_with_base_model_property_is_not_marked_peft
```

---

# 5. P1 — CrossEncoder OOM fallback repeatedly retries the bad batch size

Current reranker behavior is recursive:

```text
batch 16 OOM
  -> recursively retry current chunk at 8
return to outer loop
next chunk starts at 16 again
```

If 16 does not fit and 8 does, every outer batch can still incur an OOM.

This causes:

- unnecessary CUDA allocator pressure;
- repeated `empty_cache()`;
- slower inference;
- unreliable readiness telemetry.

`initial_batch_size` also remains a default rather than recording the first requested batch.

## Required fix

Use the same iterative strategy as the Dense retriever.

Pseudo-flow:

```python
requested_batch = batch_size
current_batch = batch_size
idx = 0

while idx < len(pairs):
    try:
        score pairs[idx:idx+current_batch]
        record successful size
        idx += actual_count
    except CUDA_OOM:
        current_batch //= 2
        retry same idx
```

Once reduced successfully:

```text
keep the smaller size for all later batches
```

Track:

```text
initial_batch_size
last_successful_batch_size
min_successful_batch_size
oom_events
```

## Mandatory test

Injected sequence:

```text
requested = 16
16 -> OOM
8 -> succeeds
all later batches run at <=8
```

Assert:

```text
initial == 16
min_successful == 8
oom_events == 1
output order preserved
output count preserved
```

---

# 6. P1 — Dense OOM/batch telemetry is split across different Dense instances

The orchestrator builds/loads one Dense retriever for:

```text
corpus indexing
train-query encoding
```

Then the final production pipeline loads another Dense retriever instance for:

```text
public-query encoding
public inference
```

Current smoke report reads Dense OOM information mainly from the earlier `dense_retriever`.

Public-stage OOM events on the final `dense_ret` can therefore be omitted.

There is also a telemetry semantics problem:

```text
dense_min_successful_batch_size
```

starts at `32`, so a successful batch=128 may still be reported as 32 even though 32 was never attempted.

## Required fix

Track metrics by stage:

```text
dense_corpus_requested_batch
dense_corpus_min_successful_batch
dense_corpus_oom_events

dense_train_query_requested_batch
dense_train_query_min_successful_batch
dense_train_query_oom_events

dense_public_query_requested_batch
dense_public_query_min_successful_batch
dense_public_query_oom_events
```

Then aggregate:

```text
dense_total_oom_events
```

Telemetry must describe actual attempted/successful sizes, not constructor defaults.

A clean implementation is to return/reset a per-call telemetry object from `encode_texts()`.

---

# 7. P1 — Runtime projection currently overcounts OOF validation queries

Current projection uses approximately:

```python
(7000 * 5) / oof_queries_per_second
```

for five-fold OOF.

That is wrong.

In standard 5-fold OOF, each of the 7,000 training queries is a held-out validation query exactly once across the five folds.

The total held-out OOF query count is therefore:

```text
7,000
```

not:

```text
35,000
```

At the same time, the projection must separately include five fold-training jobs.

The current `queries_per_second` is also an aggregate that can contain fold-training overhead and a tiny 5-step GPU-smoke training regime, so extrapolating it directly is unreliable.

## Required fix

Time stages independently:

```text
dense corpus indexing seconds
train query encoding seconds

OOF:
  pair mining seconds/fold
  training steps/sec
  held-out rerank queries/sec
  total held-out query count = 7000

doc-disjoint:
  pair mining
  training
  inference

final:
  pair mining
  training steps/sec
  public query inference
```

Project:

```text
5 fold training jobs
+ 7000 total held-out OOF inference queries
+ doc-disjoint
+ final training
+ len(public_data) public queries
+ setup/index overhead
```

Use:

```python
len(public_data)
```

instead of hard-coding 999 in the calculation.

Output assumptions explicitly in:

```text
runtime_projection.json
```

Do not allow `fits_kaggle_session_limit=true` to qualify readiness until this formula is corrected.

---

# 8. P1 — FAISS is installed but production does not require that it actually initialized

The notebook and requirements now install `faiss-cpu`, which is good.

However `DenseMacroRetriever._build_search_index()` still falls back silently to NumPy if FAISS import/index creation fails.

The orchestrator only records:

```text
faiss_index_flat_ip
or
numpy
```

without failing the production run.

With 219,460 macro chunks and thousands of retrieval calls, accidental NumPy fallback is an avoidable runtime risk.

## Required fix

After Dense build/load, in:

```text
gpu_smoke
full
```

require:

```python
dense_retriever._faiss_index is not None
```

or a separately approved and benchmarked backend.

Add to final manifest:

```json
"dense_search_backend": "faiss_index_flat_ip"
```

and fail readiness if production backend is unexpectedly `numpy`.

## Required test

```text
test_faiss_and_numpy_reference_topk_match
test_gpu_smoke_fails_if_faiss_backend_missing
```

---

# 9. P1 — Learned fusion still reports “lightgbm” even when training fell back to LinearRanker

`LightGBMRanker.fit()` can do:

```text
LightGBM fails
-> LinearRanker fallback
```

but `train_and_evaluate_fusion_cv()` currently sets:

```python
winning_model_type = "lightgbm"
```

whenever the learned ranker wins.

The manifest can therefore describe the wrong trained model.

## Required fix

Track actual model type for every fold and the final model:

```text
lightgbm
linear_fallback
rrf_weighted
```

In production, preferred behavior is:

```text
LightGBM failure -> raise
```

because LightGBM is now a required dependency.

If linear fallback is retained, it must have its own cross-fitted evaluation and may be selected only on its actual Recall@5 result.

Manifest/report must never call a LinearRanker a LightGBM model.

## Mandatory test

Force LightGBM failure and assert either:

```text
strict production raises
```

or:

```text
model type == linear_fallback
```

with correct save/reload parity.

---

# 10. P1/P2 — Chunk-backed ExactMatcher construction is unnecessarily memory-heavy

Canonical data contains about:

```text
1.15M chunk rows
```

The current ExactMatcher path can:

```python
pd.read_parquet(full_chunks_file)
-> convert statutory rows to Python dict records
```

This adds large Python-object overhead.

The matcher only needs:

```text
doc_id
article
clause
point
```

## Required fix

At minimum:

```python
pd.read_parquet(
    chunks_path,
    columns=["doc_id", "article", "clause", "point"],
)
```

Better:

- vectorized normalize;
- groupby/aggregate unique statutory values by `doc_id`;
- serialize a compact statutory index once;
- reuse it in OOF and final pipeline.

Example artifact:

```text
indexes/exact_statutory/statutory_by_doc.parquet
```

or another compact deterministic representation.

Do not repeatedly rebuild 1.15M-row Python dictionaries for every OOF/final pipeline load.

---

# 11. P1 — Strengthen actual GPU-smoke acceptance gates

A real T4×2 smoke should be more than “pipeline returned”.

After source repair, `gpu_smoke_report.json` must include and gate on:

```text
official_documents == 8532
official_train_queries == 7000
official_public_queries == 999

dense_requested == cuda:0
dense_actual == cuda:0
reranker_requested == cuda:1
reranker_actual == cuda:1

dense_search_backend == faiss_index_flat_ip

optimizer_steps > 0
param_diff > 0
adapter_checksum_verified == true
adapter_parameters > 0
final_total_parameters < 4B

query_coverage recorded
OOM metrics accurate
runtime projection valid
```

If any gate fails:

```text
READY FOR FULL KAGGLE RUN = NO
```

Recovered OOM should not automatically fail forever. Instead:

1. identify stable batch;
2. update production config;
3. rerun `gpu_smoke` starting at that stable batch;
4. require the second run to finish without avoidable OOM if practical.

---

# 12. P1 — Existing F208 tests need stronger behavioral coverage

Preserve useful tests, but replace weak implementation-mirroring tests with end-to-end behavior at component boundaries.

Required additions:

```text
test_exact_chunk_features_do_not_cross_contaminate_documents

test_query_sampler_through_real_dataloader_hits_all_qids
test_query_sampler_respects_max_steps_coverage_requirement

test_gpu_smoke_rejects_tiny_dataset
test_gpu_smoke_requires_official_counts

test_strict_peft_audit_does_not_swallow_load_failure

test_reranker_oom_reduction_persists_across_later_batches
test_dense_stage_telemetry_aggregates_public_encoder

test_runtime_projection_counts_7000_oof_validation_queries_not_35000

test_gpu_smoke_requires_faiss_backend

test_fusion_manifest_reports_actual_model_type
```

Mocks are fine around expensive Transformer kernels, but tests must invoke the actual orchestration/component boundary being claimed.

---

# 13. P2 — Documentation accuracy

Update stale wording after all correctness fixes.

The official supplied public file contains:

```text
999
```

queries, not 1000.

Do not hard-code “1,000 public predictions” in README/notebook documentation.

Always derive submission query count from:

```python
len(public_data)
```

The final submission key set must equal the public file key set exactly.

---

# 14. Score-maximization stage — only after correctness + real GPU smoke

Do not add major new architecture before the runtime gate is stable.

Historical accepted benchmark shows:

```text
Recall@5             ≈ 75.36%
Candidate Recall@50  ≈ 94.42%
Candidate Recall@150 ≈ 97.35%
```

The dominant headroom is still ranking.

After real T4×2 smoke, run small **leakage-safe sequential ablations**:

```text
A. final RRF branch weights
B. candidate retrieval branch weights
C. rerank_k = 40 vs 50 vs 80
D. BCE vs pairwise_logistic
E. candidate_k = 150 vs 200 only if candidate misses justify it
```

Rules:

- train/tune only on folds other than the evaluated fold;
- select primarily on official Recall@5;
- Precision@5 only breaks ties;
- no external data;
- do not promote an option without scorer-equivalent OOF evidence;
- do not run a large combinatorial grid on Kaggle.

---

# 15. Fresh verification sequence

The coding agent must run fresh commands; commit messages are not verification.

## Source/local gate

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('IMPORT_OK')"
pytest -q
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

Verify notebook parity:

```text
SHA(legalir_training.ipynb)
==
SHA(kaggle_kernel_task1/legalir_training.ipynb)
```

Verify source-level parameter preflight remains under 4B.

## Official Kaggle gate

Run only on official Task 1 data:

```text
LEGALIR_RUN_MODE=gpu_smoke
```

Required hardware:

```text
2 × NVIDIA T4
```

Required evidence:

```text
8532 official documents
7000 official train queries
999 public queries
FAISS active
Dense actual cuda:0
BGE PEFT actual cuda:1
real fold LoRA training
real final LoRA training
param_diff > 0
adapter checksum valid
adapter params counted
finite training
accurate OOM metrics
runtime projection
```

Only then permit:

```text
LEGALIR_RUN_MODE=full
```

---

# 16. Readiness policy

Maximum conclusion before real T4×2 execution:

```text
Source-level repair complete; ready for official Kaggle T4x2 gpu_smoke.
```

Only state:

```text
READY FOR FULL KAGGLE RUN: YES
```

after the real official-data GPU smoke passes all gates.

No GitHub commit message, mock test, or static audit is sufficient to claim FULL READY.

---

# 17. Required final report format

```markdown
# LegalIR DCC007 Final Pre-GPU-Smoke Report

## Base
- audited head: dcc007563733d24863f9c0411a0329a743188736
- new head: <sha>

## Source repair gates
| Gate | PASS/FAIL | Evidence |
|---|---|---|
| ExactMatcher cross-doc isolation | | |
| real query-aware sampler | | |
| >=99% actual query coverage gate | | |
| official dataset identity gate | | |
| tiny production smoke rejected | | |
| strict loaded PEFT audit | | |
| true PEFT detection | | |
| persistent reranker OOM fallback | | |
| Dense telemetry aggregation | | |
| corrected runtime projection | | |
| FAISS production enforcement | | |
| actual fusion model-type reporting | | |

## Fresh local verification
- compileall:
- import:
- pytest:
- CPU smoke:
- notebook SHA parity:
- preflight parameters:

## Official T4x2 GPU smoke
- executed: YES/NO
- documents:
- train queries:
- public queries:
- GPU0:
- GPU1:
- Dense requested/actual:
- Reranker requested/actual:
- Dense backend:
- Dense OOM events:
- OOF reranker OOM events:
- final reranker OOM events:
- stable Dense batches:
- stable reranker batch:
- optimizer steps:
- param_diff:
- eligible training queries:
- actual unique queries seen:
- query coverage:
- adapter SHA verified:
- adapter parameters:
- final loaded total parameters:
- fusion model type:
- projected full runtime:
- result:

## OOF / score evidence
- Candidate Recall@50:
- Candidate Recall@150:
- Reranker Recall@5:
- Fusion winner:
- Fusion Recall@5:
- Precision@5:
- Doc-disjoint Recall@5:

## Readiness
READY FOR KAGGLE GPU SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
