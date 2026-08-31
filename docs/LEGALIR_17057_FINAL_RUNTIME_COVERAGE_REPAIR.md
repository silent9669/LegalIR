# LegalIR Task 1 — 17057 Final Runtime + Coverage Repair Plan

> **For agentic workers:** implement this plan test-first. Preserve the existing LegalIR architecture and the fixes already present at HEAD `17057f347cd580653b8247a26627e8e798f7b688`. Do not claim FULL-ready from static inspection or mocked tests; the final readiness gate is an official-data Kaggle T4×2 `gpu_smoke`.

**Goal:** remove the remaining source-level blockers that would either make FULL fail after expensive training or make the GPU-smoke readiness evidence unreliable.

**Architecture:** keep the existing 5-branch retrieval → BGE LoRA reranker → cross-fitted fusion → top-5 pipeline. Repair training coverage semantics, full-mode hardware enforcement, runtime projection, final PEFT auditing, telemetry, fusion manifests, and behavioral tests. Do not redesign retrieval or replace the current models before the runtime gate is proven.

**Tech stack:** Python, PyTorch, Transformers, PEFT/LoRA, DEk21, BGE reranker, FAISS, LightGBM, pandas, pytest, Kaggle T4×2.

**Audited HEAD:** `17057f347cd580653b8247a26627e8e798f7b688`

**Previous repair base:** `dcc007563733d24863f9c0411a0329a743188736`

**Decision:** **NOT READY FOR FULL KAGGLE RUN**

---

## Global constraints

- Task 1 organizer data only.
- Official canonical identity: `8,532` documents and `7,000` train queries.
- Supplied `public-official.json`: `999` public queries.
- Final output: at most 5 unique corpus-valid document IDs per query.
- Primary score: Recall@5.
- Secondary/tie-break score: Precision@5.
- Total learned parameters: strictly `< 4,000,000,000`.
- Intended hardware: Kaggle **T4 ×2**.
- Dense model must run on `cuda:0`.
- BGE reranker must run on `cuda:1`.
- Production Dense search backend must be FAISS.
- No FULL-ready claim without a fresh real T4×2 `gpu_smoke`.

---

# What 17057 fixed correctly

Preserve these changes:

- ExactMatcher stale-variable cross-document contamination is removed.
- ExactMatcher now loads only statutory chunk columns when possible.
- official document/train-query counts are checked for `gpu_smoke` and `full`.
- `--tiny` is rejected for production smoke/full modes.
- a real `QueryBalancedSampler` is wired into DataLoader instead of `shuffle=True`.
- PEFT detection is stronger than checking only `base_model`.
- strict parameter audit can propagate force-load failures.
- reranker OOM fallback now persists a reduced batch inside a scoring call.
- FAISS is required after Dense build/load in production modes.
- public query count is checked as 999.
- five-fold OOF validation count in runtime projection is no longer hard-coded as `7000 × 5`.
- comparison reporting now exposes a learned model type.
- root and Kaggle notebook copies remain synchronized at the currently checked blob.

Do not weaken these fixes.

---

# P0-1 — FULL training is mathematically unable to pass the new coverage gate

## Root cause

Current production config:

```yaml
loss_type: bce
batch_size: 2
gradient_accumulation_steps: 8
max_steps: 500
```

Therefore a normal full run processes approximately:

```text
500 optimizer steps
× 2 rows/microbatch
× 8 microbatches/optimizer-step
= 8,000 pair-row exposures
```

Current `QueryBalancedSampler.__iter__()` schedules in pass 1:

```text
query A positive
query A negative
query B positive
query B negative
...
```

For 7,000 eligible queries, completing positive+negative once for every query requires about:

```text
14,000 row exposures
```

With the current sequence, the first 8,000 rows cover only approximately:

```text
8,000 / 2 = 4,000 distinct query IDs
4,000 / 7,000 = 57.14%
```

Then `run_kaggle_pipeline(..., run_mode="full")` hard-fails when:

```python
actual_query_coverage_pct < 99.0
```

This means FULL can spend the entire 500-step final training job and then fail by design.

This is the highest-priority blocker.

---

## Required behavior

Make coverage a pre-computable training invariant, not a post-hoc surprise.

For BCE pair training define three distinct metrics:

```text
unique_query_coverage_pct
positive_query_coverage_pct
negative_query_coverage_pct
```

Coverage must be computed against **eligible query IDs only** for eligible coverage, and separately against all organizer train query IDs.

Recommended scheduling for BCE:

```text
Phase A: one positive row for every eligible query
Phase B: one strongest hard-negative row for every eligible query
Phase C: remaining positives/negatives round-robin
```

This makes unique query coverage complete before repeats, while still guaranteeing positive and negative exposure when enough steps are allocated.

For `N` eligible queries:

```python
rows_for_unique_and_positive = N
rows_for_positive_and_negative = 2 * N
effective_rows_per_optimizer_step = batch_size * gradient_accumulation_steps
```

At the current `batch_size=2`, `grad_accum=8`:

```text
100% pos+neg coverage for 7,000 queries:
14,000 / 16 = 875 optimizer steps

99% pos+neg coverage:
13,860 / 16 = 866.25 -> 867 optimizer steps
```

Do not keep `max_steps=500` while claiming >=99% positive+negative query coverage.

---

## Production policy

Before loading BGE weights, compute:

```python
required_steps_99 = ceil(
    2 * ceil(0.99 * eligible_query_count)
    / (batch_size * gradient_accumulation_steps)
)
```

Then choose one explicit policy:

### Preferred policy

```python
effective_max_steps = max(configured_max_steps, required_steps_99)
```

with a hard upper bound configured for T4 runtime safety.

Record:

```text
configured_max_steps
coverage_required_steps
effective_max_steps
```

### Alternative policy

Fail **before expensive training** if:

```python
configured_max_steps < coverage_required_steps
```

and tell the caller the required value.

Do not discover the mismatch only after training.

---

## FULL acceptance gates

For final reranker training require:

```text
eligible_query_count > 0
unique_query_coverage_pct >= 99
positive_query_coverage_pct >= 99
negative_query_coverage_pct >= 99
param_diff > 0
optimizer_steps >= coverage_required_steps
nonfinite_loss_count == 0
```

Prefer 100% where runtime permits.

---

## Fix the current coverage formula

Current code effectively uses:

```python
actual_seen_count = len(seen_query_ids)
eligible_count = len(eligible_query_ids)
coverage = actual_seen_count / eligible_count
```

This is incorrect when ineligible query IDs are also sampled: the numerator may contain IDs not in the eligible denominator and coverage can exceed 100%.

Use:

```python
seen_eligible = seen_query_ids & eligible_query_ids
positive_seen_eligible = positive_queries_seen & eligible_query_ids
negative_seen_eligible = queries_with_negative_seen & eligible_query_ids

unique_eligible_pct = len(seen_eligible) / len(eligible_query_ids)
positive_eligible_pct = len(positive_seen_eligible) / len(eligible_query_ids)
negative_eligible_pct = len(negative_seen_eligible) / len(eligible_query_ids)
```

Also report separate all-query coverage.

---

## Files

Modify:

```text
src/training/trainer.py
src/training/train_reranker.py
src/pipeline/kaggle_train.py
configs/experiments/reranker_lora.yaml
tests/test_dcc007_pre_gpu_smoke_invariants.py
```

---

## Required red/green tests

Write these tests first and verify they fail on HEAD `17057f`:

```text
test_500_steps_cannot_claim_99pct_posneg_coverage_for_7000_queries
test_sampler_first_phase_contains_one_unique_query_each
test_required_steps_for_7000_bce_queries_is_867_for_99pct
test_eligible_coverage_never_exceeds_100pct
test_ineligible_qids_do_not_inflate_eligible_coverage
test_full_fails_before_model_load_if_steps_cannot_meet_coverage
test_full_gates_positive_and_negative_coverage_independently
```

The current test named:

```text
test_query_sampler_covers_all_eligible_qids_before_repeat
```

is insufficient because it only checks that all three query IDs occur somewhere in the first six rows. It does **not** prove “before repeat”.

A valid test for Phase A should assert:

```python
first_n_qids = qids_from_indices(indices[:eligible_count])
assert len(first_n_qids) == eligible_count
assert len(set(first_n_qids)) == eligible_count
```

Then Phase B should test negative exposure.

---

# P0-2 — GPU-smoke runtime projection still does not project the FULL run

## Root cause

The 17057 projection corrected one formula:

```text
total held-out 5-fold OOF queries = 7,000
```

but the `gpu_smoke` branch still does not extrapolate smoke training to production training.

In `gpu_smoke`:

```text
final reranker max_steps ≈ 3
OOF fold reranker max_steps ≈ 5
```

Current projection uses smoke training time nearly directly when `is_full == False`.

Therefore:

```text
3-step final training
2 tiny smoke folds
```

can be used to compute:

```text
fits_kaggle_session_limit = true
```

without projecting:

```text
5 production folds
production fold step count
doc-disjoint training
~867–875 final steps if BCE coverage is enforced
```

That readiness conclusion is not trustworthy.

There is a second problem: `cv_report["queries_per_second"]` is based on total OOF runner elapsed time, which includes fold training/pair mining, then the projection separately adds fold training again. That mixes inference throughput with training overhead and can double-count or distort runtime.

---

## Required behavior

Add explicit stage timers rather than infer everything from one aggregate QPS.

OOF runner must report at least:

```text
pair_mining_seconds_by_fold
reranker_training_seconds_by_fold
reranker_optimizer_steps_by_fold
heldout_retrieval_rerank_seconds_by_fold
heldout_queries_by_fold
heldout_queries_per_second_by_fold
doc_disjoint_pair_mining_seconds
doc_disjoint_training_seconds
doc_disjoint_optimizer_steps
doc_disjoint_inference_seconds
```

Orchestrator must report:

```text
dense_corpus_build_seconds
train_query_encoding_seconds
final_pair_mining_seconds
final_training_seconds
final_optimizer_steps
public_inference_seconds
public_queries_measured
```

---

## Projection formula for GPU smoke

Derive a per-step training rate:

```python
sec_per_fold_optimizer_step
sec_per_final_optimizer_step
```

Then project:

```text
5 × projected fold training
+ total 7,000 held-out OOF inference queries
+ document-disjoint training/inference
+ final pair mining
+ projected final training at coverage_required_steps/effective_max_steps
+ 999 public-query inference
+ Dense/index setup if not reused
+ conservative setup margin
```

Do not use:

```text
2 smoke folds
3 smoke final steps
```

as the production count.

`runtime_projection.json` must include every assumption:

```json
{
  "measured_gpu_smoke_fold_steps": 5,
  "projected_full_fold_steps": 0,
  "measured_final_steps": 3,
  "projected_final_steps": 0,
  "projected_num_folds": 5,
  "total_oof_validation_queries": 7000,
  "public_queries": 999,
  "includes_doc_disjoint": true,
  "includes_dense_build": true,
  "fits_kaggle_session_limit": false
}
```

Populate real numbers, not zeros.

---

## Files

Modify:

```text
src/pipeline/oof_runner.py
src/pipeline/kaggle_train.py
src/training/train_reranker.py
tests/test_dcc007_pre_gpu_smoke_invariants.py
```

---

## Required tests

```text
test_gpu_smoke_projection_scales_3_steps_to_full_final_steps
test_gpu_smoke_projection_uses_five_production_folds
test_projection_includes_doc_disjoint_stage
test_projection_does_not_use_oof_total_elapsed_as_pure_inference_qps
test_projection_uses_len_public_data_not_magic_1000
```

---

# P0-3 — FULL mode does not enforce the same T4×2 topology as gpu_smoke

## Root cause

Current code hard-gates dual CUDA and exact device mapping only when:

```python
is_gpu_smoke
```

A FULL run can still resolve to:

```text
cpu/cpu
cuda:0/cuda:0
reversed custom mapping
```

and proceed into expensive work.

The intended competition run is T4×2.

---

## Required behavior

Unless an explicit debug override is supplied, both:

```text
gpu_smoke
full
```

must require:

```text
torch.cuda.is_available() == True
torch.cuda.device_count() >= 2
dense_device == "cuda:0"
reranker_device == "cuda:1"
```

After model loading, verify actual parameter placement for **both** `gpu_smoke` and `full`:

```text
next(Dense.parameters()).device == cuda:0
next(BGE/PEFT.parameters()).device == cuda:1
```

Recommended debug flag:

```python
allow_nonstandard_production_devices: bool = False
```

A run with this flag enabled must be marked:

```text
NON_STANDARD_HARDWARE
NOT_A_FULL_READINESS_GATE
```

---

## Tests

```text
test_full_rejects_cpu
test_full_rejects_single_gpu
test_full_rejects_cuda0_cuda0_mapping
test_full_verifies_actual_dense_cuda0
test_full_verifies_actual_reranker_cuda1
```

---

# P1-1 — Fold-specific reranker training still wastes 10% of outer-fold training queries

## Root cause

`train_reranker()` does an internal query-level 90/10 split whenever:

```python
fold is not None
```

The current trainer does not use that inner validation set for early stopping or model selection. It trains fixed steps and evaluates validation only after training.

Therefore every outer OOF fold throws away roughly 10% of already leakage-safe training queries with no training benefit.

Example:

```text
outer training queries ≈ 5,600
inner train after 10% holdout ≈ 5,040
```

Current 500 BCE steps then process only 8,000 pair rows and do not guarantee pos+neg coverage even over those 5,040 queries.

This can lower Recall@5 and makes fold adapters less representative.

---

## Required behavior

Choose one coherent policy:

### Recommended for current fixed-step pipeline

For outer OOF fold adapters:

```text
use all outer-fold training query IDs
val_data = None
```

The outer fold itself is the validation set for scorer evaluation.

### Alternative

Keep an inner split only if it is actually used for:

```text
early stopping
checkpoint selection
learning-rate decisions
```

and is deterministic/randomized by seed rather than relying on input order.

Do not hold out 10% just to print an unused post-training BCE accuracy.

---

## Tests

```text
test_outer_fold_training_uses_all_outer_train_qids_when_no_early_stopping
test_fold_adapter_never_sees_outer_validation_qids
test_fold_coverage_requirement_is_computed_from_actual_outer_train_qids
```

---

# P1-2 — Final strict parameter audit still does not assert active adapter parameters

## Root cause

`require_loaded_models=True` verifies that components are loaded `nn.Module` objects.

That is useful, but it does not itself enforce:

```text
cross_encoder_reranker.is_peft_lora == true
adapter_parameters > 0
```

The checked-in static audit still correctly represents only the base architecture and therefore shows zero adapter parameters. That file is not proof of the final loaded system.

---

## Required behavior

After final production audit, find the cross-encoder audit entry and assert:

```python
is_peft_lora is True
adapter_parameters > 0
parameters > base_parameters
```

Also require:

```text
final total learned parameters > 702,754,049
final total learned parameters < 4,000,000,000
```

Export to:

```text
parameter_audit.json
gpu_smoke_report.json
submission_manifest.json
KaggleRunResult.audit_report
```

Add:

```text
adapter_checksum_verified = true
```

only after the strict artifact SHA check really succeeds.

---

## Tests

```text
test_strict_pipeline_audit_rejects_plain_loaded_reranker_when_adapter_requested
test_final_audit_requires_positive_adapter_parameter_count
test_gpu_smoke_report_contains_adapter_parameter_count
```

---

# P1-3 — Dense telemetry is still cumulative and mislabeled by stage

## Root cause

The original Dense retriever instance is reused for:

```text
corpus encoding
train-query encoding
```

Its `dense_oom_events` is cumulative.

The report labels that cumulative number approximately as:

```text
dense_corpus_oom_events
```

Then public query encoding happens on a second Dense instance.

Therefore stage attribution remains inaccurate.

Additionally, `dense_min_successful_batch_size` is initialized to 32 before any call. On a cached-index run where only a 128-query batch succeeds, the report can still imply a minimum successful batch of 32 although 32 was never attempted.

---

## Required behavior

Implement per-call telemetry.

Recommended object:

```python
@dataclass
class DenseEncodeTelemetry:
    requested_batch_size: int
    min_successful_batch_size: int | None
    last_successful_batch_size: int | None
    oom_events: int
    item_count: int
    elapsed_seconds: float
```

`encode_texts()` / `encode_queries()` should either return telemetry or update a named `last_encode_telemetry`.

The orchestrator must snapshot after each stage:

```text
dense_corpus
dense_train_query
dense_public_query
```

Then aggregate totals without relabeling cumulative counters.

---

## Tests

```text
test_dense_telemetry_separates_corpus_and_train_query_calls
test_cached_dense_run_does_not_report_unattempted_batch32
test_dense_total_oom_equals_sum_of_stage_ooms
```

---

# P1-4 — Fusion model type is not actually written into the fusion manifest

## Root cause

The comparison report contains:

```text
winning_model_type
```

but `manifest.json` still records:

```text
winning_method
winning_metrics
models
```

without `winning_model_type`.

The test named:

```text
test_fusion_manifest_reports_actual_model_type
```

only reads `fusion_comparison.json`, so it does not test the manifest its name claims to test.

---

## Required behavior

Write into fusion manifest:

```json
{
  "winning_method": "learned_ranker",
  "winning_model_type": "lightgbm"
}
```

Allowed actual types:

```text
lightgbm
linear_fallback
rrf_weighted
```

If fold learned models use mixed implementation types because some LightGBM folds fall back, record:

```text
fold_model_types
```

and do not present the aggregate learned OOF result as a single LightGBM result without disclosure.

Strict production loader must validate the type against the artifact actually loaded.

---

## Tests

Replace the misleading test with:

```text
test_fusion_manifest_json_contains_actual_winning_model_type
test_fusion_manifest_type_matches_loaded_ranker_type
```

---

# P1-5 — One new test is not portable to a fresh clone / Kaggle

## Root cause

`test_gpu_smoke_cli_rejects_tiny()` invokes:

```text
REPO_ROOT/.venv/bin/python
```

but `.venv/` is explicitly gitignored.

A fresh clone or CI environment is not guaranteed to contain that interpreter. `subprocess.run()` can raise `FileNotFoundError` before the intended CLI assertion is tested.

---

## Required behavior

Use:

```python
import sys
sys.executable
```

or invoke the CLI function directly.

Preferred black-box test:

```python
subprocess.run(
    [sys.executable, str(script), "--tiny", "--run-mode", "gpu_smoke"],
    ...
)
```

---

## Test quality cleanup

Strengthen tests so names match behavior.

The following current tests are weaker than their titles:

```text
test_query_sampler_covers_all_eligible_qids_before_repeat
test_fusion_manifest_reports_actual_model_type
```

Do not accept implementation-mirroring assertions.

For each repair:

1. write the behavioral test;
2. run it and confirm RED for the intended reason;
3. implement the minimum fix;
4. run it GREEN;
5. run full pytest.

---

# P2 — Report and manifest consistency

Make all final artifacts agree on:

```text
run_mode
official dataset counts
git SHA
devices requested/actual
dense backend
final optimizer steps
coverage-required steps
unique/positive/negative coverage
adapter checksum verified
adapter parameter count
total parameter count
fusion winning method
fusion winning model type
runtime projection
```

Do not let `gpu_smoke_report.json`, `submission_manifest.json`, terminal output, and `KaggleRunResult` disagree.

---

# Mandatory verification sequence

## Phase A — static/import

Run fresh:

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('IMPORT_OK')"
```

Expected:

```text
exit code 0
IMPORT_OK
```

---

## Phase B — targeted RED/GREEN tests

Run each new regression test before and after its implementation.

At minimum:

```bash
pytest -q tests/test_dcc007_pre_gpu_smoke_invariants.py
```

Do not count a test as a regression test unless it was observed failing before the fix.

---

## Phase C — full local suite

```bash
pytest -q
```

Report exact:

```text
passed
failed
skipped
duration
```

No “should pass” language.

---

## Phase D — CPU toy smoke

```bash
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

This tests orchestration only.

It is **not** a GPU readiness gate.

---

## Phase E — notebook parity

Confirm:

```text
SHA256(legalir_training.ipynb)
==
SHA256(kaggle_kernel_task1/legalir_training.ipynb)
```

Keep notebook default:

```text
RUN_MODE=gpu_smoke
```

until the GPU gate passes.

---

# Official Kaggle T4×2 readiness gate

Run on the real organizer Task 1 dataset.

Required evidence:

```text
documents = 8532
train queries = 7000
public queries = 999

CUDA devices >= 2
Dense requested = cuda:0
Dense actual = cuda:0
Reranker requested = cuda:1
Reranker actual = cuda:1

Dense backend = faiss_index_flat_ip

real DEk21 loaded
real BGE reranker loaded
real LoRA training occurs
real adapter save/reload occurs

adapter checksum verified
adapter parameters > 0
total parameters < 4B

query coverage metrics recorded
positive coverage recorded
negative coverage recorded

OOF stage timers recorded
full runtime projected from production step counts
document-disjoint runtime included
fits Kaggle session limit based on corrected projection
```

For `gpu_smoke`, sampling expensive query counts is allowed, but:

```text
the full official corpus must be indexed
the full official train/public identity must be validated
real GPU kernels and real PEFT training must run
```

---

# FULL run authorization

Only set:

```text
LEGALIR_RUN_MODE=full
```

after the real T4×2 smoke report proves:

```text
READY_FOR_FULL = true
```

with corrected runtime projection.

FULL must fail before expensive work if:

```text
hardware topology is wrong
FAISS is missing
official data identity is wrong
coverage-required steps exceed allowed runtime policy
```

---

# Score-maximization stage after runtime correctness

Once the real GPU gate passes, stop changing runtime infrastructure and optimize Recall@5 with leakage-safe OOF ablations.

Recommended order:

```text
1. rerank_k: 40 / 50 / 80
2. BCE vs pairwise_logistic
3. final/OOF training-step budget after coverage minimum
4. RRF branch weights
5. candidate_k 150 vs 200 only if candidate misses justify it
```

Promote a change only if official-scorer-equivalent OOF Recall@5 improves. Use Precision@5 only as tie-break.

Do not run a broad combinatorial search.

---

# Required final coding-agent report

```markdown
# LegalIR 17057 Repair Report

## Base
- audited head: 17057f347cd580653b8247a26627e8e798f7b688
- new head:

## Training coverage
- eligible queries:
- configured max steps:
- required steps for 99% pos+neg coverage:
- effective max steps:
- unique coverage:
- positive coverage:
- negative coverage:

## Runtime projection
- measured gpu-smoke fold steps:
- projected full fold steps:
- measured final steps:
- projected final steps:
- OOF held-out query count:
- doc-disjoint included:
- dense build included:
- projected total hours:
- Kaggle session fit:

## Hardware
- CUDA count:
- Dense requested/actual:
- Reranker requested/actual:
- Dense backend:

## Parameter audit
- adapter checksum verified:
- adapter parameters:
- final loaded parameters:
- <4B:

## Fusion
- winning method:
- winning model type:
- manifest type matches loaded type:

## Fresh verification
- compileall:
- import:
- targeted tests:
- full pytest:
- CPU smoke:
- notebook parity:

## Official T4x2 gpu_smoke
- executed:
- result:

READY FOR KAGGLE GPU SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```

The maximum valid source-only conclusion is:

```text
SOURCE-LEVEL READY FOR OFFICIAL T4×2 GPU_SMOKE
```

Never claim:

```text
FULL READY
```

without the real GPU execution evidence.
