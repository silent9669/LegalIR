# LegalIR 9506 — Kaggle Live-Integration Final Gate

> **For agentic workers:** implement test-first. Preserve the current retrieval/reranking architecture. This plan is scoped to the remaining integration, coverage, runtime-evidence, and reproducibility defects at HEAD `9506e0276a60d131f6f0e091b09207ab00d15c4e`.

**Goal:** make the repository's production notebook work with the actual Kaggle input `phucdangg/legalir-task1-clean-data`, make training coverage mathematically correct for FULL and every OOF/doc-disjoint adapter, and make the real T4×2 `gpu_smoke` a trustworthy gate for the 12-hour Kaggle FULL run.

**Architecture:** keep the existing 5-branch retrieval → BGE-v2-m3 PEFT/LoRA reranker → leakage-safe OOF fusion → deterministic top-5 system. Do not change model families before these runtime gates pass.

**Tech stack:** Python, PyTorch, Transformers, PEFT, DEk21, BGE reranker v2 M3, FAISS, LightGBM, pandas/Parquet, Kaggle T4×2.

**Audited HEAD:** `9506e0276a60d131f6f0e091b09207ab00d15c4e`

**Kaggle notebook:** `phucdangg/legalir-training`

**Actual Kaggle dataset:** `phucdangg/legalir-task1-clean-data`

**Decision:** **NOT READY FOR FULL KAGGLE RUN**

---

## Evidence boundary

The live Kaggle notebook/dataset HTML pages were not retrievable from the audit environment, so do not claim that live execution logs or attached-input UI state were inspected.

This plan is grounded in:

- the exact Kaggle notebook/dataset slugs supplied by the user;
- the repository notebook source and `kernel-metadata.json`;
- the latest GitHub HEAD;
- current Kaggle notebook documentation;
- the supplied official Task 1 source counts.

Therefore the next proof step is a real Kaggle T4×2 `gpu_smoke`.

---

# Preserve the fixes already present at 9506

Do not regress:

- official `8,532` document and `7,000` train-query checks;
- official public count = `999`;
- FULL/GPU_SMOKE strict artifacts;
- FULL/GPU_SMOKE dual-GPU topology gate;
- Dense `cuda:0`, reranker `cuda:1`;
- FAISS required in production;
- adaptive Dense OOM batching;
- persistent reranker OOM batching;
- QueryBalancedSampler instead of DataLoader shuffle;
- eligible-set coverage intersection;
- all outer-fold pairs used by `train_reranker`;
- AMP GradScaler + finite-loss/gradient checks;
- adapter checksum verification;
- actual fusion model type in `manifest.json`;
- exact top-5 final output validation;
- notebook default = `gpu_smoke`;
- root and kernel notebook byte parity.

---

# P0-1 — The production dataset discovery does not match the actual Kaggle dataset

## Root cause

The actual user dataset is:

```text
phucdangg/legalir-task1-clean-data
```

The most likely flat mount is:

```text
/kaggle/input/legalir-task1-clean-data
```

Kaggle can also use a more namespaced mount such as:

```text
/kaggle/input/datasets/phucdangg/legalir-task1-clean-data
```

depending on how the input is resolved.

Current `src/pipeline/kaggle_train.py::discover_data_dir()` does **not** list the actual slug and does not recursively search `/kaggle/input` for a complete canonical directory.

Current `discover_public_test_file()` likewise does not list the actual dataset slug.

The thin notebook delegates directly to these production functions:

```python
DATA_DIR = discover_data_dir(repo_root=REPO_ROOT)
PUBLIC_TEST_FILE = discover_public_test_file(repo_root=REPO_ROOT)
```

so the Kaggle notebook can fail before training even when the correct dataset is attached.

There is already a same-repository reference implementation in:

```text
kaggle_kernel_task1/build_notebook.py
```

that recursively scans `/kaggle/input` and explicitly includes `legalir-task1-clean-data`. Production discovery should be at least as robust.

---

## Required implementation

Create one deterministic input discovery helper.

```python
CANONICAL_REQUIRED = {
    "documents.parquet",
    "chunks.parquet",
    "queries_train.parquet",
    "qrels_train.parquet",
}
```

Search preferred roots first:

```text
/kaggle/input/legalir-task1-clean-data
/kaggle/input/legalir-task1-clean-data/artifacts/task1/data
/kaggle/input/legalir-task1-clean-data/artifacts/shared/canonical/v2
/kaggle/input/datasets/phucdangg/legalir-task1-clean-data
/kaggle/input/datasets/phucdangg/legalir-task1-clean-data/artifacts/task1/data
/kaggle/input/datasets/phucdangg/legalir-task1-clean-data/artifacts/shared/canonical/v2
```

Then recursively scan `/kaggle/input` for directories containing **all four** canonical files.

For `public-official.json`, first search relative to the selected dataset root and its parents, then recursively search `/kaggle/input`.

If exactly one complete canonical set exists, use it.

If multiple complete sets exist:

1. prefer an explicit `data_dir`;
2. otherwise prefer the actual attached dataset slug;
3. otherwise raise an ambiguity error listing every candidate.

Never silently select the first arbitrary recursive match.

Log:

```text
selected canonical root
selected public file
Kaggle input root
all complete canonical candidates
```

---

## Mandatory tests

```text
test_discover_actual_flat_kaggle_dataset_slug
test_discover_actual_namespaced_kaggle_dataset_slug
test_discover_nested_canonical_under_actual_dataset
test_public_file_discovered_from_same_dataset
test_recursive_discovery_requires_all_four_parquets
test_ambiguous_kaggle_inputs_raise
```

Use `tmp_path` to model the exact mount trees.

---

# P0-2 — `kernel-metadata.json` still attaches the wrong Kaggle dataset

Current repository metadata contains:

```json
"dataset_sources": [
  "phucdangg/legalir"
]
```

but the dataset supplied for the real run is:

```text
phucdangg/legalir-task1-clean-data
```

If the notebook is pushed/versioned from repository metadata, this can attach the wrong input or no longer reproduce the user's current live notebook setup.

## Required fix

Update:

```text
kaggle_kernel_task1/kernel-metadata.json
```

to:

```json
"dataset_sources": [
  "phucdangg/legalir-task1-clean-data"
]
```

Keep:

```text
enable_gpu = true
enable_internet = true
```

The notebook itself must still verify at runtime that the mounted input contains:

```text
8,532 docs
7,000 train queries
999 public queries
```

Do not trust metadata alone.

## Test

```text
test_kernel_metadata_uses_actual_clean_dataset_source
```

---

# P0-3 — The 99% BCE coverage step calculation is still mathematically wrong for the actual sampler

## Current implementation

`compute_coverage_required_steps()` currently calculates:

```python
2 * ceil(target_pct * N)
```

rows.

For:

```text
N = 7000
target = 99%
batch = 2
grad_accum = 8
```

this returns:

```text
13,860 rows
867 optimizer steps
```

The new test explicitly locks in:

```python
assert steps_99 == 867
```

## Why this is wrong

The actual sampler order is:

```text
Phase A: one positive for every query
Phase B: one negative for every query
Phase C: repeats
```

To reach **99% negative coverage**, Phase A must first consume all `N` positives.

Required rows are therefore:

```python
N + ceil(0.99 * N)
```

For 7,000 queries:

```text
7,000 + 6,930 = 13,930 rows
13,930 / 16 = 870.625
=> 871 optimizer steps minimum
```

At 867 steps:

```text
867 × 16 = 13,872 rows
negative rows reached = 13,872 - 7,000 = 6,872
negative coverage = 6,872 / 7,000 = 98.17%
```

FULL will therefore still fail its own:

```text
negative_query_coverage_pct >= 99
```

gate after training.

---

## Preferred repair

Use the simple robust production invariant:

```text
Phase A + Phase B must finish completely before training may stop.
```

Thus:

```python
coverage_required_rows = 2 * eligible_query_count
coverage_required_steps = ceil(
    coverage_required_rows /
    (batch_size * gradient_accumulation_steps)
)
```

For 7,000 queries:

```text
14,000 / 16 = 875 steps
```

This guarantees 100% positive and 100% negative eligible-query exposure and removes boundary ambiguity.

If configurable `<100%` coverage is truly desired, calculate it from the **sampler phase order**, not `2 * target_queries`.

---

## Also fix eligible/ineligible scheduling

`QueryBalancedSampler` currently builds:

```python
qids = eligible_sorted + other_sorted
```

then Phase A loops all qids and Phase B loops all qids.

Ineligible positive-only queries can therefore consume rows before eligible-query negative Phase B.

Production coverage guarantees must schedule:

```text
eligible Phase A positives
eligible Phase B negatives
then non-eligible rows
then remaining eligible/non-eligible repeats
```

The step calculation and sampler must use exactly the same scheduling contract.

---

## Mandatory tests

Delete/replace the incorrect `867` test.

Add:

```text
test_7000_bce_queries_require_875_steps_for_complete_posneg_cycle
test_867_steps_do_not_reach_99pct_negative_coverage
test_sampler_finishes_eligible_positive_phase_before_negative_phase
test_sampler_finishes_eligible_negative_phase_before_ineligible_rows
test_step_formula_matches_real_sampler_prefix
```

The strongest test should iterate the sampler prefix for a synthetic query set and compare actual observed coverage to the formula.

---

# P0-4 — Full OOF fold adapters are still under-covered relative to the final adapter

## Root cause

The final FULL adapter now receives a dynamically expanded step budget.

But in `OOFRunner` each full fold still calls approximately:

```python
train_reranker(
    ...,
    max_steps=5 if self.smoke else None,
)
```

For FULL, `None` resolves back to the config:

```text
max_steps = 500
```

A typical 5-fold outer training set contains roughly:

```text
5,600 train queries
```

With the BCE sampler:

```text
500 × 2 × 8 = 8,000 rows
```

After the first 5,600 positive rows, only about 2,400 negative rows are reached:

```text
2,400 / 5,600 ≈ 42.9% negative query coverage
```

The final model will be trained with near/full positive+negative coverage, while OOF fold models are not.

This makes the OOF reranker/fusion selection distribution materially different from the final inference model.

---

## Required fix

Coverage-aware step budgeting must apply to:

```text
each OOF fold adapter
document-disjoint adapter
final adapter
```

Before each training job:

1. inspect the actual pair file;
2. derive eligible query IDs with >=1 positive and >=1 negative;
3. compute required full Phase-A/B cycle steps;
4. choose:

```python
effective_steps = max(configured_steps, coverage_required_steps)
```

within a measured T4 runtime ceiling.

Report for each fold:

```text
eligible_training_queries
coverage_required_steps
effective_max_steps
unique coverage
positive coverage
negative coverage
optimizer steps
```

Do not compute fold steps from the global 7,000 count.

---

## Required tests

```text
test_fold_step_budget_uses_actual_fold_eligible_qids
test_doc_disjoint_step_budget_uses_actual_eligible_qids
test_fold_negative_coverage_gate_is_not_left_at_500_steps
test_oof_report_exports_fold_posneg_coverage
```

---

# P0-5 — Validate mined-pair coverage before loading/training BGE

The current pipeline dynamically sizes steps from expected train-query counts before final pair mining.

However some organizer query may theoretically produce:

```text
no positive pair
no hard negative
missing pair rows
```

The true training coverage ceiling depends on the mined pair file.

## Required fix

Add a cheap `audit_pair_coverage(pairs_df, expected_qids)` before loading the BGE training model.

Report:

```text
expected_qids
qids_in_pairs
qids_with_positive
qids_with_negative
eligible_qids
missing_qids
positive_missing_qids
negative_missing_qids
```

For FULL final training require:

```text
qids_in_pairs == 7000
positive coverage == 100%
negative coverage >= 99% (prefer 100%)
```

or fail before model load with explicit IDs/counts.

Then compute `coverage_required_steps` from **actual eligible qids**, not only `len(df_queries)`.

Apply equivalent fold-safe validation for OOF/doc-disjoint pair files.

---

# P1-1 — Dense per-stage telemetry exists but is not actually used by the GPU report

`DenseEncodeTelemetry` and `stage_telemetry` are now implemented.

But `gpu_smoke_report.json` still reads legacy cumulative fields such as:

```text
dense_oom_events
dense_initial_batch_size
dense_min_successful_batch_size
```

and labels them as corpus/public metrics.

Also public query encoding currently uses the default query stage name, so it can be recorded as:

```text
train_query
```

instead of:

```text
public_query
```

## Required fix

Pass explicit stage names:

```python
dense_retriever.fit(... stage_name="corpus")        # extend fit if needed
dense_retriever.encode_queries(... stage_name="train_query")
dense_ret.encode_queries(... stage_name="public_query")
```

Snapshot the actual telemetry object immediately after each stage.

Write:

```text
dense_corpus.requested_batch_size
dense_corpus.min_successful_batch_size
dense_corpus.oom_events
dense_corpus.item_count
dense_corpus.elapsed_seconds

dense_train_query.*
dense_public_query.*
```

Then:

```python
dense_total_oom_events = sum(stage.oom_events ...)
```

Do not infer stage metrics from cumulative object properties.

---

# P1-2 — Runtime projection still says Dense is included without measuring Dense build time

The latest projection correctly:

- uses 7,000 total held-out OOF queries;
- projects 5 folds;
- scales smoke training steps.

But it still sets:

```json
"includes_dense_build": true
```

while the projection does not add an observed Dense corpus build duration.

It also does not explicitly add:

```text
train-query encoding time
final pair-mining time
```

and instead adds a generic 300-second setup allowance.

This is not evidence-grade enough to authorize a long FULL run.

## Required fix

Add orchestrator timers:

```text
canonical load/validation
BM25 raw build/load
BM25 PyVi build/load
Dense corpus build/load
train-query encoding
OOF total by separated stages
fusion training
full-memory build
final pair mining
final BGE training
final pipeline load/audit
public inference
submission validation/package
```

For a cached stage, explicitly record:

```text
cache_hit = true
measured_seconds = ...
```

The GPU-smoke FULL projection must distinguish:

```text
cold-start FULL
warm-cache FULL
```

Kaggle Save & Run All starts a clean session; do not assume `/kaggle/working` indexes from an interactive run are available unless they are an attached persisted dataset/input.

---

## Kaggle time budget

Current Kaggle notebook documentation states:

```text
12 hours for CPU/GPU notebook sessions
T4 x2 = 2 Tesla T4 + 29 GB host RAM
```

The current hard-coded gate:

```python
projected_total_sec < 32400  # 9 h
```

is conservative, but it can falsely reject a 9–12 hour run.

Use configuration such as:

```python
KAGGLE_MAX_SECONDS = 12 * 3600
SAFETY_FACTOR = 0.90
PRODUCTION_RUNTIME_BUDGET = KAGGLE_MAX_SECONDS * SAFETY_FACTOR
```

This yields a ~10.8-hour planned ceiling while retaining headroom.

Do not use the full 12h as the planned runtime target.

---

# P1-3 — Notebook clone is not pinned to the audited commit

The thin notebook currently does:

```bash
git clone https://github.com/silent9669/LegalIR.git
```

then runs whatever default-branch HEAD exists at execution time.

A notebook reviewed at commit `9506e027...` may therefore execute a different later commit during Kaggle Save & Run All.

## Required fix

Add:

```python
EXPECTED_COMMIT = os.environ.get(
    "LEGALIR_COMMIT_SHA",
    "<approved full 40-char SHA>"
)
```

After clone:

```bash
git fetch --all
git checkout --detach $EXPECTED_COMMIT
```

Then verify:

```python
actual = git rev-parse HEAD
assert actual == EXPECTED_COMMIT
```

For development, explicitly allow:

```text
LEGALIR_COMMIT_SHA=main
```

but mark it non-reproducible and not a FULL-readiness run.

The committed notebook used for the production run should pin the approved repaired SHA after this plan is implemented.

---

# P1-4 — Host-RAM duplication risk on Kaggle T4×2

Kaggle's documented T4×2 host RAM is approximately:

```text
29 GB
```

The pipeline can hold multiple large retrieval stacks over ~1.15M chunks:

1. orchestrator-level BM25/PyVi/Dense;
2. `OOFRunner` loads its own BM25/PyVi/Dense/ExactMatcher/dataframes;
3. final `LegalIRPipeline.load_pipeline()` loads another retrieval stack.

The old objects are not aggressively released before the final stack is loaded.

This is a credible host-memory risk even if GPU VRAM is healthy.

## Required fix

Add process RSS telemetry using `psutil`:

```text
after canonical load
after BM25 indexes
after Dense index
before OOF
peak during OOF
after OOF cleanup
before final pipeline load
after final pipeline load
peak RSS
```

After OOF/fusion outputs are fully persisted and only scalar reports are still required:

```python
doc_disjoint_report = dict(oof_runner.doc_disjoint_report)
del oof_runner
gc.collect()
torch.cuda.empty_cache()
```

Also delete redundant orchestrator BM25/PyVi objects when no longer needed, or inject already-built retrievers into OOFRunner rather than reloading them.

Do not refactor the whole architecture solely for this; first instrument RSS, then remove clearly redundant copies.

GPU-smoke must demonstrate a safe host-RAM margin.

---

# P1-5 — Runtime audit must explicitly prove the active PEFT adapter

The strict loader checks adapter files, training manifest, parameter diff, optimizer steps, and SHA checksum. This is good.

But after:

```python
final_audit_report = pipeline.audit_parameters(...)
```

the production orchestrator should explicitly assert the runtime cross-encoder entry has:

```text
is_peft_lora == true
adapter_parameters > 0
total cross-encoder parameters > base parameters
```

The repository-level static `parameter_audit.json` is still the preflight architecture count:

```text
702,754,049
adapter_parameters = 0
```

which is fine as a static budget artifact, but it is not proof of the loaded final adapter.

## Required final runtime gate

Find the model entry whose role is:

```text
cross_encoder_reranker
```

and require:

```python
entry["is_peft_lora"] is True
entry["adapter_parameters"] > 0
final_audit_report["total_learned_parameters"] > 702_754_049
final_audit_report["total_learned_parameters"] < 4_000_000_000
```

Write these values into `gpu_smoke_report.json`.

---

# P1-6 — Fix stale Kaggle documentation

Update README/notebook comments after source fixes.

Current README still says:

```text
Dataset: LegalIR mounted at /kaggle/input/legalir
1,000 public predictions
T4 x2 (or T4)
```

Production reality is:

```text
dataset source = phucdangg/legalir-task1-clean-data
public file = 999 queries
production gate = T4 x2
```

Do not hard-code one physical mount as the only supported path; document discovery instead.

---

# P1-7 — Stronger tests for the actual integration boundary

New tests must exercise behavior, not just arithmetic constants.

Required additions:

```text
test_kaggle_actual_dataset_slug_is_discoverable
test_kernel_metadata_matches_actual_dataset_slug

test_867_steps_fail_real_sampler_negative_coverage
test_875_steps_finish_real_7000_query_phase_ab
test_ineligible_queries_do_not_delay_eligible_phase_b

test_fold_pair_audit_drives_fold_effective_steps
test_final_pair_audit_fails_before_model_load

test_public_dense_stage_is_named_public_query
test_gpu_report_uses_stage_telemetry_not_legacy_counters

test_runtime_projection_contains_real_dense_and_pair_mining_times
test_cold_start_projection_does_not_claim_unmeasured_stage_included

test_notebook_checkout_is_pinned_to_expected_sha
test_runtime_parameter_audit_requires_active_lora
```

For 7,000-query sampler tests, no transformer/model download is needed; use integer pair records.

---

# P2 — Score optimization only after real GPU smoke

Do not start another large architecture rewrite.

After the official-data T4×2 runtime gate passes, ranking remains the main historical headroom.

Run sequential leakage-safe OOF ablations:

```text
1. rerank_k = 40 vs 50 vs 80
2. BCE vs pairwise_logistic
3. train steps above the coverage minimum
4. RRF branch weights
5. candidate_k = 150 vs 200 only if candidate misses justify it
```

Primary promotion criterion:

```text
Recall@5
```

Tie-break:

```text
Precision@5
```

No promotion without scorer-equivalent held-out OOF evidence.

---

# Mandatory verification order

## 1. Static/import

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('IMPORT_OK')"
```

## 2. Targeted test-first repair

For every bug:

```text
write failing behavioral test
run and observe RED for the intended reason
implement minimal fix
run and observe GREEN
```

## 3. Full local tests

```bash
pytest -q
```

Report exact passed/failed/skipped counts.

## 4. CPU orchestration smoke

```bash
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

This is not GPU readiness evidence.

## 5. Notebook parity

Require:

```text
SHA256(legalir_training.ipynb)
==
SHA256(kaggle_kernel_task1/legalir_training.ipynb)
```

## 6. Kernel metadata

Require:

```text
dataset_sources = ["phucdangg/legalir-task1-clean-data"]
```

## 7. Official Kaggle T4×2 gpu_smoke

Use the actual attached dataset:

```text
phucdangg/legalir-task1-clean-data
```

Required run evidence:

```text
actual canonical root printed
8,532 docs
7,000 train queries
999 public queries
2 CUDA GPUs
Dense actual cuda:0
BGE/PEFT actual cuda:1
FAISS active

real fold LoRA training
real doc-disjoint LoRA training
real final LoRA training
adapter save/reload
adapter SHA verified
adapter parameters > 0

fold coverage-required steps projected
final coverage-required steps projected
positive/negative coverage telemetry

per-stage Dense telemetry
per-stage OOF timers
final pair-mining timer
host RSS telemetry
GPU peak VRAM
cold-start runtime projection
```

Only then authorize FULL.

---

# FULL readiness policy

A source-only repair may report:

```text
READY FOR OFFICIAL KAGGLE GPU_SMOKE: YES
READY FOR FULL KAGGLE RUN: NO
```

A real `gpu_smoke` may report FULL-ready only if:

```text
official dataset resolved correctly
actual dual-GPU placement verified
no unresolved OOM
host RAM has safe margin
active PEFT audit passes
coverage math and pair coverage pass
cold-start projected runtime <= configured safety budget
```

No commit message, mock test, or static parameter JSON can replace this proof.

---

# Required final agent report

```markdown
# LegalIR 9506 Kaggle Integration Gate Report

## Base
- audited head: 9506e0276a60d131f6f0e091b09207ab00d15c4e
- new head:

## Kaggle integration
- kernel dataset source:
- discovered canonical root:
- documents:
- train queries:
- public queries:
- pinned commit:

## Coverage
- final eligible qids:
- final required steps:
- final effective steps:
- unique coverage:
- positive coverage:
- negative coverage:
- fold required/effective steps:
- doc-disjoint required/effective steps:

## Hardware / memory
- GPU0:
- GPU1:
- Dense requested/actual:
- Reranker requested/actual:
- FAISS:
- GPU0 peak VRAM:
- GPU1 peak VRAM:
- peak host RSS:

## Dense telemetry
- corpus:
- train_query:
- public_query:
- total OOM events:

## PEFT audit
- adapter SHA verified:
- adapter parameters:
- final total learned parameters:
- <4B:

## Runtime projection
- cold-start:
- warm-cache:
- measured stages:
- projected total hours:
- configured safety budget:
- fits budget:

## OOF
- Candidate Recall@50:
- Candidate Recall@150:
- Reranker Recall@5:
- Fusion winner:
- Fusion model type:
- Fusion Recall@5:
- Precision@5:
- Doc-disjoint Recall@5:

## Fresh verification
- compileall:
- import:
- targeted RED/GREEN:
- pytest:
- CPU smoke:
- notebook parity:
- kernel metadata:

## Official T4x2 gpu_smoke
- executed:
- result:

READY FOR OFFICIAL KAGGLE GPU_SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
