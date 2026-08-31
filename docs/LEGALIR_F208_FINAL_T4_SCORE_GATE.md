# LegalIR Task 1 — F208 Final T4 Runtime + Score Gate

**Repository:** `silent9669/LegalIR`  
**Audited HEAD:** `f2080e4881fbb54e41dff33836de014252719076`  
**Previous audited HEAD:** `cd40702519c9418bbdfd2195ff62196e89031fe1`  
**Target:** UIT DSC 2026 Task 1, Kaggle T4 ×2  
**Primary metric:** Recall@5  
**Secondary metric:** Precision@5  
**Decision:** **NOT READY FOR FULL KAGGLE RUN YET**

The F208 commit resolves most of the CD407 correctness contract. Do not redesign the stack. The remaining work is now concentrated in **real T4 safety, exact production auditing, training coverage, runtime feasibility, and a few score-critical feature gaps**.

Official local task files verified for this contract:

```text
train.json              : 7,000 queries
selected-contexts.zip   : 8,532 context JSON files
public-official.json    : 999 public queries
```

Do not hard-code the README's old “1,000 public queries” statement. The supplied official public file contains **999** query IDs.

---

# 0. Preserve the fixes already present

Keep all of these F208 improvements:

- valid run-mode whitelist;
- `gpu_smoke` rejects CPU and <2 CUDA devices;
- explicit canonical `data_dir` requires all four canonical parquet files;
- canonical validation is strict in FULL/GPU_SMOKE;
- preflight and final audit files are separated;
- FULL invalid submission raises;
- AMP `GradScaler` and finite loss/gradient checks exist;
- actual query coverage fields exist;
- adapter SHA-256 is verified;
- exact top-5 filling is enforced in production;
- learned fusion does not silently degrade to RRF when strict;
- fusion feature schema is checked;
- Dense/public embedding failures fail fast;
- root and Kaggle notebook copies remain byte-identical;
- notebook default remains `gpu_smoke`.

Do not remove or weaken these gates.

---

# 1. P0 — Final PEFT parameter audit is still performed before the adapter is actually loaded

`LegalIRPipeline.load_pipeline()` constructs:

```python
CrossEncoderReranker(...)
```

but the cross-encoder is lazy: `reranker.model` remains `None` until `_load_model()` / scoring.

Immediately after constructing the pipeline, the orchestrator runs:

```python
final_audit_report = pipeline.audit_parameters(...)
```

`audit_parameters()` therefore sees no instantiated reranker model and falls back to the BGE model-name string.

Result: the supposed final loaded-system audit can still count only:

```text
DEk21 base + BGE base
```

instead of:

```text
DEk21 base + BGE base + PEFT adapter parameters
```

The checked static audit still reports `702,754,049` with adapter parameters `0`.

## Required fix

Expose a supported public loader:

```python
CrossEncoderReranker.ensure_loaded()
```

Do not call a private method from orchestration if avoidable.

Before final production audit:

```python
if pipeline.reranker is not None:
    pipeline.reranker.ensure_loaded()

final_audit_report = pipeline.audit_parameters(...)
```

In FULL/GPU_SMOKE assert:

```text
reranker.model is not None
actual reranker device is correct
PEFT adapter is active
adapter parameter count > 0
final total > base-only 702,754,049
final total < 4,000,000,000
```

The final `parameter_audit.json`, submission manifest, terminal report, and `KaggleRunResult.audit_report` must all use this loaded-model audit.

Keep:

```text
preflight_parameter_audit.json
```

as a separate architecture preflight.

## Required behavioral test

Use a tiny PEFT model:

```text
base model -> apply LoRA -> save adapter -> reload through LegalIRPipeline
```

Then assert:

```text
adapter_parameters > 0
final total > base-only total
final report is compliant
```

A mock ranker with `model=None` is not sufficient.

---

# 2. P0 — DEk21 corpus indexing uses a T4-risky batch size of 128 at length 512 with FP32 activations

The orchestrator currently sets approximately:

```python
dense_batch = 128 if CUDA else 32
dense_retriever.fit(..., batch_size=dense_batch)
```

`DenseMacroRetriever.encode_texts()`:

- uses a ~135M parameter transformer;
- allows sequence length 512;
- runs `torch.no_grad()`;
- does not currently use CUDA AMP;
- has no adaptive OOM retry.

On a 16 GB T4, `128 × 512` transformer inference is an unnecessary high-risk first-run configuration.

## Required fix

For T4 corpus indexing use a conservative initial batch:

```text
16 or 32
```

Recommended:

```python
dense_corpus_batch_size = 32
```

Implement CUDA AMP:

```python
with torch.autocast("cuda", dtype=torch.float16, enabled=device_is_cuda):
    ...
```

Implement deterministic adaptive OOM handling:

```text
32 -> 16 -> 8 -> 4 -> 2 -> 1
```

Record:

```text
dense_initial_batch_size
dense_min_successful_batch_size
dense_oom_events
dense_peak_allocated_vram
dense_peak_reserved_vram
```

If batch 1 fails, raise.

Do not silently drop Dense in FULL/GPU_SMOKE.

Train/public query encoding can use a separately benchmarked larger batch because queries are usually shorter, but it must also have OOM retry.

## Required test

Inject a fake CUDA OOM at the first Dense batch and verify:

```text
event count increments
batch size halves
all embeddings are returned in original order
no query/chunk is lost or duplicated
```

---

# 3. P0 — FAISS is optional in code but effectively required for full-scale runtime

`DenseMacroRetriever` attempts:

```python
import faiss
```

and otherwise falls back to:

```python
np.dot(all_embeddings, query)
np.argsort(...)
```

for every query.

The notebook dependency preflight installs:

```text
lightgbm
sentencepiece
bm25s
pyvi
peft
accelerate
```

but does not guarantee `faiss`.

The historical canonical audit describes about **1.15M chunks**. A full NumPy scan + full sort for thousands of OOF/public queries is a major runtime regression.

## Required fix

Notebook/runtime preflight must verify:

```python
import faiss
```

Install:

```text
faiss-cpu
```

if missing.

After Dense index construction/loading, production modes must verify:

```python
dense_retriever._faiss_index is not None
```

unless an explicitly benchmarked alternative search backend is selected.

Prefer:

```text
FAISS IndexFlatIP
```

for exact cosine/IP behavior and scorer reproducibility.

Record backend in run manifest:

```json
"dense_search_backend": "faiss_index_flat_ip"
```

Do not silently switch to NumPy for FULL/GPU_SMOKE.

## Required test

Load a small persisted Dense index and assert FAISS and NumPy reference results have the same top-k ordering within floating tolerance.

---

# 4. P0 — `gpu_smoke` requires two GPUs but does not enforce the intended GPU mapping

The entry gate now requires `>=2` CUDA devices, but it does not require:

```text
Dense    = cuda:0
Reranker = cuda:1
```

Later device verification is conditional:

```python
if dense_device == "cuda:0":
    verify actual cuda:0
if reranker_device == "cuda:1":
    verify actual cuda:1
```

A caller can therefore request reversed devices and bypass the intended topology.

## Required fix

For production `gpu_smoke`:

```python
if dense_device != "cuda:0":
    raise RuntimeError(...)
if reranker_device != "cuda:1":
    raise RuntimeError(...)
```

Then verify actual parameter devices exactly match requested devices.

If a developer genuinely needs another topology, add an explicit debug-only flag such as:

```text
allow_nonstandard_gpu_mapping=False
```

It must default to `False` and must never qualify a run as FULL-ready.

---

# 5. P0 — A “gpu_smoke” can still use toy data and can run without the official public file

The smoke CLI creates toy data when:

```text
--tiny
```

or when a real data directory was not supplied.

The orchestrator also allows GPU_SMOKE to fall back to sampled train questions when `public-official.json` is absent.

That tests hardware, but not the **actual competition production path**.

## Required fix

Define two concepts clearly:

### `smoke`
CPU/mock/toy is allowed.

### `gpu_smoke`
Must use the **official Task 1 corpus and query files** while sampling only the expensive stages.

For `gpu_smoke`, require independently:

```text
train queries = 7,000
canonical documents = 8,532
public-official.json present
public query IDs = exactly those in the file
```

The supplied official public file has **999** queries. Do not require 1000.

Prefer a canonical dataset fingerprint/manifest generated from the source files rather than relying only on row counts:

```text
source train SHA
source contexts SHA
canonical schema version
document count
train query count
```

GPU smoke may evaluate only 20 public queries and small validation subsets, but retrieval/indexes must come from the full official corpus.

If toy GPU testing is useful, name it explicitly:

```text
gpu_smoke_toy
```

and mark:

```text
NON_PRODUCTION
NOT_A_FULL_READINESS_GATE
```

---

# 6. P1 — Actual final training coverage is measured but not guaranteed

F208 adds:

```text
actual_unique_queries_seen
actual_query_coverage_pct
```

which is useful, but the underlying sampler is still row-shuffled pair training.

Official train data contains **7,000 queries**. The current final settings are:

```text
batch_size = 2
gradient_accumulation_steps = 8
max_steps = 500
```

This yields only about:

```text
500 × 2 × 8 = 8,000 row exposures
```

The mined dataset contains one or more positives plus multiple hard negatives per query, so 8,000 random pair rows cannot reliably cover all 7,000 query IDs.

Therefore the documentation phrase:

```text
train final reranker on all 7,000 queries
```

is not proven merely because all queries exist in the input parquet.

## Required fix

Implement a **query-balanced training sampler**.

Minimum behavior:

1. group pair rows by `query_id`;
2. in the first cycle, schedule every query at least once;
3. prioritize at least one positive + one strong negative for each query;
4. only after coverage is complete, sample additional hard negatives/repeats;
5. keep deterministic seed = 42.

For FULL mode require:

```text
actual_query_coverage_pct >= 99%
```

Prefer exactly:

```text
100%
```

unless a query has no usable training pair, which must be separately reported.

Do not blindly increase `max_steps` without T4 timing evidence.

A simple target estimate with effective batch 16:

```text
~14,000 pair exposures for 2 examples/query
≈ 875 optimizer steps
```

but derive the production step count from the balanced sampler and measured T4 throughput.

Record:

```text
eligible_train_queries
actual_unique_queries_seen
coverage_pct
positive_queries_seen
queries_with_negative_seen
optimizer_steps
examples_seen
```

---

# 7. P1 — Query coverage instrumentation is incomplete for pairwise/listwise group mode

The pair collator exposes:

```text
query_ids
```

but the grouped/listwise collator does not consistently provide query IDs to the trainer.

`RerankerGroupDataset` also uses grouped items rather than `.records`.

Therefore the new coverage calculation can become zero/incorrect if loss is switched from BCE to pairwise/listwise.

## Required fix

Make both collators return:

```python
"query_ids": [...]
```

and give each dataset an explicit:

```python
unique_query_ids
```

or:

```python
num_unique_queries
```

Coverage computation must be loss-mode independent.

Add tests for:

```text
BCE
pairwise_logistic
listwise
```

with known query IDs and exact expected coverage.

---

# 8. P1 — ExactMatcher's article/clause/point features are mostly dead in production

`ExactMatcher` can index:

```text
article
clause
point
```

but only when those fields are present in the records supplied to its constructor.

Production pipeline constructs it from `documents.parquet` metadata:

```text
doc_id
title
name_raw
legal_number
year
doc_type
link
```

The statutory article/clause/point fields live in chunks, not document rows.

Result: features such as:

```text
exact_article
exact_clause
exact_point
```

are usually always false even when a query explicitly cites legal structure.

## Required fix

Build a chunk-backed statutory index.

Options:

```python
ExactMatcher(
    documents=documents,
    chunks=chunks,
)
```

or preaggregate:

```text
doc_id -> article set / clause set / point set
```

from canonical micro/macro chunks.

Avoid scanning 1M+ chunks at query time; build the map once.

Add behavioral tests:

```text
query mentions "Điều 12"
correct doc gets exact_article=1

query mentions "khoản 3 Điều 12"
correct doc gets exact_article=1 and exact_clause=1

query mentions "điểm b khoản 3 Điều 12"
correct doc receives all available statutory features
```

This should improve both retrieval and learned-fusion features.

---

# 9. P1 — Accentless slug titles weaken title matching for Vietnamese queries

Canonical metadata currently derives titles from slug-like `name_raw` values by replacing hyphens with spaces.

Official source names frequently look like:

```text
Nghi-dinh-...
Thong-tu-...
Quyet-dinh-...
```

while user queries use accented Vietnamese:

```text
Nghị định
Thông tư
Quyết định
```

NFC/lowercasing alone does not make these forms equal.

## Required fix

Preserve original metadata but add a parallel search-normalized representation:

```text
title_display
title_search
title_ascii_folded
```

Use Unicode accent folding only for retrieval/matching features, never to overwrite the original title.

A robust match can compare both:

```text
accent-preserving normalized tokens
accent-folded normalized tokens
```

Use the existing `prettify_doc_title()` where appropriate, but do not rely on it to reconstruct arbitrary Vietnamese accents that are not encoded in the slug.

Apply this parallel representation to:

```text
ExactMatcher title overlap
raw BM25 title field
optional learned fusion title-match feature
```

Required regression:

```text
"Nghị định 31/2021/NĐ-CP"
```

must match/index the equivalent ASCII-slug metadata.

---

# 10. P1 — OOM reporting only covers the final public reranker instance

`gpu_smoke_report.json` currently reads:

```python
pipeline.reranker.oom_events
```

This is the final loaded reranker used for public inference.

OOF uses fold-specific reranker instances that are deleted after each fold. Recovered OOMs there are lost.

## Required fix

Each OOF fold must report:

```text
reranker_oom_events
initial_batch_size
min_successful_batch_size
peak_vram
```

Aggregate:

```text
oof_reranker_oom_events
final_reranker_oom_events
total_reranker_oom_events
```

into `gpu_smoke_report.json`.

A run must not say:

```text
oom=false
```

if any fold recovered from an OOM.

For the production gate, prefer zero OOM events. If a fallback was required, store the stable configuration and rerun smoke with that configuration from the start.

---

# 11. P1 — Reranker batch-size telemetry is not yet trustworthy

`CrossEncoderReranker` initializes:

```text
initial_batch_size = 16
min_successful_batch_size = 16
```

but does not reliably update `initial_batch_size` to the actual requested batch on first inference.

With no OOM, a call using batch 4 can still report stable batch 16.

## Required fix

At the start of a top-level score operation:

```python
self.initial_batch_size = requested_batch_size
```

Track successful batches separately:

```text
last_successful_batch_size
minimum_successful_batch_size
```

Recursive OOM retries must not reset the initial requested size.

Add an injected-OOM test:

```text
requested 16
first call OOM
retry 8 succeeds
report initial=16, min_successful=8, oom_events=1
```

---

# 12. P1 — Full runtime remains a serious risk; measure it before FULL

Historical canonical audit records approximately:

```text
8,532 documents
~1.15M chunks
7,000 train queries
```

Full OOF uses all training queries and:

```text
candidate_k = 150
rerank_k = 50
```

The evidence builder can create a complete pack plus additional chunk-level pairs per document.

Therefore BGE OOF inference can approach hundreds of thousands to around a million cross-encoder sequences, before:

```text
5 fold trainings
doc-disjoint training/evaluation
final training
public inference
```

GPU0 is primarily used for Dense work; much of the later BGE workload remains on GPU1.

## Required fix

GPU smoke must calculate a runtime projection:

```text
dense index time
query embedding throughput
BGE pairs/sec
OOF queries/sec
final public queries/sec
training steps/sec
projected 5-fold OOF time
projected final training time
projected total full runtime
```

Write:

```text
runtime_projection.json
```

Define a safety margin against the Kaggle session limit.

If projected full runtime is unsafe, optimize before FULL.

Preferred options in order:

1. after Dense corpus/query embeddings are finished, offload/free DEk21 model from GPU0;
2. keep Dense embeddings/FAISS index CPU-resident;
3. optionally replicate the same final/fold BGE model across GPU0 + GPU1 for **inference only** and split rerank batches/queries;
4. use a leakage-safe OOF query subset for fusion development if full 7k OOF is infeasible;
5. evaluate rerank budget 40/50 based on Recall@k evidence.

Do not add a second learned architecture; duplicating the same BGE inference model across GPUs does not change model-family compliance, but parameter-accounting policy should still be documented.

---

# 13. P1 — Several F208 tests still do not exercise the behavior their names imply

Replace/strengthen these tests.

Current weak patterns include:

- OOM test only checks initial `oom_events == 0`;
- final PEFT audit test uses mock smoke and only checks `<4B`;
- FULL invalid-submission test directly calls validator rather than the orchestrator failure path;
- tiny GPU-smoke device test recreates the CLI ternary expression instead of invoking the CLI/main path.

## Required real behavioral tests

Add:

```text
test_final_audit_force_loads_peft_and_counts_adapter
test_dense_cuda_oom_halves_batch_and_preserves_order
test_gpu_smoke_requires_exact_cuda0_cuda1_mapping
test_gpu_smoke_requires_official_dataset_identity
test_gpu_smoke_requires_public_official_file
test_production_dense_search_requires_faiss
test_reranker_oom_telemetry_tracks_recursive_retry
test_oof_aggregates_reranker_oom_events
test_query_balanced_sampler_reaches_100pct_coverage
test_group_loss_coverage_tracking
test_exact_matcher_uses_chunk_article_clause_point_index
test_ascii_slug_title_matches_accented_query
test_full_orchestrator_raises_on_invalid_submission
```

Mocks are acceptable at transformer compute boundaries, but the orchestration path under test must really execute.

---

# 14. P2 — Score optimization after the runtime/correctness gate

Historical accepted benchmark:

```text
Recall@5             ≈ 75.36%
Candidate Recall@50  ≈ 94.42%
Candidate Recall@150 ≈ 97.35%
```

The largest headroom remains ranking.

Only after P0/P1 and real GPU smoke, perform **sequential leakage-safe ablation**, not a combinatorial grid.

## 14.1 Candidate/RRF branch weights

Current HybridSearch defaults are fixed:

```text
raw BM25 = 1.0
PyVi     = 1.0
Dense    = 1.2
Memory   = 2.0
Exact    = 2.5
```

Tune branch weights cross-fitted to maximize:

```text
Candidate Recall@50 first
Candidate Recall@150 as guardrail
```

Never tune on the same fold being evaluated.

## 14.2 Final fusion RRF weights

The fusion function accepts `rrf_weights`, but the production call currently uses defaults.

Implement small fold-isolated weight search for the final RRF comparator.

Primary selection:

```text
official Recall@5
```

Tie-break:

```text
Precision@5
```

## 14.3 Rerank budget

Sequentially compare:

```text
rerank_k = 40
rerank_k = 50
rerank_k = 80
```

Do not choose larger k without Recall gain sufficient to justify runtime.

## 14.4 Loss

After query-balanced training is correct, compare:

```text
BCE
pairwise_logistic
```

Do not promote listwise/group training until its coverage accounting and memory profile are verified.

## 14.5 Candidate cutoff

Only if candidate misses justify it:

```text
candidate_k = 150 vs 200
```

Candidate@150 is historically already ~97.35%, so ranking quality is more valuable than indiscriminately increasing candidate volume.

---

# 15. Fix learned-fusion model-type reporting

`train_and_evaluate_fusion_cv()` currently labels the learned winner as:

```text
winning_model_type = "lightgbm"
```

even though `LightGBMRanker.fit()` is allowed to fall back to `LinearRanker`.

## Required fix

After training use the actual model state:

```python
actual_model_type = full_ranker.model_type
```

Manifest must report exactly one of:

```text
lightgbm
linear_fallback
rrf_weighted
```

If LightGBM failure occurs in FULL/GPU_SMOKE, preferred behavior is to raise rather than silently promote a weaker fallback unless the fallback itself has independently won the cross-fitted Recall@5 comparison.

---

# 16. Required verification order

## Local/source gate

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('IMPORT_OK')"
pytest -q
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

Verify notebook parity:

```text
SHA(root legalir_training.ipynb)
==
SHA(kaggle_kernel_task1/legalir_training.ipynb)
```

Do not claim tests passed from commit messages. Use fresh command output.

## Real Kaggle T4 ×2 gate

Run on official data:

```text
LEGALIR_RUN_MODE=gpu_smoke
```

Required evidence:

```text
2 x NVIDIA T4
official docs = 8,532
official train queries = 7,000
official public file present = 999 queries
Dense requested/actual = cuda:0
BGE requested/actual = cuda:1
Dense search backend = FAISS
Dense OOM events accurately reported
Reranker OOM events aggregated across OOF + final
AMP training finite
optimizer steps > 0
param_diff > 0
adapter SHA verified
loaded PEFT adapter_parameters > 0
final total params <4B
2 cross-fit smoke folds
query coverage recorded
fusion model round-trip valid
peak VRAM per GPU recorded
runtime projection recorded
```

Only after all of this passes:

```text
LEGALIR_RUN_MODE=full
```

---

# 17. FULL readiness conditions

Report:

```text
READY FOR KAGGLE GPU SMOKE: YES
```

only after fresh local compile/tests/mock smoke pass.

Report:

```text
READY FOR FULL KAGGLE RUN: YES
```

only after a **real official-data T4×2 GPU smoke** proves:

- correct hardware mapping;
- no unresolved OOM;
- production Dense backend;
- loaded PEFT parameter audit;
- adapter integrity;
- finite training;
- acceptable query coverage mechanism;
- fusion round-trip;
- projected total runtime fits Kaggle budget.

Without real T4×2 evidence, maximum allowed conclusion is:

```text
Source-level repair complete; ready for official Kaggle T4x2 gpu_smoke.
```

---

# 18. Final report format

```markdown
# LegalIR F208 Final T4 Score Gate Report

## Base
- audited head: f2080e4881fbb54e41dff33836de014252719076
- new head: <sha>

## Source gates
| Gate | PASS/FAIL | Evidence |
|---|---|---|
| PEFT force-load before final audit | | |
| adapter params counted | | |
| T4-safe Dense AMP/adaptive batch | | |
| FAISS production backend | | |
| exact cuda:0/cuda:1 mapping | | |
| official-data gpu_smoke identity | | |
| public-official required for gpu_smoke | | |
| query-balanced final training | | |
| group-mode coverage | | |
| chunk-backed statutory exact features | | |
| accent-folded title matching | | |
| aggregated OOM telemetry | | |
| full submission hard gate | | |

## Fresh local verification
- compileall:
- pytest:
- CPU/mock smoke:
- notebook SHA parity:
- preflight parameter total:
- final loaded PEFT parameter total:
- adapter parameters:

## Official T4x2 GPU smoke
- executed: YES/NO
- corpus docs:
- train queries:
- public queries:
- GPU0:
- GPU1:
- Dense requested/actual:
- Reranker requested/actual:
- Dense backend:
- Dense initial/stable batch:
- Dense OOM events:
- Reranker initial/stable batch:
- OOF reranker OOM events:
- final reranker OOM events:
- GPU0 peak VRAM:
- GPU1 peak VRAM:
- optimizer steps:
- param_diff:
- query coverage:
- adapter SHA verified:
- final PEFT params:
- fusion round-trip:
- projected full runtime:
- result:

## OOF / score evidence
- candidate Recall@50:
- candidate Recall@150:
- reranker OOF Recall@5:
- fusion winner:
- actual fusion model type:
- fusion Recall@5:
- Precision@5:
- doc-disjoint Recall@5:

## Readiness
READY FOR KAGGLE GPU SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
