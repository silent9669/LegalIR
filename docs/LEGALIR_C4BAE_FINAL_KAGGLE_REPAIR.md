# LegalIR Task 1 — Remaining Pre-Kaggle Repair Contract

**Repository:** `silent9669/LegalIR`  
**Audited HEAD:** `c4bae606bff891bd3c995f9f9359aa700031632d`  
**Previous audited HEAD:** `ca67b671d40a8e211af7869b275026007b0f7a5e`  
**Decision:** **NOT READY FOR FULL KAGGLE RUN**

## 0. Preserve what is already fixed

Do not redesign the pipeline again. Keep the improvements already present at `c4bae606`:

- tuple/list QuestionMemory parsing;
- explicit `cuda:N` resolver;
- separate dense/reranker device API;
- post-reranker OOF feature extraction;
- fold-local document-frequency priors;
- dedicated document-disjoint reranker path;
- official public-file requirement in full mode;
- `year` / `doc_type` metadata in final loader;
- strict artifact mode;
- byte-identical notebooks;
- <4B parameter stack.

This file contains only remaining blockers/gaps found in the source audit.

---

# 1. P0 — `src/training/build_pairs.py` currently cannot import

The file declares:

```python
query_embeddings: Mapping[str, Any] | None = None
```

but imports only:

```python
from typing import Any
```

and has no `from __future__ import annotations`.

On normal Python 3.10/3.11 this evaluates `Mapping` when defining the function and raises:

```text
NameError: name 'Mapping' is not defined
```

Because `src.pipeline.kaggle_train` imports `build_training_pairs` at module import time, this can prevent the Kaggle orchestrator and the new test module from importing at all.

## Fix

Use either:

```python
from collections.abc import Mapping
```

or:

```python
from typing import Any, Mapping
```

Prefer `collections.abc.Mapping`.

## Required verification

Before anything else:

```bash
python -m py_compile src/training/build_pairs.py
python -c "from src.training.build_pairs import build_training_pairs; print('OK')"
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('OK')"
```

Add a repository compile/import gate:

```bash
python -m compileall -q src scripts
```

This must run in CI/tests before expensive tests.

---

# 2. P0 — Kaggle notebook resolves reranker config relative to the wrong CWD

`run_kaggle_pipeline()` defaults to:

```python
runtime_config_path="configs/kaggle.yaml"
reranker_config_path="configs/experiments/reranker_lora.yaml"
```

but converts them directly with `Path(...)`.

The notebook discovers/clones:

```text
REPO_ROOT=/kaggle/working/LegalIR
```

and adds it to `sys.path`, but it does **not** `chdir(REPO_ROOT)`.

Therefore, if the notebook CWD is `/kaggle/working`, these paths are tested as:

```text
/kaggle/working/configs/kaggle.yaml
/kaggle/working/configs/experiments/reranker_lora.yaml
```

They do not exist.

The current code then silently falls back to:

```text
REPO_ROOT/configs/pipeline.yaml
```

This is dangerous because `pipeline.yaml` is not the LoRA training config. It has nested ranking settings and batch size 16 under `ranking.reranker`, but `train_reranker()` expects top-level training fields.

Consequences in full OOF:

```text
batch_size -> trainer default 16
grad accumulation -> default 1
max_steps -> None
epochs -> 2
```

For BGE reranker v2-m3 at length 512 on a T4, this is likely to be extremely slow and/or OOM.

## Required fix

Resolve all configured paths relative to `repo_root`:

```python
def resolve_repo_path(value, repo_root):
    p = Path(value)
    if not p.is_absolute():
        p = Path(repo_root) / p
    return p.resolve()
```

Then:

```python
resolved_runtime_config = resolve_repo_path(
    runtime_config_path, root_path
)
resolved_reranker_config = resolve_repo_path(
    reranker_config_path, root_path
)
```

In FULL or GPU_SMOKE mode, do **not** silently replace a missing reranker experiment config with `pipeline.yaml`.

Use:

```python
if not resolved_reranker_config.is_file():
    raise FileNotFoundError(...)
```

The effective full reranker config must prove:

```text
base_model=BAAI/bge-reranker-v2-m3
batch_size=2
gradient_accumulation_steps=8
max_length=512
max_steps=500
fp16=true
gradient_checkpointing=true
learning_rate=2e-5
device=cuda:1
```

## Required test

Run from a CWD outside the repo:

```python
monkeypatch.chdir(tmp_path)
run_kaggle_pipeline(repo_root=REPO_ROOT, ...)
```

and assert the experiment config resolves to:

```text
REPO_ROOT/configs/experiments/reranker_lora.yaml
```

Do not test only whether the YAML exists when CWD == repo root.

---

# 3. P0 — `gpu_smoke` uses one OOF fold but cross-fitted fusion requires >=2 folds

Current orchestrator sets:

```python
num_folds=1 if is_gpu_smoke else ...
```

Then it always calls:

```python
train_and_evaluate_fusion_cv(oof_df, ...)
```

Cross-fitting uses for each validation fold:

```python
train_data = oof_df[oof_df["fold"] != f_idx]
```

With exactly one fold, `train_data` is empty.

LightGBM training can fail and the linear fallback will also receive zero samples.

Therefore the newly added `gpu_smoke` path is structurally unable to validate cross-fitted fusion correctly.

## Required fix

Preferred:

```text
gpu_smoke = 2 tiny folds
```

Use ~10–20 validation queries per fold and 2–5 LoRA optimizer steps so it remains cheap.

Alternative only if necessary:

- explicitly skip learned fusion in one-fold smoke;
- mark fusion as `NOT_TESTED`;
- never call that run sufficient for FULL readiness.

For the intended readiness gate, use **2 folds**.

## Required test

Create a tiny 2-fold OOF table and execute the actual fusion function.

Also add:

```python
if len(unique_folds) < 2:
    raise ValueError("Cross-fitted fusion requires at least 2 folds")
```

so it fails clearly instead of entering empty training.

---

# 4. P0/P1 — Learned-fusion public inference still has feature mismatch

OOF fusion training includes these core features:

```text
query_length
train_doc_freq
```

OOF correctly supplies:

```python
query_text=q_text
doc_freq_map=fold_train_doc_freq
```

But final `LegalIRPipeline.predict_one()` currently calls:

```python
self.ranker.predict(candidates)
```

without:

```text
query_id
query_text
doc_freq_map
```

`LightGBMRanker.predict()` therefore reconstructs public features with:

```text
query_length = 0
train_doc_freq = 0
```

unless those fields happen to already exist in candidate records.

They normally do not.

This means the selected learned fusion is trained on one feature distribution and receives another in the official public run.

## Required fix

Load/compute full allowed training-document frequency in `LegalIRPipeline.load_pipeline()` from `qrels_train.parquet`.

Store it on the pipeline, e.g.:

```python
self.doc_freq_map
```

Then final prediction must call:

```python
ranked = self.ranker.predict(
    candidates,
    query_id=query_id,
    query_text=question,
    doc_freq_map=self.doc_freq_map,
)
```

For RRF, existing behavior can remain.

## Required test

Train/use a deterministic fake learned ranker and inspect the actual inference feature frame.

Assert:

```text
query_length == len(real query text)
train_doc_freq != forced zero for known training-positive docs
```

The current schema-only test is insufficient.

---

# 5. P1 — Public query embeddings are still encoded twice

The repair contract required one DEk21 query embedding to be shared between:

```text
Dense retrieval
Question Memory
```

but final pipeline methods do not accept `q_emb`.

`predict_one()` calls HybridSearch without `q_emb`; Dense encodes the query, then Question Memory can encode the same query again.

The orchestrator also does not batch-precompute public embeddings.

## Required fix

Extend:

```python
predict_one(..., q_emb=None)
predict_single(..., q_emb=None)
predict_batch(..., query_embeddings=None)
```

and pass:

```python
q_emb=q_emb
```

to HybridSearch.

In public inference:

```python
public_embeddings = dense.encode_queries(all_public_texts, ...)
```

once on GPU0, then reuse each vector for Dense + Memory.

Also allow `OOFRunner` to receive the already-created `train_query_embs` mapping from the orchestrator instead of recomputing all 7,000 embeddings again.

## Required behavioral test

Use a counting fake encoder and run the actual pipeline for one query.

Assert the query is encoded once, not once per branch.

The current test that calls only `TrainQuestionMemory.search(q_emb=...)` does not verify pipeline reuse.

---

# 6. P1 — GPU smoke still does not prove actual GPU placement

The code now preserves `cuda:0` / `cuda:1`, but current tests mostly verify strings/mocks.

The required real Kaggle gate must prove:

```text
DEk21 actual parameter device = cuda:0
BGE/PEFT actual parameter device = cuda:1
```

After each lazy model load, inspect:

```python
next(model.parameters()).device
```

Record:

```text
requested device
actual device
peak allocated VRAM
peak reserved VRAM
```

for GPU0/GPU1.

In `gpu_smoke`, set:

```python
strict_artifacts=True
```

or equivalent production-strict validation.

A GPU smoke that silently drops Dense/Memory/fusion is not a valid readiness proof.

---

# 7. P1 — Strengthen strict production artifact validation

Current strict checks are improved, but still incomplete.

## Final adapter

`training_manifest.json` must be **required**, not optional.

Require:

```text
status == completed
param_diff > 0
adapter_checksum non-empty
unique_training_queries > 0
optimizer_steps > 0
```

## Dense

Require:

```text
embeddings.npy exists
chunks_meta.parquet exists
len(embeddings) > 0
len(doc_ids) > 0
```

not only an existing file.

## Fusion

If learned fusion was selected, require:

```text
manifest.json exists
winning_method == learned_ranker
feature_columns match model
actual model file exists
model loads successfully
```

Add:

```text
feature_training_stage = post_rerank
```

to the fusion manifest and validate it on load.

---

# 8. P1 — Fix final parameter audit overwrite/coverage

`LegalIRPipeline.audit_parameters()` looks for:

```python
self.hybrid_engine.dense_retriever
```

but `HybridSearchEngine` stores Dense on:

```python
self.dense
```

As a result, the final pipeline audit can omit DEk21.

Because `load_pipeline(... audit_output_json=...)` writes the audit file again, this may overwrite the correct earlier 702,754,049 parameter report with an incomplete report.

## Required fix

Audit both supported attributes, preferably canonicalize HybridSearch to expose:

```python
@property
def dense_retriever(self):
    return self.dense
```

Then verify final audit includes:

```text
DEk21: 134,998,272
BGE reranker: 567,755,777
total baseline: 702,754,049 + adapter overhead as counted by policy
```

Never allow a later audit to reduce the report by accidentally omitting a model.

---

# 9. P1 — Replace weak invariant tests with real path tests

Several new tests do not prove the behavior named in their titles.

Examples:

### Current device test

It passes:

```text
dense_device="cpu"
reranker_device="cpu"
```

for both, so it does not prove distinct routing.

### Current OOF reranker-score test

It manually constructs candidate records already containing:

```python
reranker_score=0.9542
```

then tests `extract_candidate_features()`.

It does not prove OOFRunner reranks before feature extraction.

### Current doc-disjoint trained-reranker test

It runs with:

```python
train_reranker_per_fold=False
use_reranker=False
```

and therefore does not train a dedicated reranker.

Another test writes a dummy report JSON instead of exercising the evaluator.

### Current max_steps test

It expects any exception from an invalid full run, so it does not prove config routing or `max_steps=500`.

### Current public q_emb test

It tests QuestionMemory directly, not `LegalIRPipeline`.

## Required replacements

Add real-path tests with tiny/fake components:

```text
test_kaggle_config_resolution_from_non_repo_cwd
test_build_pairs_module_imports
test_oof_runner_calls_reranker_before_feature_extraction
test_doc_disjoint_actually_trains_and_loads_adapter
test_pipeline_passes_query_text_and_doc_freq_to_learned_ranker
test_pipeline_shares_one_q_emb_between_dense_and_memory
test_gpu_smoke_requires_two_folds_for_cross_fit
test_final_parameter_audit_contains_dense_and_reranker
```

Use mocks only at expensive model boundaries, not to bypass the orchestration being tested.

---

# 10. P2 — Submission manifest copy order

The orchestrator copies files to:

```text
/kaggle/working/
```

before `submission_manifest.json` is created.

Therefore the root-level manifest may be missing or, worse, a stale file from an older run may be copied.

Create the manifest first, then copy:

```text
submission.json
submission.zip
submission_manifest.json
```

after all validation succeeds.

---

# 11. Reporting correctness

`cv_report["mean_recall@5"]` is the direct reranker-order OOF score.

The final system may instead use the winner from:

```text
cross-fitted RRF vs LightGBM
```

Do not label raw `cv_report` Recall@5 as the final system OOF score.

Report separately:

```text
Reranker OOF Recall@5
Fusion winner
Fusion winner full cross-fitted OOF Recall@5
Fusion winner Precision@5
Candidate Recall@150
Document-disjoint trained-system Recall@5
```

For model promotion and expected leaderboard quality, use the cross-fitted fusion winner metric.

---

# 12. Mandatory verification order

After repair:

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline"
pytest -q
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

Then on real Kaggle T4×2:

```bash
python scripts/smoke_kaggle_pipeline.py \
  --run-mode gpu_smoke \
  --data-dir <real-canonical-data>
```

GPU smoke must use:

```text
2 tiny OOF folds
real DEk21
real BGE
real PEFT LoRA
cuda:0 Dense
cuda:1 reranker
real adapter save/reload
strict artifacts
post-rerank OOF features
cross-fitted fusion
```

Only after the real T4×2 GPU smoke succeeds should full training run.

---

# 13. Readiness criteria

The coding agent may report:

```text
READY FOR KAGGLE GPU SMOKE: YES
```

only after compile/import/tests/mock-smoke pass.

It may report:

```text
READY FOR FULL KAGGLE RUN: YES
```

only after a **real Kaggle T4×2 GPU smoke** proves:

```text
actual GPU placement
real model loading
real LoRA parameter changes
real adapter checksum
no OOM
non-empty memory
post-rerank OOF feature generation
2-fold fusion smoke
strict artifact loading
```

No real GPU evidence -> maximum conclusion:

```text
Source-level repair complete; ready for Kaggle GPU smoke.
```

---

# 14. Final repair report

Return:

```markdown
# LegalIR C4BAE Final Repair Report

## Base
- audited: c4bae606bff891bd3c995f9f9359aa700031632d
- new commit: <sha>

## Critical fixes
| Fix | PASS/FAIL | Verification |
|---|---|---|
| build_pairs Mapping/import | | |
| repo-root config resolution | | |
| T4-safe effective reranker config | | |
| gpu_smoke 2-fold fusion | | |
| learned-fusion inference parity | | |
| public q_emb reuse | | |
| final parameter audit completeness | | |
| strict artifacts | | |

## Verification
- compileall:
- pytest:
- mock smoke:
- notebook SHA parity:
- parameter total:

## Real T4x2 GPU smoke
- executed: YES/NO
- Dense requested/actual:
- Reranker requested/actual:
- LoRA param_diff:
- adapter checksum:
- GPU0 peak VRAM:
- GPU1 peak VRAM:
- OOM: YES/NO
- fusion smoke: PASS/FAIL/NOT_RUN

## Readiness
READY FOR KAGGLE GPU SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
