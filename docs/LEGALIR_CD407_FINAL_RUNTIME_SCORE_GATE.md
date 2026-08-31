# LegalIR Task 1 — CD407 Final Runtime + Score Gate

**Repository:** `silent9669/LegalIR`  
**Audited HEAD:** `cd40702519c9418bbdfd2195ff62196e89031fe1`  
**Previous audited HEAD:** `57e2d121f7622053e50a8d8eba17e5fed477106a`  
**Target hardware:** Kaggle T4 ×2  
**Decision:** **DO NOT START THE FULL RUN YET**

The latest commit fixes most of the previous 57E2 gate contract. Preserve those fixes. This document contains only remaining issues found in the fresh source audit.

---

## 1. What is already fixed — preserve it

Do not redesign these components:

- `/kaggle/input/legalir` discovery was added.
- `gpu_smoke` defaults to strict artifacts.
- Dense/public query embedding failures fail fast in full/gpu_smoke.
- submission validation now checks official corpus IDs.
- learned-fusion save/load is safer and strict inference refuses an unloaded model.
- adapter SHA-256 is recomputed and verified.
- production prediction fills to exactly 5 valid unique IDs.
- document-disjoint report is retained.
- notebook preflight checks LightGBM + SentencePiece.
- notebook default is `gpu_smoke`.
- root and Kaggle notebooks remain synchronized.
- checked baseline learned parameter count is still safely below 4B.

---

# 2. P0 — `gpu_smoke` can still run without T4 ×2

Current device mismatch checks only run when:

```python
torch.cuda.is_available() and torch.cuda.device_count() >= 2
```

If `gpu_smoke` is accidentally launched on CPU or one GPU, it does **not** immediately reject the environment.

That defeats the purpose of the real T4×2 readiness gate.

## Required fix

At the beginning of `run_kaggle_pipeline()`:

```python
if is_gpu_smoke:
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_smoke requires CUDA")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("gpu_smoke requires Kaggle T4 x2 / >=2 CUDA devices")
```

Require:

```text
dense_device == cuda:0
reranker_device == cuda:1
```

unless an explicit debug-only override is passed.

FULL mode should also warn/fail when the requested T4×2 production profile is not available.

## Important CLI bug

`scripts/smoke_kaggle_pipeline.py` currently does:

```python
devices=["cpu", "cpu"] if args.tiny else None
```

So:

```bash
--tiny --run-mode gpu_smoke
```

can become a CPU run while still being called `gpu_smoke`.

Fix the CLI:

```text
--tiny + smoke       -> CPU/mock allowed
--run-mode gpu_smoke -> NEVER force CPU
```

Add:

```text
test_gpu_smoke_rejects_cpu
test_gpu_smoke_rejects_single_gpu
test_gpu_smoke_requires_cuda0_cuda1
test_tiny_gpu_smoke_does_not_force_cpu
```

---

# 3. P0 — Canonical validation is still FULL-only

The orchestrator currently raises on failed canonical validation only for FULL mode.

`gpu_smoke` is supposed to prove the exact production data path. It must use a valid complete canonical dataset too.

## Required fix

Change to:

```python
if not val_report["is_valid"] and (is_full or is_gpu_smoke):
    raise ValueError(...)
```

Also fix explicit `data_dir` behavior. Today an existing but incomplete directory can be returned by `discover_data_dir()`.

For an explicitly supplied production path, require:

```text
documents.parquet
chunks.parquet
queries_train.parquet
qrels_train.parquet
```

or raise immediately.

Do not silently return an incomplete directory.

---

# 4. P0 — Final parameter report uses the preflight count, not the loaded PEFT system count

The orchestrator first creates:

```python
audit_report = audit_system_parameters(...)
```

from config/model names.

Later `LegalIRPipeline.load_pipeline(... audit_output_json=...)` performs an audit after the real final reranker adapter is loaded.

However the submission manifest, terminal summary, and returned `KaggleRunResult.audit_report` still use the earlier `audit_report`.

This can report:

```text
702,754,049
```

while the actual loaded PEFT model contains base + adapter parameters.

The system is still far below 4B, but the compliance artifact must be exact.

## Required fix

After final pipeline load:

```python
final_audit_report = pipeline.audit_parameters(
    output_json=working_path / "parameter_audit.json",
    raise_on_violation=True,
)
```

Use `final_audit_report` for:

```text
submission_manifest.parameter_total
terminal summary
KaggleRunResult.audit_report
ablation report
readiness report
```

Retain the first audit separately as:

```text
preflight_parameter_audit.json
```

Add a test proving final PEFT audit total is >= base DEk21 + base BGE and contains adapter parameters when loaded.

---

# 5. P0 — Unknown run modes are silently accepted

Current logic derives:

```python
is_smoke
is_gpu_smoke
is_full
```

but does not reject any other string.

A typo such as:

```text
LEGALIR_RUN_MODE=ful
```

can enter a hybrid execution path with wrong strictness and only 5 final training steps.

## Required fix

Immediately enforce:

```python
VALID_RUN_MODES = {"smoke", "gpu_smoke", "full"}
if run_mode_str not in VALID_RUN_MODES:
    raise ValueError(...)
```

Add a regression test.

---

# 6. P0/P1 — GPU smoke report can claim `oom=false` without measuring OOM events

`gpu_smoke_report.json` currently writes:

```json
"oom": false
```

unconditionally.

The reranker has automatic OOM batch-size fallback, so a recoverable OOM can occur and the final report still says no OOM.

## Required fix

Instrument the reranker/inference path.

Record:

```text
oom_events
initial_batch_size
minimum_successful_batch_size
dense_peak_vram
reranker_peak_vram
```

Set:

```python
oom = oom_events > 0
```

For the final readiness gate, either:
- require zero OOM events; or
- explicitly report recovered OOM and the stable batch size.

Never fabricate `oom=false`.

---

# 7. P1 — FP16 training has no GradScaler

The training loop uses:

```python
torch.autocast(... dtype=torch.float16)
loss.backward()
optimizer.step()
```

but no gradient scaler.

On a T4, FP16 without `GradScaler` can underflow gradients or produce non-finite updates, especially for a large cross-encoder.

## Required fix

Use modern AMP:

```python
scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)
```

with:

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

Track:

```text
nonfinite_loss_count
grad_norm
scaler_scale
```

Fail FULL/GPU_SMOKE if loss or trainable parameters become NaN/Inf.

CPU/mock behavior must remain unchanged.

---

# 8. P1 — “trained on all 7,000 queries” is not currently proven

The final pair miner receives all training queries, but the final trainer is bounded to:

```text
batch_size = 2
gradient_accumulation = 8
max_steps = 500
```

That is about 8,000 row examples at most before partial-batch effects.

The pair file contains positives plus multiple negatives, so `unique_training_queries` in the manifest currently means:

```text
queries present in the input DataFrame
```

not:

```text
queries actually observed by optimizer steps
```

This distinction matters for the claim that the final adapter trained on all 7,000 queries.

## Required fix

Instrument actual training coverage:

```text
actual_seen_query_ids
actual_unique_queries_seen
actual_query_coverage_pct
actual_examples_seen
```

Do not infer coverage from the input DataFrame.

For final training, target near-100% query coverage without exploding T4 runtime.

Preferred approach:
- query-balanced sampling;
- ensure every training query appears before repeated oversampling;
- then spend remaining steps on hard negatives.

If 500 steps cannot guarantee adequate coverage, determine a bounded step count from the query-balanced sampler and GPU-smoke throughput.

Do not blindly increase to thousands of steps before measuring T4 runtime.

---

# 9. P1 — Strengthen learned-fusion schema validation

Strict fusion validation checks the manifest and loads the model, but also require exact agreement between:

```text
manifest.feature_columns
loaded ranker.feature_cols
```

A stale model + newer manifest must fail.

For both LightGBM and Linear fallback:

```python
if list(loaded_feature_cols) != list(manifest_feature_cols):
    raise ValueError(...)
```

Also record the actual selected model type:

```text
lightgbm
linear_ridge
rrf
```

Do not report `lightgbm` if training fell back to linear.

---

# 10. P1 — Current new tests still contain weak gates

Replace tests that only duplicate implementation logic.

Current `test_gpu_smoke_defaults_to_strict_artifacts()` manually recreates:

```python
strict_artifacts = is_full or is_gpu_smoke
```

instead of exercising the orchestrator.

Current document-disjoint persistence test uses:

```text
use_reranker=False
```

so it does not prove the trained document-disjoint reranker path.

## Required behavioral tests

Add:

```text
test_gpu_smoke_orchestrator_passes_strict_artifacts_true
test_gpu_smoke_rejects_cpu_and_one_gpu
test_document_disjoint_trains_and_loads_dedicated_adapter
test_final_audit_uses_loaded_peft_model
test_invalid_run_mode_raises
test_gpu_report_does_not_hardcode_oom_false
test_final_training_reports_actual_query_coverage
test_fusion_manifest_feature_schema_matches_loaded_model
```

Also test an actual tiny LightGBM train/save/reload round trip, not only LinearRanker serialization.

---

# 11. P1 — Full mode should fail hard on invalid final submission

The full pipeline currently computes:

```python
is_submission_valid = ...
```

and can return a result with `is_valid=False`.

For a production competition run, invalid submission artifacts must terminate the run.

## Required fix

After JSON + ZIP validation:

```python
if is_full and not is_submission_valid:
    raise RuntimeError(
        f"Final official submission failed validation: ..."
    )
```

Do not copy or mark anything submittable before this passes.

---

# 12. P2 — Score maximization after correctness gate

Only do this after all P0/P1 gates pass and a real T4×2 GPU smoke gives runtime headroom.

The historical system has much higher candidate recall than final Recall@5, so ranking remains the main opportunity.

Run leakage-safe OOF ablations for:

```text
rerank_k: 50 vs 80
candidate_k: 150 vs 200
loss: BCE vs pairwise logistic
RRF weights: default vs cross-fitted tuned
```

Rules:
- tune on folds != f;
- evaluate only on held-out fold f;
- primary selection = official Recall@5;
- Precision@5 only tie-break;
- preserve Task1-only data;
- do not promote any configuration without scorer-equivalent OOF evidence.

Do not run a combinatorial grid. Use sequential ablation and stop options that do not improve Recall.

---

# 13. Mandatory verification sequence

Local/source gate:

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('IMPORT_OK')"
pytest -q
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

Notebook gate:

```text
root notebook SHA == kaggle_kernel_task1 notebook SHA
default RUN_MODE == gpu_smoke
```

Real Kaggle gate:

```text
2 x NVIDIA T4 detected
gpu_smoke refuses CPU/1GPU
Dense actual == cuda:0
BGE/PEFT actual == cuda:1
strict_artifacts == true
2 OOF folds
real LoRA optimizer steps > 0
param_diff > 0
adapter checksum verified
actual query coverage recorded
final parameter audit uses loaded adapter
fusion save/reload passes
peak VRAM recorded
OOM events accurately recorded
```

Only after this passes:

```bash
LEGALIR_RUN_MODE=full
```

---

# 14. Readiness policy

Without real T4×2 smoke evidence, maximum allowed conclusion:

```text
Source-level repair complete; ready for Kaggle GPU smoke.
```

Only report:

```text
READY FOR FULL KAGGLE RUN: YES
```

after a real T4×2 smoke run succeeds.

---

# 15. Final report format

```markdown
# LegalIR CD407 Final Runtime Gate Report

## Base
- audited head: cd40702519c9418bbdfd2195ff62196e89031fe1
- new head: <sha>

## Source gates
| Gate | PASS/FAIL | Evidence |
|---|---|---|
| valid run-mode enforcement | | |
| gpu_smoke requires 2 CUDA GPUs | | |
| canonical gpu_smoke validation | | |
| final loaded-model parameter audit | | |
| accurate OOM reporting | | |
| AMP GradScaler + finite training | | |
| actual query coverage reporting | | |
| fusion schema round-trip | | |
| full submission hard fail | | |

## Verification
- compileall:
- pytest:
- CPU smoke:
- notebook SHA parity:
- preflight params:
- final loaded params:

## T4x2 GPU smoke
- executed:
- GPU names:
- Dense requested/actual:
- Reranker requested/actual:
- GPU0 peak VRAM:
- GPU1 peak VRAM:
- OOM events:
- stable reranker batch:
- optimizer steps:
- param_diff:
- actual unique train queries seen:
- query coverage:
- adapter SHA verified:
- fusion round-trip:
- result:

## OOF / score evidence
- candidate Recall@150:
- reranker OOF Recall@5:
- fusion winner:
- fusion winner Recall@5:
- Precision@5:
- doc-disjoint Recall@5:

## Readiness
READY FOR KAGGLE GPU SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
