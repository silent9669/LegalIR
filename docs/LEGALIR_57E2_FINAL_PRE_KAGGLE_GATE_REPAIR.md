# LegalIR Task 1 — Final Pre-Kaggle Gate Repair

**Repository:** `silent9669/LegalIR`  
**Audited HEAD:** `57e2d121f7622053e50a8d8eba17e5fed477106a`  
**Previous audited HEAD:** `c4bae606bff891bd3c995f9f9359aa700031632d`  
**Target:** Kaggle T4 ×2  
**Decision:** **NOT READY FOR FULL KAGGLE RUN YET**

## 0. Preserve the fixes already present

Do **not** redesign the pipeline. The latest commit correctly fixes most of the C4BAE contract:

- `Mapping` import in `build_pairs.py`;
- repo-root config resolution;
- explicit reranker experiment config;
- 2-fold cross-fitted GPU-smoke structure;
- post-rerank OOF features;
- public `q_emb` plumbing;
- query text + document-frequency features at learned-fusion inference;
- Dense audit exposure;
- stronger adapter/dense artifact checks;
- submission manifest creation before root copy;
- root/Kaggle notebook byte parity.

The remaining work below is the final correctness / deployment / GPU-readiness gate.

---

# 1. P0 — Kaggle data discovery does not support the documented `/kaggle/input/legalir` mount

The README tells the user to attach the official **LegalIR** Kaggle dataset mounted at:

```text
/kaggle/input/legalir
```

but `discover_data_dir()` currently checks paths such as:

```text
/kaggle/input/legalir-task1/...
/kaggle/input/legalir-task-1/...
/kaggle/input/uit-dsc-2026-task1/...
/kaggle/input/legalir-dataset/...
```

and does **not** check:

```text
/kaggle/input/legalir
/kaggle/input/legalir/artifacts/task1/data
```

The same mismatch exists for raw `train.json`, `selected-contexts.zip`, and `public-official.json`.

This can make the exact notebook setup documented in README fail before training begins.

## Required fix

Add the documented mount explicitly and implement deterministic discovery under `/kaggle/input`.

Preferred approach:

```python
CANONICAL_REQUIRED = {
    "documents.parquet",
    "chunks.parquet",
    "queries_train.parquet",
    "qrels_train.parquet",
}
```

Search candidate directories and accept only a directory containing the complete canonical set.

Also discover a raw source set containing:

```text
selected-contexts.zip
train.json
public-official.json
```

Prefer all three from the same Kaggle dataset directory.

If multiple complete sets are found, either:
1. prefer an explicitly supplied `data_dir` / `public_json_path`; or
2. raise an ambiguity error listing candidates.

Never silently select an arbitrary recursive match.

Add at minimum:

```text
/kaggle/input/legalir
/kaggle/input/legalir/artifacts/task1/data
/kaggle/input/legalir/artifacts/shared/canonical/v2
```

## Tests

```text
test_discover_documented_kaggle_legalir_mount
test_discover_public_official_from_documented_mount
test_kaggle_discovery_requires_complete_canonical_set
test_ambiguous_kaggle_dataset_discovery_raises
```

---

# 2. P0 — `gpu_smoke` is still not a production-strict readiness gate

Current default behavior is effectively:

```python
strict_artifacts = is_full
```

so `gpu_smoke` is not strict.

A GPU smoke that can silently continue without a required production artifact does not prove that the full run is safe.

## Required fix

Default to:

```python
strict_artifacts = is_full or is_gpu_smoke
```

The real GPU-smoke gate must require:

```text
Legal BM25 non-empty
PyVi BM25 non-empty
DEk21 Dense embeddings + metadata non-empty
Question Memory non-empty
real PEFT adapter + manifest + checksum
fusion artifact if learned fusion won
```

Do not weaken the gate because the smoke dataset is small.

---

# 3. P0 — GPU smoke still does not prove actual GPU placement or VRAM

The source now routes requested devices correctly, but there is still no real runtime proof that the loaded model parameters actually reside on the intended GPUs.

Required runtime proof:

```text
Dense requested:    cuda:0
Dense actual:       cuda:0

Reranker requested: cuda:1
Reranker actual:    cuda:1
```

Inspect actual model parameters after lazy loading:

```python
next(model.parameters()).device
```

For PEFT:

```python
next(reranker.model.parameters()).device
```

Record:

```text
torch.cuda.max_memory_allocated(0)
torch.cuda.max_memory_reserved(0)
torch.cuda.max_memory_allocated(1)
torch.cuda.max_memory_reserved(1)
```

Reset peak statistics before GPU smoke.

Save a machine-readable file:

```text
gpu_smoke_report.json
```

containing:

```json
{
  "dense_requested": "cuda:0",
  "dense_actual": "cuda:0",
  "reranker_requested": "cuda:1",
  "reranker_actual": "cuda:1",
  "gpu0_peak_allocated_bytes": 0,
  "gpu1_peak_allocated_bytes": 0,
  "optimizer_steps": 0,
  "param_diff": 0,
  "adapter_checksum": "...",
  "strict_artifacts": true,
  "fusion_crossfit_folds": 2,
  "oom": false
}
```

If requested != actual, GPU smoke fails.

---

# 4. P0 — Final submission validation does not independently validate corpus IDs

The full orchestrator currently calls roughly:

```python
validate_submission(
    sub_json,
    expected_qids=expected_qids,
)
```

but does not supply `corpus_doc_ids` or `data_dir`.

The pipeline selector normally filters IDs, but the final validation gate must be independent of the producer.

## Required fix

Use:

```python
official_doc_ids = set(df_docs["doc_id"].astype(str))

val_res = validate_submission(
    sub_json,
    expected_qids=expected_qids,
    corpus_doc_ids=official_doc_ids,
)
```

Full mode must fail on any unknown ID.

## Tests

```text
test_full_submission_rejects_unknown_corpus_id
test_full_submission_validates_official_qids_and_doc_ids_independently
```

---

# 5. P0 — Learned-fusion fallback can silently reload as RRF

`LightGBMRanker.fit()` catches any LightGBM failure and falls back to `LinearRanker`.

However, when the fallback is saved using a `.txt` target, the linear JSON payload can be written into a `.txt` file.

`LightGBMRanker.load()` only recognizes `linear_ridge` metadata when the selected path has `.json` suffix. A `.txt` fallback is then passed to:

```python
lgb.Booster(model_file=...)
```

which fails. The exception is swallowed and the ranker remains unloaded.

Then `predict()` silently falls back to RRF because both:

```text
self.model is None
self.fallback_model is None
```

This can make OOF select a learned model but public inference actually run RRF.

## Required fix

Choose one robust policy.

### Preferred competition policy

Ensure LightGBM is installed and make FULL/GPU_SMOKE LightGBM failures explicit:

```python
strict_training=True
```

If learned fusion is selected, the selected model must serialize and reload successfully before public inference.

### If retaining linear fallback

Give it a real typed artifact:

```text
model.json
model_type=linear_ridge
```

and make `LightGBMRanker.load()` support the exact fallback artifact regardless of the original requested suffix.

Add a round-trip prediction parity test:

```python
pred_before == pred_after_reload
```

within numerical tolerance.

## Strict learned-fusion invariant

If learned fusion won:

```text
ranker.model is not None OR ranker.fallback_model is not None
```

must be true after loading.

Never silently substitute RRF.

---

# 6. P0/P1 — Kaggle notebook does not guarantee LightGBM availability

`requirements.txt` includes:

```text
lightgbm
sentencepiece
```

but the thin Kaggle notebook's minimal dependency check currently installs only:

```text
bm25s
pyvi
peft
accelerate
```

If LightGBM is unavailable, the silent fallback path above can be triggered.

The BGE/XLM-R tokenizer stack can also require SentencePiece depending on the Kaggle image.

## Required fix

The notebook dependency preflight must verify at least:

```text
lightgbm
sentencepiece
bm25s
pyvi
peft
accelerate
```

Do not reinstall PyTorch.

After installation, import each required package and fail immediately if unavailable.

Save dependency versions to the run manifest.

---

# 7. P1 — Strict fusion validation is still permissive

Current strict code validates the fusion manifest only **if the manifest exists**.

For a learned-fusion winner, require:

```text
manifest.json exists
winning_method == learned_ranker
feature_training_stage == post_rerank
feature_columns non-empty
model artifact exists
model artifact successfully loads
loaded feature columns match manifest
```

Do not allow:

```text
missing manifest -> continue
failed LightGBM load -> warning -> RRF
```

The final pipeline must fail instead.

---

# 8. P1 — Adapter checksum is recorded but not verified against the current file

Strict adapter loading currently checks that:

```text
adapter_checksum
```

exists in `training_manifest.json`, but it does not recompute the SHA-256 of the adapter weights and compare it to the manifest.

## Required fix

Locate:

```text
adapter_model.safetensors
```

or:

```text
adapter_model.bin
```

Compute SHA-256 and require:

```python
actual_checksum == manifest["adapter_checksum"]
```

Mismatch must fail.

Also require:

```text
status == completed
param_diff > 0
optimizer_steps > 0
unique_training_queries > 0
```

---

# 9. P1 — FULL mode still swallows important Dense errors before failing much later

Examples:

```python
try:
    dense_retriever = DenseMacroRetriever.load(...)
except Exception:
    print("Warning...")
```

and query-embedding precomputation also catches broad exceptions.

In FULL mode this can cause expensive BM25 / OOF / pair-building work before a later strict-artifact failure.

## Required behavior

In:

```text
FULL
GPU_SMOKE
```

required learned-component failures must raise immediately.

Only normal CPU/mock `smoke` may use degraded branches.

Contract:

```python
if is_full or is_gpu_smoke:
    dense load/build failure -> raise
    train query embedding failure -> raise
    public q_emb batch precompute failure -> raise
```

The production path must never silently drop DEk21.

---

# 10. P1 — Final output should default to exactly five documents

Official Recall cannot decrease when adding a unique valid candidate while staying at `<=5`, and the project documentation itself specifies default top-5.

The current selector guarantees `1..5`, but it only backfills when below `min_k`, normally `1`.

If retrieval returns 2–4 valid documents, final submission can therefore contain fewer than 5.

## Required fix

For the **competition submission path**, enforce:

```text
exactly 5 unique valid corpus IDs
```

when the corpus contains at least 5 documents.

Do not change generic selector semantics if other tests depend on `min_k`; instead add an explicit final-submission `fill_to_k=5` policy.

Backfill order must be deterministic and corpus-valid.

## Tests

```text
test_full_submission_every_answer_has_exactly_five_ids
test_top5_backfill_is_unique_deterministic_and_corpus_valid
```

---

# 11. P1 — Persist document-disjoint report correctly

`run_document_disjoint_evaluation()` writes a report file and returns the report, but the orchestrator later checks:

```python
oof_runner.doc_disjoint_report
```

The runner should explicitly retain:

```python
self.doc_disjoint_report = final_report
```

Initialize the attribute in `__init__`.

Do not label this as full final-system fusion robustness unless fusion is evaluated leakage-safely on that split. The current safe label is:

```text
trained_reranker_system
```

---

# 12. P2 — Remove duplicated terminal summary block

`run_kaggle_pipeline()` currently prints two pipeline-complete summaries, including an older block that again labels raw OOF values as "Full OOF".

Keep one authoritative summary containing:

```text
Reranker OOF Recall@5
Cross-fitted fusion winner
Fusion winner Recall@5
Fusion winner Precision@5
Candidate Recall@150
Doc-disjoint trained-reranker Recall@5
parameter total
submission status
```

This is reporting cleanup, not an algorithm change.

---

# 13. P2 — Add leakage-safe RRF weight tuning after correctness is locked

The source calls the comparison "Tuned RRF", but the production call currently uses default RRF weights.

After all P0/P1 fixes and only if runtime allows, add a small deterministic cross-fitted search over:

```text
bm25
bm25_pyvi
dense
memory
exact
reranker
```

weights.

For each validation fold:
- tune weights using folds != f only;
- evaluate on fold f;
- concatenate held-out predictions;
- compare official Recall@5, then Precision@5 tie-break.

Do **not** tune weights on the same fold being scored.

This is an optional score-maximization upgrade; it must not delay the correctness gate.

---

# 14. Required verification

Run in this order:

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('IMPORT_OK')"
pytest -q
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

Required source tests:

```text
test_discover_documented_kaggle_legalir_mount
test_discover_public_official_from_documented_mount
test_gpu_smoke_defaults_to_strict_artifacts
test_full_submission_rejects_unknown_corpus_id
test_learned_fusion_roundtrip_preserves_predictions
test_selected_learned_fusion_cannot_silently_fallback_to_rrf
test_adapter_checksum_is_verified
test_full_submission_every_answer_has_exactly_five_ids
test_doc_disjoint_report_is_persisted
test_notebook_installs_required_non_torch_dependencies
test_notebook_byte_level_parity
```

Then on real Kaggle T4×2:

```bash
LEGALIR_RUN_MODE=gpu_smoke
```

GPU smoke must use:
- official Task 1 data;
- real DEk21;
- real BGE reranker;
- real PEFT LoRA;
- 2 tiny OOF folds;
- strict artifacts;
- real fusion train/save/reload;
- actual `cuda:0` / `cuda:1` verification;
- VRAM recording;
- final tiny inference;
- no OOM.

Only after GPU smoke passes:

```bash
LEGALIR_RUN_MODE=full
```

---

# 15. Notebook safety

Until a real T4×2 GPU smoke has passed, prefer the notebook default:

```python
RUN_MODE = os.environ.get("LEGALIR_RUN_MODE", "gpu_smoke")
```

After GPU smoke passes, the user can explicitly set:

```text
LEGALIR_RUN_MODE=full
```

Do not accidentally launch the multi-hour full training path as the first hardware test.

---

# 16. Readiness gate

The coding agent may report:

```text
READY FOR KAGGLE GPU SMOKE: YES
```

only after:
- compile/import succeeds;
- all tests pass;
- CPU/mock smoke passes;
- notebook parity passes.

It may report:

```text
READY FOR FULL KAGGLE RUN: YES
```

only after a **real T4×2 gpu_smoke** verifies:
- actual devices;
- non-zero VRAM on both GPUs as expected;
- real LoRA optimizer steps;
- `param_diff > 0`;
- adapter SHA verification;
- learned-fusion round trip if selected;
- strict artifacts;
- no OOM;
- valid tiny output.

Without that evidence, the maximum allowed conclusion is:

```text
Source-level repair complete; ready for Kaggle GPU smoke.
```

---

# 17. Final report format

```markdown
# LegalIR 57E2 Final Gate Report

## Base
- audited head: 57e2d121f7622053e50a8d8eba17e5fed477106a
- new head: <sha>

## Repair status
| Gate | PASS/FAIL | Evidence |
|---|---|---|
| documented Kaggle data discovery | | |
| strict gpu_smoke | | |
| actual dual-GPU placement | | |
| corpus-ID submission validation | | |
| learned fusion round-trip | | |
| LightGBM dependency | | |
| adapter checksum verification | | |
| exact top-5 submission | | |
| doc-disjoint report persistence | | |

## Verification
- compileall:
- pytest:
- CPU smoke:
- notebook parity:
- parameter total:

## T4x2 GPU smoke
- executed:
- Dense requested/actual:
- Reranker requested/actual:
- GPU0 peak allocated:
- GPU1 peak allocated:
- optimizer steps:
- param_diff:
- adapter checksum verified:
- fusion save/reload:
- OOM:
- result:

## Readiness
READY FOR KAGGLE GPU SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
