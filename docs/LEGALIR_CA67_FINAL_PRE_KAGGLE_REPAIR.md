# LegalIR Task 1 — Final Pre-Kaggle Repair Contract

**Repository:** `silent9669/LegalIR`  
**Audited HEAD:** `ca67b671d40a8e211af7869b275026007b0f7a5e`  
**Previous audited HEAD:** `3c6b3261027286907fedfad24f53c478a32a4eed`  
**Target:** Kaggle T4 ×2  
**Audit decision:** **NOT READY FOR FULL KAGGLE RUN**

---

## 0. Mission

Do **not** redesign the pipeline again. Preserve the fixes already implemented in `ca67b671`:

- thin 5-cell Kaggle notebook calling one `run_kaggle_pipeline(...)` orchestrator;
- root and Kaggle notebooks byte-identical;
- explicit `build_training_pairs(data_dir, index_dir, output_dir, ...)`;
- explicit `train_reranker(pairs_file=...)` with no silent 50-query fallback;
- final pair generation using `fold=None, use_all_queries=True`;
- real PEFT/LoRA training and adapter loading;
- PyVi BM25 in OOF/final retrieval;
- BM25 metadata enrichment;
- no duplicated `[QUESTION]` inside sequence B;
- cross-fitted fusion implementation;
- `peft` and `accelerate` dependencies;
- strict submission helpers and <4B parameter audit.

This repair contract addresses only the remaining integration bugs that can still make a long Kaggle run wrong, wasteful, or misleading.

---

# 1. P0 — Fix TrainQuestionMemory tuple/qid parsing

## Root cause

`OOFRunner` and `run_kaggle_pipeline()` pass fold/full memory records as tuples:

```python
(qid, question_text, q_emb)
```

but `TrainQuestionMemory._query_records()` does not parse tuple/list records. Non-mapping records are converted to:

```python
qid = index
text = normalize_text(record)
embedding = None
```

A record such as:

```python
("146300", "Dự án đầu tư là gì?", embedding)
```

can therefore become qid `"0"`, then be discarded because qrels are keyed by official IDs. The fold-local and final Question Memory can silently become empty.

## Required fix

Teach `_query_records()` to support:

```python
(qid, text)
(qid, text, embedding)
```

Example:

```python
elif isinstance(record, (tuple, list)):
    if len(record) == 3:
        qid, text, embedding = record
    elif len(record) == 2:
        qid, text = record
        embedding = None
    else:
        raise ValueError(
            "question tuple/list must be (qid, text) or (qid, text, embedding)"
        )
    parsed.append((str(qid), normalize_text(text), embedding))
```

Also prefer explicit mapping records in new production code:

```python
{
    "query_id": qid,
    "question_norm": question_text,
    "q_emb": embedding,
}
```

## Hard invariants

For each OOF fold:

```python
set(memory.qids) == set(fold_train_qids_with_labels)
set(memory.qids).isdisjoint(set(val_qids))
len(memory.qids) > 0
```

For final memory:

```python
set(memory.qids) == set(all_labeled_train_qids)
len(memory.qids) == expected_labeled_query_count
```

## Mandatory tests

```python
test_question_memory_tuple_records_preserve_qids
test_oof_memory_nonempty_and_fold_safe
test_final_memory_contains_all_labeled_train_qids
```

---

# 2. P0 — Preserve explicit CUDA indices

## Root cause

`resolve_kaggle_devices()` returns the desired:

```text
dense_device    = cuda:0
reranker_device = cuda:1
```

but `src/models/device.py::resolve_device()` only recognizes exact `cpu`, `cuda`, and `mps`. `cuda:1` falls through and becomes generic `cuda`, normally GPU 0.

The notebook can therefore print “GPU 1: reranker” while both learned models run on GPU 0.

## Required fix

`resolve_device()` must preserve indexed CUDA devices and validate the index:

```python
resolve_device("cuda:0") == "cuda:0"
resolve_device("cuda:1") == "cuda:1"
```

Pseudo-contract:

```python
if req.startswith("cuda:"):
    if not torch.cuda.is_available():
        raise RuntimeError(...)
    idx = int(req.split(":", 1)[1])
    if idx < 0 or idx >= torch.cuda.device_count():
        raise RuntimeError(...)
    return f"cuda:{idx}"
```

Support `torch.device` input too.

## Mandatory tests

```python
test_resolve_device_cuda_zero
test_resolve_device_cuda_one
test_resolve_device_invalid_cuda_index_raises
```

The existing test that only checks `resolve_kaggle_devices(["cuda:0", "cuda:1"])` is insufficient because it never verifies the model-level resolver.

---

# 3. P0 — Separate dense_device and reranker_device end-to-end

Fixing `resolve_device()` is not enough.

## Current coupling

`LegalIRPipeline.load_pipeline(..., device=...)` passes the same device to both:

```python
DenseMacroRetriever.load(..., device=device)
CrossEncoderReranker(..., device=device)
```

The orchestrator currently passes `reranker_device`, so final inference would place both DEk21 and BGE on the reranker device.

`OOFRunner` also has one generic `device` and uses it to load dense retrieval and reranker components.

`train_reranker()` resolves its own device from config rather than accepting the orchestrator's explicit `reranker_device`.

## Required APIs

### `train_reranker`

Add:

```python
def train_reranker(..., device: str | None = None):
    resolved_device = resolve_device(
        device if device is not None else cfg.get("device", "auto")
    )
```

### `OOFRunner`

Use:

```python
OOFRunner(
    ...,
    dense_device: str | None = None,
    reranker_device: str | None = None,
)
```

Contract:

```text
DEk21 corpus/query embeddings -> dense_device
Question embedding cache      -> dense_device
fold LoRA training            -> reranker_device
fold BGE inference            -> reranker_device
```

### `LegalIRPipeline.load_pipeline`

Use:

```python
LegalIRPipeline.load_pipeline(
    ...,
    dense_device=None,
    reranker_device=None,
)
```

Then:

```python
DenseMacroRetriever.load(..., device=dense_device)
CrossEncoderReranker(..., device=reranker_device)
```

Backward compatibility may map an old single `device` into both only outside the production orchestrator.

## Runtime proof

After lazy loading, log and assert:

```text
Dense requested: cuda:0
Dense actual:    cuda:0
Reranker requested: cuda:1
Reranker actual:    cuda:1
```

Inspect at least one actual model parameter device.

## Mandatory tests

```python
test_train_reranker_honors_explicit_device
test_oof_accepts_distinct_dense_and_reranker_devices
test_final_pipeline_accepts_distinct_dense_and_reranker_devices
```

---

# 4. P0 — Separate Kaggle runtime config from reranker training config

## Root cause

The orchestrator currently passes `configs/kaggle.yaml` as the training config.

That file contains T4 runtime settings such as:

```yaml
batch_size: 2
gradient_accumulation_steps: 8
max_length: 512
fp16: true
gradient_checkpointing: true
```

but it does not carry the full reranker optimization contract.

The actual experiment file contains:

```yaml
learning_rate: 2.0e-5
max_steps: 500
warmup_ratio: 0.1
loss_type: bce
```

Without an explicit `max_steps`, `RerankerTrainer` falls back to epoch mode (`num_epochs=2`). On full folds this can multiply T4 runtime dramatically and is not the intended bounded run.

## Required fix

Use two explicit config paths:

```python
run_kaggle_pipeline(
    ...,
    runtime_config_path="configs/kaggle.yaml",
    reranker_config_path="configs/experiments/reranker_lora.yaml",
)
```

Pass `reranker_config_path` to every fold and final `train_reranker()` call.

Before training, save the effective config, including:

```json
{
  "base_model_name": "BAAI/bge-reranker-v2-m3",
  "loss_type": "bce",
  "learning_rate": 0.00002,
  "batch_size": 2,
  "gradient_accumulation_steps": 8,
  "max_length": 512,
  "max_steps": 500,
  "fp16": true,
  "gradient_checkpointing": true,
  "device": "cuda:1"
}
```

Full mode must fail if training work is not explicit:

```python
if effective_max_steps is None:
    raise ValueError("Full reranker training requires explicit max_steps")
```

## Mandatory tests

```python
test_full_training_uses_explicit_reranker_config
test_effective_reranker_max_steps_is_explicit
```

---

# 5. P1 — Extract OOF fusion features AFTER reranking

## Root cause

Current OOF flow extracts features before the fold reranker runs:

```python
candidates = hybrid_engine.search_candidates(...)
feat_df = extract_candidate_features(candidates)
candidates = reranker.rerank(...)
```

But fusion features include:

```text
reranker_score
reranker_second_score
reranker_margin
```

At extraction time they are missing/sentinel. The learned fusion is therefore trained on pre-reranker features, while final inference gives it post-reranker records. This is a training/inference mismatch.

## Required flow

```python
retrieval_candidates = hybrid_engine.search_candidates(...)

reranked_candidates = fold_reranker.rerank(
    query=q_text,
    candidates=retrieval_candidates,
    evidence_builder=self.evidence_builder,
    top_k=self.rerank_k,
)

post_rerank_candidates = merge_reranker_features(
    retrieval_candidates,
    reranked_candidates,
)

feat_df = extract_candidate_features(
    query_id=qid,
    candidate_records=post_rerank_candidates,
    query_text=q_text,
    doc_freq_map=fold_train_doc_freq_map,
    qrels=...,
)
```

Top `rerank_k` rows must contain real reranker scores. Rows outside the budget may retain the explicit sentinel.

The fusion comparison must evaluate the same post-reranker candidate representation:

```text
A. weighted RRF including reranker contribution
B. cross-fitted LightGBM using retrieval + exact + memory + reranker features
```

## Required test

Use a deterministic fake reranker with distinctive scores and assert those exact values land in `oof_features.parquet`:

```python
test_oof_features_are_extracted_after_reranking
```

The current synthetic LightGBM unit test that manually inserts `reranker_score` does not test this wiring.

---

# 6. P1 — Document-disjoint must evaluate a trained reranker

## Root cause

When `train_reranker_per_fold=True`, the normal OOF loop trains fold adapters, but `global_reranker` remains `None`.

The document-disjoint call receives that `None`, so its reported score is retrieval-only rather than trained-reranker system robustness.

## Required path

```text
doc-disjoint train IDs
  -> fold-safe Question Memory
  -> hard-negative mining on train IDs only
  -> dedicated LoRA adapter trained on train IDs only
  -> evaluate doc-disjoint val IDs
  -> same post-rerank ranking/fusion policy
```

Save separately:

```text
cv/doc_disjoint/reranker_adapter/
cv/doc_disjoint/report.json
```

If useful, report both:

```json
{
  "retrieval_only": {...},
  "trained_reranker_system": {...}
}
```

Never call retrieval-only metrics final-system document-disjoint metrics.

---

# 7. P0 — Full mode must require the official public file

## Root cause

Current full path can fall back to the first train queries if `public-official.json` is missing and then validate using:

```python
expected_qids = set(predictions.keys())
```

That validates predictions against themselves instead of an independent official query set.

## Required behavior

### Full mode

```python
if public_test_file is None or not public_test_file.exists():
    raise FileNotFoundError(
        "Full mode requires official public-official.json; refusing to generate submission"
    )
```

Then:

```python
official_public_qids = set(public_data.keys())
assert set(predictions.keys()) == official_public_qids
validate_submission(sub_json, expected_qids=official_public_qids)
```

Never derive the expected set from predictions.

### Smoke mode

A train-query fallback can remain for unit tests, but mark it explicitly:

```text
NON_SUBMITTABLE_SMOKE
```

Do not present it as an official competition submission.

## Mandatory tests

```python
test_full_mode_missing_public_file_raises
test_submission_validation_uses_independent_public_qid_set
test_smoke_train_fallback_is_non_submittable
```

---

# 8. P1 — Strict production artifact loading

Compatibility fallbacks are useful in tests but dangerous in `full` mode.

Add:

```python
LegalIRPipeline.load_pipeline(..., strict_artifacts: bool = False)
```

Production must call:

```python
strict_artifacts=True
```

In strict mode require:

```text
Legal BM25 index exists and is non-empty
PyVi BM25 index exists and is non-empty
dense embeddings + metadata exist and are non-empty
Question Memory exists and qids are non-empty
final adapter_config.json exists
adapter_model.safetensors or adapter_model.bin exists
training_manifest.status == completed
training_manifest.param_diff > 0
training_manifest.adapter_checksum != null
selected fusion model exists if learned fusion won
```

Canonical validation failure must raise in production; do not only print warnings.

If a trained adapter is expected, never silently load untouched base BGE.

## Mandatory tests

Parameterized tests should delete one required artifact at a time and assert strict production load raises.

---

# 9. P1 — Restore exact-match metadata parity at public inference

## Root cause

The final loader currently reads only:

```python
["doc_id", "title", "name_raw", "legal_number"]
```

but `ExactMatcher` uses `year` and `doc_type` for:

```text
exact_year
exact_doc_type
title + year matching
```

OOF uses full document rows, so public inference is weaker/different.

## Required fix

Load at least:

```python
[
    "doc_id",
    "title",
    "name_raw",
    "legal_number",
    "year",
    "doc_type",
    "link",
]
```

Add a test asserting final loaded exact matching can set `exact_year=True` and `exact_doc_type=True` when the query and metadata support them.

---

# 10. P1 — Make DEk21 query embedding reuse real end-to-end

Current pipeline still re-encodes queries in multiple places.

Required shared cache contract:

```python
qid -> normalized DEk21 embedding
```

Use the same vector for:

```text
dense retrieval
Question Memory dense similarity
hard-negative mining
OOF retrieval
public inference
```

Extend pair mining:

```python
build_training_pairs(
    ...,
    query_embeddings: Mapping[str, np.ndarray] | None = None,
)
```

and pass `q_emb` to dense and hybrid calls.

Public inference should precompute the public query matrix once on GPU0 and pass each vector into the pipeline.

`TrainQuestionMemory.load()` must not re-encode all stored train questions when `train_embeddings.npy` already exists. Implement `fit(..., encode_dense=False)` or a dedicated saved-state loader that restores TF-IDF + saved dense matrix without calling DEk21.

## Mandatory tests

Use a counting fake encoder:

```python
test_memory_load_with_saved_embeddings_does_not_reencode
test_public_query_embedding_shared_by_dense_and_memory
test_pair_mining_uses_precomputed_query_embedding
```

---

# 11. P1 — Add a real Kaggle GPU smoke mode

Current `smoke` uses mock models and does not exercise fold-specific real LoRA or dual-GPU placement.

Keep:

```text
smoke = CPU/mock/fast integration
```

Add:

```text
gpu_smoke = real DEk21 + real BGE + real PEFT, tiny workload
full      = competition run
```

Minimum `gpu_smoke`:

```text
1 fold
20–50 train queries
candidate_k 20–50
rerank_k 10
2–5 LoRA optimizer steps
real DEk21 on cuda:0
real BGE reranker on cuda:1
real adapter save + reload
```

Must record:

```text
dense actual device = cuda:0
reranker actual device = cuda:1
memory qids > 0
LoRA param_diff > 0
adapter checksum != null
OOF real reranker_score count > 0
feature stage = post_rerank
peak VRAM GPU0
peak VRAM GPU1
```

`gpu_smoke` is non-submittable. Do not launch `full` until it passes.

---

# 12. OOF feature parity contract

Cross-fitted fusion training and public inference must use the same semantics and feature order:

```text
raw_bm25_rank
raw_bm25_score
pyvi_bm25_rank
pyvi_bm25_score
dense_rank
dense_score
dense_second_score
dense_margin
memory_rank
memory_similarity
memory_vote_count
exact_score
exact_legal_number
exact_article
exact_clause
exact_point
exact_year
exact_doc_type
exact_title_overlap
source_count
rrf_score
reranker_score
reranker_second_score
reranker_margin
query_length
train_doc_freq
```

For fold `f`, `train_doc_freq` must come from fold-train qrels only.

Fusion manifest must record:

```json
{
  "feature_schema_version": "...",
  "feature_columns": ["..."],
  "training_stage": "post_rerank"
}
```

The production loader must validate this schema before prediction.

---

# 13. T4-safe explicit training contract

Initial production reranker config:

```yaml
base_model_name: BAAI/bge-reranker-v2-m3
loss_type: bce
use_lora: true
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
learning_rate: 2.0e-5
batch_size: 2
gradient_accumulation_steps: 8
max_steps: 500
warmup_ratio: 0.1
max_length: 512
fp16: true
gradient_checkpointing: true
```

With a single reranker GPU, effective batch is:

```text
2 × 8 = 16
```

Do not multiply by 2 merely because the machine has two GPUs; the architecture assigns different models to the two GPUs rather than DDP-training BGE across both.

---

# 14. Mandatory integration tests before readiness

The final suite must include at least:

```text
test_question_memory_tuple_records_preserve_qids
test_oof_memory_nonempty_and_fold_safe
test_final_memory_contains_all_labeled_train_qids
test_memory_load_with_saved_embeddings_does_not_reencode

test_resolve_device_cuda_zero
test_resolve_device_cuda_one
test_resolve_device_invalid_cuda_index_raises
test_train_reranker_honors_explicit_device
test_pipeline_uses_distinct_dense_and_reranker_devices

test_full_training_uses_explicit_reranker_config
test_effective_reranker_max_steps_is_explicit

test_oof_features_contain_actual_fold_reranker_scores
test_fusion_feature_training_inference_schema_match
test_fold_train_doc_frequency_has_no_val_label_leakage

test_doc_disjoint_reranker_uses_train_side_only
test_doc_disjoint_report_contains_trained_reranker_metrics

test_strict_load_missing_bm25_raises
test_strict_load_missing_pyvi_raises
test_strict_load_missing_dense_raises
test_strict_load_empty_memory_raises
test_strict_load_missing_final_adapter_raises
test_strict_load_missing_selected_fusion_raises
test_invalid_canonical_dataset_raises

test_full_mode_missing_public_file_raises
test_submission_validation_uses_official_public_keys
test_smoke_train_fallback_is_non_submittable

test_final_exact_matcher_has_year_and_doc_type
test_public_query_embedding_shared_dense_memory
test_pair_mining_uses_precomputed_query_embedding

test_kaggle_notebook_byte_level_parity
```

---

# 15. Full-mode fail-fast rules

In `run_mode="full"`, abort for any of:

```text
canonical validation failed
official public file missing
Legal BM25 missing/empty
PyVi BM25 missing/empty
dense index missing/empty
Question Memory missing/empty
final adapter missing
final adapter checksum missing
final adapter param_diff <= 0
selected learned fusion missing
submission query keys differ from official public keys
answer length > 5
invalid document ID
parameter total >= 4B
```

Silent degradation belongs only in explicit unit/smoke modes.

---

# 16. Recommended orchestrator contract

```python
result = run_kaggle_pipeline(
    data_dir=DATA_DIR,
    working_dir=WORKING_DIR,
    run_mode=RUN_MODE,  # smoke | gpu_smoke | full
    hf_token=hf_token,
    public_json_path=PUBLIC_TEST_FILE,
    repo_root=REPO_ROOT,

    runtime_config_path="configs/kaggle.yaml",
    reranker_config_path="configs/experiments/reranker_lora.yaml",

    dense_device="cuda:0",
    reranker_device="cuda:1",
    strict_artifacts=True,
)
```

Production flow:

```text
1. validate mode/config/data/public input
2. verify CUDA topology
3. parameter audit
4. build/load metadata-enriched Legal BM25
5. build/load metadata-enriched PyVi BM25
6. build/load DEk21 on GPU0
7. precompute train query embeddings once
8. each OOF fold:
   - non-empty fold-safe memory
   - cached-q_emb hard-negative mining
   - fold LoRA on GPU1
   - param_delta/checksum verification
   - fold adapter reload
   - val retrieval
   - fold reranking
   - POST-RERANK OOF feature extraction
   - metrics
   - release reranker VRAM
9. cross-fit weighted RRF vs LightGBM
10. dedicated doc-disjoint trained reranker evaluation
11. full memory on all eligible train queries
12. all-query final pair mining
13. final LoRA with explicit 500-step config on GPU1
14. verify/reload exact final adapter
15. fit final LightGBM only if it won
16. require official public file
17. precompute public q_emb once on GPU0
18. full 5-branch + BGE-LoRA + selected fusion inference
19. validate against independent official qid set
20. package exactly submission.json at ZIP root
21. write manifests/hashes/reports
```

---

# 17. Preserve previous fixes — regression guards

Do not regress these already-correct behaviors:

```text
explicit training-pair data/index/output paths
explicit reranker pairs_file
no hidden 50-query production fallback
final fold=None / use_all_queries=True
no duplicated [QUESTION] in passage sequence B
PyVi BM25 participation
BM25 metadata enrichment
root/Kaggle notebook byte identity
requirements include peft + accelerate
explicit final adapter loading
strict 1–5 unique string document IDs
submission ZIP root contains only submission.json
parameter audit remains <4B
```

---

# 18. Small cleanup

The notebook says “4-Branch Hybrid Candidate Retrieval” but lists:

```text
Legal BM25
PyVi BM25
DEk21
Question Memory
Exact Matcher
```

Change wording to **5-Branch Hybrid Candidate Retrieval**.

For convenience, after a valid `full` run, also copy:

```text
/kaggle/working/submission.json
/kaggle/working/submission.zip
/kaggle/working/submission_manifest.json
```

while retaining detailed artifacts in `/kaggle/working/legalir_run/`.

---

# 19. Verification sequence

Do not launch the expensive full run immediately after code changes.

First:

```bash
pytest -q
```

Then mock integration:

```bash
LEGALIR_RUN_MODE=smoke python scripts/smoke_kaggle_pipeline.py
```

Then on Kaggle T4 ×2:

```text
LEGALIR_RUN_MODE=gpu_smoke
```

Only if real GPU smoke passes:

```text
LEGALIR_RUN_MODE=full
```

---

# 20. Readiness rules

The coding agent may report:

```text
READY FOR KAGGLE GPU SMOKE: YES
```

when source-level tests and mock smoke pass.

It may report:

```text
READY FOR FULL KAGGLE RUN: YES
```

only after a real T4×2 GPU smoke proves:

```text
dense model really runs on cuda:0
reranker really runs on cuda:1
real DEk21 loads
real BGE loads
real PEFT LoRA update occurs
real adapter reload works
Question Memory is non-empty
post-rerank OOF feature path works
no T4 OOM at configured settings
```

Do not infer this from CPU/tiny/mock tests.

---

# 21. Final coding-agent report format

```markdown
# Final LegalIR Pre-Kaggle Repair Report

## Audited base
- Base commit: ca67b671d40a8e211af7869b275026007b0f7a5e
- New commit: <sha>

## P0 fixes
| Item | Status | Test |
|---|---|---|
| QuestionMemory tuple/qid contract | PASS/FAIL | ... |
| cuda:N preservation | PASS/FAIL | ... |
| distinct dense/reranker devices | PASS/FAIL | ... |
| explicit reranker training config | PASS/FAIL | ... |
| official public input hard requirement | PASS/FAIL | ... |

## P1 fixes
| Item | Status | Test |
|---|---|---|
| post-rerank OOF features | PASS/FAIL | ... |
| trained-reranker doc-disjoint | PASS/FAIL | ... |
| strict artifact loading | PASS/FAIL | ... |
| exact metadata parity | PASS/FAIL | ... |
| query embedding reuse | PASS/FAIL | ... |

## Test results
- pytest: <passed>/<failed>
- mock smoke: PASS/FAIL
- notebook byte parity: PASS/FAIL
- parameter audit: <count>

## GPU smoke
- Ran on real Kaggle T4x2: YES/NO
- dense actual device: ...
- reranker actual device: ...
- real LoRA param_diff: ...
- real adapter checksum: ...
- peak GPU0 VRAM: ...
- peak GPU1 VRAM: ...
- status: PASS/FAIL/NOT_RUN

## Readiness
- READY FOR KAGGLE GPU SMOKE: YES/NO
- READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
- ...
```

If real T4×2 smoke has not run, the strongest allowed conclusion is:

```text
Source-level repair complete; ready for Kaggle GPU smoke.
```

---

# 22. Final decision on audited commit

`ca67b671` is **much closer** to the intended high-score architecture, but it is still **not safe for the complete expensive Kaggle run**.

The blockers are not cosmetic:

```text
Question Memory can silently be empty;
cuda:1 is not preserved by the model-level device resolver;
dense/reranker device APIs remain coupled;
the wrong YAML can turn bounded training into two full epochs;
learned fusion currently sees pre-reranker OOF features;
document-disjoint does not evaluate the trained reranker system;
full mode can validate against its own wrong query keys;
production artifacts can silently degrade;
public exact matching loses year/doc_type metadata.
```

Fix these, run `gpu_smoke`, then run full.
