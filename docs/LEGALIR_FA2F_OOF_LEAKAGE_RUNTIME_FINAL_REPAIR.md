# LegalIR FA2F OOF Leakage/Runtime Final Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use test-driven-development during implementation and verification-before-completion before claiming success.

**Goal:** eliminate the remaining OOF validation-label leakage on the real split-less Kaggle input, restore the duplicate-document false-negative blacklist, correct smoke→FULL runtime projection, and make the T4×2 gate trustworthy.

**Architecture:** preserve the existing 5-branch hybrid retrieval → DEk21 Dense → BGE reranker-v2-m3 PEFT/LoRA → OOF fusion → deterministic top-5 system. Do not change models or ranking architecture in this repair.

**Tech Stack:** Python, pandas/PyArrow, PyTorch, Transformers, PEFT, FAISS, LightGBM, Kaggle T4×2.

**Spec:** `docs/LEGALIR_B49B_PUBLIC1000_KAGGLE_FINAL_GATE.md`

## Global constraints

- Audited release HEAD: `fa2f2da482607b22ca8561520a6fbd430ea3ab9b`
- Pinned runtime SHA: `1817e712542871469851dffd63637e91fa417af4`
- Kaggle dataset: `phucdangg/legalir-task1-clean-data`
- Canonical v2: 8,532 docs; 1,153,876 chunks; 934,416 micro; 219,460 macro; 7,000 train queries; 7,637 qrels
- Public queries: exactly 1,000
- Learned parameters: strictly `<4,000,000,000`
- Production: Dense=`cuda:0`, reranker=`cuda:1`, FAISS required
- Real official-data `gpu_smoke` is mandatory before `full`

---

## Audit verdict

**NOT READY FOR OFFICIAL GPU SMOKE OR FULL YET.**

The previous B49B Public-1000 repair is now substantially present: the runtime accepts 1,000 public queries, notebook bootstrap is fail-closed, notebook Cell 3 uses Parquet metadata rather than fully materializing the largest Parquets, `Sequence` and Dense `stage_name` defects are fixed, `psutil` exists, and notebook copies are byte-identical.

The remaining P0 is a fold-isolation defect reachable on the actual Kaggle layout. The clean input does not visibly contain `splits/` or `duplicate_groups.json`. The repository does contain the deterministic derived artifacts under `artifacts/task1/data/`.

---

## Task 1 — P0: bind OOF pair mining to the authoritative fold train IDs

**Files**
- Modify: `src/pipeline/oof_runner.py`
- Test: `tests/test_fa2f_oof_leakage_runtime_gate.py`

### Root cause

`OOFRunner` knows `train_ids` and `val_ids`, but the fold pair builder is called by `fold=f_idx` without explicit `train_query_ids`.

`build_training_pairs()` independently searches:

```text
data_dir/splits/random_5fold.json
data_dir/random_5fold.json
```

and if neither exists it falls back to **all train queries**.

On the user's split-less Kaggle input, the fold reranker can therefore train on held-out validation queries and their gold labels.

### Required implementation

- [ ] Write a RED test with a canonical fixture that intentionally has no `splits/`.
- [ ] Capture `build_training_pairs()` arguments and require exact fold train QIDs.
- [ ] Change the fold call to:

```python
_, pairs_df = build_training_pairs(
    data_dir=self.data_dir,
    index_dir=self.index_dir,
    output_dir=pairs_dir,
    fold=f_idx,
    train_query_ids=sorted(train_ids),
    use_all_queries=False,
    limit=self.smoke_sample_size if self.smoke else None,
    query_embeddings=self.train_query_embeddings,
    duplicate_groups_path=self.duplicate_groups_path,
)
```

- [ ] Immediately assert:

```python
pair_qids = set(pairs_df["query_id"].astype(str))
train_set = set(map(str, train_ids))
val_set = set(map(str, val_ids))

unknown = pair_qids - train_set
leaked = pair_qids & val_set
if unknown or leaked:
    raise AssertionError(
        f"Fold {f_idx} pair isolation failed: "
        f"unknown={sorted(unknown)[:10]}, leaked={sorted(leaked)[:10]}"
    )
```

- [ ] Export per fold:

```text
training_query_count
pair_query_count
validation_query_count
pair_unknown_train_count
pair_validation_leakage_count
pair_validation_leakage_ids
```

Required in production:

```text
pair_unknown_train_count = 0
pair_validation_leakage_count = 0
```

### Required tests

```text
test_oof_pair_builder_receives_exact_fold_train_ids_without_input_splits
test_oof_pair_qids_are_subset_of_train_ids
test_oof_pair_qids_have_zero_validation_overlap
```

---

## Task 2 — deterministic split provenance

**Files**
- Modify: `src/pipeline/kaggle_train.py`
- Modify: `src/pipeline/oof_runner.py`
- Test: `tests/test_fa2f_oof_leakage_runtime_gate.py`

Add:

```python
@dataclass(frozen=True)
class SplitArtifactResolution:
    random_5fold_path: Path
    doc_disjoint_path: Path
    random_source: str
    doc_disjoint_source: str
```

Add a resolver with this exact order:

```text
1. canonical_data_dir/splits/<file>.json
2. repo_root/artifacts/task1/data/splits/<file>.json
3. generate deterministic seed=42 files in working_dir/splits/
```

Generated artifacts must pass:

```python
verify_fold_isolation(...)
verify_document_disjoint_isolation(...)
```

Pass absolute resolved paths into `OOFRunner`; never rely on ambient CWD.

Record SHA-256 and source in CV/GPU-smoke reports:

```json
"split_provenance": {
  "random_5fold": {"source": "input|repo|generated", "sha256": "..."},
  "doc_disjoint": {"source": "input|repo|generated", "sha256": "..."}
}
```

### Required tests

```text
test_split_resolver_prefers_input
test_split_resolver_uses_repo_artifact_when_input_has_no_splits
test_split_resolver_generates_seed42_working_split
test_split_sha_is_reported
```

---

## Task 3 — restore the canonical four-group duplicate blacklist

**Files**
- Modify: `src/training/build_pairs.py`
- Modify: `src/pipeline/oof_runner.py`
- Modify: `src/pipeline/kaggle_train.py`
- Test: `tests/test_fa2f_oof_leakage_runtime_gate.py`

Current pair builder only checks `data_dir/duplicate_groups.json` and `data_dir/splits/duplicate_groups.json`, then silently uses `{}`.

The repository canonical artifact has four groups:

```text
[158189, 184972, 206810]
[254937, 280171]
[277743, 35337]
[121575, 84226]
```

Add:

```python
duplicate_groups_path: str | Path | None = None
```

to `build_training_pairs()`.

Production resolution order:

```text
canonical_data_dir/duplicate_groups.json
canonical_data_dir/splits/duplicate_groups.json
repo_root/artifacts/task1/data/duplicate_groups.json
```

For `gpu_smoke`/`full`, missing metadata is fatal.

Validate:

```text
group count == 4
every group has >=2 IDs
every ID exists in official corpus
```

Pass the same resolved file into OOF fold mining, document-disjoint mining, and final all-query pair mining.

Pair manifests must include:

```text
duplicate_groups_source
duplicate_groups_count
duplicate_doc_ids_count
excluded_duplicate_cases_count
```

### Required tests

```text
test_duplicate_blacklist_uses_repo_artifact_when_input_lacks_it
test_production_requires_four_duplicate_groups
test_duplicate_ids_must_exist_in_corpus
test_duplicate_of_gold_is_never_mined_as_negative
```

---

## Task 4 — scale pair-mining runtime from smoke query counts to FULL

**Files**
- Modify: `src/training/build_pairs.py`
- Modify: `src/pipeline/oof_runner.py`
- Modify: `src/pipeline/kaggle_train.py`
- Test: `tests/test_fa2f_oof_leakage_runtime_gate.py`

Instrument each pair-builder call with:

```text
setup_seconds
query_loop_seconds
queries_attempted
queries_with_positive_pairs
total_seconds
seconds_per_attempted_query
```

Add:

```python
def project_pair_mining_seconds(
    *,
    setup_seconds: float,
    query_loop_seconds: float,
    measured_queries: int,
    target_queries: int,
    safety_factor: float = 1.10,
) -> float:
    per_query = query_loop_seconds / max(1, measured_queries)
    return setup_seconds + per_query * target_queries * safety_factor
```

Project separately for:

```text
each of 5 production OOF fold train sets
document-disjoint train set
final all-7,000-query pair mining
```

Do not add the 20-query/100-query smoke mining duration unchanged as a FULL estimate.

### Required tests

```text
test_pair_projection_scales_query_loop_not_setup
test_20_query_smoke_scales_to_fold_train_count
test_100_query_final_smoke_scales_to_7000
```

---

## Task 5 — make the cold-start projection component-complete

**File**
- Modify: `src/pipeline/kaggle_train.py`
- Test: `tests/test_fa2f_oof_leakage_runtime_gate.py`

The runtime measures fusion, question-memory build, and final pair mining, but the current cold-start sum does not explicitly include all those stages.

Build one named map:

```python
projection_components = {
    "canonical_load": canonical_load_time,
    "bm25_legal": bm25_legal_time,
    "bm25_pyvi": bm25_pyvi_time,
    "dense_index": dense_build_time,
    "train_query_encoding": train_query_enc_time,
    "projected_oof": projected_5fold_oof_sec,
    "fusion_training": fusion_time,
    "projected_doc_disjoint": projected_doc_disjoint_sec,
    "question_memory": qm_time,
    "projected_final_pair_mining": projected_final_pair_mining_sec,
    "projected_final_reranker": projected_final_training_sec,
    "final_pipeline_load_audit": pipe_load_audit_time,
    "projected_public_inference": projected_public_infer_sec,
    "submission_packaging": pkg_time,
    "safety_overhead": 60.0,
}
cold_start_total_sec = sum(projection_components.values())
```

Do not double-count fold/doc-disjoint pair mining if already included in those projected components.

Export:

```text
projection_components_seconds
cold_start_total_seconds
cold_start_total_hours
production_runtime_budget_hours
fits_kaggle_session_limit
```

Keep the safety budget:

```text
12h * 0.90 = 10.8h
```

### Required tests

```text
test_runtime_projection_total_equals_component_sum
test_projection_contains_fusion_memory_and_final_pair_mining
```

Tests must call production helpers, not reproduce independent arithmetic.

---

## Exact-layout regression fixture

Model exactly:

```text
legalir-task1-clean-data/
  audit_report.json
  chunks.parquet
  documents.parquet
  manifest.json
  public-official.json
  qrels_train.parquet
  queries_train.parquet
  train.json
  selected-contexts/
```

Intentionally omit:

```text
splits/
duplicate_groups.json
```

The regression must still prove:

```text
deterministic split source resolves
OOF pairs are train-only
validation overlap = 0
repo duplicate blacklist is active
public count = 1000
```

---

## Mandatory fresh verification

Run:

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('PIPELINE_IMPORT_OK')"
python -c "from src.pipeline.oof_runner import OOFRunner; print('OOF_IMPORT_OK')"
python -c "from src.training.build_pairs import build_training_pairs; print('PAIR_IMPORT_OK')"

pytest -q tests/test_fa2f_oof_leakage_runtime_gate.py
pytest -q

python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
python scripts/audit_parameters.py

sha256sum legalir_training.ipynb kaggle_kernel_task1/legalir_training.ipynb
```

Report exact passed/failed/skipped counts and durations. Commit messages are not verification evidence.

---

## Release procedure

Use two commits:

```text
Commit A = runtime fixes
Commit B = regenerate both notebooks and pin them to Commit A
```

Keep notebook default:

```text
LEGALIR_RUN_MODE=gpu_smoke
```

---

## Official Kaggle T4×2 gate

Run on `phucdangg/legalir-task1-clean-data`.

Required proof before FULL:

```text
runtime SHA == pinned SHA
official v2 identity passes
public = 1000

random split source + SHA recorded
doc-disjoint source + SHA recorded

all OOF folds:
  pair_validation_leakage_count = 0
  pair_unknown_train_count = 0

duplicate groups:
  source recorded
  group count = 4
  all IDs corpus-valid
  blacklist active

2 x NVIDIA T4
Dense actual = cuda:0
reranker actual = cuda:1
FAISS active

optimizer steps > 0
param_diff > 0
adapter checksum verified
adapter_parameters > 0
total learned parameters <4B

peak host RSS recorded
GPU0/GPU1 peak VRAM recorded

pair-mining smoke→FULL scaling recorded
fusion + question memory + final pair mining included
cold-start projection <10.8h
```

Only after this passes may `LEGALIR_RUN_MODE=full` be used.

---

## Required final agent report

```markdown
# LegalIR FA2F OOF Leakage/Runtime Repair Report

## Base
- audited release HEAD: fa2f2da482607b22ca8561520a6fbd430ea3ab9b
- audited runtime SHA: 1817e712542871469851dffd63637e91fa417af4
- repaired runtime SHA:
- final release SHA:

## OOF isolation
- random split source:
- random split SHA:
- folds:
- max pair validation leakage:
- max unknown pair QIDs:
- all pair QIDs subset train IDs:

## Duplicate blacklist
- source:
- group count:
- invalid IDs:
- excluded duplicate negatives:

## Runtime projection
- measured pair queries:
- target query counts:
- projected OOF pair mining:
- projected doc-disjoint pair mining:
- projected final pair mining:
- fusion:
- question memory:
- cold-start hours:
- 10.8h budget fit:

## Fresh verification
- compileall:
- imports:
- targeted tests:
- full pytest:
- CPU smoke:
- parameter audit:
- notebook parity:

## Official T4x2 gpu_smoke
- executed:
- result:

READY FOR OFFICIAL KAGGLE GPU_SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
