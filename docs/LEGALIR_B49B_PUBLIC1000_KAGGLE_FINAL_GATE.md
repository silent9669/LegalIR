# LegalIR B49B — Public-1000 Kaggle Final Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use test-driven development and verification-before-completion. Preserve the existing retrieval/reranking architecture. Do not start FULL until local verification and the official-data Kaggle T4×2 `gpu_smoke` both pass.

**Goal:** Correct the remaining production mismatch between the actual Drive/Kaggle dataset and the latest pinned runtime, remove avoidable notebook memory duplication, and make the final GPU-smoke gate trustworthy.

**Architecture:** Keep the current 5-branch hybrid retrieval → DEk21 Dense → BGE reranker-v2-m3 PEFT/LoRA → cross-fitted fusion → deterministic top-5 pipeline. This plan changes only dataset identity handling, notebook preflight memory behavior, production tests/documentation, and readiness projection.

**Tech Stack:** Python, PyTorch, Transformers, PEFT, FAISS, LightGBM, pandas/Parquet, PyArrow, Kaggle T4×2.

**Release HEAD:** `b49b359dd474f9fe31ec2a7f7ec286c86451afb2`  
**Pinned runtime SHA:** `a76723ebb4ab766b4d50101ad9041ba641fab6bc`  
**Kaggle dataset source:** `phucdangg/legalir-task1-clean-data`

---

## Source-of-truth dataset evidence

The newly shared Google Drive folder is directly inspectable and resolves the last public-query-count ambiguity.

Actual folder contents include:

```text
selected-contexts/
train.json
manifest.json
public-official.json
qrels_train.parquet
queries_train.parquet
documents.parquet
chunks.parquet
audit_report.json
```

Actual `manifest.json`:

```text
dataset                 task1_canonical
version                 v2
documents               8,532
chunks                   1,153,876
micro chunks             934,416
macro chunks             219,460
train queries            7,000
qrels                    7,637
duplicate groups         4
empty documents          20
schema                   hierarchical_micro_macro_v2
normalization            nfc_whitespace_preserve_legal_ids
```

Actual `audit_report.json`:

```text
is_valid                 true
documents                8,532
chunks                   1,153,876
micro chunks             934,416
macro chunks             219,460
train queries            7,000
qrels                    7,637
empty documents          20
errors                    []
```

Direct parsing of the actual Drive files also gives:

```text
train.json               7,000 query entries
public-official.json     1,000 query entries
```

The public file contains normal query-ID keys and records shaped like:

```json
{
  "question": "...",
  "answer": null
}
```

Therefore **1,000 is the actual public-query count for the dataset the user is running**, not 999.

This is now the authoritative production input for the Kaggle notebook.

---

# What B49B/A767 fixed correctly

Preserve these changes:

- `Sequence` import is fixed.
- `DenseMacroRetriever.fit(..., stage_name=...)` is fixed.
- notebook runtime is pinned to `a76723e...`.
- notebook checkout is fail-closed instead of silently falling back to `main`.
- `psutil` is installed/preflighted.
- true `ru_maxrss` peak RSS helper exists.
- FULL hardware topology requires T4×2 with Dense=`cuda:0`, reranker=`cuda:1`.
- FULL pair coverage uses `target_coverage_pct=1.0`.
- `train_reranker()` can enforce pair-derived coverage-required steps.
- Dense corpus/train/public telemetry stages exist.
- BM25/Dense/query/final-pair/final-training timers are substantially improved.
- root and Kaggle notebook copies are byte-identical.
- kernel metadata points to `phucdangg/legalir-task1-clean-data`.
- static parameter budget remains well below 4B.

Do not regress any of these.

---

# P0 — Production hardcodes 999 public queries, but the actual file contains 1,000

This is a deterministic production failure.

Current runtime contains:

```python
if len(public_data) != 999:
    raise ValueError(...)
```

Current notebook Cell 3 also contains:

```python
print(... expected 999)
...
if len(public_data) != 999:
    raise ValueError(...)
```

The actual shared `public-official.json` has exactly:

```text
1,000
```

top-level query IDs.

Therefore the current pinned notebook will fail before/at production inference on the actual dataset.

## Required fix

Create one source of truth in production code:

```python
OFFICIAL_PUBLIC_QUERY_COUNT = 1000
```

or, preferably, derive the submission key set from the actual official file and use the constant only as an identity sanity check:

```python
official_public_qids = set(public_data.keys())

if run_mode in {"gpu_smoke", "full"}:
    if len(official_public_qids) != 1000:
        raise ValueError(...)
```

The critical submission invariant is:

```python
set(predictions.keys()) == official_public_qids
```

for FULL.

Do not manufacture, truncate, or drop one query to preserve the old 999 assumption.

## Modify all production-facing occurrences

At minimum:

```text
src/pipeline/kaggle_train.py
scripts/generate_kaggle_notebook.py
legalir_training.ipynb
kaggle_kernel_task1/legalir_training.ipynb
README.md
tests/test_595e_drive_kaggle_final_blockers.py
```

Historical repair documents may remain historical, but current operational documentation must no longer state 999.

## Mandatory RED/GREEN tests

```python
def test_actual_public_fixture_has_1000_queries(actual_public_fixture):
    assert len(actual_public_fixture) == 1000
```

```python
def test_gpu_smoke_accepts_1000_public_qids(...):
    ...
```

```python
def test_full_submission_requires_exact_1000_file_keyset(...):
    assert set(predictions) == set(public_data)
```

```python
def test_999_query_public_file_fails_current_official_identity_gate(...):
    ...
```

Use a realistic synthetic 1000-key JSON fixture if the real competition file must not be committed.

---

# P1 — Notebook Cell 3 duplicates the largest Parquet tables in host RAM

The latest notebook performs identity verification by materializing:

```python
df_docs = pd.read_parquet(DATA_DIR / "documents.parquet")
df_chunks = pd.read_parquet(DATA_DIR / "chunks.parquet")
df_queries = pd.read_parquet(DATA_DIR / "queries_train.parquet")
df_qrels = pd.read_parquet(DATA_DIR / "qrels_train.parquet")
```

Then Cell 4 calls:

```python
run_kaggle_pipeline(...)
```

which loads the same canonical DataFrames again.

Because Cell 3 variables remain in notebook global scope, the first copies can remain alive during the whole orchestrator run.

The actual Drive files are large:

```text
documents.parquet  ~313 MB compressed
chunks.parquet     ~457 MB compressed
```

The in-memory pandas representation can be much larger than compressed size.

This adds unnecessary cold-start I/O and creates avoidable host-RAM pressure before the 1.15M-chunk retrieval stack is constructed.

## Required fix

Cell 3 must be a **lightweight identity preflight**, not a second data load.

Use:

```python
manifest_data = json.loads(...)
audit_data = json.loads(...)
public_data = json.loads(...)
```

For row counts use PyArrow metadata only:

```python
import pyarrow.parquet as pq

docs_rows = pq.ParquetFile(DATA_DIR / "documents.parquet").metadata.num_rows
chunks_rows = pq.ParquetFile(DATA_DIR / "chunks.parquet").metadata.num_rows
queries_rows = pq.ParquetFile(DATA_DIR / "queries_train.parquet").metadata.num_rows
qrels_rows = pq.ParquetFile(DATA_DIR / "qrels_train.parquet").metadata.num_rows
```

Do not call `pd.read_parquet()` in notebook Cell 3.

For micro/macro counts, use the verified v2 manifest/audit values. The orchestrator will validate the real chunk table once during its normal load.

## Acceptance test

The generated notebook Cell 3 must not contain:

```text
pd.read_parquet
df_chunks =
df_docs =
```

It should contain:

```text
pyarrow.parquet
ParquetFile
metadata.num_rows
```

Add:

```text
test_notebook_identity_preflight_does_not_materialize_full_parquets
```

---

# P1 — Move strict canonical identity into one reusable production helper

Current identity logic is split between the notebook and orchestrator.

The notebook prints manifest/audit fields, but its hard gate is mostly row counts.

The orchestrator checks:

```text
documents == 8,532
train queries == 7,000
manifest version == v2
```

and calls the canonical validator, but the exact official identity is not represented by one reusable function.

This duplication caused the 999/1000 divergence.

## Required fix

Create one helper in `src/pipeline/kaggle_train.py` or a small focused module:

```python
def validate_official_task1_identity(
    *,
    data_dir: Path,
    public_json_path: Path,
    strict: bool,
) -> dict[str, Any]:
    ...
```

It must verify:

```text
manifest.dataset == task1_canonical
manifest.version == v2
manifest.schema == hierarchical_micro_macro_v2

manifest.total_documents == 8532
manifest.total_chunks == 1153876
manifest.total_micro_chunks == 934416
manifest.total_macro_chunks == 219460
manifest.total_queries == 7000
manifest.total_qrels == 7637

audit_report.is_valid == true
audit_report.errors == []
audit counts agree with manifest

public query count == 1000
public keys are unique
```

Then:

```text
notebook Cell 3 -> calls lightweight version/helper
run_kaggle_pipeline -> calls same helper
gpu_smoke_report -> stores returned identity report
```

Avoid having three different count tables in source strings.

---

# P1 — Current dataset-identity unit test is self-fulfilling and contains the wrong count

Current test constructs a local dictionary:

```python
manifest = {
    ...
    "public_queries": 999
}
assert manifest["public_queries"] == 999
```

This does not invoke production validation and now contradicts the actual Drive file.

## Required replacement

Test the actual helper:

```python
def test_official_identity_accepts_exact_v2_1000_fixture(tmp_path):
    ...
    report = validate_official_task1_identity(...)
    assert report["is_valid"]
    assert report["public_queries"] == 1000
```

Also test:

```text
wrong version
wrong schema
audit is_valid=false
audit errors nonempty
999 public keys
1001 public keys
manifest/audit disagreement
```

The test name must match the behavior it executes.

---

# P1 — Runtime projection overestimates every FULL fold as 875 steps

Current GPU-smoke projection uses:

```python
full_final_steps = 875
full_fold_steps = 875
```

for non-FULL projection.

875 is correct for a final 7,000-query full Phase A+B cycle.

It is conservative but not representative for a normal 5-fold outer-training set. A fold with approximately 5,600 training queries needs about:

```text
5,600 * 2 / 16 = 700 optimizer steps
```

Document-disjoint training may have a similar but split-specific requirement.

The current projection can therefore overstate training time by roughly 25% for each fold and may falsely mark a viable FULL run as exceeding the 10.8-hour planning budget.

## Required fix

Do not use one global fold step count.

Project fold requirements from split sizes:

```python
projected_fold_steps = [
    compute_coverage_required_steps(
        eligible_query_count=len(fold["train_query_ids"]),
        batch_size=2,
        gradient_accumulation_steps=8,
        target_coverage_pct=1.0,
        require_pos_and_neg=True,
    )
    for fold in production_folds
]
```

Prefer pair-derived eligible counts if full pair-audit statistics are available.

For the document-disjoint split, derive its own count.

`runtime_projection.json` should include:

```text
projected_fold_steps_by_fold
projected_doc_disjoint_steps
projected_final_steps
```

This is primarily a readiness-accuracy fix; do not weaken the 10.8-hour safety budget.

---

# P1 — GPU smoke report should carry the now-verified official dataset identity

The new Drive folder eliminates the previous uncertainty.

Add to `gpu_smoke_report.json`:

```json
"dataset_identity": {
  "dataset": "task1_canonical",
  "version": "v2",
  "schema": "hierarchical_micro_macro_v2",
  "documents": 8532,
  "chunks": 1153876,
  "micro_chunks": 934416,
  "macro_chunks": 219460,
  "train_queries": 7000,
  "qrels": 7637,
  "public_queries": 1000,
  "audit_valid": true,
  "audit_errors": []
}
```

This makes the GPU-smoke artifact sufficient to prove that the run used the correct dataset.

---

# P1 — Do not read the same official public file twice with conflicting logic

Current orchestrator discovers the public file near startup, then discovers/loads it again later before inference.

After adding the shared identity helper, load it once:

```python
public_test_file = discover_public_test_file(...)
public_data = load/validate once
official_public_qids = set(public_data)
```

Carry this data to inference.

This removes another place where count logic can diverge.

The file is small, so memory is not the issue; consistency is.

---

# Verification order

## 1. Static/import

```bash
python -m compileall -q src scripts
python -c "from src.training.trainer import audit_pair_coverage; print('TRAINER_IMPORT_OK')"
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline, validate_official_task1_identity; print('PIPELINE_IMPORT_OK')"
```

## 2. Targeted RED/GREEN

Run the new public-count/identity/notebook-memory tests first.

Required regressions:

```text
1000 public keys accepted
999 public keys rejected
1001 public keys rejected
notebook Cell 3 contains no full pd.read_parquet
actual identity helper validates manifest/audit/schema/counts
runtime projection uses split-specific fold steps
```

## 3. Full suite

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

Do not repeat commit-message claims as verification.

## 4. CPU smoke

```bash
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

## 5. Notebook parity

```bash
sha256sum legalir_training.ipynb kaggle_kernel_task1/legalir_training.ipynb
```

Hashes must match.

## 6. Release pin

Use the two-commit release pattern:

```text
runtime repair commit = RUNTIME_SHA
notebook pin release  = RELEASE_SHA
```

The release notebook must pin `RUNTIME_SHA` and fail closed.

---

# Official Kaggle T4×2 gate

Run:

```text
LEGALIR_RUN_MODE=gpu_smoke
```

on:

```text
phucdangg/legalir-task1-clean-data
```

Required evidence:

```text
runtime SHA == pinned runtime SHA

dataset = task1_canonical
version = v2
schema = hierarchical_micro_macro_v2
documents = 8,532
chunks = 1,153,876
micro = 934,416
macro = 219,460
train queries = 7,000
qrels = 7,637
public queries = 1,000
audit valid = true
audit errors = []

2 x NVIDIA T4
Dense actual = cuda:0
BGE/PEFT actual = cuda:1
FAISS active

Dense cold-start completes
real LoRA optimizer steps > 0
param_diff > 0
adapter checksum verified
adapter_parameters > 0
total learned parameters < 4B

peak host RSS recorded
GPU0/GPU1 peak VRAM recorded
all major stage timings recorded
split-specific FULL runtime projection < 10.8h
```

Only after this succeeds should FULL be started.

---

# Readiness policy

After this source repair but before real GPU execution, maximum conclusion:

```text
SOURCE-LEVEL READY FOR OFFICIAL KAGGLE T4×2 GPU_SMOKE
```

Only after the real official-data smoke passes:

```text
READY FOR FULL KAGGLE RUN
```

---

# Required final coding-agent report

```markdown
# LegalIR B49B Public-1000 Final Gate Report

## Base
- audited release HEAD: b49b359dd474f9fe31ec2a7f7ec286c86451afb2
- old pinned runtime SHA: a76723ebb4ab766b4d50101ad9041ba641fab6bc
- repaired runtime SHA:
- final release SHA:

## Dataset identity
- dataset:
- version:
- schema:
- documents:
- chunks:
- micro:
- macro:
- train:
- qrels:
- public:
- audit valid:
- audit errors:

## Public-query fix
- old expected count:
- actual official count:
- orchestrator:
- notebook:
- README/current docs:
- exact submission keyset validation:

## Notebook memory
- Cell 3 full parquet materialization removed:
- metadata-only row-count check:
- notebook SHA parity:

## Runtime projection
- projected fold steps:
- projected doc-disjoint steps:
- projected final steps:
- cold-start projected hours:
- planning budget:
- fits:

## Fresh verification
- compileall:
- trainer import:
- pipeline import:
- targeted tests:
- full pytest:
- CPU smoke:
- notebook parity:

## Official T4x2 gpu_smoke
- executed:
- runtime SHA:
- GPU0:
- GPU1:
- FAISS:
- peak host RSS:
- GPU0 peak VRAM:
- GPU1 peak VRAM:
- adapter params:
- total learned params:
- result:

READY FOR OFFICIAL KAGGLE GPU_SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
