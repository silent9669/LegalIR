# LegalIR 595E — Drive/Kaggle Final Blocker Repair Plan

> **For agentic workers:** implement test-first. Preserve the current retrieval/reranking architecture. Do not start FULL until the source gate and the official-data T4×2 `gpu_smoke` both pass.

**Repository HEAD:** `595e24fd2ea5e6e3815b64f4e189f77471e9a8a4`  
**Notebook-pinned runtime SHA:** `98e3d17bba42116375b7601d816960f641ee43d6`  
**Kaggle dataset:** `phucdangg/legalir-task1-clean-data`  
**Google Drive archive:** `LegalIR-dataset.zip`  
**Decision:** **NOT READY FOR FULL**

The latest GitHub HEAD has not moved beyond `595e24fd...`. This means the two definite source blockers found in the previous audit are still present.

The supplied Google Drive file is a 747,387,708-byte ZIP. The Drive connector cannot download files larger than 256 MiB in this environment, so this audit cannot inspect the ZIP's internal `manifest.json` or `audit_report.json` directly. Treat the ZIP's exact canonical version as unverified until Kaggle prints the manifest/audit contents or the two JSON files are shared separately.

Repository canonical expectation:

```text
dataset                 task1_canonical
version                 v2
documents               8,532
chunks                   1,153,876
micro chunks             934,416
macro chunks             219,460
train queries            7,000
qrels                    7,637
schema                   hierarchical_micro_macro_v2
public queries           999
```

---

# P0-1 — `src.training.trainer` currently fails to import

Current source imports:

```python
from typing import Any, Mapping
```

but later defines:

```python
def audit_pair_coverage(
    pairs_data: pd.DataFrame | list[dict[str, Any]],
    expected_qids: Sequence[str] | set[str] | None = None,
) -> dict[str, Any]:
```

`Sequence` is not imported and there is no postponed-annotation future import. Module import can therefore fail immediately with:

```text
NameError: name 'Sequence' is not defined
```

This prevents:

```python
from src.pipeline.kaggle_train import run_kaggle_pipeline
```

because `kaggle_train.py` imports `train_reranker`, which imports this module.

## Required fix

```python
from typing import Any, Mapping, Sequence
```

No workaround; fix the missing symbol at source.

## Mandatory tests

```python
def test_trainer_module_imports():
    import importlib
    mod = importlib.import_module("src.training.trainer")
    assert callable(mod.audit_pair_coverage)
```

Then verify:

```bash
python -c "from src.training.trainer import audit_pair_coverage; print('TRAINER_IMPORT_OK')"
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('PIPELINE_IMPORT_OK')"
```

---

# P0-2 — cold-start Dense build has a caller/callee signature mismatch

Production orchestrator calls:

```python
dense_retriever.fit(
    macro_chunks.to_dict("records"),
    batch_size=dense_batch,
    stage_name="corpus",
)
```

but `DenseMacroRetriever.fit()` currently is:

```python
def fit(
    self,
    corpus: Any,
    batch_size: int = 32,
    max_length: int = 512,
):
    self.encode_corpus(
        corpus,
        batch_size=batch_size,
        max_length=max_length,
    )
    return self
```

The attached Kaggle clean dataset contains canonical data, not a prebuilt `/kaggle/working/legalir_run/indexes/dense_dek21` index. A clean `Save & Run All` therefore enters the Dense build branch and raises:

```text
TypeError:
DenseMacroRetriever.fit() got an unexpected keyword argument 'stage_name'
```

## Required fix

Make `fit()` backward-compatible:

```python
def fit(
    self,
    corpus: Any,
    batch_size: int = 32,
    max_length: int = 512,
    stage_name: str = "corpus",
):
    self.encode_corpus(
        corpus,
        batch_size=batch_size,
        max_length=max_length,
        stage_name=stage_name,
    )
    return self
```

## Mandatory behavioral test

Do not only inspect the function signature.

```python
def test_dense_fit_propagates_corpus_stage(monkeypatch):
    retriever = make_tiny_dense_retriever(monkeypatch)
    retriever.fit(
        [
            {"chunk_id": "c1", "doc_id": "d1", "text_norm": "alpha"},
            {"chunk_id": "c2", "doc_id": "d2", "text_norm": "beta"},
        ],
        batch_size=2,
        stage_name="corpus",
    )
    assert "corpus" in retriever.stage_telemetry
    assert retriever.stage_telemetry["corpus"].item_count == 2
```

Also execute the actual orchestrator Dense-build boundary with Transformer compute mocked but **without mocking `DenseMacroRetriever.fit`**.

---

# P0-3 — notebook commit pin is still fail-open

The notebook defaults to:

```text
EXPECTED_COMMIT =
98e3d17bba42116375b7601d816960f641ee43d6
```

which is better than tracking `main`.

However current bootstrap does:

```python
try:
    git fetch --all
    git checkout --detach EXPECTED_COMMIT
except Exception:
    print("... using default branch")
```

This means a checkout failure silently executes unverified source.

A second path also exists: if `possible_repo_paths` already contains a repository, the clone/checkout block is skipped and the notebook can run whatever HEAD happens to be there.

## Required fix

Before importing any `src.*` module:

```python
actual_commit = git rev-parse HEAD
```

For production:

```python
if EXPECTED_COMMIT != "main" and actual_commit != EXPECTED_COMMIT:
    raise RuntimeError(...)
```

If checkout/fetch fails:

```text
gpu_smoke -> FAIL
full      -> FAIL
```

No fallback to default branch.

## Release procedure

Use two commits:

```text
commit A = all runtime source repairs
commit B = notebook generation/pin update to commit A
```

The final notebook release commit may be newer than the runtime SHA it intentionally executes.

## Tests

```text
test_notebook_bootstrap_raises_on_checkout_failure
test_notebook_bootstrap_raises_on_sha_mismatch
test_existing_repo_path_is_also_sha_verified
test_generated_notebook_contains_no_default_branch_fallback
```

---

# P1-1 — FULL coverage policy is still internally inconsistent

`train_reranker()` computes a complete Phase A+B requirement with:

```python
target_coverage_pct=1.0
```

For 7,000 eligible queries, batch 2, grad accumulation 8:

```text
14,000 rows / 16 rows per optimizer step = 875 steps
```

But FULL preflight currently computes:

```python
target_coverage_pct=0.99
```

which gives:

```text
7,000 positive rows
+ 6,930 negative rows
= 13,930 rows
=> 871 steps
```

The orchestrator explicitly passes the preflight count into final training. When `max_steps` is explicitly supplied, `train_reranker()` currently accepts that lower value instead of `max(requested, pair_derived_required)`.

This may satisfy a >=99% gate, but it does **not** satisfy the commit/documentation claim that the complete 875-step Phase A+B cycle is enforced.

For a competition run, four extra optimizer steps are negligible. Prefer a single unambiguous invariant.

## Required policy

For FULL:

```python
target_coverage_pct = 1.0
```

and:

```python
effective_steps = max(
    configured_steps,
    requested_steps or 0,
    pair_derived_coverage_required_steps,
)
```

For intentionally abbreviated `gpu_smoke`, use an explicit:

```python
enforce_full_coverage_steps=False
```

rather than overloading `max_steps`.

Apply the same actual-pair-derived rule to:

```text
each OOF fold
document-disjoint training
final training
```

## Required tests

```text
test_7000_full_cycle_requires_875_steps
test_explicit_max_steps_cannot_undercut_full_pair_coverage
test_fold_steps_derive_from_fold_eligible_pairs
test_doc_disjoint_steps_derive_from_eligible_pairs
```

---

# P1-2 — runtime projection still needs evidence-grade stage timing

The current orchestrator initializes values such as:

```text
train_query_enc_time = 0.1
final_pair_mining_time = 0.1
```

but the visible train-query encoding block does not assign a measured elapsed time.

Current cold-start projection also uses a broad setup allowance while not clearly summing all cold-start stages.

The Kaggle dataset is canonical data only; `/kaggle/working` starts clean. FULL cold-start time must therefore include index construction.

## Required timings

Measure with `time.perf_counter()`:

```text
canonical load + validation
Legal BM25 build/load
PyVi BM25 build/load
Dense build/load
train query encoding
OOF pair mining
OOF reranker training
OOF inference
fusion training
doc-disjoint pair mining/training/inference
full question-memory build
final pair mining
final reranker training
final pipeline load/audit
public inference
submission validation/package
```

Each cacheable stage must report:

```json
{
  "seconds": 0.0,
  "cache_hit": false
}
```

## Projection rule

`cold_start_total_seconds` must be an explicit sum of the stages it claims to include.

Continue using a conservative planning ceiling:

```text
Kaggle nominal GPU session = 12 h
planning safety factor      = 0.90
FULL planning budget        = 10.8 h
```

Do not mark FULL-ready if cold-start projection exceeds that budget.

---

# P1-3 — RAM telemetry can silently be invalid

`get_process_rss_mb()` uses `psutil`, but `psutil` is not currently listed in `requirements.txt`. On import failure it silently returns:

```text
0.0 MB
```

That makes the memory gate meaningless.

The current report field named `"peak"` is also only a current-RSS snapshot after later pipeline work, not a true historical process peak.

## Required fix

Add:

```text
psutil>=5.9.0
```

to requirements and Kaggle notebook dependency preflight.

For true peak RSS on Linux:

```python
import resource

def get_peak_process_rss_mb() -> float:
    return resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024.0
```

Export both:

```text
current RSS snapshots
true peak RSS
```

---

# P1-4 — verify the exact Drive ZIP dataset version before score benchmarking

The supplied Drive archive is too large for this connector to unpack directly.

Therefore do **not** assume it contains the repository's v2 canonical manifest solely because the file name matches.

At Kaggle Cell 3, print and validate:

```python
DATA_DIR
manifest.json
audit_report.json
len(documents)
len(chunks)
len(queries_train)
len(qrels_train)
len(public_official)
```

Required repository v2 identity:

```text
version           v2
documents         8,532
chunks            1,153,876
micro chunks      934,416
macro chunks      219,460
queries           7,000
qrels             7,637
public            999
schema            hierarchical_micro_macro_v2
audit is_valid    true
audit errors      []
```

If the attached dataset reports a different canonical version or chunk count, do not silently continue and do not compare its scores directly against the repository v2 benchmark.

Prefer failing `gpu_smoke` with a clear identity mismatch.

---

# P1-5 — add a real Kaggle-layout cold-start regression

Model the exact attached structure:

```text
/kaggle/input/legalir-task1-clean-data/
  selected-contexts/
  audit_report.json
  chunks.parquet
  documents.parquet
  manifest.json
  public-official.json
  qrels_train.parquet
  queries_train.parquet
  train.json
```

Required behavior:

```text
discover_data_dir -> flat dataset root
discover_public_test_file -> same root/public-official.json
production identity -> PASS only for exact official counts
cold Dense build -> reaches fit successfully
```

Mock expensive neural kernels, not the orchestration interfaces under test.

---

# Fresh local verification order

Run exactly:

```bash
python -m compileall -q src scripts
python -c "from src.training.trainer import audit_pair_coverage; print('TRAINER_IMPORT_OK')"
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('PIPELINE_IMPORT_OK')"
pytest -q tests/test_595e_drive_kaggle_final_blockers.py
pytest -q
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
```

Then notebook parity:

```bash
sha256sum \
  legalir_training.ipynb \
  kaggle_kernel_task1/legalir_training.ipynb
```

Both hashes must match.

Fresh verification output, not commit-message prose, is the evidence.

---

# Official T4×2 `gpu_smoke` gate

Run after local fixes only:

```text
LEGALIR_RUN_MODE=gpu_smoke
```

Required evidence:

```text
runtime SHA == notebook EXPECTED_COMMIT
dataset source == phucdangg/legalir-task1-clean-data

manifest version/schema/counts verified
audit_report valid
8,532 docs
1,153,876 chunks
7,000 train queries
7,637 qrels
999 public queries

2 × NVIDIA T4
Dense requested/actual cuda:0
BGE requested/actual cuda:1
FAISS active

Dense cold-start succeeds
real LoRA optimizer steps > 0
param_diff > 0
adapter checksum verified
adapter_parameters > 0
total learned parameters <4B

GPU peak VRAM recorded
true peak host RSS recorded
stage timings recorded
cold-start FULL projection <10.8 h
```

Only after this run passes may the next notebook version switch to:

```text
LEGALIR_RUN_MODE=full
```

---

# Readiness policy

Before a real Kaggle GPU smoke, the strongest allowed conclusion is:

```text
SOURCE-LEVEL READY FOR OFFICIAL T4×2 GPU_SMOKE
```

Do not report:

```text
READY FOR FULL KAGGLE RUN
```

without actual T4×2 execution evidence.

---

# Required final agent report

```markdown
# LegalIR 595E Drive/Kaggle Repair Report

## Base
- audited release HEAD: 595e24fd2ea5e6e3815b64f4e189f77471e9a8a4
- old pinned runtime SHA: 98e3d17bba42116375b7601d816960f641ee43d6
- repaired runtime SHA:
- final notebook release SHA:

## P0 gates
- Sequence import fixed:
- pipeline import:
- Dense fit stage_name fixed:
- cold-start Dense boundary:
- notebook pin fail-closed:

## Dataset identity
- canonical root:
- manifest version:
- schema:
- documents:
- chunks:
- micro:
- macro:
- train queries:
- qrels:
- public:
- audit valid:

## Coverage
- final eligible queries:
- final required steps:
- final effective steps:
- unique coverage:
- positive coverage:
- negative coverage:
- OOF fold coverage steps:
- doc-disjoint coverage steps:

## Runtime / memory
- BM25 Legal:
- BM25 PyVi:
- Dense:
- train query encoding:
- OOF:
- doc-disjoint:
- final pair mining:
- final training:
- public inference:
- peak host RSS:
- GPU0 peak VRAM:
- GPU1 peak VRAM:
- cold-start projected hours:
- 10.8h budget fit:

## PEFT / compliance
- optimizer steps:
- param_diff:
- adapter SHA verified:
- adapter params:
- total learned params:
- <4B:

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
- result:

READY FOR OFFICIAL KAGGLE GPU_SMOKE: YES/NO
READY FOR FULL KAGGLE RUN: YES/NO

## Remaining risks
...
```
