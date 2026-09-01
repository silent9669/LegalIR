# LegalIR 76BB — Final Release Provenance & Colab Contract Repair Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use test-driven-development for every behavioral change and verification-before-completion before declaring the release approved.

**Goal:** make the already-green CI → Colab T4 → Kaggle FULL workflow evidence-correct: Kaggle must execute the exact Colab-approved runtime, Colab must exercise real production pair-mining/reload/reranking contracts rather than synthetic substitutes, and the final release must carry verifiable SHA provenance.

**Architecture:** preserve the current scoring architecture and production hyperparameters. Do not retune retrieval, reranking, fusion, or LoRA in this repair. The work is release provenance + smoke integrity only.

**Audited latest HEAD:** `76bb4c3a483ee7946bc5a8b19c42a0b6e5e0f2a4`  
**Live Colab-smoked runtime:** `573d36d12230b6fa27f6f0bd7bf97b81f7dbc59c`  
**Current Kaggle notebook default pin:** `2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12`  
**Current CI run:** `33517774417` — `LegalIR CI`, completed `success`, HEAD `76bb4c3...`  
**Official dataset:** Task1 canonical v2, 8,532 docs, 1,153,876 chunks, 934,416 micro, 219,460 macro, 7,000 train queries, 7,637 qrels, 1,000 public queries.

---

## Audit verdict

**DO NOT START KAGGLE FULL YET.**

The core production architecture is in good shape and GitHub CI is genuinely green. The live Tesla T4 test also proves the real DEk21 model and real BGE+LoRA can run on a T4.

However, the current release has four P0 evidence/provenance defects.

### P0-1 — Kaggle notebook is pinned to a stale runtime

The current generated Kaggle notebook and generator still contain:

```python
EXPECTED_COMMIT = os.environ.get(
    "LEGALIR_COMMIT_SHA",
    "2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12",
)
```

But the actual Colab T4 smoke report was generated from:

```text
573d36d12230b6fa27f6f0bd7bf97b81f7dbc59c
```

`573d36d...` is **19 commits ahead** of `2c1b6e8...`.

Those 19 commits are not documentation-only. They include changes to:

```text
src/pipeline/kaggle_train.py
src/retrieval/dense_macro.py
src/ranking/reranker.py
src/training/train_reranker.py
```

including a production Dense max-length clamp added specifically to prevent a CUDA assert and a production public-file discovery fix.

Therefore, running the current Kaggle notebook without overriding `LEGALIR_COMMIT_SHA` will execute source that is older than the source validated on Colab.

**This alone blocks FULL.**

---

### P0-2 — Colab pair mining no longer exercises production `build_training_pairs()`

Commit `573d36d...` removed the production pair-builder path and replaced it with direct records:

```python
neg_docs = set(sub_docs["doc_id"].astype(str)) - pos_docs
for ndoc in list(neg_docs)[:4]:
    ...
```

Problems:

1. this is not the production hard-negative miner;
2. it does not prove production BM25/candidate hard-negative behavior;
3. it does not apply the canonical duplicate-group blacklist;
4. `set` iteration is not a proper deterministic ranking/mining policy;
5. a smoke PASS therefore cannot prove the production pair-mining contract.

The Colab smoke should remain fast, but it must invoke the real pair-mining module on a smaller official-data mini-corpus.

---

### P0-3 — Colab public “prediction validation” does not run retrieval/reranking

Current Stage 5 effectively does:

```python
cand_docs = list(valid_doc_ids)[:5]
predictions[qid] = cand_docs
```

It then checks only:

```text
1–5 docs
unique docs
valid corpus IDs
```

That is a formatting test, not an inference smoke.

The report therefore shows:

```text
prediction_validation.valid = true
prediction_eval_sec = 0.0
```

without proving that the saved LoRA adapter can be reloaded and used to rerank actual candidates.

The CI CPU smoke already covers submission formatting. The T4 smoke must cover the neural runtime contract.

---

### P0-4 — Adapter “SHA verified” / duplicate blacklist claims are stronger than the implementation

Current Colab code obtains or computes an adapter SHA, but it does not load the saved adapter into a fresh production reranker and prove inference succeeds.

Current duplicate reporting does:

```python
dup_count = 4 if (dup_path and dup_path.exists()) else 4
```

which reports `4` even when no file exists.

Therefore:

```text
Adapter SHA Verified
Duplicate Blacklist Count = 4
```

must not be treated as evidence until the code actually verifies these conditions.

---

## What is already valid and must be preserved

Do not regress:

- GitHub workflow `LegalIR CI`.
- Fresh CI: 393 tests passed at current release.
- `compileall` and imports pass.
- parameter preflight: 702,754,049 / 4B.
- CPU tiny orchestration smoke passes.
- root/Kaggle notebook byte parity passes.
- Tesla T4 detected.
- CUDA/PyTorch real GPU execution works.
- DEk21 runs on `cuda:0`.
- FAISS is active.
- Dense embeddings are finite.
- BGE reranker-v2-m3 + PEFT LoRA trains on `cuda:0`.
- 10 real optimizer steps ran.
- final loss is finite.
- `param_diff > 0`.
- reranker peak VRAM is only ~6.1 GB.
- official dataset manifest/audit is valid.
- public query count is exactly 1,000.
- protected scoring configuration remains isolated from smoke limits.
- Kaggle FULL must remain Dense=`cuda:0`, reranker=`cuda:1`, T4×2.

---

# Task 1 — Establish explicit runtime/release provenance

**Files**
- Create: `src/release/provenance.py`
- Create: `tests/test_76bb_final_release_gate.py`
- Modify: `scripts/generate_kaggle_notebook.py`
- Modify: `scripts/generate_colab_smoke_notebook.py`
- Modify: `scripts/run_colab_t4_smoke.py`

## Required model

Use distinct concepts:

```text
runtime_sha
    exact source tree executed by Colab and Kaggle

release_sha
    later artifact-only commit that records approval and generated notebooks

colab_report_sha256
    hash of the exact PASS report

ci_run_id
    successful LegalIR CI run for runtime_sha
```

Never label an artifact-only release SHA as the T4-tested runtime SHA.

- [ ] Add:

```python
@dataclass(frozen=True)
class ReleaseApproval:
    runtime_sha: str
    colab_report_sha256: str
    colab_result: str
    ci_run_id: int
    ci_conclusion: str
```

- [ ] Validate all SHA strings as exact 40-hex Git SHAs.

- [ ] Add a test proving `76bb4c3...` and `573d36d...` are not silently treated as the same SHA.

- [ ] Add a release allowlist check: if `release_sha != runtime_sha`, every change after runtime approval must be in explicitly allowed release-only paths.

Recommended allowlist:

```text
artifacts/task1/colab_smoke_report.json
artifacts/task1/release_approval.json
parameter_audit.json
scripts/generate_kaggle_notebook.py
legalir_training.ipynb
kaggle_kernel_task1/legalir_training.ipynb
```

If any `src/**`, production config, requirements, or scoring file changes, Colab PASS is invalid and must be rerun.

---

# Task 2 — Restore real production pair mining while keeping Colab fast

**Files**
- Modify: `src/pipeline/colab_smoke.py`
- Modify: `configs/colab_smoke.yaml`
- Test: `tests/test_76bb_final_release_gate.py`

Do **not** restore full-corpus PyVi/BM25 construction.

Create a deterministic pair-mining mini-corpus from the already official subset.

Recommended smoke limits:

```yaml
pair_mining_max_documents: 256
pair_mining_train_queries: 32
pair_mining_max_micro_chunks: 20000
```

Selection rule:

```text
all positives for selected smoke training queries
+
deterministic seed-42 distractors
```

Then invoke the real:

```python
build_training_pairs(
    data_dir=pair_subset_dir,
    index_dir=pair_index_dir,
    output_dir=pairs_dir,
    train_query_ids=selected_pair_train_qids,
    use_all_queries=True,
    duplicate_groups_path=resolved_duplicate_groups_path,
    ...
)
```

The real miner may require a small legal-BM25 index. Build it on the bounded mini-corpus using the existing production index/retriever classes.

Required postconditions:

```text
pair_qids <= selected_pair_train_qids
pair_qids ∩ validation_qids == ∅
positive coverage > 0
negative coverage > 0
hard negatives exist
```

### Duplicate blacklist must be proven

Resolve:

```text
official input duplicate_groups.json
→ repo artifacts/task1/data/duplicate_groups.json
→ FAIL
```

Require exactly four groups.

For every training pair with label 0:

```python
negative_doc not in duplicate_closure(any_positive_doc_for_query)
```

Report:

```text
duplicate_blacklist_source
duplicate_blacklist_count
duplicate_blacklist_valid
excluded_duplicate_negative_count
```

No hard-coded `else 4`.

---

# Task 3 — Add a real adapter save/reload verification

**Files**
- Modify: `src/pipeline/colab_smoke.py`
- Test: `tests/test_76bb_final_release_gate.py`

After LoRA training:

1. locate the actual adapter weight file;
2. compute SHA-256 from bytes;
3. compare it to `training_manifest.json` when the manifest contains an expected SHA;
4. destroy the training model/references;
5. `gc.collect()` and `torch.cuda.empty_cache()`;
6. instantiate a **fresh production `CrossEncoderReranker`** using the saved adapter;
7. call its normal `ensure_loaded()` / production load path;
8. verify an active PEFT adapter exists;
9. score at least 16 real `(query, evidence)` pairs;
10. require every score to be finite.

Report:

```json
"adapter_verification": {
  "file_sha256": "...",
  "manifest_sha256": "...",
  "sha_match": true,
  "fresh_reload": true,
  "active_peft": true,
  "finite_scores": true
}
```

`result=PASS` requires every field above.

---

# Task 4 — Replace fake public predictions with real neural inference

**Files**
- Modify: `src/pipeline/colab_smoke.py`
- Test: `tests/test_76bb_final_release_gate.py`

Keep this stage efficient.

For each of the 16 selected official public queries:

1. encode/search with the saved/reloaded Dense subset index;
2. aggregate Dense chunk hits to unique candidate document IDs;
3. take bounded candidate set, e.g. up to production `rerank_k` capped by available subset candidates;
4. build evidence using production evidence construction;
5. score with the **freshly reloaded LoRA reranker**;
6. sort deterministically by reranker score, then stable doc-ID tie-break;
7. emit top 1–5 unique official doc IDs.

This is not intended to estimate competition Recall because public labels are hidden.

It is intended to prove:

```text
Dense artifact reload
query encoding
FAISS search
candidate aggregation
evidence construction
fresh adapter reload
BGE inference
deterministic ranking
top-5 validation
```

Report:

```text
prediction_pipeline = dense_faiss_plus_reloaded_bge
prediction_eval_sec > 0
public_queries_executed = 16
finite_reranker_scores = true
```

Delete the arbitrary `list(valid_doc_ids)[:5]` path.

---

# Task 5 — Make GPU telemetry evidence-based

**Files**
- Modify: `src/pipeline/colab_smoke.py`
- Test: `tests/test_76bb_final_release_gate.py`

Current report hard-codes:

```python
"dense_oom_events": 0
```

Instead read the real `DenseEncodeTelemetry`:

```text
requested batch
min successful batch
last successful batch
OOM events
item count
elapsed seconds
```

Likewise report reranker adaptive-batch/OOM telemetry if the production reranker exposes it.

PASS may allow recovered OOM, but the report must not fabricate zero.

---

# Task 6 — Unify parameter audit semantics

**Files**
- Modify: `src/pipeline/colab_smoke.py`
- Modify: `src/models/parameter_audit.py` only if necessary
- Test: `tests/test_76bb_final_release_gate.py`

Current static audit reports:

```text
702,754,049
```

while the live Colab report says:

```text
570,379,266
```

because `total_learned_params` currently comes from the reranker training result rather than the whole learned system.

Do not use the same field name for different scopes.

Report separately:

```text
dense_loaded_parameters
reranker_base_loaded_parameters
adapter_parameters
system_learned_parameters
static_preflight_parameters
```

`system_learned_parameters` must use one documented counting rule and include every learned component without double-counting the PEFT base model.

Hard gate:

```text
system_learned_parameters < 4_000_000_000
```

Also report whether loaded and static counts agree within the expected modeling definition.

---

# Task 7 — Strengthen dataset identity in the Colab report

The actual Drive dataset has been independently rechecked and is valid.

Current report only records:

```json
{"canonical_v2": true, "parent_dir": "..."}
```

Replace with:

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
  "duplicate_groups": 4,
  "audit_valid": true,
  "audit_errors": [],
  "manifest_sha256": "...",
  "public_sha256": "..."
}
```

Read this from the real files; do not hard-code `canonical_v2=True`.

---

# Task 8 — Pin Kaggle to the exact newly approved runtime

This must happen **after** the repaired Colab smoke passes.

Use a two-commit release:

```text
Commit A = runtime + smoke-contract repairs
```

Push Commit A.

Wait for:

```text
LegalIR CI = GREEN for Commit A
```

Run Colab T4 using **Commit A**.

Require repaired report `PASS`.

Then:

```text
Commit B = release-only commit
```

Commit B may:

- add the exact PASS report;
- add `release_approval.json`;
- update `parameter_audit.json`;
- update Kaggle notebook generator default `EXPECTED_COMMIT` to Commit A;
- regenerate both identical Kaggle notebooks.

It must **not** change `src/**` or protected production config.

Kaggle notebook must contain:

```python
EXPECTED_COMMIT = "<COMMIT_A_RUNTIME_SHA>"
```

and fail closed if actual Git HEAD differs.

Do not pin to Commit B unless Commit B itself is what Colab executed.

---

# Task 9 — Add a release approval artifact

Create:

```text
artifacts/task1/release_approval.json
```

Example schema:

```json
{
  "schema_version": 1,
  "runtime_sha": "<Commit A>",
  "release_sha": "<Commit B>",
  "ci": {
    "workflow": "LegalIR CI",
    "runtime_sha": "<Commit A>",
    "run_id": 0,
    "conclusion": "success"
  },
  "colab": {
    "runtime_sha": "<Commit A>",
    "report_sha256": "...",
    "result": "PASS",
    "gpu": "Tesla T4"
  },
  "dataset": {
    "version": "v2",
    "public_queries": 1000
  },
  "production": {
    "kaggle_expected_commit": "<Commit A>",
    "dual_gpu_required": true,
    "dense_device": "cuda:0",
    "reranker_device": "cuda:1"
  },
  "approved_for_kaggle_full": true
}
```

Add a validator:

```bash
python scripts/verify_release_approval.py
```

CI on Commit B must run it.

It must fail if:

```text
report runtime SHA != approved runtime SHA
Kaggle EXPECTED_COMMIT != approved runtime SHA
CI runtime SHA != approved runtime SHA
Colab result != PASS
release diff contains non-allowlisted runtime files
```

---

# Task 10 — Behavioral tests that would catch every current false-positive

**Create**
- `tests/test_76bb_final_release_gate.py`

Mandatory tests:

```text
test_kaggle_expected_commit_equals_approved_runtime_sha
test_approved_runtime_is_not_stale_2c1b6e8

test_colab_pair_mining_calls_production_build_training_pairs
test_colab_pair_mining_has_zero_validation_qids
test_duplicate_blacklist_missing_is_failure
test_duplicate_count_is_not_hardcoded
test_duplicate_equivalent_gold_never_becomes_negative

test_adapter_sha_compares_manifest_and_file
test_fresh_adapter_reload_is_required_for_pass
test_fresh_reloaded_reranker_scores_are_finite

test_colab_predictions_use_dense_search
test_colab_predictions_use_reloaded_reranker
test_arbitrary_first_five_doc_fallback_does_not_exist

test_dense_oom_telemetry_comes_from_retriever
test_system_parameter_total_includes_dense_and_reranker

test_dataset_identity_report_contains_full_official_counts

test_release_only_commit_allowlist_rejects_src_changes
test_release_approval_shas_all_match
```

Tests must execute production boundaries or mock those exact interfaces. Source-string-only assertions are not sufficient for critical gates.

---

# Fresh verification before Commit A push

```bash
python -m compileall -q src scripts

python -c "from src.pipeline.colab_smoke import run_colab_t4_smoke_pipeline; print('COLAB_IMPORT_OK')"
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('KAGGLE_IMPORT_OK')"

pytest -q tests/test_76bb_final_release_gate.py
pytest -q

python scripts/audit_parameters.py
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
python scripts/check_notebook_parity.py
```

Report exact test counts.

---

# Gate A — GitHub CI on Commit A

Required:

```text
LegalIR CI
head_sha = Commit A
status = completed
conclusion = success
```

No Colab run before this is green.

---

# Gate B — repaired live T4 Colab smoke on Commit A

Required evidence:

```text
report.runtime_sha = Commit A
CI verified green for Commit A
Tesla T4

official full dataset identity recorded
public = 1000

Dense real model on cuda:0
FAISS active
Dense telemetry real

production pair builder used
duplicate groups = 4, real file validated
zero validation leakage
duplicate negatives excluded

real BGE LoRA optimizer steps > 0
finite loss
param_diff > 0

adapter SHA file == manifest
fresh adapter reload = true
active PEFT = true
fresh reranker finite inference = true

16 public smoke queries:
real Dense search
real candidate aggregation
real reloaded BGE scoring
1–5 unique valid docs

system learned params <4B

report result = PASS
```

Target runtime should remain under ~20 minutes on T4. Use bounded official mini-corpora for pair mining rather than the full PyVi corpus.

---

# Gate C — Commit B release and CI

Commit only release artifacts/notebook pin.

Run CI again.

`verify_release_approval.py` must prove Commit B points Kaggle at Commit A and that the Colab PASS belongs to Commit A.

---

# Manual Kaggle FULL

Only after Commit B CI is green.

Run the generated notebook with:

```text
LEGALIR_RUN_MODE=full
```

The notebook must print:

```text
Approved Runtime SHA: <Commit A>
Actual Runtime SHA:   <Commit A>
Release Approval: PASS
```

Production remains:

```text
T4×2
Dense cuda:0
Reranker cuda:1
full v2 corpus
5-fold OOF
doc-disjoint
full hard-negative mining
full LoRA training
1,000 public queries
official top-5 validation
```

Do not run Kaggle's old `gpu_smoke` gate again.

---

# Score policy

Do **not** retune score-affecting parameters in this repair.

After provenance is fixed, the currently accepted score configuration remains production.

Future score changes still require:

```text
Recall@5 primary
Precision@5 tie-break
Candidate Recall@50/150 guardrails
doc-disjoint robustness guardrail
```

This repair's purpose is to ensure the Kaggle run executes the exact code actually approved by CI + T4, not to introduce a new unvalidated ranking configuration.

---

# Required final agent report

```markdown
# LegalIR Final Release Gate Report

## Runtime
- old latest release:
- old stale Kaggle pin:
- repaired runtime SHA (Commit A):
- final release SHA (Commit B):

## CI
- Commit A CI run:
- Commit A result:
- Commit B CI run:
- Commit B result:
- pytest:

## Dataset
- manifest SHA:
- version:
- documents:
- chunks:
- micro:
- macro:
- train:
- qrels:
- public:
- audit valid:

## Colab T4
- runtime SHA:
- GPU:
- wall time:
- Dense backend/device:
- Dense OOM telemetry:
- pair builder = production:
- duplicate source/count:
- validation leakage:
- optimizer steps:
- loss finite:
- param_diff:
- adapter SHA match:
- fresh reload:
- PEFT active:
- finite reload scores:
- real public neural predictions:
- adapter params:
- system learned params:
- result:

## Release approval
- colab report SHA256:
- runtime SHA parity:
- Kaggle EXPECTED_COMMIT:
- release-only diff allowlist:
- approval validator:

## Kaggle production
- FULL remains dual-T4:
- Dense device:
- reranker device:
- notebook parity:

READY FOR MANUAL KAGGLE FULL: YES/NO

## Remaining risks
...
```
