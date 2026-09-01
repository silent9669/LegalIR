# LegalIR CI → Colab T4 → Kaggle FULL Verification Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** replace the expensive Kaggle GPU-smoke gate with mandatory green GitHub CI followed by a real single-T4 Colab contract smoke, while keeping Kaggle T4×2 as the only FULL production run and preserving the best-known scoring configuration.

**Architecture:** add a CPU-only GitHub Actions gate, introduce a `colab_smoke` execution profile that runs the same production modules on a deterministic official-data subset using one T4 sequentially, and keep `full` dual-T4 semantics unchanged. Score-affecting configuration is protected from smoke overrides and remains governed by leakage-safe OOF promotion.

**Tech Stack:** GitHub Actions, Python 3.12, pytest, PyTorch, Transformers, PEFT/LoRA, FAISS, LightGBM, Google Colab T4, Kaggle T4×2.

**Spec:** `LEGALIR_CI_COLAB_KAGGLE_ARCHITECTURE_SPEC.md`

## Global Constraints

- Baseline release: `377630cc169e080caaca0395c9822844066c05b9`.
- Official dataset is Task 1 canonical v2: 8,532 docs, 1,153,876 chunks, 7,000 train queries, 7,637 qrels, 1,000 public queries.
- Learned parameter budget is strictly `<4B`.
- GitHub CI is CPU/source-only and must not download pretrained models.
- Colab readiness hardware is one NVIDIA T4 using `cuda:0` sequentially.
- Kaggle FULL remains exactly Dense=`cuda:0`, reranker=`cuda:1`, requiring two CUDA GPUs.
- Colab smoke must use production modules and real model weights; no stub neural model qualifies.
- Smoke-only overrides may change limits/devices/telemetry but not score-affecting production settings.
- Every changed commit must re-pass CI and Colab before Kaggle FULL.
- Seed = 42.

---

## File map

Create:

```text
.github/workflows/ci.yml
configs/colab_smoke.yaml
scripts/check_notebook_parity.py
scripts/verify_github_ci.py
scripts/build_colab_smoke_subset.py
scripts/run_colab_t4_smoke.py
scripts/generate_colab_smoke_notebook.py
colab/legalir_t4_smoke.ipynb
src/pipeline/colab_smoke.py
tests/test_ci_colab_architecture.py
docs/CI_COLAB_KAGGLE_WORKFLOW.md
```

Modify:

```text
src/pipeline/kaggle_train.py
src/pipeline/oof_runner.py
src/training/train_reranker.py
src/retrieval/dense_macro.py
README.md
```

Do not change the production Kaggle notebook to single-GPU behavior.

---

### Task 1: Add the GitHub CI correctness gate

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `scripts/check_notebook_parity.py`
- Test: `tests/test_ci_colab_architecture.py`

**Interfaces:**
- Produces GitHub workflow named `LegalIR CI`.
- Produces `check_notebook_parity.main() -> int`.

- [ ] **Step 1: Write tests for required CI structure**

Parse `.github/workflows/ci.yml` as YAML and assert:

```text
name == LegalIR CI
push trigger exists
pull_request trigger exists
workflow_dispatch exists
ubuntu-latest
Python 3.12
pip caching configured
timeout-minutes <= 35
compileall command
pipeline/oof/pair import checks
pytest
audit_parameters.py
CPU tiny smoke
notebook parity check
```

Also assert the workflow does not reference `HF_TOKEN`.

- [ ] **Step 2: Run the test and verify RED**

```bash
pytest -q tests/test_ci_colab_architecture.py -k "ci_workflow"
```

Expected: FAIL because `.github/workflows/ci.yml` does not exist.

- [ ] **Step 3: Create `scripts/check_notebook_parity.py`**

Implement SHA-256 comparison for:

```text
legalir_training.ipynb
kaggle_kernel_task1/legalir_training.ipynb
```

Exit 0 only when identical.

- [ ] **Step 4: Create `.github/workflows/ci.yml`**

Use this structure:

```yaml
name: LegalIR CI

on:
  push:
    branches: ["main"]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: legalir-ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 35
    env:
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      TOKENIZERS_PARALLELISM: "false"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"
          cache-dependency-path: requirements.txt
      - run: python -m pip install --upgrade pip
      - run: pip install -r requirements.txt
      - run: python -m compileall -q src scripts
      - run: python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('PIPELINE_IMPORT_OK')"
      - run: python -c "from src.pipeline.oof_runner import OOFRunner; print('OOF_IMPORT_OK')"
      - run: python -c "from src.training.build_pairs import build_training_pairs; print('PAIR_IMPORT_OK')"
      - run: pytest -q
      - run: python scripts/audit_parameters.py
      - run: python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
      - run: python scripts/check_notebook_parity.py
```

Do not cache model weights or secrets.

- [ ] **Step 5: Run CI architecture tests GREEN**

```bash
pytest -q tests/test_ci_colab_architecture.py -k "ci_workflow or notebook_parity"
```

- [ ] **Step 6: Run the whole local source gate**

```bash
python -m compileall -q src scripts
pytest -q
python scripts/audit_parameters.py
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
python scripts/check_notebook_parity.py
```

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/ci.yml scripts/check_notebook_parity.py tests/test_ci_colab_architecture.py
git commit -m "ci: add mandatory LegalIR correctness gate"
```

---

### Task 2: Add CI-status verification for Colab

**Files:**
- Create: `scripts/verify_github_ci.py`
- Test: `tests/test_ci_colab_architecture.py`

**Interfaces:**

```text
python scripts/verify_github_ci.py --repo silent9669/LegalIR --sha <40-char SHA>
```

Optional environment: `GITHUB_TOKEN`.

Exit 0 only if workflow `LegalIR CI` has a completed successful run for exactly that SHA.

- [ ] **Step 1: Write mocked API tests**

Cover:

```text
green completed workflow -> exit 0
failed workflow -> nonzero
in_progress -> nonzero
no matching workflow -> nonzero
SHA mismatch -> nonzero
403/rate limit -> explain GITHUB_TOKEN and fail closed
```

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_ci_colab_architecture.py -k "verify_github_ci"
```

- [ ] **Step 3: Implement with `urllib.request`**

Use GitHub REST:

```text
GET /repos/{owner}/{repo}/actions/runs?head_sha={sha}
```

Find a workflow run whose `name == "LegalIR CI"`, `head_sha == sha`, `status == completed`, `conclusion == success`.

Never accept a green run from a different commit.

- [ ] **Step 4: GREEN tests**

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_github_ci.py tests/test_ci_colab_architecture.py
git commit -m "feat(colab): require green CI for target SHA"
```

---

### Task 3: Protect production score configuration from smoke overrides

**Files:**
- Create: `configs/colab_smoke.yaml`
- Create: `src/pipeline/colab_smoke.py`
- Test: `tests/test_ci_colab_architecture.py`

**Interfaces:**
- `PROTECTED_SCORE_KEYS: tuple[str, ...]`
- `validate_smoke_overrides(production: Mapping, smoke: Mapping) -> None`

Protected categories must include:

```text
candidate retrieval/RRF branch weights
candidate_k except subset-cardinality runtime cap
rerank_k except min(production, available candidates)
reranker model name
LoRA rank/alpha/dropout
loss type
learning rate
fusion feature columns
fusion selection policy
top-k output logic
```

Allowed smoke overrides:

```text
sample query/doc counts
fold count
optimizer step cap
single-GPU device mapping
output/work directories
telemetry
```

- [ ] **Step 1: RED test that score-key overrides are rejected**

Test changes to RRF weight, loss type, LoRA rank, fusion features.

- [ ] **Step 2: RED test allowed bounds**

- [ ] **Step 3: Implement protected-key validation**

- [ ] **Step 4: Create `configs/colab_smoke.yaml`**

```yaml
seed: 42
train_queries: 64
validation_queries: 32
public_queries: 16
max_documents: 2000
folds: 2
reranker_optimizer_steps: 10
dense_batch_size: 16
reranker_batch_size: 8
```

Do not duplicate production model names or ranking weights here.

- [ ] **Step 5: GREEN tests**

- [ ] **Step 6: Commit**

```bash
git add configs/colab_smoke.yaml src/pipeline/colab_smoke.py tests/test_ci_colab_architecture.py
git commit -m "feat(colab): isolate smoke limits from score config"
```

---

### Task 4: Build a deterministic official-data smoke subset

**Files:**
- Create: `scripts/build_colab_smoke_subset.py`
- Extend: `src/pipeline/colab_smoke.py`
- Test: `tests/test_ci_colab_architecture.py`

**Interfaces:**
- `build_colab_subset(data_dir: Path, out_dir: Path, config: ColabSmokeConfig) -> ColabSubsetManifest`
- Output canonical-format Parquets plus `subset_manifest.json`.

- [ ] **Step 1: Write fixture tests**

Require:

```text
same seed -> same QIDs/docs
selected train QIDs come from official train set
all qrel-positive docs for selected queries included
distractor docs corpus-valid
subset chunks belong only to selected docs
no synthetic qrels
validation QIDs never enter smoke pair-training QIDs
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement deterministic selection**

Selection policy:

1. sample train IDs deterministically across at least two official folds;
2. include every positive doc from selected qrels;
3. add deterministic corpus distractors until `max_documents`;
4. include all macro chunks for selected docs;
5. include selected-doc micro chunks, capped deterministically only if memory requires;
6. create subset qrels without changing labels.

- [ ] **Step 4: Write `subset_manifest.json`**

Include parent dataset identity, source SHA/file metadata, selected QIDs/doc IDs, seed and counts.

- [ ] **Step 5: GREEN tests**

- [ ] **Step 6: Commit**

```bash
git add scripts/build_colab_smoke_subset.py src/pipeline/colab_smoke.py tests/test_ci_colab_architecture.py
git commit -m "feat(colab): build deterministic official-data smoke subset"
```

---

### Task 5: Implement the real single-T4 Colab smoke runner

**Files:**
- Create: `scripts/run_colab_t4_smoke.py`
- Extend: `src/pipeline/colab_smoke.py`
- Modify production components only where reusable interfaces are needed.
- Test: `tests/test_ci_colab_architecture.py`

**Interfaces:**

```bash
python scripts/run_colab_t4_smoke.py \
  --data-dir <official-v2-dir> \
  --work-dir <output-dir> \
  --target-sha <sha>
```

Output: `colab_smoke_report.json`.

Exit nonzero on any readiness failure.

- [ ] **Step 1: Write hardware contract tests with mocked CUDA**

Require:

```text
no CUDA -> fail
T4 -> readiness-capable
L4/A100 without override -> fail readiness
explicit non-T4 override -> execute but result NOT_A_T4_READINESS_GATE
```

- [ ] **Step 2: Write report-schema tests**

- [ ] **Step 3: Implement preflight**

Order:

```text
verify target git SHA
verify GREEN GitHub CI
verify HF_TOKEN exists without printing it
verify official v2 identity
verify T4
build deterministic subset
```

- [ ] **Step 4: Exercise Dense on the real T4**

Use production DEk21 code/tokenizer.

Require:

```text
cuda:0 actual placement
FP16/autocast production path
finite embeddings
FAISS active
real search returns corpus-valid docs
OOM telemetry
peak allocated/reserved VRAM
```

Persist subset FAISS/embedding artifacts.

- [ ] **Step 5: Free Dense before BGE**

```python
del dense_model_or_retriever_model
gc.collect()
torch.cuda.empty_cache()
```

Record VRAM before and after cleanup.

- [ ] **Step 6: Exercise real BGE+LoRA training on cuda:0**

Use real mined official-subset pairs.

Require:

```text
optimizer_steps >= 1
all losses finite
param_diff > 0
adapter saved
adapter SHA-256 computed
adapter reload succeeds
adapter_parameters > 0
total learned params <4B
```

- [ ] **Step 7: Exercise reranking + artifact reload**

Rerank held-out subset queries. Validate finite scores and stable output count/order.

- [ ] **Step 8: Exercise prediction contract**

On 16 official public queries, validate:

```text
keys exactly equal selected public keys
1..5 unique doc IDs/query
all doc IDs official-corpus-valid
```

This is a format/contract smoke, not a score estimate.

- [ ] **Step 9: Emit `colab_smoke_report.json`**

Required fields:

```text
git_sha
ci_green
gpu_name
CUDA/PyTorch versions
dataset identity
subset manifest hash
split provenance/SHA
duplicate blacklist source/count
Dense actual device/backend/VRAM/OOM
reranker actual device/VRAM/OOM
optimizer steps
finite loss
param_diff
adapter SHA verified
adapter params
total learned params
prediction validation
stage timings
result
```

- [ ] **Step 10: Component tests GREEN**

Pytest mocks expensive kernels; real kernels are validated in Colab.

- [ ] **Step 11: Commit**

```bash
git add scripts/run_colab_t4_smoke.py src/pipeline/colab_smoke.py tests/test_ci_colab_architecture.py
git commit -m "feat(colab): add real single-T4 contract smoke"
```

---

### Task 6: Generate the Colab notebook

**Files:**
- Create: `scripts/generate_colab_smoke_notebook.py`
- Create: `colab/legalir_t4_smoke.ipynb`
- Test: `tests/test_ci_colab_architecture.py`

**Interfaces:**
The notebook is a thin wrapper around `scripts/run_colab_t4_smoke.py`.

- [ ] **Step 1: Test notebook content**

Require cells for:

```text
GPU preflight / nvidia-smi
google.colab.userdata HF_TOKEN
optional GITHUB_TOKEN
Google Drive mount
exact TARGET_SHA
clone + detached checkout
dependency install without replacing Colab torch
CI verification
run_colab_t4_smoke.py
report summary
```

Reject any cell that prints secret values.

- [ ] **Step 2: Implement generator**

Use:

```python
from google.colab import drive, userdata
```

Secrets:

```python
HF_TOKEN = userdata.get("HF_TOKEN")
GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
```

Never embed values into notebook JSON.

- [ ] **Step 3: Require explicit official data path**

Document a default example:

```text
/content/drive/MyDrive/legalir-task1-clean-data
```

User should need to edit only:

```text
TARGET_SHA
DATA_DIR
OUTPUT_DIR
```

- [ ] **Step 4: Regenerate notebook**

- [ ] **Step 5: GREEN tests**

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_colab_smoke_notebook.py colab/legalir_t4_smoke.ipynb tests/test_ci_colab_architecture.py
git commit -m "feat(colab): add reproducible T4 smoke notebook"
```

---

### Task 7: Preserve Kaggle FULL as strict dual-T4 production

**Files:**
- Modify: `src/pipeline/kaggle_train.py` only if shared mode parsing is necessary.
- Test: `tests/test_ci_colab_architecture.py`

- [ ] **Step 1: RED isolation tests**

Assert:

```text
Colab smoke accepts one T4
FULL rejects one GPU
FULL still requires Dense cuda:0
FULL still requires reranker cuda:1
Colab limits never enter FULL config
```

- [ ] **Step 2: Implement minimal isolation**

Prefer separate Colab orchestration over adding many `if colab_smoke` branches to `kaggle_train.py`.

- [ ] **Step 3: GREEN tests**

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/kaggle_train.py src/pipeline/colab_smoke.py tests/test_ci_colab_architecture.py
git commit -m "test(production): keep Kaggle full dual-T4 strict"
```

---

### Task 8: Add score-promotion guardrails

**Files:**
- Create: `scripts/check_score_promotion.py`
- Create: `configs/production_score_guard.json`
- Test: `tests/test_ci_colab_architecture.py`
- Modify: `README.md`

**Interfaces:**
The script compares candidate OOF evidence against the accepted baseline. It does not generate scores.

- [ ] **Step 1: Write metric comparison tests**

Rules:

```text
higher Recall@5 -> eligible
equal Recall@5 + higher Precision@5 -> eligible
lower Recall@5 -> reject
material Candidate Recall@50/150 regression -> reject/warn by configured tolerance
missing doc-disjoint metric -> reject production promotion
```

- [ ] **Step 2: Create guard config**

Use existing accepted benchmark metrics already stored in the repository as the initial baseline. Do not invent new targets.

- [ ] **Step 3: Implement comparison**

- [ ] **Step 4: Document sequential score-ablation order**

```text
1. retrieval/RRF branch weights
2. rerank_k = 40 / 50 / 80
3. BCE vs pairwise_logistic
4. steps above full-query coverage minimum
5. candidate_k = 150 / 200 only after miss analysis
```

No broad combinatorial grid.

- [ ] **Step 5: Commit**

```bash
git add scripts/check_score_promotion.py configs/production_score_guard.json tests/test_ci_colab_architecture.py README.md
git commit -m "feat(score): gate production changes on leakage-safe OOF"
```

---

### Task 9: Document the operating workflow

**Files:**
- Modify: `README.md`
- Create: `docs/CI_COLAB_KAGGLE_WORKFLOW.md`
- Test: `tests/test_ci_colab_architecture.py`

- [ ] **Step 1: Update primary release sequence**

```text
1. push
2. wait for LegalIR CI green
3. run Colab T4 smoke for same SHA
4. inspect colab_smoke_report.json
5. manually run Kaggle FULL for same SHA
```

Legacy Kaggle `gpu_smoke` may remain available only as an optional diagnostic.

- [ ] **Step 2: Document Colab setup**

```text
Runtime → Change runtime type → GPU
confirm nvidia-smi says Tesla T4
add HF_TOKEN to Colab Secrets
optional GITHUB_TOKEN
mount Drive
set official data path
set target SHA
Run all
```

- [ ] **Step 3: Document invalidation rule**

Any source commit after Colab PASS invalidates the prior smoke result.

- [ ] **Step 4: GREEN docs tests**

- [ ] **Step 5: Commit**

```bash
git add README.md docs/CI_COLAB_KAGGLE_WORKFLOW.md tests/test_ci_colab_architecture.py
git commit -m "docs: adopt CI Colab Kaggle workflow"
```

---

## Final source verification

Run before push:

```bash
python -m compileall -q src scripts
python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('PIPELINE_IMPORT_OK')"
python -c "from src.pipeline.colab_smoke import *; print('COLAB_IMPORT_OK')"
pytest -q
python scripts/audit_parameters.py
python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
python scripts/check_notebook_parity.py
python scripts/generate_colab_smoke_notebook.py
git diff --exit-code colab/legalir_t4_smoke.ipynb
```

Then push.

## GitHub CI monitoring gate

After push:

1. record the exact 40-character SHA;
2. monitor workflow `LegalIR CI`;
3. if RED, inspect the failed step, fix, commit and push again;
4. never proceed to Colab with a red/in-progress/missing workflow;
5. when green, use exactly that SHA in Colab.

## Colab execution gate

Colab PASS requires:

```text
CI green for exact SHA
Tesla T4
official v2 identity
real DEk21 GPU inference
FAISS
real BGE LoRA optimizer steps
finite loss
param_diff > 0
adapter save/reload/SHA
adapter params > 0
<4B
zero fold leakage
duplicate blacklist active
valid 1–5 doc prediction lists
```

A Colab PASS is valid only for its exact SHA.

## Kaggle authorization

After Colab PASS, manually run Kaggle with:

```text
LEGALIR_RUN_MODE=full
```

using the same approved runtime SHA.

Kaggle FULL remains dual T4 and full corpus/training. Do not copy Colab limits into production.

## Required final implementation report

```markdown
# LegalIR CI-Colab-Kaggle Architecture Report

## Git
- implementation SHA:
- final release SHA:

## GitHub CI
- workflow:
- status:
- compileall:
- imports:
- pytest:
- parameter audit:
- CPU smoke:
- notebook parity:

## Colab T4
- target SHA:
- CI verified:
- GPU:
- official dataset:
- subset queries/docs/chunks:
- Dense actual device:
- FAISS:
- Dense peak VRAM:
- Dense OOM:
- reranker actual device:
- optimizer steps:
- finite loss:
- param_diff:
- adapter SHA verified:
- adapter params:
- total params:
- reranker peak VRAM:
- prediction validation:
- wall time:
- report result:

## Production protection
- protected score config unchanged:
- FULL still requires T4×2:
- full Dense device:
- full reranker device:

## Score promotion
- baseline report:
- candidate report:
- Recall@5 delta:
- Precision@5 delta:
- Candidate Recall@50 delta:
- Candidate Recall@150 delta:
- doc-disjoint delta:
- promotion decision:

READY FOR COLAB T4 SMOKE: YES/NO
READY FOR MANUAL KAGGLE FULL: YES/NO

## Remaining risks
...
```
