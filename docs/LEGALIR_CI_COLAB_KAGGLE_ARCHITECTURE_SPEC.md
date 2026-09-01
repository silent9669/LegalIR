# LegalIR Verification & Production Architecture — GitHub CI → Colab T4 → Kaggle T4×2

**Repository baseline:** `silent9669/LegalIR`  
**Audited release HEAD:** `377630cc169e080caaca0395c9822844066c05b9`  
**Pinned runtime at that release:** `2c1b6e8bcfb3738ccd369d181a92ac68f3f98f12`

## 1. Objective

Replace the expensive Kaggle `gpu_smoke` workflow with a three-gate release process:

```text
Push code
  ↓
GitHub CI — CPU/source correctness
  ↓ GREEN only
Google Colab — one real T4, sequential GPU contract smoke
  ↓ PASS only
Kaggle — manual T4×2 FULL production run
```

The verification architecture must not weaken or fork the competition pipeline. Colab smoke must execute the same production modules with bounded data/step limits. Kaggle FULL remains the only authoritative full-corpus/full-training production run.

The scoring objective remains:

1. maximize official Recall@5;
2. use Precision@5 only as tie-break;
3. never output more than five document IDs;
4. preserve Task-1-only data isolation;
5. keep total learned parameters strictly below 4B.

## 2. Why this change

The current Kaggle smoke is too expensive as a readiness test. A real run log reached:

```text
Legal BM25 ready                 ~354.6 s
PyVi BM25 ready                  ~5432.8 s
DEk21 Dense index only started   after ~90 minutes
```

The PyVi index alone consumed roughly 84.6 minutes. A smoke gate should validate contracts, CUDA, model loading, real LoRA updates, FAISS, artifact integrity and inference—not rebuild the entire production retrieval stack.

GitHub currently has no `.github/workflows/` directory at the audited HEAD, so there is no actual CI gate yet.

## 3. Gate A — GitHub CI

### Purpose

Catch every source/data-contract error that does not require a GPU.

### Trigger

```text
push to main
pull_request
workflow_dispatch
```

### Required CI environment

- Ubuntu GitHub-hosted runner.
- Python 3.12.
- `actions/setup-python` with pip dependency cache keyed by `requirements.txt`.
- No HF token.
- No model downloads.
- `HF_HUB_OFFLINE=1`.
- `TRANSFORMERS_OFFLINE=1`.
- CI must use mocks/fixtures for heavyweight Transformer kernels.

### Required commands

```bash
python -m compileall -q src scripts

python -c "from src.pipeline.kaggle_train import run_kaggle_pipeline; print('PIPELINE_IMPORT_OK')"
python -c "from src.pipeline.oof_runner import OOFRunner; print('OOF_IMPORT_OK')"
python -c "from src.training.build_pairs import build_training_pairs; print('PAIR_IMPORT_OK')"

pytest -q

python scripts/audit_parameters.py

python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke

python scripts/check_notebook_parity.py
```

CI must verify:

- official v2 identity constants and 1,000 public-query contract;
- OOF pair-QID train-only invariant;
- zero fold validation overlap;
- doc-disjoint isolation;
- four-group duplicate blacklist contract;
- strict top-5 submission validation;
- fusion manifest/load roundtrip;
- parameter budget `<4B`;
- notebook-generation parity;
- Colab-smoke configuration cannot alter production score hyperparameters.

### Required outcome

Only a commit whose CI workflow concludes `success` may be used in Colab.

The Colab notebook must pin that exact commit SHA.

## 4. Gate B — Google Colab single-T4 contract smoke

### Purpose

Validate real GPU execution cheaply without duplicating full Kaggle work.

### Hardware contract

Normal readiness path:

```text
CUDA available
at least one visible GPU
GPU name contains "T4"
```

A stronger GPU may be allowed only with an explicit debug override and must produce:

```text
NOT_A_T4_READINESS_GATE
```

The single T4 is used sequentially:

```text
Stage 1: DEk21 Dense on cuda:0
  → encode bounded real subset
  → build/search FAISS
  → persist embeddings/index
  → unload Dense model
  → torch.cuda.empty_cache()

Stage 2: BGE reranker + LoRA on cuda:0
  → real pair training
  → real optimizer steps
  → adapter save
  → SHA-256
  → reload
  → rerank
```

Do not require two GPUs in Colab. Do not change the Kaggle FULL topology.

### Colab data contract

Use the actual canonical v2 dataset:

```text
8,532 documents
1,153,876 chunks
934,416 micro chunks
219,460 macro chunks
7,000 train queries
7,637 qrels
1,000 public queries
audit is_valid=true
```

Identity verification reads manifest/audit + Parquet metadata only.

The smoke then constructs a deterministic bounded workspace from official data using seed 42. It must include selected queries' positive documents plus deterministic distractors. No synthetic labels are permitted.

Recommended initial bounds:

```text
smoke train queries       64
smoke validation queries  32
smoke public queries      16
smoke documents           2,000 maximum
OOF folds                 2
LoRA optimizer steps      8–12 per smoke training job
Dense batch size          16
Reranker batch size       8
seed                      42
```

The limits are smoke-only. Model names, tokenizer settings, retrieval features, RRF/fusion feature definitions and production scoring hyperparameters must come from the production configuration.

### Required real GPU assertions

```text
GPU is T4
DEk21 parameters actually on cuda:0
Dense FAISS backend active
Dense embeddings finite
Dense OOM events recorded

BGE/PEFT parameters actually on cuda:0
real optimizer_steps > 0
finite loss
param_diff > 0
adapter file exists
adapter SHA verified
adapter_parameters > 0
final learned parameters <4B
reranker outputs finite
artifact reload succeeds
```

### Required pipeline assertions

- split provenance resolved and SHA recorded;
- smoke pair QIDs are a subset of smoke fold train QIDs;
- zero validation leakage;
- duplicate blacklist active;
- prediction IDs are official corpus IDs;
- each prediction list has 1–5 unique IDs;
- deterministic rerun of the same bounded sample produces identical selected QIDs and compatible artifact manifests.

### Colab report

Create `colab_smoke_report.json` containing at minimum:

```text
git_sha
ci_workflow_name
ci_green
gpu_name
cuda_version
torch_version
dataset identity
subset query/doc/chunk counts
split SHA
duplicate blacklist source/count
Dense device/backend/VRAM/OOM
reranker device/VRAM/OOM
optimizer steps
loss finite
param_diff
adapter SHA
adapter parameter count
total learned parameter count
wall-clock stage timings
prediction validation
result = PASS|FAIL
```

Save the report to the mounted Drive output directory and print a concise PASS/FAIL summary.

### Time target

The Colab smoke should target approximately 10–20 minutes after dependencies/models are cached. It must never build the full 934,416-document PyVi index merely to prove GPU correctness.

## 5. Gate C — manual Kaggle FULL

Kaggle is no longer used as a smoke environment.

Prerequisites:

```text
same target commit has GREEN GitHub CI
same target commit has PASS Colab T4 smoke
```

Production remains:

```text
Dense = cuda:0
BGE/PEFT reranker = cuda:1
full official corpus
full five-fold OOF
document-disjoint evaluation
full pair mining
full final reranker training
full public inference
exact 1,000-query submission
```

Kaggle FULL still requires:

- official v2 identity;
- FAISS;
- OOF leakage counters = 0;
- duplicate blacklist = four canonical groups;
- full pair-derived training coverage;
- adapter SHA and positive adapter params;
- `<4B`;
- exact top-5 submission validation.

No smoke-specific sample limit may be read by FULL mode.

## 6. Production score protection

Verification changes must not silently change ranking quality.

Create one immutable production config layer containing score-affecting settings:

```text
candidate branch weights
candidate_k
rerank_k
BGE loss type
learning rate / LoRA config
fusion features
fusion selection policy
top-5 logic
```

`colab_smoke` may override only:

```text
query counts
document/chunk subset size
fold count
optimizer step cap
device topology
output/work directories
telemetry verbosity
```

It must not override ranking weights or feature definitions.

Add a configuration-diff test that fails if a smoke override changes a protected production key.

## 7. Score-maximization workflow

The CI/Colab architecture verifies correctness; it does not select better ranking settings.

Any score-affecting change must pass a separate promotion gate using leakage-safe OOF evidence.

Promotion order:

```text
1. Candidate retrieval branch/RRF weights
2. rerank_k: 40 vs 50 vs 80
3. BCE vs pairwise_logistic
4. training steps above full-query coverage minimum
5. candidate_k 150 vs 200 only if candidate misses justify it
```

Run sequential ablations, not a combinatorial grid.

Promotion rule:

```text
higher official-scorer-equivalent Recall@5 wins
Precision@5 breaks ties
Candidate Recall@50/150 must not regress materially
doc-disjoint Recall@5 is a robustness guardrail
```

Keep the best known production config pinned. A runtime/CI refactor is not allowed to replace it without OOF evidence.

## 8. Secrets and reproducibility

GitHub CI uses no HF secret.

Colab uses Colab Secrets:

```text
HF_TOKEN
optional GITHUB_TOKEN
```

Read through `google.colab.userdata`, never print secret values.

The Colab notebook must pin an exact commit SHA. If a public GitHub API check cannot prove the target SHA's CI is green, the notebook must stop and ask for a token rather than silently continue.

Kaggle also pins the exact approved runtime SHA.

## 9. Final release state machine

```text
CI_RED
  → fix source → push again

CI_GREEN
  → run Colab T4 smoke

COLAB_FAIL
  → fix source → push → CI_GREEN again → rerun Colab

COLAB_PASS
  → authorize manual Kaggle FULL

KAGGLE_FULL
  → submission artifact + final reports
```

A source change after Colab PASS invalidates that PASS because the commit SHA changed.
