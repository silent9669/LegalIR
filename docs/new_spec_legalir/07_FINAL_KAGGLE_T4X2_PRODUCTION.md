# Final Kaggle T4×2 Production Run

## Purpose

The final Kaggle notebook is a production trainer, not a validation laboratory.

## Inputs

```text
canonical Task1 v2 dataset
approved production bundle
exact approved runtime SHA
HF token if required for model download
```

## It must not run

```text
full BM25 build
full PyVi build
full Dense corpus encoding
5-fold OOF
doc-disjoint validation
hyperparameter search
fusion selection
```

## Production sequence

### K0 — SHA and environment

Verify:

```text
actual git SHA == approved runtime SHA
2 CUDA devices
both GPUs are usable
```

### K1 — Dataset identity

Verify exact canonical identity.

### K2 — Bundle

Verify every production-bundle fingerprint.

Fail closed on mismatch.

### K3 — Final pairs

Read `final_training_pairs.parquet`.

Audit:

```text
all 7,000 expected qids represented where eligible
positive coverage
negative coverage
duplicate blacklist
```

### K4 — Final BGE+LoRA

Train one adapter on all 7,000 queries.

Protected semantics:

```text
BGE reranker-v2-m3
LoRA config
BCE unless promoted otherwise
max_length = 512
effective batch = 16
coverage-derived minimum steps
FP16
gradient checkpointing
```

Use the largest T4-tested microbatch factorization that preserves effective batch 16.

Examples:

```text
8 × grad_acc 2
4 × grad_acc 4
2 × grad_acc 8
```

### K5 — Adapter verification

Require:

```text
optimizer_steps > 0
loss finite
param_diff > 0
adapter file exists
adapter SHA-256
fresh reload
active PEFT
finite reranker scores
system params <4B
```

### K6 — Public reranking

Read public candidates/evidence from bundle.

Option A — single GPU:
- rerank all 1,000 on one GPU.

Option B — preferred:
- load the final adapter on both T4s after training;
- split public queries across GPU0/GPU1;
- merge deterministic outputs.

### K7 — Fusion/top-5

Apply approved production lock.

Never select new weights in final run.

### K8 — Submission validation

For every public query:

```text
key exists
1..5 predictions
unique doc IDs
all IDs valid
```

Exact keyset = all 1,000 public IDs.

### K9 — Package

Create:

```text
submission.json
submission.zip
final_run_manifest.json
```

## Runtime target

Target:

```text
<= 9 hours
```

Hard reject before FULL if measured projection exceeds target.

The remaining margin is for:

- model download;
- Kaggle variability;
- checkpoint write;
- reranking;
- packaging;
- recovery from adaptive batch fallback.
