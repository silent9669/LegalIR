# Validation Sharding: 5-Fold OOF + Document-Disjoint

## Principle

Validation remains full and leakage-safe, but it runs as resumable jobs.

## Random five-fold

For each fold:

```text
train IDs = authoritative split train IDs
val IDs   = authoritative split val IDs
```

Required invariants:

```text
pair_qids ⊆ train_ids
pair_qids ∩ val_ids = ∅
memory_qids ⊆ train_ids
memory_qids ∩ val_ids = ∅
```

## Fold job contract

Inputs:

```text
runtime SHA
canonical dataset fingerprints
split SHA
static cache SHA
evidence algorithm version
production scoring config
fold ID
```

Outputs:

```text
adapter/
training_manifest.json
oof_features.parquet
predictions.json
fold_metrics.json
job_manifest.json
```

## Resume behavior

If `job_manifest.json` says PASS and all output hashes verify, the fold is reusable.

Otherwise rerun only that fold.

## Parallelism

If the environment supports two T4s:

```text
wave 1: fold0 GPU0 + fold1 GPU1
wave 2: fold2 GPU0 + fold3 GPU1
wave 3: fold4 GPU0
```

Each fold should run in an isolated process so GPU and host memory return to the OS at process exit.

Do not keep multiple BGE models in the parent process.

## Document-disjoint

Run as a separate job.

It is a robustness gate, not a replacement for five-fold OOF.

Report:

```text
Recall@1
Recall@3
Recall@5
Precision@5
Candidate Recall@20/50/100/150
```

## OOF aggregate

Build aggregate metrics from fold files, not from in-memory accumulation.

Use Arrow/Parquet dataset scans.

## Fusion

Fusion training consumes only OOF-safe feature files.

No public labels exist and none are inferred.

## Promotion

A new production config is eligible only if:

1. official-equivalent Recall@5 improves;
2. or Recall@5 ties and Precision@5 improves;
3. Candidate Recall@50/150 does not materially regress;
4. document-disjoint does not show unacceptable robustness regression;
5. all leakage checks pass.
