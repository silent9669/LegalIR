# Artifact Factory Pipeline

## Goal

Build expensive reusable artifacts once, with resumability and provenance.

## Stage F0 — Preflight

Validate:

- canonical identity;
- 7,000 train queries;
- 7,637 qrels;
- 1,000 public queries;
- split files;
- duplicate groups;
- parameter budget;
- model availability.

Output:

```text
factory/preflight.json
```

## Stage F1 — Query embeddings

Encode:

- 7,000 train queries;
- 1,000 public queries.

Write:

```text
train_query_embeddings.npy
public_query_embeddings.npy
query_embedding_manifest.json
```

Do not repeatedly encode the same queries per fold.

## Stage F2 — Static retrieval

Build once:

- Legal BM25;
- PyVi BM25;
- DEk21 Dense/FAISS;
- Exact Matcher.

Retrieve branch-level results for all train/public queries.

Write incrementally:

```text
static_candidates_train.parquet
static_candidates_public.parquet
```

The cache builder must not accept qrels.

## Stage F3 — Release heavy retrieval memory

After static caches are safely written:

```text
delete corpus DataFrames
delete BM25 objects
delete PyVi objects
delete Dense transformer
delete Dense float matrix where FAISS/cache makes it redundant
gc.collect
malloc_trim
cuda.empty_cache
```

Record host memory before and after.

## Stage F4 — Evidence store

Initialize one Arrow-backed macro evidence store.

Do not preprocess the entire macro corpus.

Prepare compact document→row index.

Use bounded LRU preprocessing.

## Stage F5 — Fold pair artifacts

For each fold independently:

```text
static branch cache
+
fold-local Question Memory
+
duplicate blacklist
+
lazy evidence
=
fold candidate/pair artifact
```

Write:

```text
fold_N/
  train_pairs.parquet
  validation_candidates.parquet
  pair_manifest.json
```

A failed fold can resume without rebuilding static retrieval.

## Stage F6 — Fold model jobs

Each fold:

1. verify pair manifest;
2. train BGE+LoRA;
3. save adapter;
4. rerank validation candidates;
5. write OOF features;
6. write metrics;
7. terminate process.

The fold job must be idempotent.

## Stage F7 — Document-disjoint

Run as a separate shard.

It has the same artifact contract as a fold.

## Stage F8 — Fusion selection

Read OOF feature Parquets and reports.

Train/evaluate fusion according to the protected feature schema.

Select the production config by promotion policy.

Output:

```text
production_lock.json
fusion_model/
validation_summary.json
```

## Stage F9 — Final materialization

Build:

```text
final_training_pairs.parquet
public_candidates.parquet
public_evidence.parquet
```

using all 7,000 train queries and the approved production config.

## Stage F10 — Bundle

Create immutable production bundle.

Every file gets:

- SHA-256;
- size;
- schema;
- row count where relevant.

The bundle is approved only if the verifier passes.
