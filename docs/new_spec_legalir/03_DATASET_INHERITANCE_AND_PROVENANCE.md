# Dataset Inheritance and Provenance

## Canonical source

The refresh inherits the existing official Task 1 canonical v2 dataset unchanged.

Expected identity:

```text
dataset            task1_canonical
version            v2
schema             hierarchical_micro_macro_v2
documents          8,532
chunks             1,153,876
micro chunks       934,416
macro chunks       219,460
train queries      7,000
qrels              7,637
public queries     1,000
duplicate groups   4
```

## Read-only rule

The refresh may:

- read canonical data;
- derive indexes;
- derive branch caches;
- derive split-local artifacts;
- derive pair files;
- derive OOF reports.

The refresh may not:

- alter official labels;
- alter public IDs;
- alter document IDs;
- add external documents;
- mix Task 2 data;
- rewrite official truth to improve metrics.

## Required source fingerprints

Every production bundle must record:

```text
manifest.json SHA256
audit_report.json SHA256
documents.parquet fingerprint
chunks.parquet fingerprint
queries_train.parquet fingerprint
qrels_train.parquet fingerprint
public-official.json SHA256
duplicate_groups artifact SHA256
split artifacts SHA256
```

For huge Parquets, use:

- file SHA-256 when practical;
- otherwise size + metadata + row count + schema + optional chunked hash.

Production should prefer full SHA-256 if the artifact-building environment can afford it.

## Provenance graph

```text
canonical dataset fingerprint
        +
runtime commit SHA
        +
retrieval model revisions
        +
production config SHA
        ↓
factory artifacts
        ↓
validation reports
        ↓
production lock
        ↓
production bundle
        ↓
final adapter
        ↓
submission
```

Each arrow must be represented in a manifest.

## No silent reuse

An artifact is reusable only if all declared fingerprints match.

A mismatch in any of these invalidates reuse:

- dataset fingerprint;
- code/runtime SHA;
- tokenizer/preprocessing version;
- retrieval model ID/revision;
- branch weights;
- candidate depth;
- split SHA;
- evidence algorithm version;
- reranker config;
- fusion config.

## Public-file contract

Exactly 1,000 public query IDs are required.

Submission keyset must exactly equal the public query keyset.

No truncation, filtering, or accidental missing query is permitted.
