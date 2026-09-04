# Static Retrieval Cache and Lazy Evidence

## 1. Static branch cache

Static branches:

```text
Legal BM25
PyVi BM25
DEk21 Dense
Exact Matcher
```

They are label-free and fold-independent.

Question Memory is excluded because it depends on fold-local train labels.

## Cache schema

Recommended normalized Parquet:

```text
query_id          string
branch            categorical/string
rank              int16/int32
doc_id            string
score             float32
best_chunk_id     nullable string
second_score      nullable float32
mean_score        nullable float32
extra_json        optional compact metadata
```

Alternative list-column format is allowed if parity tests pass.

Use `ParquetWriter` in bounded batches.

## Depth

Cache enough branch depth to reproduce the production fused candidate depth.

If production `candidate_k=150`, cache branch depths large enough that truncation cannot change the top-150 fused union.

Determine depth empirically and lock it.

## Live-vs-cache parity

For at least 100 deterministic real train queries:

```text
branch doc IDs exact
branch rank exact
branch score tolerance <= chosen serialization tolerance
fused top-150 doc IDs exact
```

Any mismatch blocks promotion.

## 2. Lazy MacroEvidenceStore

### Problem

Legacy code preprocesses all macro chunks into large Python object graphs.

### New interface

```python
class MacroEvidenceStore:
    def get_doc_chunks(self, doc_id: str) -> list[MacroChunk]
    def get_preprocessed_doc(self, doc_id: str) -> PreprocessedDoc
    def clear_cache(self) -> None
    def cache_bytes(self) -> int
```

### Backend

Use PyArrow/string buffers.

Keep only required columns.

Build a compact doc ID → row-range/index map.

### LRU

Recommended initial cap:

```text
512 MB
```

Also cap by document count, e.g.:

```text
256–512 docs
```

Eviction must be deterministic enough not to affect output semantics.

## 3. Positive localization

Reimplement current PositiveLocalizer scoring lazily.

Do not change weights.

Parity gate:

```text
same selected positive chunk ID
```

for 100+ official query/gold-doc pairs.

## 4. Evidence pack

Reimplement EvidencePackBuilder against the same store.

Parity gate:

```text
same selected chunk IDs
same evidence text
```

for 100+ deterministic official candidate pairs.

## 5. Duplicate blacklist

The four canonical duplicate groups remain mandatory.

No negative may be equivalent to any gold positive under duplicate closure.

This must be checked after pair materialization.
