# Legacy Decisions: What We Keep and What We Replace

## Keep

From the old system:

- canonical v2 dataset;
- Task1-only data isolation;
- Legal BM25;
- PyVi BM25;
- DEk21;
- Exact Matcher;
- fold-safe Question Memory;
- BGE reranker-v2-m3;
- LoRA;
- duplicate-group blacklist;
- 5-fold OOF;
- document-disjoint validation;
- learned fusion;
- Recall@5-first promotion;
- strict top-5 submission validation;
- GitHub CI;
- Colab T4 smoke;
- exact runtime pinning;
- <4B parameter audit.

## Replace

### Monolithic Kaggle FULL

Old:

```text
index everything
→ five-fold
→ doc-disjoint
→ final train
→ public inference
```

New:

```text
factory validates/builds
→ frozen bundle
→ Kaggle final train + public inference
```

### Repeated retrieval

Old:

```text
retrievers loaded repeatedly by pipeline/OOF/pair builder
```

New:

```text
static retrieval once
→ cache
→ unload
```

### Corpus-wide evidence preprocessing

Old:

```text
PositiveLocalizer full corpus
+
EvidencePackBuilder full corpus
```

New:

```text
one lazy Arrow evidence store
```

### Non-resumable validation

Old:

```text
one long process
```

New:

```text
independent artifact jobs
```

## Historical failure retained as a design constraint

The old Kaggle FULL run died during Fold-0 pair setup after hours of completed indexing.

The refresh treats this as evidence that:

- memory ownership must be explicit;
- reusable work must be cached;
- final Kaggle should not contain validation workloads;
- external kernel death must be replaced by proactive memory guards.
