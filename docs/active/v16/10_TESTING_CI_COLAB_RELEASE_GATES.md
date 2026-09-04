# Testing, CI, Colab, and Release Gates

## Test layers

### Unit

- canonical identity;
- hashing;
- manifests;
- memory guard;
- cache serialization;
- promotion comparator.

### Parity

Highest priority after correctness.

Required parity:

```text
live vs cached Legal BM25
live vs cached PyVi
live vs cached Dense
live vs cached Exact
live vs cached fused candidates
legacy vs lazy positive localization
legacy vs lazy evidence text
Dense before/after matrix drop
```

### Leakage

Required:

```text
pair_qids subset fold train
pair_qids disjoint fold validation
memory_qids subset fold train
memory_qids disjoint validation
duplicate-equivalent gold never negative
doc-disjoint isolation
```

### Memory

Required:

```text
no full chunks DataFrame in pair materializer
no retriever loads in cached pair path
bounded evidence LRU
bounded pair writer
bounded OOF writer
low-memory guard
```

### Integration

Required:

```text
tiny full factory flow
single fold job
doc-disjoint job
bundle build/verify
final production dry run
```

## GitHub CI

Must run on every push/PR:

```text
compileall
imports
pytest
parameter audit
CPU tiny integration
notebook parity
release verifier
```

No heavy pretrained downloads.

## Colab T4

Colab is the real GPU contract gate.

Must test:

- approved runtime SHA;
- T4 hardware;
- real DEk21;
- FAISS;
- lazy evidence;
- cached pair materialization;
- real BGE+LoRA;
- throughput probe;
- adapter reload;
- public reranking path;
- host memory telemetry;
- parameter budget.

## Colab pass requirements

```text
result PASS
loss finite
param_diff >0
fresh reload true
finite scores true
no leakage
effective batch 16
stable microbatch recorded
peak GPU memory within safe T4 margin
host RSS safe
```

## Validation-factory acceptance

Before freezing a production bundle:

```text
5/5 folds complete
doc-disjoint complete
all artifact hashes valid
all leakage counters zero
promotion report exists
production_lock.json exists
```

## Release rule

A source change after GPU approval invalidates the old approval.

Use:

```text
Commit A = runtime
CI GREEN
Colab PASS
Commit B = release-only notebook/artifacts
CI GREEN
```

Kaggle must execute Commit A.
