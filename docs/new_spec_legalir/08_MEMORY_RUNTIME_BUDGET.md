# Memory and Runtime Budget

## Host-RAM rule

Never design against a hard-coded Kaggle RAM number.

Detect runtime memory.

Acceptance:

```text
peak RSS <= 70% of physical RAM
available RAM >= 3 GiB at stage boundaries
```

## Required telemetry

At every major stage log:

```text
process RSS
system used
system available
system total
GPU0 allocated/reserved/peak
GPU1 allocated/reserved/peak
```

## Memory guard

Pseudo-contract:

```python
if available_ram < max(3 * GiB, 0.10 * total_ram):
    release_memory()
    recheck()
    if still_low:
        raise MemoryError(stage_diagnostics)
```

Failing with a Python report is preferable to an external kernel kill.

## Factory memory design

Heavy state is sequential:

```text
Parquet scan
→ retriever
→ branch cache
→ retriever unload
```

Avoid simultaneous residency of all corpus structures.

## Evidence memory design

One Arrow store + bounded LRU.

No global token counters for all 219k macro chunks.

## Fold jobs

Run in child processes.

Process exit is the strongest memory cleanup.

## Dense memory

After static cache is complete, release:

- DEk21 model;
- corpus embedding matrix;
- FAISS if no longer needed.

If Dense search must remain temporarily, allow dropping redundant matrix after FAISS parity.

## Pair files

Write incrementally.

Recommended flush:

```text
2,000–5,000 rows
```

## OOF features

Write fold Parquets.

Do not hold all folds simultaneously.

## Runtime budgets

### Factory

Factory jobs may span multiple sessions because they are resumable.

No single shard should be designed to require the entire validation run.

### Kaggle final

Target total:

```text
<= 9 h
```

Expected dominant component:

```text
one final all-query LoRA training
```

Before final release, project using measured T4 step time from the current runtime/config.

Reject the final run if:

```text
projected_total > 9 h
```

## Training throughput probe

Measure on T4:

```text
microbatch 8 / accumulation 2
microbatch 4 / accumulation 4
microbatch 2 / accumulation 8
```

Always preserve:

```text
effective batch = 16
```

Choose fastest stable option with conservative VRAM margin.
