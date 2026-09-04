# Failure Recovery and Resume Strategy

## Principle

No expensive completed stage should be lost because a later stage fails.

## Artifact states

Every long job writes:

```text
RUNNING
PASS
FAIL
```

with a manifest.

## Resume validation

A PASS artifact is reusable only when:

- runtime SHA matches;
- source fingerprints match;
- config SHA matches;
- output file hashes match;
- upstream manifest hashes match.

## Static retrieval failure

Resume by query batch or branch shard.

Suggested shards:

```text
train qids 0..999
1000..1999
...
public qids
```

Merge only after every shard verifies.

## Fold failure

Rerun only the failed fold.

## Doc-disjoint failure

Rerun only doc-disjoint.

## Bundle failure

Rebuild only invalid derived files when upstream hashes are valid.

## Kaggle final failure

Persist checkpoints under `/kaggle/working`.

If competition rules/session environment permit manual continuation, recovery may reuse the final adapter checkpoint only when its training manifest and runtime SHA verify.

Do not reuse partially written submission artifacts.

## Memory failure

The new runtime should fail before external kill with:

```text
stage
RSS
available RAM
largest live artifact classes where instrumented
GPU memory
```

Use that evidence to adjust implementation, not score semantics.

## Timeout failure

Do not increase Kaggle complexity.

Move remaining reusable computation into the artifact factory.

The final Kaggle run must remain simple.
