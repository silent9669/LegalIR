# Operations Runbook

## Phase 1 — Fresh branch

```bash
git checkout main
git pull
git switch -c refresh/factory-v1
```

Prefer a clean worktree.

## Phase 2 — Implement foundation

Complete:

```text
canonical/provenance
memory telemetry
static cache
lazy evidence
pair materializer
```

Run parity tests before building full artifacts.

## Phase 3 — Build static cache

Run the official dataset.

Verify cache manifest.

Do not proceed on parity failure.

## Phase 4 — Validation shards

Run fold jobs 0..4.

Track manifest status.

Then run doc-disjoint.

## Phase 5 — Select production config

Aggregate metrics.

Apply promotion policy.

Freeze `production_lock.json`.

## Phase 6 — Build production bundle

Build:

```text
final pairs
public candidates
public evidence
fusion
manifests
```

Run bundle verifier.

## Phase 7 — CI

Push runtime commit.

Wait for GitHub CI GREEN.

## Phase 8 — Colab T4

Run exact runtime SHA.

Require:

```text
PASS
stable T4 microbatch
effective batch 16
adapter reload
host memory safe
```

## Phase 9 — Release

Commit only:

```text
Colab report
approval artifact
generated notebooks
runtime pin
```

Wait for CI GREEN.

## Phase 10 — Kaggle T4×2 final

Attach:

```text
canonical dataset
production bundle
```

Enable T4×2.

Run all.

Do not run OOF.

## Phase 11 — Final audit

Verify:

```text
runtime SHA
bundle SHA
adapter SHA
1,000 public keys
1..5 predictions/query
submission zip
```

Then submit.
