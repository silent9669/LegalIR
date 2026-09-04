# Score Optimization and Promotion Policy

## Primary objective

Official-equivalent Recall@5.

## Tie-break

Precision@5.

## Guardrails

Monitor:

```text
Recall@1
Recall@3
Candidate Recall@20
Candidate Recall@50
Candidate Recall@100
Candidate Recall@150
document-disjoint Recall@5
```

## Baseline

Keep the accepted leakage-safe baseline as the minimum reference until a candidate beats it.

Do not erase historical metrics.

## Experimental order

Run sequential ablations, not combinatorial sweeps.

Recommended order:

1. branch/RRF weights;
2. candidate branch depth;
3. rerank depth 40/50/80;
4. BCE vs pairwise objective;
5. training steps above minimum coverage;
6. evidence-pack settings only after parity-safe refactor;
7. candidate_k 150 vs 200 only if miss analysis supports it.

## Promotion rule

Promote if:

```text
candidate Recall@5 > baseline Recall@5
```

or:

```text
Recall@5 ties within declared tolerance
AND Precision@5 improves
```

Reject if:

- leakage audit fails;
- Candidate Recall@50/150 regresses beyond tolerance;
- doc-disjoint robustness collapses;
- runtime becomes incompatible with production budget;
- memory exceeds production target.

## Separation of concerns

The fresh architecture refactor must first reproduce existing results.

Do not mix score tuning with cache/evidence/memory changes until parity is demonstrated.

## Production lock

After promotion, freeze:

```text
runtime SHA
config SHA
fusion artifact SHA
static cache schema/version
evidence version
training config
selected metrics
```

Final Kaggle consumes this lock and cannot retune.
