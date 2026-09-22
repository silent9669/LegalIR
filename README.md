# LegalIR Task 1

Vietnamese legal document retrieval for the UIT Data Science Challenge 2026. The pipeline combines legal BM25, PyVi BM25, DEk21 dense retrieval, exact matching, fold-local question memory, and a BGE LoRA cross-encoder.

## Status — updated 2026-09-22

- **HEAD:** `f867ab4` — evidence bundle for runtime `6b57ed7` (Kaggle dual-T4 **PASS v73**: 24.23s, weight delta 285.00, genuine Tesla T4 ×2 execution).
- **Working tree:** 4 approved surgical fixes on top of HEAD (pending commit) — doc-disjoint duplicate expansion, `validate_submission_zip(exact_answer_count)`, MPS OOM matcher narrowing, strict-mode warm-start rank-lock. See `TEAMMATE.md` §3.
- **Tests:** **587 passed, 2 skipped** (full suite). Preflight gates green: `validate_score_push.py`, parameter audit, notebook zero-drift.
- **Launch:** no release required. Single entrypoint `python scripts/modal/run_full.py --warm --private --push-config` (preflight runs automatically). Details in `TEAMMATE.md`.
- **Targets, not demonstrated results:** full private run (5-fold OOF + disjoint + final + 2080-query inference) has not completed on A100 yet; OOF/private recall and end-to-end runtime are projections until a real run finishes.
- **Parameter audit:** 702,754,049 parameters across the dense encoder and reranker (~715M with the r=64 LoRA adapter), below the competition's 4B ceiling. Re-run the audit if the models change.
- A timeout caps duration, not spend; retries/re-runs bill extra. There is no checkpoint-resume — a killed run restarts from scratch (Volume holds forensics only).

## Essential documents

| Document | Purpose |
|---|---|
| [TEAMMATE.md](TEAMMATE.md) | **Start here.** How to run, what was built, workflow, verification, recovery |
| [Architecture](docs/ARCHITECTURE.md) | Components, data boundaries, training/evaluation, and configuration sources |
| [Release workflow](docs/REPRODUCIBLE_TRAINING_WORKFLOW.md) | Local tests → Kaggle smoke → evidence bundle → A100 run |
| [A100 launch guide](docs/README_A100_LAUNCH.md) | Modal supervision, consent, timeouts, recovery, and stop procedures |
| [Historical timing evidence](docs/A100_SCALE_DOWN_AND_OPTIMIZATION_REPORT.md) | Old A100 measurements; historical record, not launch approval |

Launch instructions live in `TEAMMATE.md` and the launch guide. Historical reports and architecture descriptions are not launch approval.

## Pipeline

```text
Canonical Task 1 corpus and queries
  → lexical / dense / exact retrieval + fold-local question memory
  → candidate fusion and query-aware evidence selection
  → fold-specific BGE LoRA training and batched held-out inference
  → five-fold OOF + document-disjoint evaluation and fusion evaluation
  → dedicated final training on all training queries
  → final-model reload, private inference, submission validation
  → durable artifacts and verified delivery receipts
```

Static mining candidates can be reused across folds without reusing fold labels. This reduces some repeated searches; it does not eliminate every retrieval or index load. Batched inference and parallel PyVi indexing are implemented optimization mechanisms, not measured speedup guarantees.

The query-balanced sampler prioritizes an interleaved positive/negative pair per eligible query. Query coverage is not full exposure to every available pair, and neither guarantees model quality. The top-five oracle measures a ceiling, not achieved Recall@5.

## Data and evaluation rules

- Use only canonical Task 1 training queries, qrels, and legal corpus. Private queries are for inference only.
- No external legal corpus, Task 2 data, crawling, synthetic LLM examples, or external inference APIs.
- Preserve all five folds, document-disjoint evaluation, and fold-local supervised memory/mining.
- Never initialize honest held-out evaluation from a final adapter trained on all held-out labels.
- Parameter counts, dataset identities, model revisions, split identities, and effective configuration must be recorded and checked.

## Local verification

Use the repository virtual environment; no GPU allocation is needed for these commands:

```bash
.venv/bin/python scripts/validate_score_push.py
.venv/bin/python scripts/generate_notebooks.py --check-drift
.venv/bin/python scripts/audit_parameters.py --check-only
.venv/bin/python scripts/verify_release_approval.py --repo-root .
.venv/bin/pytest tests/leakage/ tests/integration/test_doc_disjoint.py tests/contracts/test_modal_cli.py tests/integration/test_modal_delivery.py -q
git status --short
```

## Repository map

- `src/`: retrieval, evidence, training, ranking, evaluation, pipeline, and release contracts.
- `configs/`: algorithm, experiment, and backend runtime configuration.
- `scripts/`: local verification, notebook generation, hardware gates, and backend launchers.
- `notebooks/`: generated Kaggle smoke and Colab A100 notebooks; do not hand-edit.
- `tests/`: unit, contracts, dataset, notebook, parity, leakage, memory, integration, and release suites.
- `kaggle_dataset/`: canonical local dataset layout; heavy data is not a Git deliverable.
- `artifacts/task1/`: release evidence and run outputs; distinguish historical evidence from the active candidate.

Persisted stage artifacts are not automatic remote resume. Modal creates a new UUID attempt directory on every invocation; Colab recovery is best effort and does not survive VM loss by itself. Exact optimizer-state resume is not provided.
