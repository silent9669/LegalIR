# LegalIR Task 1

Vietnamese legal document retrieval for the UIT Data Science Challenge 2026. The pipeline combines legal BM25, PyVi BM25, DEk21 dense retrieval, exact matching, fold-local question memory, and a BGE LoRA cross-encoder.

## Status — updated 2026-09-23

- **Policy A:** Basic GitHub CI green on the exact checked-out Git SHA is the authoritative release policy. Upstream Kaggle dual-T4 report and production freeze tuples are optional historical attachments; set `LEGALIR_STRICT_GATES=1` to opt into legacy fail-closed lineage requirements.
- **Working tree:** Teammate readiness implementation in progress; verified read-only HF preflight (no `create_repo`), exact-5 ZIP gate enforcement, hardened volume cache identity, and zero-drift notebooks.
- **Launch:** Single entrypoint `python scripts/modal/run_full.py --warm --private --push-config --hf-repo <repo> --hf-allow-public-repo` (preflight runs automatically). Details in `docs/TEAMMATE.md`.
- **Targets, not demonstrated results:** Full private run (5-fold OOF + disjoint + final + 2080-query inference) has not completed on A100 yet; OOF/private recall and end-to-end runtime are projections until a real run finishes.
- **Parameter audit:** 702,754,049 parameters across the dense encoder and reranker (~715M with the r=64 LoRA adapter), below the competition's 4B ceiling. Re-run the audit if the models change.
- A timeout caps duration, not spend; retries/re-runs bill extra. There is no checkpoint-resume — a killed run restarts from scratch (Volume holds forensics only).

## Essential documents

| Document | Purpose |
|---|---|
| [TEAMMATE.md](docs/TEAMMATE.md) | **Start here.** How to run, what was built, workflow, verification, recovery |
| [Task 1 Plan](docs/TASK1_MAX_SCORE_MIN_TIME_PLAN.md) | Quality and efficiency plan, benchmarks, and M0-M4 milestones |

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
