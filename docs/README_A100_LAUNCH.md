# Modal A100 Launch Guide

Reviewed 2026-09-22. Production execution guide for LegalIR Task 1 on Modal A100-80GB.
No release required — see [TEAMMATE.md](../TEAMMATE.md) for the full runbook. This guide covers launch mechanics only.

## Pre-Launch Requirements

1. HEAD `f867ab4` (runtime `6b57ed7`, Kaggle dual-T4 PASS v73) or newer; full test suite green (587 passed).
2. `.env` with `HF_TOKEN_WRITE` + Kaggle credentials; Modal secrets `kaggle-secret` and `huggingface-secret` on the dashboard; `.venv/bin/modal setup` done.
3. Canonical dataset resolvable (warm Volume or Kaggle download); pinned base-model revisions unchanged.
4. Run `--dry-run` green before every dispatch.

## Modal A100 Execution

### Launch Commands (via `scripts/modal/run_full.py` — runs preflight first)

```bash
# Max-score private run (warm + 2080 queries + v3 config + detached, recommended):
.venv/bin/python scripts/modal/run_full.py --warm --private --push-config --detach

# Public smoke (1000 queries, cheap pipeline validation):
.venv/bin/python scripts/modal/run_full.py --warm --detach

# Warm shared Volume only (CPU, no A100 billing):
.venv/bin/python scripts/modal/run_full.py --warm-only
```

`--detach` lets the app survive client disconnect (record app ID, watch `modal app logs <id>`, stop with `modal app stop <id> --yes`). Default attached mode dies with the client. `--private` selects the 2,080-query private set with exact-5 submission validation; `--push-config` selects `reranker_lora_v3_push.yaml` (listwise, LoRA r=64 cold start).

Strict legacy behavior (exact-SHA + clean-tree + fail-closed provenance) is opt-in via `LEGALIR_STRICT_GATES=1`; default is advisory and any run label is accepted for Volume paths.

### Resources, Lifetime, and Persistence

- **Allocated Hardware**: 1 × NVIDIA A100-80GB GPU, 8 dedicated vCPUs (`cpu=8.0`), 32 GiB host RAM (`memory=32768`).
- **Timeout**: `86400` seconds (24h platform max = effectively no limit). Set `MODAL_TIMEOUT_SECONDS` only to cap spend. Timeout caps duration, not spend — retries bill extra, and there is no checkpoint-resume.
- **Volume Mount**: `/root/legalir_volume/<label>/attempts/<uuid>/` on persistent Volume `legalir-production`. Every attempt is a fresh UUID; no cross-attempt resume.
- **Output Artifacts**: Checkpoints, `submission.zip`, manifests, logs, and `recovery.tar.gz` are written directly to the persistent Volume.

### Supervision and Monitoring

1. **Log Streaming**:
   ```bash
   modal app logs <app-id>
   ```
2. **Retrieve Completed Artifacts**:
   ```bash
   modal volume get legalir-production <label>/attempts/<uuid>/ ./local_artifacts/
   ```
3. **Emergency Stop**:
   ```bash
   modal app stop <app-id> --yes
   ```

## FULL Completion and Later Reuse

The required workload includes cold retrieval/indexing, five fold trainings/evaluations, document-disjoint training/evaluation, fusion evaluation, dedicated final training, final-model reload, test inference, validation, and durable delivery.

Before declaring success, verify all expected query IDs (2,080 for private round), required model/tokenizer artifacts, immutable base-model identity, checksums, valid submissions, delivery receipts, and provider shutdown. Preserve the original attempt path and logs for any failure investigation. If training finished but inference died, reuse the attempt's adapters inference-only (see `TEAMMATE.md` §7) instead of re-dispatching.
