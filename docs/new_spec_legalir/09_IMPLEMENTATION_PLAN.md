# LegalIR Fresh Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build a resumable validation/artifact factory and a memory-safe final Kaggle trainer that inherit the official canonical Task1 v2 dataset while preserving leakage-safe ranking semantics.

**Architecture:** static label-free retrieval is computed once and frozen; evidence is lazy/Arrow-backed; validation is sharded; a verified production bundle feeds a final Kaggle run that performs only one final all-query LoRA training plus public reranking.

**Tech Stack:** Python 3.12, PyArrow, Pandas, NumPy, PyTorch, Transformers, PEFT, FAISS, LightGBM, pytest, GitHub Actions, Google Colab, Kaggle T4×2.

**Spec:** `01_FINAL_ARCHITECTURE.md`

## Global Constraints

- Task1 canonical v2 only.
- 8,532 docs, 1,153,876 chunks, 7,000 train, 7,637 qrels, 1,000 public.
- Learned params <4B.
- Recall@5 primary, Precision@5 tie-break.
- No validation leakage.
- Effective reranker batch remains 16 unless a separate score-promotion experiment approves change.
- Final Kaggle run performs no 5-fold OOF or doc-disjoint.

---

### Task 1: Canonical source adapter and provenance

**Files**
- Create `src/data/canonical.py`
- Create `src/core/hashing.py`
- Create `src/core/manifests.py`
- Test `tests/unit/test_canonical.py`

- [ ] Define a canonical identity dataclass.
- [ ] Read Parquet metadata without full materialization.
- [ ] Verify manifest/audit/public counts.
- [ ] Implement file hashing/fingerprints.
- [ ] Write RED tests for count/schema/public mismatch.
- [ ] Make tests GREEN.
- [ ] Commit.

### Task 2: Host memory telemetry

**Files**
- Create `src/core/memory.py`
- Test `tests/memory/test_memory_guard.py`

- [ ] Implement RSS/system/GPU snapshot.
- [ ] Implement `release_memory()`.
- [ ] Implement low-memory guard.
- [ ] Test mocked low-memory failure.
- [ ] Commit.

### Task 3: Static candidate cache

**Files**
- Create `src/retrieval/static_cache.py`
- Create `scripts/build_static_cache.py`
- Test `tests/parity/test_static_cache.py`

- [ ] Define cache schema.
- [ ] Build cache writer that accepts no qrels.
- [ ] Stream all train/public branch outputs.
- [ ] Implement cache reader.
- [ ] Prove live-vs-cache parity.
- [ ] Commit.

### Task 4: Dense lifecycle

**Files**
- Modify dense retriever implementation.
- Test `tests/parity/test_dense_lifecycle.py`

- [ ] Add explicit unload.
- [ ] Add optional matrix drop after FAISS.
- [ ] Preserve count metadata when matrix is dropped.
- [ ] Prove top-k parity.
- [ ] Commit.

### Task 5: Lazy MacroEvidenceStore

**Files**
- Create `src/evidence/macro_store.py`
- Test `tests/parity/test_macro_store.py`

- [ ] Read only macro-required columns.
- [ ] Build compact doc→row index.
- [ ] Add bounded LRU.
- [ ] Add cache-byte accounting.
- [ ] Test eviction.
- [ ] Commit.

### Task 6: Lazy positive localization/evidence

**Files**
- Create/modify `src/evidence/selector.py`
- Test `tests/parity/test_evidence_parity.py`

- [ ] Reproduce legacy localization formula lazily.
- [ ] Reproduce legacy evidence formula lazily.
- [ ] Validate exact selected IDs/text on deterministic official sample.
- [ ] Commit.

### Task 7: Cached pair materializer

**Files**
- Create `src/evidence/pair_materializer.py`
- Create `scripts/build_fold_pairs.py`
- Test `tests/leakage/test_pair_materializer.py`

- [ ] Fuse static cache + fold memory.
- [ ] Preserve negative-source policy.
- [ ] Apply duplicate blacklist.
- [ ] Stream pair rows.
- [ ] Assert train-only qids.
- [ ] Assert zero validation overlap.
- [ ] Commit.

### Task 8: Sharded fold job

**Files**
- Create `src/validation/fold_job.py`
- Create `scripts/run_fold.py`
- Test `tests/integration/test_fold_job.py`

- [ ] Define job manifest.
- [ ] Train adapter in isolated process.
- [ ] Produce OOF feature Parquet.
- [ ] Produce metrics/report.
- [ ] Implement resume verification.
- [ ] Commit.

### Task 9: Document-disjoint job

**Files**
- Create `src/validation/doc_disjoint_job.py`
- Create `scripts/run_doc_disjoint.py`
- Test `tests/integration/test_doc_disjoint.py`

- [ ] Reuse same pair/cache/evidence contracts.
- [ ] Enforce document-disjoint isolation.
- [ ] Produce robustness report.
- [ ] Commit.

### Task 10: Fusion/promotion

**Files**
- Create `src/validation/promotion.py`
- Create `scripts/select_production_config.py`
- Test `tests/unit/test_promotion.py`

- [ ] Aggregate fold Parquets.
- [ ] Train/evaluate fusion.
- [ ] Compare Recall@5 first.
- [ ] Apply Precision@5 tie-break.
- [ ] Enforce candidate/doc-disjoint guardrails.
- [ ] Write `production_lock.json`.
- [ ] Commit.

### Task 11: Final artifact materialization

**Files**
- Create `scripts/build_production_bundle.py`
- Create `src/bundle/builder.py`
- Create `src/bundle/verifier.py`
- Test `tests/release/test_bundle.py`

- [ ] Build final all-query pair file.
- [ ] Build public candidate/evidence files.
- [ ] Add all fingerprints.
- [ ] Hash every bundle file.
- [ ] Fail closed on mismatch.
- [ ] Commit.

### Task 12: Final Kaggle trainer

**Files**
- Create `src/production/final_train.py`
- Create `src/production/public_rerank.py`
- Create `src/production/submission.py`
- Create `scripts/run_kaggle_final.py`
- Test `tests/integration/test_final_production.py`

- [ ] Verify runtime/dataset/bundle.
- [ ] Train one final adapter.
- [ ] Fresh-reload adapter.
- [ ] Rerank public candidates.
- [ ] Support two-GPU public inference.
- [ ] Strict submission validation.
- [ ] Commit.

### Task 13: T4 throughput probe

**Files**
- Add to Colab smoke runner/config.
- Test runtime configuration logic.

- [ ] Probe 8×2.
- [ ] Fallback 4×4.
- [ ] Fallback 2×8.
- [ ] Require effective batch 16.
- [ ] Persist stable profile.
- [ ] Commit.

### Task 14: CI + release gates

**Files**
- Update `.github/workflows/ci.yml`
- Create `scripts/verify_release.py`
- Tests under `tests/release/`

- [ ] Full pytest.
- [ ] Parity tests.
- [ ] Leakage tests.
- [ ] Bundle verification.
- [ ] Notebook pin verification.
- [ ] Commit.

### Task 15: Final notebooks

**Files**
- Generate `notebooks/colab_t4_smoke.ipynb`
- Generate `notebooks/kaggle_final.ipynb`

- [ ] Colab runs final-training contract on bounded real data.
- [ ] Kaggle runs only final production.
- [ ] Both pin exact runtime.
- [ ] No secret printing.
- [ ] Commit.

## Final source gate

```bash
python -m compileall -q src scripts
pytest -q
python scripts/verify_dataset.py
python scripts/verify_production_bundle.py --bundle <bundle>
python scripts/verify_release.py
```

Then CI → Colab T4 → release pin → final Kaggle.
