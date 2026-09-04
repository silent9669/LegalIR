# LegalIR Operating Rules & Agent Boundaries

## 1. Non-Negotiable Competition Invariants
- **Data Boundary**: Organizer Task 1 data only (`artifacts/task1/data/` or symlinked `data/task1_canonical_v2/`). No Task 2 data, no external legal corpora, no external LLM APIs.
- **Learned Parameter Budget**: Total learned parameters must strictly remain `< 4,000,000,000` (4B). Monitored on every build via `scripts/audit_parameters.py`.
- **Ranking Semantics**: Official evaluation metric is Recall@5 primary, with Precision@5 as secondary tie-break. Maximum 5 predicted document IDs per query. Exactly 1,000 public queries must be predicted.
- **Zero Validation Leakage**:
  - Training pairs must strictly contain only fold training query IDs (`pair_qids ⊆ train_qids`, `pair_qids ∩ val_qids = ∅`).
  - Question Memory must be strictly fold-local (`memory_qids ⊆ train_qids`, `memory_qids ∩ val_qids = ∅`).
  - Validation queries must never appear in fold training pairs or question memory.
  - Negative candidate selection must enforce the transitive duplicate closure blacklist (`duplicate_groups.json`).

## 2. Architectural Separation of Concerns
The pipeline is strictly decoupled into two systems:
1. **Validation / Artifact Factory (Offline / Sharded)**:
   - Computes static label-free retrieval (Legal BM25, PyVi BM25, DEk21 Dense, Exact Matcher) once across all 8,000 queries without qrels.
   - Serves document macro chunks lazily via Arrow-backed `MacroEvidenceStore` with bounded LRU cache (<=512 MB).
   - Executes 5-fold OOF and document-disjoint validation as independent, process-isolated resumable jobs.
   - Selects and freezes production configuration (`production_lock.json`) only if Recall@5 improves or ties with higher Precision@5 without candidate recall degradation.
   - Assembles and verifies immutable production bundle.
2. **Kaggle Final Production Trainer**:
   - Pure production runner (`notebooks/kaggle_final.ipynb` / `scripts/run_kaggle_final.py`).
   - Verifies runtime Git commit SHA and canonical dataset identity.
   - Verifies production bundle fingerprints.
   - Trains exactly one final BGE LoRA adapter on all 7,000 queries with effective batch 16.
   - Reranks public candidates using frozen fusion.
   - Validates strict submission criteria (1–5 unique document IDs per query for all 1,000 public queries) and packages `submission.zip`.
   - Never reruns 5-fold OOF, doc-disjoint validation, or heavy index builds on Kaggle.

## 3. Release Governance Protocol
Release follows a strict sequential gate workflow:
1. **Gate A — GitHub Actions CI (`LegalIR CI`)**:
   - Must be GREEN on `main`.
   - Runs compileall, unit/parity/leakage/memory/integration/release tests (480+ tests), parameter audit (<4B), and notebook parity checks.
2. **Gate B — Google Colab Single-T4 Contract Smoke**:
   - Executes `scripts/run_colab_t4_smoke.py` on real Tesla T4 hardware.
   - Validates real DEk21 FAISS, pair mining, BGE LoRA forward/backward training probe, adapter reload, and public inference path.
3. **Gate C — Commit B Release Notebook Pin**:
   - Pins the approved 40-character Git commit SHA in `notebooks/kaggle_final.ipynb` and `legalir_training.ipynb`.
   - Validates byte-for-byte identity.
   - Verifies release approval consistency via `scripts/verify_release_approval.py`.
