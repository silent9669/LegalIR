## Task 4 Report

Status: DONE

Implemented the unified hybrid candidate search in `/Users/phucdang/Documents/LegalIR - Public Test/src/retrieval/hybrid_search.py`.

- Added `HybridSearchEngine.search(query, top_k_candidates=100, rrf_k=60, exclude_qid=None)`.
- Queries BM25 Micro, Dense Macro (DEk21-compatible), Train Question Memory, and Exact Matcher branches when supplied.
- Supports legacy `QuestionMemory` and the new `TrainQuestionMemory` result/method signatures.
- Deduplicates candidates by string `doc_id` within and across branches.
- Applies weighted RRF using the configured smoothing constant and retains branch ranks, contributions, metadata, and existing flattened candidate features.
- Preserved `search_candidates(..., top_k=...)` compatibility for pipeline callers.
- Added a four-branch RRF/metadata test covering unique IDs and exact score calculation.
- Extended `CandidateRecord` typing for the new branch metadata fields.

Commits:
- `4822422 feat(retrieval): unify 4-branch candidate search and RRF fusion`

Test summary: `PYTHONPATH=. ./.venv/bin/pytest tests/test_candidate_union.py tests/test_retrieval_branches.py -v` — 5 passed.
