# LEGALIR Refresh Final Architecture Package

This ZIP is a complete fresh-start handoff for rebuilding Task 1 around the existing canonical v2 dataset.

Start with `00_START_HERE.md`.

The package intentionally separates:
- validation/artifact production;
- final Kaggle training/submission.

It is designed to preserve score semantics while removing the memory/runtime failure modes of the old monolithic run.
