# Final Acceptance Checklist

## Dataset

- [ ] canonical dataset = Task1 v2
- [ ] documents = 8,532
- [ ] chunks = 1,153,876
- [ ] micro = 934,416
- [ ] macro = 219,460
- [ ] train = 7,000
- [ ] qrels = 7,637
- [ ] public = 1,000
- [ ] duplicate groups = 4
- [ ] audit valid

## Static retrieval

- [ ] Legal cache parity
- [ ] PyVi cache parity
- [ ] Dense cache parity
- [ ] Exact cache parity
- [ ] fused candidate parity
- [ ] no qrels accepted by static-cache builder

## Evidence

- [ ] Arrow-backed
- [ ] lazy per-document preprocessing
- [ ] bounded LRU
- [ ] positive localization parity
- [ ] evidence text parity

## Leakage

- [ ] 5/5 folds train-only pairs
- [ ] 5/5 folds zero val pair overlap
- [ ] 5/5 folds memory train-only
- [ ] doc-disjoint isolation
- [ ] duplicate blacklist active

## Validation

- [ ] all five folds PASS
- [ ] doc-disjoint PASS
- [ ] aggregate OOF report
- [ ] promotion policy applied
- [ ] production lock frozen

## Bundle

- [ ] all file hashes
- [ ] canonical fingerprints
- [ ] runtime SHA
- [ ] config SHA
- [ ] static cache
- [ ] final pairs
- [ ] public candidates
- [ ] public evidence
- [ ] fusion artifact
- [ ] verifier PASS

## CI

- [ ] full pytest
- [ ] parity tests
- [ ] leakage tests
- [ ] memory tests
- [ ] release verifier
- [ ] GitHub CI GREEN

## Colab

- [ ] Tesla T4
- [ ] real DEk21
- [ ] real FAISS
- [ ] real BGE+LoRA
- [ ] stable microbatch
- [ ] effective batch 16
- [ ] finite loss
- [ ] param_diff >0
- [ ] adapter reload
- [ ] host RAM safe
- [ ] PASS

## Kaggle

- [ ] exact approved runtime
- [ ] T4×2
- [ ] final mode
- [ ] no OOF
- [ ] one final adapter
- [ ] public rerank
- [ ] <9h projection
- [ ] <4B
- [ ] exact 1,000 public keyset
- [ ] 1..5 unique docs/query
- [ ] submission.zip generated

## Final decision

```text
READY FOR SUBMISSION: YES / NO
```
