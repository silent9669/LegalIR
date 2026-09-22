# INSTRUCTION — Toàn cảnh LegalIR: từ fix, đẩy điểm, tới warm Volume (cho agent review)

> File này là single source of truth để một agent khác review lại toàn bộ công việc.
> Đọc file này + chạy các lệnh validate trong mục 7 là đủ hiểu và kiểm chứng.
> Ngôn ngữ: Vietnamese. Repo local: `/Users/phucdang/Documents/LegalIR - Public Test`.

---

## 1. Bối cảnh

- **Bài toán:** UIT Data Science Challenge 2026, Task 1 — Vietnamese Legal Information Retrieval. Private test: **2.080 queries**, mỗi query nộp đúng 5 doc IDs.
- **Pipeline:** Legal BM25 + PyVi BM25 + DEk21 dense + exact match + fold-local question memory → fusion RRF → BGE LoRA cross-encoder rerank → 5-fold OOF + doc-disjoint eval → final train trên 7.000 queries → inference + submission.
- **Base models (pin revision, không được đổi):**
  - Reranker: `BAAI/bge-reranker-v2-m3` @ `953dc6f...` (~567M params)
  - Dense: `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` @ `99a2963...` (~135M params)
  - Budget: ~702M–715M / 4.0B competition limit (luôn PASS).
- **HF repo (production):** `dangphuc2109/legalir-task1-reranker` — root là canonical adapter đang phục vụ; `runs/` chứa lịch sử:
  - `runs/run-04-oof89.6/` — Run04, Modal A100-40GB, 5.4h, OOF 89.53%, candidate@150 98.82%, disjoint 87.49%.
  - `runs/run_7020249ea5cd7aed_20260922_040108/` — Run05, Modal A100-80GB, full ~7h **timeout-kill đúng lúc inference**, rescue inference-only 1.6h, OOF 90.44%, candidate@200 99.03%, disjoint 88.03%. File `submission.zip` trong đó chính là bản nộp private mới nhất.
- **Điểm private hiện tại: 0.8953** — cải thiện rất nhỏ so với bản cũ dù OOF tăng +0.91pp. Đây là vấn đề trung tâm.

## 2. Chẩn đoán (đã kiểm chứng từ logs HF + code)

1. **Trần nằm ở ranking, không phải retrieval.** Candidate@200 đã 99.03% mà OOF chỉ 90.44% → ~87% lỗi còn lại là reranker xếp gold ở hạng ≥6.
2. **Warm-start khóa rank ở r=8 (phát hiện quan trọng nhất).** Final model warm-start từ HF adapter r=8 nên mọi config `lora.r` lớn hơn đều bị PEFT lờ đi trong im lặng cho chính model giao bài.
3. **BCE pointwise sai objective.** Metric hỏi "gold có hơn 4 thằng khác?", BCE hỏi "pair này relevant?" → không dạy top-5 boundary.
4. **Negatives thiếu boundary.** Thứ tự mining cũ ưu tiên branch lẻ; `medium_neg` (rank 20-80) gần như không bao giờ được sample với query 1 gold; không có dải near-miss rank 3-15.
5. **Lệch train/infer:** train `max_length 512` vs infer default 384; `max_chunks` 3 vs 2; `rerank_k=100` vứt 100 docs đã retrieve (gán -999).
6. **Fusion RRF cố định thiên lexical:** `exact 2.5 / memory 2.0` đè `rerank 1.8` → mang bias memorize sang private (disjoint rớt ~2.4pp so với OOF là bằng chứng).
7. **Timeout 7h kill oan:** killer duy nhất là `TIMEOUT_SECONDS=25200` trong Modal function; Volume chỉ giữ forensics, không resume → Run05 chết đúng lúc inference sau khi train xong.
8. **Gates SHA/release tốn thời gian:** exact-40-char SHA + freeze/report digest + chuỗi Kaggle→Colab→A100 bắt buộc release mới cho mỗi lần sửa.

## 3. Những gì đã sửa (theo phase)

### Phase A — Bỏ gates/time limits (mặc định advisory; `LEGALIR_STRICT_GATES=1` bật lại chế độ cũ)
| File | Đổi gì |
|---|---|
| `scripts/modal/run_modal_a100.py` | Timeout 7h → 86400 (24h, coi như không giới hạn); SHA bất kỳ làm run label; provenance preflight chỉ warn; thêm step `warm_cache` (§4) |
| `src/release/fingerprints.py` | `assert_exact_git_sha` advisory mặc định + hàm `strict_gates_enabled()` |
| `scripts/colab/bootstrap.py` | `verify_launch` advisory (chỉ missing file mới fail); bọc `verify_prior_gate_reports` về warn |
| `src/release/acceptance.py` | Time/quality gates advisory (ghi `details`, không fail verdict); default `TIME_GATE 86400` |
| `src/pipeline/kaggle_train.py` | `STRICT_GATE` thành advisory forecast bound 86400; không abort ở đâu |
| `scripts/verify_release_approval.py` | `--allow-runtime-changes` pass ở chế độ thường |
| `scripts/verify_prepush.py` | Fast mặc định (<1 phút); full gates chỉ với `--strict` |
| `scripts/modal/run_modal_cli.sh` | Bỏ bắt clean-tree/SHA; thêm `--push-config`, `--warm`, `--warm-only` |

### Phase B — Tăng tốc
- `configs/experiments/reranker_lora.yaml`: `inference_batch_size 64→128`, `rerank_k 100→200` (= `candidate_k`, hết truncation -999).
- `src/pipeline/kaggle_train.py`: shared index cache `LEGALIR_INDEX_CACHE_DIR` (`warm_index_cache`/`persist_index_cache`) — attempt UUID mới không rebuild BM25×2 + DEk21 + query-embs (~2.5k s); A100 tự set về Volume `shared/indexes`.
- `src/pipeline/kaggle_train.py`: `LEGALIR_RERANKER_CONFIG` override config reranker.

### Phase C — Đẩy điểm (max-score, không giới hạn gì)
- **Loss listwise làm default** (base + v3); `RerankerGroupDataset` nhận `max_negatives_per_group=12` từ config (trước cap cứng 7) — `src/training/trainer.py`.
- **Cold-start mặc định** (`pretrained_lora_path: null` cả 2 config) để thoát rank-lock r=8; hedge warm-start vẫn mở qua env `LEGALIR_WARM_START_ADAPTER` + cảnh báo rank-mismatch trong `setup_peft_model`.
- **Base:** LoRA r=32/alpha=64, `coverage_epochs: 2` (mới, trong `src/training/train_reranker.py`: `effective = max(cfg, requested, req*epochs)`).
- **V3 push** (`configs/experiments/reranker_lora_v3_push.yaml`): r=64/alpha=128, listwise, 2 epochs (~1750 steps final), cold.
- **Mining** (`hard_negative_miner.py`, `build_pairs.py`, `pair_materializer.py`): thêm dải `near_miss` (hybrid rank 3-15); thứ tự ưu tiên exact → hybrid → near_miss → dense → memory → medium → branches; budget 12/query 1-gold = exact2+hybrid3+near3+dense1+memory1+medium2; `negatives_per_positive` 8/10 → 12.
- **Evidence/length đồng bộ:** `pipeline.yaml` + `legalir_v2.yaml`: reranker `max_length 384→512`, `max_chunks 2→3`, `max_chars 1200→1600`.
- **Fusion:** `w_rerank 1.8→2.5` (`src/ranking/fusion.py` + 2 fallback dicts trong `src/production/public_rerank.py`), override qua `LEGALIR_FUSION_W_RERANK`.
- **Ensemble 6 adapters** (`src/ranking/ensemble.py` mới): trung bình điểm final + 5 fold qua `score_fn` injection (không đụng aggregation/fusion/selector); chỉ dùng cho **private inference** (`predict.py` param mới `reranker_ensemble_adapter_paths`, `kaggle_train.py` resolve từ `cv/fold_i/reranker_adapter`, tắt bằng `LEGALIR_ENSEMBLE=0`, smoke không ensemble). OOF/disjoint giữ single-adapter → số đo trung thực, không leak. Wrapper giữ `model/tokenizer/adapter_path` của member[0] để audit + reload probe hoạt động.

### Phase D — Warm Volume trước khi bật A100 (mới nhất)
- **`scripts/modal/warm_volume.py` (mới):** Modal function CPU rẻ (4 CPU/16GB, timeout 2h, image nhẹ không torch) nạp sẵn vào `legalir-production:shared/`:
  - `shared/models/huggingface/` — 3 pinned models qua `download_models()` có sẵn + `manifest.json` id→snapshot; kèm LoRA adapter nếu xin warm.
  - `shared/dataset/` — dataset Kaggle 616MB qua `prepare_dataset()` (verify fingerprint, đủ file thì skip).
  - `shared/warm_manifest.json` — tổng hợp models/revisions/dung lượng/timestamp.
- A100 tự dùng cache qua `attach_warmed_cache()` (helpers + step `warm_cache` trong `run_modal_a100.py`): set `HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE`/`TRANSFORMERS_CACHE`/`HF_HOME` về thư mục warmed, mirror manifest vào `artifacts/local/models/huggingface/manifest.json`, ghim `LEGALIR_MODAL_DATASET_DIR` về dataset warmed. **Thiếu/dở cache → rơi về download như cũ, không bao giờ crash.**
- `run_modal_cli.sh` + `run_full.py`: cờ `--warm` (warm rồi dispatch) và `--warm-only` (chỉ warm).

### Entrypoint cho teammate
- `python scripts/modal/run_full.py --warm --private --push-config --detach` (preflight: `validate_score_push.py` + audit <4B + check token/modal CLI).
- `scripts/validate_score_push.py` (mới): **34 checks** coherence (configs, fusion, miner fill/priority/diversity, ensemble math/resolve, budget r32/r64).
- `docs/TEAMMATE_RUNBOOK.md`: runbook 6 bước.

## 4. Kết quả test (đã chạy thật trên CPU local)

- Full suite xanh: unit/contracts/release **440 passed**, integration **81**, parity/leakage/memory/dataset/notebook **65**; skipped 2 (cũ).
- Mới: `tests/unit/test_score_push.py` (9 passed), `tests/unit/test_warm_volume.py` (6 passed), +3 tests CLI warm trong `tests/contracts/test_modal_cli.py`.
- `smoke_kaggle_pipeline --tiny` SUCCESS; `verify_prepush` PASS; `check_no_fallbacks` PASS; notebook drift/parity PASS; audit <4B PASS (17.57%); `run_full.py --dry-run --warm-only` PASS.
- Trong quá trình test đã bắt và fix 1 bug thật (`epochs` ngoài scope khi coverage tắt, `train_reranker.py`).

## 5. Dự định re-construct HF repo (để agent review cho ý kiến)

Layout hiện tại: root = canonical adapter + `runs/run-04-oof89.6/` + `runs/run_7020249..._20260922_040108/` (mỗi run: `RUN_SUMMARY.md`, `LOGS.md`, `MODELS.md`, `submission.json/.zip`, `reports/`, `logs/`).
Đề xuất gọn lại: root giữ duy nhất production adapter + `README.md` (benchmark) + `training_manifest.json`; mỗi run mới là `runs/<run-id>/` tự chứa đầy đủ (model files riêng nếu khác root, không duplicate); thêm `runs/INDEX.md` liệt kê OOF/disjoint/hardware/submission từng run.

## 6. Checklist cho agent review (soi kỹ những chỗ này)

1. **OOF honesty:** ensemble chỉ dùng ở private inference? Fold adapter nào cũng mù đúng val-fold của nó ở OOF? (`oof_runner.py`, `kaggle_train.py` § ensemble, `predict.py`).
2. **Warm-start rank-lock:** cold-start default có làm mất điểm so với warm r=8 không? Có nên 1 run hedge `LEGALIR_WARM_START_ADAPTER`?
3. **Listwise groups:** `max_negatives_per_group=12` vs mining 12/query có khớp mọi trường hợp multi-gold không? (`trainer.py:337-372`).
4. **Mining priority:** phân bổ 2+3+3+1+1+2 cho query 1-gold có bỏ sót error-mode nào của dense không?
5. **Fusion 2.5:** là guess chưa qua GPU-validate — có nên tune trên disjoint thay vì fix cứng?
6. **Ensemble cost:** 6× inference (~5-6h) có đáng so với gain bagging kỳ vọng? `LEGALIR_ENSEMBLE=0` fallback đủ chưa?
7. **Warm fallback:** mọi đường thiếu cache có thật sự không-crash? (`attach_warmed_cache`, `warm_shared_cache`, manifest partial).
8. **Budget 4B:** ensemble 6 adapters + r=64 tính đúng luật thi không? (ước tính local 0.715B).
9. **Strict-mode parity:** `LEGALIR_STRICT_GATES=1` có khôi phục đúng hành vi cũ cho release qualify? (tests strict đã cover).
10. **HF re-construct (§5):** layout đề xuất có mất provenance nào agent khác/BTC cần không?

## 7. Lệnh kiểm chứng nhanh (agent review chạy các lệnh này)

```bash
.venv/bin/python scripts/validate_score_push.py
.venv/bin/python scripts/verify_prepush.py
.venv/bin/python scripts/audit_parameters.py --check-only
.venv/bin/python -m pytest tests/unit/test_score_push.py tests/unit/test_warm_volume.py tests/contracts/test_modal_cli.py -q
.venv/bin/python scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
.venv/bin/python scripts/modal/run_full.py --dry-run --warm --private --push-config
```

## 8. Rủi ro tồn tại (nói thẳng)

- Private 0.8953 có thể không theo OOF: distribution shift 2080 queries + multi-gold penalty + RRF fixed. Ensemble + w_rerank 2.5 là cược pro-generalization nhưng **chỉ run GPU thật mới biết**.
- Cold-start r=64/listwise/2-epochs chưa từng chạy GPU: 2-3 ngày đầu nên có 1 run hedge (warm r=8 hoặc base r=32) song song để so.
- Fusion 2.5 và phân bổ mining là heuristic từ log cũ, chưa grid-search trên disjoint.
- Ensemble ×6 nhân đôi VRAM load (~8GB, vẫn thoải mái trên 80GB) và ~5-6h inference — nằm trong trần 24h nhưng không còn margin nhiều nếu OOF phình.
