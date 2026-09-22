# Teammate runbook — Modal A100-80GB max-score run (no release needed)

1. Chuẩn bị (1 lần):
   - `cp .env.example .env` rồi điền `HF_TOKEN_WRITE`, `KAGGLE_USERNAME`/`KAGGLE_KEY`.
   - Tạo Modal secrets `kaggle-secret`, `huggingface-secret` trên dashboard.
   - `.venv/bin/modal setup` để authenticate CLI.

2. Warm shared Volume trước (CPU rẻ, không tốn giờ A100):
   - `python scripts/modal/run_full.py --warm-only`
   - Tải sẵn: 3 pinned models + dataset 616MB + manifest vào
     `legalir-production:shared/`. Chạy 1 lần, mọi run sau tái dùng.

3. Chạy max-score: warm + private + v3 push + ensemble, detached (khuyến nghị):
   - `python scripts/modal/run_full.py --warm --private --push-config --detach`
   - `--warm` warm trước rồi mới bật A100 → A100 không ngồi download.
   - Preflight tự chạy: `validate_score_push.py` (configs/fusion/miner/ensemble/budget) + audit <4B.
   - Ghi lại app ID + Volume attempt path. Theo dõi: `modal app logs <app-id>`.
   - Dừng: `modal app stop <app-id> --yes`. Không timeout tự kill (mặc định 24h platform max).

3. Recipe max-score của run này (đã validate, không cần chỉnh):
   - Reranker listwise (base r=32 / v3 r=64 cold start), 2 coverage epochs,
     12 negatives/query (ưu tiên boundary hybrid-top + near-miss rank 3-15,
     giữ 1 dense + 1 memory cho đa dạng lỗi), max_length 512, chunks 3.
   - Inference: candidate 200 = rerank 200, batch 128, ensemble 6 adapters
     (final + 5 fold) trung bình điểm — chỉ cho private, OOF/disjoint giữ
     single-adapter nên số đo vẫn trung thực.
   - Fusion RRF w_rerank 2.5 (reranker lấn át lexical trên query lạ).
   - Thời gian ước tính A100-80GB: train ~2-3h + inference ensemble ~5-6h.

4. Biến môi trường hữu ích:
   - `LEGALIR_RERANKER_CONFIG` — override config reranker (`--push-config` = v3).
   - `LEGALIR_ENSEMBLE=0` — tắt ensemble, chỉ dùng final adapter.
   - `LEGALIR_WARM_START_ADAPTER=dangphuc2109/legalir-task1-reranker` — hedge:
     continual fine-tune từ adapter cũ (lưu ý: rank bị khóa ở r=8).
   - `LEGALIR_FUSION_W_RERANK` — override trọng số rerank (mặc định 2.5).
   - `MODAL_TIMEOUT_SECONDS` — chỉ để giới hạn chi tiêu (mặc định 86400).
   - `LEGALIR_STRICT_GATES=1` — bật lại gates cũ khi cần qualify release.
   - `LEGALIR_INDEX_CACHE_DIR` — remote tự set về Volume `shared/indexes`.

5. Nếu run chết ở inference sau khi train xong (như Run05):
   - Adapter + indexes còn trên Volume `legalir-production:<label>/attempts/<id>/`.
   - Ensemble chỉ cần adapter dirs → chạy lại inference-only từ attempt đó.

6. Kết quả: `submission.json`/`submission.zip` (2080×5, unique, đúng corpus IDs) + `RUN_SUMMARY.md`/`LOGS.md`/`MODELS.md` đẩy lên HF `runs/`.
