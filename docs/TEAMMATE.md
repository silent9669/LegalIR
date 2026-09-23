# TEAMMATE.md — Hướng dẫn chạy + những gì đã làm + workflow

> Runbook lịch sử; **chưa phải lệnh GO cho fresh production**. Tại `a342d7f`, HF forwarding đã push và offline suite có 655 passed/2 skipped, nhưng strict release gate vẫn FAIL vì freeze khóa runtime `6b57ed7`; chưa có GPU benchmark/điểm mới. Chỉ dispatch FULL sau khi gate Kaggle dual-T4 + approval đúng SHA cuối, kiểm tra account/quota/secrets/repo public + write và ngân sách được duyệt. A100 có thể là 40 GB. Thiếu đích HF tường minh thì dry-run exit 2.
> Mốc lịch sử: release `f867ab4` (runtime `6b57ed7`, Kaggle dual-T4 PASS v73); 4 surgical fixes đã commit ở `30631f4`. Số **587 passed, 2 skipped** thuộc lần kiểm tra trước, không phải kiểm chứng cho SHA mới.

---

## 1. Lệnh tham khảo — chỉ chạy FULL sau khi GO theo kế hoạch fresh-account

```bash
# 0. Chuẩn bị 1 lần duy nhất (xem §2). MÁY MỚI: .venv không nằm trong git — chạy setup:
bash scripts/setup.sh
# Tạo .env CHỈ khi chưa có (không ghi đè file đã tồn tại):
[ -f .env ] || cat > .env <<'EOF'
HF_TOKEN_WRITE=hf_token_write_cua_ban
KAGGLE_USERNAME=username_cua_ban
KAGGLE_KEY=kaggle_key_cua_ban
HF_REPO_ID=dangphuc2109/legalir-task1-reranker
EOF
.venv/bin/modal setup  # authenticate Modal CLI

# 0b. Preflight read-only (không tạo repo/upload/GPU) — BLOCKED exit 2 nếu thiếu đích HF,
# token, secrets hoặc tree bẩn; UNVERIFIED cho quota/budget; FAIL cho strict gate:
.venv/bin/python scripts/preflight_teammate.py --hf-repo dangphuc2109/legalir-task1-reranker --hf-allow-public-repo

# 1. Dry-run kiểm tra trước khi tốn tiền (CPU local, ~1 phút; BLOCKED exit 2 nếu thiếu --hf-repo)
.venv/bin/python scripts/modal/run_full.py --dry-run --warm --private --push-config --hf-repo dangphuc2109/legalir-task1-reranker --hf-allow-public-repo

# 2. Chỉ sau khi strict release gate PASS và đã duyệt chi phí: warm Volume + private 2080q + config v3 + detached + public opt-in
.venv/bin/python scripts/modal/run_full.py --warm --private --push-config --detach --hf-repo dangphuc2109/legalir-task1-reranker --hf-allow-public-repo
# → ghi lại app ID + Volume attempt path in ra màn hình
# → theo dõi:  modal app logs <app-id>
# → dừng:     modal app stop <app-id> --yes
```

Các biến thể:

```bash
# Chỉ warm Volume, không bật A100 (nên chạy 1 lần trước run đầu tiên)
.venv/bin/python scripts/modal/run_full.py --warm-only --hf-repo dangphuc2109/legalir-task1-reranker --hf-allow-public-repo

# Lệnh public 1000q lịch sử: không phải single-fold pilot, vẫn có thể tốn full training; chỉ chạy khi duyệt chi phí
.venv/bin/python scripts/modal/run_full.py --warm --detach --hf-repo dangphuc2109/legalir-task1-reranker --hf-allow-public-repo

# Tắt ensemble (ít inference passes hơn; tác động điểm CHƯA đo) — xem §5
LEGALIR_ENSEMBLE=0 .venv/bin/python scripts/modal/run_full.py --warm --private --push-config --detach --hf-repo dangphuc2109/legalir-task1-reranker --hf-allow-public-repo
```

---

## 2. Chuẩn bị một lần (checklist)

- [ ] Chạy `bash scripts/setup.sh` (idempotent; không động vào `.env`/secrets/cloud), rồi tạo `.env` thủ công từ `.env.example` nếu chưa có.
- [ ] Chạy read-only preflight `scripts/preflight_teammate.py --hf-repo dangphuc2109/legalir-task1-reranker --hf-allow-public-repo`: không BLOCKED; ghi nhận các mục UNVERIFIED và strict FAIL (nếu còn) cho quyết định GO.
- [ ] `.env` có `HF_TOKEN_WRITE` (token Hugging Face quyền WRITE trên đúng repo public) và `KAGGLE_USERNAME`/`KAGGLE_KEY`.
- [ ] Trên Modal dashboard tạo 2 secrets: `kaggle-secret` (`KAGGLE_USERNAME`, `KAGGLE_KEY`) và `huggingface-secret` (`HF_TOKEN`).
- [ ] `.venv/bin/modal setup` thành công.
- [ ] Chạy `--warm-only` 1 lần: nạp sẵn 3 pinned models (`MODEL_REGISTRY`) + dataset + manifest vào Volume `legalir-production:shared/`. Mọi run sau tái dùng khi manifest verified (đủ model + đúng revision + SHA nguồn khớp); thiếu/lệch thì A100 tự download và ghi nhận fallback trong `warm_cache_summary.json`.
- [ ] Chạy `--dry-run` xanh (preflight §4) trước mỗi lần dispatch: chỉ xác nhận cú pháp đích HF tường minh và kiểm tra local; **không** xác nhận remote write access/profile/secrets/quota hoặc release approval. Thiếu repo explicit là BLOCKED exit 2.
- [ ] Tài khoản riêng: `--hf-repo dangphuc2109/legalir-task1-reranker` + `--hf-allow-public-repo` (repo public có chủ ý; token bạn phải có write) — xem `scripts/check_hf_repo.py --repo dangphuc2109/legalir-task1-reranker --allow-public-repo` phải exit 0 (chi tiết §8).

---

## 3. Những gì đã làm (để hiểu vì sao lệnh chạy như vậy)

### 3a. Bối cảnh bài toán

UIT DSC 2026 Task 1 — Vietnamese Legal IR. Corpus 8.532 văn bản / 1.15M passages, 7.000 queries train, private test **2.080 queries × đúng 5 doc_ids** (`submission.zip` chỉ chứa `submission.json` ở root). Trần tham số **< 4.0B** — hệ thống dùng ~0.70–0.72B (PASS).

### 3b. Recipe max-score của run này (đã validate offline, không cần chỉnh)

- **Reranker:** BAAI/bge-reranker-v2-m3, loss **listwise** (1 positive + 12 negatives), LoRA **r=64/alpha=128 cold-start**, 2 coverage epochs (~1750 steps final). Base config r=32. File: `configs/experiments/reranker_lora_v3_push.yaml` (`--push-config` chọn file này).
- **Mining:** 12 negatives/query, ưu tiên boundary (hybrid-top + near-miss rank 3–15), giữ 1 dense + 1 memory cho đa dạng lỗi.
- **Đồng bộ train/infer:** `max_length 512`, evidence `max_chunks 3`, `candidate_k = rerank_k = 200` (không truncate doc nào thành -999).
- **Fusion:** RRF `w_rerank 2.5` (reranker lấn át lexical trên query lạ).
- **Ensemble 6 adapters** (final + 5 fold, trung bình điểm) — **chỉ cho private inference**; OOF/disjoint giữ single-adapter nên số đo trung thực, không leak.
- **Warm Volume:** models + dataset nạp sẵn trên CPU trước, A100 chỉ train/infer (xem §6).

### 3c. 4 surgical fixes vừa review APPROVED (so với `f867ab4`)

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `src/pipeline/oof_runner.py:1343-1375` | `dup_data.get("duplicate_groups", [])` sai schema — file thật là `dict{gid:[docs]}` nên duplicate-expansion chết lặng, near-duplicate của held-out docs lọt vào train | Unwrap cả dict/list/wrapper `{"doc_ids":...}`, ép `str()`, `except` chỉ bắt `JSONDecodeError` + log `N val docs -> M after closure (G groups)` |
| 2 | `src/evaluation/submission.py:142` | `validate_submission_zip()` chỉ check lỏng 1–5 docs, không enforce exact 5 | Thêm `exact_answer_count=None` (backward-compatible), forward vào `validate_submission()` |
| 3 | `src/ranking/reranker.py:460` | Matcher OOM chứa bare `"mps"` — nuốt nhầm mọi lỗi MPS khác thành halve-batch | Thu hẹp thành `"mps out of memory"` / `"mps backend out of memory"`; logic halve + giữ `idx` nguyên |
| 4 | `src/training/trainer.py:505-529` | Warm-start adapter cũ r=8 vs config r=64 chỉ warning rồi lặng lẽ dùng r=8 | `LEGALIR_STRICT_GATES=1` + rank mismatch → `raise RuntimeError`; tắt strict thì giữ warning cũ; lỗi load thật vẫn fallback |

Kèm 2 unit tests mới trong `tests/leakage/test_warm_start_isolation.py` (zip exact-count, strict rank-lock).

---

## 4. Preflight tự chạy mỗi lần dispatch (không cần chạy tay)

`run_full.py` tự chạy trước khi gọi Modal:

1. `scripts/validate_score_push.py` — coherence configs/fusion/miner/ensemble/budget (ALL CHECKS PASSED mới cho đi tiếp).
2. `scripts/audit_parameters.py --check-only` — tổng tham số < 4B.
3. Check `.env` token/Kaggle creds + `modal CLI` authenticated.

Muốn chạy tay từng cái (CPU local, không GPU):

```bash
.venv/bin/python scripts/validate_score_push.py
.venv/bin/python scripts/audit_parameters.py --check-only
.venv/bin/python scripts/generate_notebooks.py --check-drift
.venv/bin/pytest tests/leakage/ tests/integration/test_doc_disjoint.py tests/contracts/test_modal_cli.py tests/integration/test_modal_delivery.py -q
```

---

## 5. Biến môi trường hữu ích

| Biến | Tác dụng |
|---|---|
| `HF_REPO_ID` | Đích HF tường minh (thiếu → dry-run exit 2 BLOCKED). Đích có chủ ý là repo PUBLIC `dangphuc2109/legalir-task1-reranker` + `--hf-allow-public-repo`; KHÔNG tự tạo repo khác hay để preflight tạo repo nếu chưa được phép riêng |
| `LEGALIR_RERANKER_CONFIG` | Override config reranker (`--push-config` = v3 push) |
| `LEGALIR_ENSEMBLE=0` | Tắt ensemble 6-adapter, chỉ dùng final adapter (nhanh, điểm thấp hơn) |
| `LEGALIR_WARM_START_ADAPTER` | Hedge warm-start từ adapter cũ — **lưu ý rank-lock r=8** (§3c-fix 4) |
| `LEGALIR_FUSION_W_RERANK` | Override trọng số rerank (mặc định 2.5) |
| `MODAL_TIMEOUT_SECONDS` | Chỉ giới hạn thời gian MỘT job (mặc định 86400 = 24h platform max), KHÔNG phải trần chi tiêu — retries/re-runs/builds vẫn tốn tiền, cần budget + stop tay |
| `LEGALIR_STRICT_GATES=1` | Bật lại gates cũ (exact-SHA, clean-tree, fail-closed) khi cần qualify release |
| `LEGALIR_INDEX_CACHE_DIR` | Remote tự set về Volume `shared/indexes` (không cần đụng) |

---

## 6. Workflow của một run (chuyện gì xảy ra trên cloud)

```text
run_full.py (local preflight, HF repo fail-closed khi thiếu explicit)
  → run_modal_cli.sh (HF repo resolve flag>env>.env, local provenance check + MODAL_ARGS)
    → run_modal_a100.py :: main() (local entrypoint, forward private/reranker_config/hf_repo)
      → run_production_training.remote() trên A100 (SKU không đảm bảo 80GB — report mới nhất thấy 40GB):
          checkout SHA → warm cache attach (models+dataset từ Volume shared/, verified hoặc fallback download)
          → verify_launch (advisory) → HF preflight (fail trước khi tốn dataset) → prepare_dataset
          → 5-fold OOF (mỗi fold train LoRA riêng, cold-start,brochure)
          → doc-disjoint eval → final train 7000q → ensemble resolve
          → private inference 2080q → submission.json/zip (exact 5, unique, đúng corpus)
          → manifest + checksums → commit Volume → (HF upload)
```

- Mỗi attempt là thư mục UUID mới: `/root/legalir_volume/<label>/attempts/<id>/` trên Volume `legalir-production`. Không resume cross-attempt.
- Không checkpoint-resume: timeout kill → chạy lại từ đầu (Volume chỉ giữ forensics).
- Mặc định ATTACHED (`modal run`): tắt client là kill job. Muốn thả tay thì `--detach` + tự giám sát (app ID, logs, stop).

## 7. Lấy kết quả + rescue khi run chết giữa chừng

```bash
# Tải artifacts về local
modal volume get legalir-production <label>/attempts/<id>/ ./local_artifacts/

# Trong attempt: submission.json/.zip, checkpoints/reranker_final/,
# cv/fold_*/reranker_adapter/, manifests, training.log
```

Nếu run train xong nhưng chết ở inference (từng xảy ra — Run05): adapter + indexes còn nguyên trên Volume. Không cần dispatch lại — chạy inference-only từ attempt dir đó với `LEGALIR_RESCUE_ADAPTER_DIR` (xem docstring `run_full.py`).

Kết quả chuẩn: `submission.json`/`submission.zip` (2080×5, unique, 100% IDs trong corpus) + adapter đẩy lên HF `dangphuc2109/legalir-task1-reranker` + `RUN_SUMMARY.md`/`LOGS.md`/`MODELS.md` trong `runs/`.

## 8. Chạy trên tài khoản riêng (teammate không dùng chung account)

Repo GitHub là **PUBLIC** → clone nặc danh được, không cần cấp quyền:

```bash
git clone https://github.com/silent9669/LegalIR.git
cd LegalIR
```

Mỗi tài khoản Modal/HF/Kaggle là không gian riêng — không chia sẻ gì ngoài code:

1. **Modal (tài khoản của bạn):** `modal setup` bằng account bạn → tự tạo secrets `kaggle-secret`, `huggingface-secret` trong dashboard của bạn. Volume `legalir-production` tự tạo mới (rỗng) trong workspace của bạn → **khuyến nghị chạy `--warm-only` 1 lần** trước run đầu tiên để nạp cache (không bắt buộc: thiếu cache thì A100 tự download + ghi nhận fallback, nhưng tốn GPU idle).
2. **Hugging Face (repo PUBLIC có chủ ý):** đích là repo public của LegalIR để ban giám khảo xác minh — KHÔNG đổi sang private, tạo repo khác, hay để preflight tạo repo mới nếu chưa được phép riêng. Mọi lệnh dispatch **bắt buộc** `--hf-repo dangphuc2109/legalir-task1-reranker` (hoặc `HF_REPO_ID` env/`.env`); thiếu đích tường minh là dry-run exit 2 BLOCKED. Production đi kèm `--hf-allow-public-repo` tường minh, và token của bạn phải có quyền WRITE trên đúng repo đó (kiểm tra read-only, không tạo repo: `scripts/check_hf_repo.py --repo dangphuc2109/legalir-task1-reranker --allow-public-repo` phải exit 0). Preflight thật có thể `create_repo` — là thao tác ghi ra ngoài, cần chấp thuận trước khi chạy.
3. **Kaggle (key của bạn):** dataset canonical dùng credential của bạn; không chia sẻ secret trong log.
4. **Chưa chạy y hệt §1** trên account mới: trước hết giải quyết blocker repo đích, xác minh Modal profile, Volume/secrets riêng và kiểm định SHA trên GitHub. Attempt path/Volume/app ID độc lập hoàn toàn với owner.

Checklist teammate sẵn sàng: clone đúng SHA đã push + `--dry-run` xanh **và** kiểm chứng remote profile/secrets/repo ID/visibility thật + warm asset fingerprint đúng + budget/stop được duyệt.

## 9. Rủi ro nói thẳng

- OOF/private recall và tổng thời gian là **ước tính** (train ~2–3h + inference ensemble ~5–6h) — chỉ run GPU thật mới biết.
- Ensemble ×6 tốn ~5–6h inference nhưng nằm trong trần 24h; tắt bằng `LEGALIR_ENSEMBLE=0` nếu cần nhanh.
- Cold-start r=64/listwise/2-epochs chưa từng chạy GPU end-to-end — cân nhắc 1 run hedge song song (base r=32 hoặc warm r=8).
- Fusion 2.5 và phân bổ mining là heuristic từ log cũ, chưa grid-search trên disjoint.
