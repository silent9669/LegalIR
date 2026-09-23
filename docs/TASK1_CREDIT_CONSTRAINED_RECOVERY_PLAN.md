# LegalIR Task 1 — kế hoạch cứu kết quả khi hết credit A100

> **Lưu trữ lịch sử (account cũ).** Người vận hành đã chọn fresh run trên tài khoản mới. Kế hoạch đang áp dụng: [TASK1_FRESH_RUN_TIME_AND_SCORE_PLAN.md](TASK1_FRESH_RUN_TIME_AND_SCORE_PLAN.md). Không dùng các bước tái sử dụng Volume/adapter dưới đây cho fresh run.

**Ngày lập:** 2026-09-23  
**Trạng thái:** kế hoạch, chưa triển khai/chưa chạy GPU.  
**Nguồn chính:** báo cáo phiên chạy `/Users/phucdang/Downloads/report.md` (bằng chứng do người vận hành cung cấp; chưa đối chiếu trực tiếp Modal Volume/log).  
**Checkout lúc lập kế hoạch:** `e337785a86060517da002f427c596bac65b0de5e`, working tree sạch. Đây **không** phải xác nhận SHA của container đã chạy: báo cáo ghi checkout HEAD `e337785` cùng ba sửa đổi local chưa commit (`report.md:5–8`), cần đối chiếu artifact/config thực trước khi tái sử dụng.

## Quyết định trước mắt

**Không chạy lại** `run_full.py --warm --private --push-config --detach`: full pipeline sẽ huấn luyện lại 5 fold, doc-disjoint và final adapter; không tự resume khi một attempt bị dừng. Mục tiêu mới là tạo **submission hợp lệ với chi phí thấp nhất có thể**, rồi chỉ thử các thay đổi inference/fusion được chứng minh tăng điểm trên validation. Không hứa đạt điểm cao hơn hoặc thời gian cụ thể trước khi đo.

Báo cáo nêu lần chạy trên **A100-SXM4-40GB** (không phải 80 GB), đã hoàn thành fold 0 (Recall@5 92,30%, 4.406,5 s) và fold 1 (91,48%, 4.267,0 s); fold 2 train xong nhưng OOF inference mới đến batch 42/44 lúc bị dừng. Chưa chạy fold 3–4, doc-disjoint, final training, private inference; chưa có submission hay upload mới (`report.md:18–38`). Hai fold hoàn thành **không phải** số đo OOF năm fold hoặc dự đoán điểm private. Chi phí fold 0+1 quan sát là 8.673,5 s (~2,41 giờ); không suy ra toàn bộ lần chạy chỉ từ phép nhân vì các stage khác cũng tốn thời gian.

## Những gì cần xác thực, không được giả định

1. Xem `launcher_state.json`, `resolved_config.yaml`, training log, checkpoint manifests và listing của đúng attempt `e337785a86060517da002f427c596bac65b0de5e/attempts/746144731ed54f4f8df7fb26626c7b67/` trên Volume. Chỉ tải metadata và file nhỏ trước, không tải toàn bộ model. Kiểm tra complete marker, prediction/metric coverage, checksum, tokenizer và base revision của từng fold. Fold 2 chỉ được tính hoàn tất nếu có đủ outputs/coverage; có adapter không đồng nghĩa fold hoàn tất. Xác nhận app cũ đã stop và không còn GPU tính phí. Đối chiếu report với log, không coi report tự thân là receipt độc lập.
2. Kiểm tra xem có **final adapter hoàn chỉnh từ lần chạy trước** (ví dụ run-04) trên Hugging Face/Volume không; xác minh nguồn gốc, cấu hình LoRA, base revision, tokenizer và quyền truy cập bằng credential sẽ dùng thật. Không dùng adapter MOCK của local smoke run làm model production. Nếu không có final adapter đủ tin cậy, fold 0/1 đã hoàn tất vẫn có thể là ứng viên *inference-only tạm thời* trên private (chúng chưa được train trên toàn bộ 7.000 train queries); phải ghi rõ đây không phải final model v3 và không thể lấy OOF của fold đó làm chứng cứ cho private score. Fold 2 là dự phòng **chỉ sau khi** kiểm tra checkpoint không dở dang và nạp lại được.
3. `recovery.tar.gz` **không tự bảo đảm chứa fold adapters**; archive theo danh sách release riêng. `scripts/modal/run_full.py:27–31` mô tả `run_modal_rescue.py`, nhưng entrypoint đó hiện không có trong repo; biến `LEGALIR_RESCUE_ADAPTER_DIR` không có đường inference-only đang được kiểm chứng. Không đưa ra lệnh rescue như thể đã chạy được.
4. Các hướng dẫn cũ ở `TEAMMATE.md`/`docs/README_A100_LAUNCH.md` về A100-80GB, zero-download, SHA và rescue có thể đã lỗi thời so với báo cáo mới; trước dispatch, tin vào thông số và artifact của attempt đã kiểm chứng, không tin một dự báo. Repo Hugging Face public hay private và quyền ghi cần kiểm tra live; không tự thêm cờ public override hoặc upload artifact.

## Lộ trình ưu tiên theo chi phí

### P0 — Cứu một submission hợp lệ, **không train**

- Sau bước xác thực, chọn **một adapter hoàn chỉnh**: ưu tiên final adapter có provenance tốt; nếu không có, cân nhắc fold 0 hoặc fold 1 như baseline suy luận có giới hạn. Không warm-start adapter fold vào training fold khác hoặc dùng fold đã nhìn thấy validation queries để báo OOF.
- Thiết kế **entrypoint inference-only tối thiểu** nhận đường adapter trên Volume, model/tokenizer revision đã pin, canonical dataset/index cache, private query file, output attempt mới và chế độ `no_upload` mặc định. Nó phải bỏ qua mining, OOF, doc-disjoint và final training; không sửa `run_full.py` để lặng lẽ đổi ngữ nghĩa của lệnh full. Trước khi trả tiền GPU, tạo unit/integration test bằng dữ liệu nhỏ cho resume provenance, load adapter, exact-five validation và packaging; chạy dry-run CPU. Đây là **việc cần implement**, không phải chức năng đã có.
- Benchmark một lát cắt private nhỏ **chỉ để đo throughput và kiểm tra schema** (không có private labels, không dùng để chọn hyperparameter); đo setup/index-cache hit và query/s trên chính GPU định thuê. Dự báo chi phí 2.080 queries + buffer theo throughput đo được rồi mới xin duyệt ngân sách. Không khẳng định warm cache giúp zero-download. Nếu vẫn vượt ngân sách, dừng thay vì khởi chạy cả batch.
- Chạy toàn bộ private **một lần đã được phê duyệt**; validate 2.080 qids đúng expected set, đúng 5 doc IDs duy nhất/query, tất cả nằm trong 8.532 corpus docs, ZIP chỉ chứa `submission.json`; ghi checksum, model/config SHA, attempt ID, runtime và trạng thái cuối. Chưa upload HF hay nộp bài trừ khi có chấp thuận riêng.

**Điều kiện thành công P0:** một `submission.zip` có chứng cứ kiểm tra đầy đủ, model lineage minh bạch, không phát sinh full retraining. Nếu không tìm được adapter hợp lệ, không thể cam kết nhánh zero-training sẽ làm ra submission; chuyển sang quyết định P2 hoặc dừng.

### P1 — Tìm tăng điểm rẻ, không huấn luyện lại

- Thu thập feature/prediction OOF **của fold 0 và 1 đã hoàn tất** từ đúng run mới, kèm qrels và coverage; chỉ dùng fold 2 nếu có đủ outputs. Phân biệt chúng với báo cáo các run cũ (cấu hình cũ `rerank_k=100`/rank khác), không trộn các run để tuyên bố cải thiện. **Nếu P0 chọn final adapter từ run cũ, OOF của các fold r=64 mới không đại diện cho model đó**: phải dùng held-out predictions được tạo bởi chính adapter/pipeline tương ứng trước khi tuning fusion. Nếu artifact chỉ có metric tổng hợp, cần thêm scoring/validation có giới hạn; không thể tuning đáng tin từ hai con số Recall@5.
- So sánh trên cùng held-out query set các **chính sách fusion định trước** (ví dụ trọng số reranker/RRF hiện tại với một vài mức cố định) dùng score features đã lưu; không dựa vào nhãn private. Tách phần chọn chính sách và phần xác nhận theo query/fold (ví dụ fold 0 chọn, fold 1 xác nhận; rồi đổi chiều như sensitivity check). Với chỉ hai fold, ước lượng không ổn định: chỉ nhận thay đổi nếu chênh lệch nhất quán, coverage bằng nhau, multi-gold Recall@5 không suy giảm và không phá doc-disjoint constraint; nếu không thì giữ baseline. Chọn trên tất cả nhãn rồi báo cùng số OOF như đánh giá độc lập là sai.
- Kiểm tra recall/quality thực ở các độ sâu rerank **100/150/200** trên dữ liệu validation có per-candidate score phù hợp. Chỉ hạ `rerank_k` từ 200 khi giảm thời gian đo được **và** mất Recall@5 nằm trong ngưỡng mà người vận hành chấp nhận; chưa có bằng chứng cho một ngưỡng miễn phí. Thay `candidate_k` hoặc `max_length` cũng có thể đổi score, không được gọi là optimization thuần tốc độ. Current push config đã có listwise, LoRA r=64, `rerank_k=200`, max_length 512, inference batch 128; không đề xuất lại như phát hiện mới.
- So sánh final-only với ensemble **chỉ khi** mọi adapter thành viên tồn tại, đủ provenance và có validation hợp lệ cho phép đo công bằng. `LEGALIR_ENSEMBLE=0` có thể giảm compute inference, **không được hứa** cải thiện điểm; bản thân ensemble 5 fold + final không có đủ adapter từ attempt bị dừng. Selector hiện chỉ loại trùng exact doc IDs; không có quy tắc chặn văn bản cùng luật để “gỡ” mà tăng điểm. Với multi-gold, cần đếm qrels và quan sát miss per-query thay vì suy từ tổng Precision/Recall.

**Điều kiện thành công P1:** cấu hình inference chọn trước khi chạy private, có bảng baseline/variant về Recall@5, độ trễ, số query và sai số/độ nhạy; nếu chưa có bằng chứng tăng điểm đáng tin, dùng baseline P0. Thay đổi chính sách scoring cần test parity và kiểm tra lại giới hạn <4B, submission contract, release lineage theo quy trình dự án.

### P2 — Chỉ khi người vận hành có thêm ngân sách và chấp nhận train

- Cân nhắc huấn luyện **một final adapter** từ toàn bộ train set nếu P0 không có model chấp nhận được hoặc baseline quá thấp; bỏ 5-fold OOF/doc-disjoint **chỉ nếu** mục tiêu là tạo submission tạm thời và báo rõ thiếu các chỉ số đánh giá bắt buộc. Đây là một pipeline mới cần thiết kế/test, không phải cờ sẵn có; tuyệt đối không xuất report đầy đủ từ các fold thiếu. Cấu hình push có **2 coverage epochs**; giảm riêng `max_steps: 250` không giúp nếu coverage floor vẫn nâng effective steps (`src/training/train_reranker.py:118–143`). Giảm epochs hoặc batch/negatives là thay đổi recipe có nguy cơ giảm điểm, phải thử trên holdout trước khi đánh đổi.
- Tái dùng fold 0/1 qua Volume để hoàn thành FULL 5-fold là hướng phát triển riêng, đòi identity/complete-marker checks và tính lại fold 2, doc-disjoint, final train. `src/pipeline/kaggle_train.py:1929` cố ý tắt stage reuse cho FULL; attempt mới có UUID riêng, không optimizer checkpoint-resume. Không được tự bật lại reuse, không ghép artifacts khác config/SHA; đây không phải đường tiết kiệm GPU đã được kiểm chứng.
- Trước mọi run trả tiền: chạy tests/preflight, đo pilot và xin chấp thuận riêng về GPU loại gì, ngân sách/thời gian trần, timeout/stop, HF visibility, upload. Không chạy full 8–10 giờ chỉ để “thử xem”.

## Thứ tự triển khai và cổng quyết định

| Thứ tự | Công việc | Kiểm chứng / điều kiện dừng |
|---|---|---|
| 1 | Audit artifact của attempt đã dừng + tìm adapter hoàn chỉnh | Metadata, file marker, SHA/revision, adapter load test; nếu không có model hợp lệ thì báo BLOCKED P0. |
| 2 | Viết/test đường inference-only tối thiểu | Unit/integration tests + local dry-run, không chạy full stages; kiểm tra config, <4B và leak-safe metrics. |
| 3 | Benchmark ngắn với chi phí được duyệt | Đo GPU/VRAM, queries/s, cache/setup, cost estimate cho 2.080q và reserve; dừng nếu vượt trần. |
| 4 | P1 offline trên held-out thật nếu có dữ liệu đầy đủ | Baseline vs thay đổi đã định trước; score và latency, query coverage, multi-gold; không có tín hiệu thì giữ baseline. |
| 5 | Private inference một lần, xác minh và giao artifact | Contract validator + checksum + receipt + app terminal status; báo rõ mô hình/fold nào được dùng. |

**Không thực hiện trong yêu cầu lập kế hoạch này:** sửa pipeline/config, launch Modal, đổi HF visibility, upload HF, chạy Kaggle hay nộp submission. Repo hiện sạch; nội dung báo cáo chưa được xác thực trực tiếp với remote Volume. Để thực thi P0/P1, cần phê duyệt riêng cho phần code mới và mọi chi phí cloud/outward-facing action.
