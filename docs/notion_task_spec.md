
========================================
# 01 — Competition Rules & Scoring
========================================
01 — Competition Rules & Scoring

  > 💡 
    Trang này là source-of-truth cho mọi quyết định pipeline. Nếu một experiment vi phạm các constraint dưới đây thì không dùng cho submission.

  ## Scoring thực tế
  Scorer đọc answer của từng query, yêu cầu số query dự đoán phải khớp reference, sau đó tính:
  ```Python
recall_i = |gold ∩ pred| / |gold|
precision_i = |gold ∩ pred| / |pred|
  ```
  Điều kiện hợp lệ cho một query là 0 < len(pred) <= 5. Nếu output rỗng hoặc quá 5 IDs thì scorer cho score câu đó bằng 0.

  ### Hệ quả trực tiếp
  * Ranking bên trong top 5 không được scorer thưởng thêm; quan trọng là gold có nằm trong set dự đoán hay không.
  * Recall là metric chính; Precision là metric phụ/tie-breaker theo overview.
  * Với mục tiêu leaderboard, chọn top 5 là default hợp lý vì false positives không làm giảm Recall miễn số lượng không vượt 5.
  * Tuy nhiên ở bước cuối có thể tune adaptive 1–5 để cải thiện Precision nếu validation chứng minh Recall giữ nguyên.
  * Output phải là unique document IDs; duplicate không tăng intersection nhưng làm mẫu prediction xấu đi.

  ## Model/system constraints
  * Tổng số tham số của toàn hệ thống trong một task phải < 4B.
  * Quantization/LoRA không làm model >4B trở thành hợp lệ nếu bản chất model vẫn >4B parameters.
  * Không dùng API, kể cả API phi lợi nhuận.
  * Chỉ dùng mô hình/hệ thống có thể tải, kiểm soát và tái lập.
  * BTC cho phép pretrained/open models phù hợp license; pretrained corpus không bị xem là external dataset trực tiếp theo phần giải thích của BTC.

  ## Data constraints
  * Chỉ dùng dữ liệu BTC cung cấp cho Task 1 để xây phương pháp.
  * Không dùng external legal corpus.
  * Không dùng synthetic/data augmentation từ nguồn ngoài.
  * Không dùng dữ liệu của Task 2 cho Task 1.
  * BTC xác nhận train có một số passage rỗng/trùng; public/private không có đáp án rơi vào context rỗng/trùng. Vì vậy phải xử lý noise train cẩn thận.

  ## Submission constraints
  * File cuối là submission.zip chứa duy nhất submission.json.
  * Format mỗi key:
  ```JSON
{
  "147194": {
    "answer": ["177504", "740"]
  }
}
  ```
  * Kiểm tra đủ toàn bộ query IDs trước khi zip.
  * Kiểm tra mỗi answer có 1–5 unique string IDs.

  ## Checklist trước khi nộp
  - [ ] Tổng neural params < 4B.
  - [ ] Không gọi API ở inference/training pipeline.
  - [ ] Không dùng data ngoài BTC.
  - [ ] Mọi query có answer.
  - [ ] Không query nào >5 IDs.
  - [ ] Không duplicate ID trong một answer.
  - [ ] Reproduce được bằng README/script sạch.
  - [ ] Model/license/version được ghi rõ.

  ## Concrete scoring examples

  ### Case 1 — gold chỉ có 1 document
  Gold = [A].
    | Precision | Recall | Nhận xét | Prediction |
    | 1.0 | 1.0 | Hoàn hảo | [A] |
    | 0.2 | 1.0 | Recall không giảm dù có 4 false positives | [A,B,C,D,E] |
    | 0.0 | 0.0 | Miss gold | [B,C,D,E,F] |
    | 0.0 | 0.0 | >5 IDs → scorer cho 0 toàn câu | [A,B,C,D,E,F] |

  ### Case 2 — gold có 2 documents
  Gold = [A,B].
  * Predict [A,C,D,E,F] → Recall = 1/2 = 0.5, Precision = 1/5 = 0.2.
  * Predict [A,B,C,D,E] → Recall = 2/2 = 1.0, Precision = 2/5 = 0.4.
  ```Mermaid
flowchart LR
    G["Gold docs"] --> I["Intersection with predicted set"]
    P["Predicted docs, 1..5 IDs"] --> I
    I --> R["Recall = hit / #gold"]
    I --> PR["Precision = hit / #pred"]
    P --> C{"#pred > 5?"}
    C -->|Yes| Z["Recall=0, Precision=0"]
    C -->|No| I
  ```

  ## Submission shape — premise example
  BTC overview mô tả submission là JSON Object, ví dụ:
  ```JSON
{
  "147194": {
    "answer": ["177504", "740"]
  }
}
  ```
  Checklist bắt buộc:
  * Có đúng toàn bộ query IDs của test.
  * Mỗi answer là list.
  * 1 <= len(answer) <= 5.
  * IDs unique trong mỗi list.
  * Không trả passage, name hay answer text; chỉ trả document IDs.

  ## Parameter-budget examples
  BTC tính tổng tham số của toàn hệ thống.
    | Ví dụ stack | Xấp xỉ tổng params | Hợp lệ theo giới hạn 4B? |
    | BGE-M3 + BGE reranker v2 m3 | ~1.14B | ✅ Có |
    | GTE multilingual base + BGE reranker | ~0.87B | ✅ Có |
    | 3.5B retriever + 0.8B reranker | ~4.3B | ❌ Không |
    | 6B model quantized 4-bit | vẫn 6B params | ❌ Không |
  ```Mermaid
flowchart TD
    A["Candidate system"] --> B["Cộng params tất cả neural models"]
    B --> C{"Total < 4B?"}
    C -->|No| X["Không hợp lệ dù quantized/LoRA"]
    C -->|Yes| D{"Có dùng external train data / augmentation?"}
    D -->|Yes| X2["Không hợp lệ theo rules"]
    D -->|No| E{"Có gọi API trung gian?"}
    E -->|Yes| X3["Không hợp lệ"]
    E -->|No| OK["Candidate hợp lệ về các constraint chính"]
  ```

  ## Tư duy metric đúng cho Task 1
  Scorer không quan tâm vị trí trong list khi tính Recall/Precision; nó dùng set intersection. Vì vậy ranking 1→5 chỉ có ý nghĩa gián tiếp: top candidate cần đủ mạnh để tất cả gold lọt vào 5 slots. Đây là lý do tài liệu tập trung vào candidate coverage + reranking + diversity, không phải MRR hay generation.


========================================
# 02 — Data Audit & Preprocessing
========================================
02 — Data Audit & Preprocessing

  ## Mục tiêu
  Biến selected-contexts thành một corpus có cấu trúc tốt cho sparse + dense retrieval mà không mất các tín hiệu pháp lý quan trọng.

  ## Audit đã quan sát từ source hiện tại
  * Train có hàng nghìn query và phần lớn chỉ có 1 gold document.
  * Corpus có nhiều document dài; một số passage cực dài nên không thể chỉ lấy 512 token đầu.
  * Train có passage rỗng/trùng; BTC xác nhận đây là noise của train và public/private gold không rơi vào trường hợp này.
  * Một số context thiếu name, nhưng vẫn còn link; URL slug có thể dùng làm metadata fallback.
  * Nhiều luật/văn bản xuất hiện làm positive cho nhiều câu hỏi khác nhau, tạo cơ hội rất lớn cho query-memory retrieval.

  ## Normalization
  Giữ song song hai phiên bản:
  1. Original Vietnamese: phục vụ neural model và exact legal text.
  1. Normalized search text: lowercase, chuẩn hóa whitespace, Unicode NFC, có thể thêm accent-stripped field cho fuzzy lexical search.

  ### Tuyệt đối giữ
  * Số hiệu văn bản: 44/2023/NĐ-CP, 58/2020/TT-BCA.
  * Điều/Khoản/Điểm.
  * Năm, ngày tháng.
  * Số tiền, tỷ lệ phần trăm, số tháng/ngày.
  * Tên tỉnh/thành, chức danh, mẫu biểu, viết tắt như ABTC, BHXH, VSD.

  ## Metadata fallback
  Nếu name thiếu:
  ```Python
title = ctx.get("name")
if not title:
    title = slug_from_url(ctx["link"])
  ```
  Từ URL có thể phục hồi nhiều signal như loại văn bản, số hiệu, năm và chủ đề.

  ## Legal-aware chunking
  Ưu tiên hierarchy:
  1. Chương
  1. Mục
  1. Điều
  1. Khoản
  1. Fallback token window

  ### Chunk template
  ```Plain Text
[DOC TITLE]
[DOC NUMBER / YEAR if found]
[CHAPTER / SECTION]
[ARTICLE HEADING]
[CONTENT]
  ```

  ### Kích thước fallback
  * Khoảng 700–1200 tokens/chunk.
  * Overlap 100–200 tokens.
  * Không cắt ngang heading/Điều nếu có thể.

  ## Index representation
  Mỗi chunk lưu:
  ```Plain Text
chunk_id
document_id
title
link_slug
legal_number
year
chapter
article
body
normalized_body
  ```

  ## Train noise handling
  * Passage rỗng: không dùng body để train positive span; dùng title/URL metadata hoặc loại khỏi span-supervision.
  * Exact duplicate passages: deduplicate khi xây index, nhưng giữ mapping về tất cả document IDs cần thiết trong train để không phá labels.
  * Conflicting/near-duplicate train questions: đánh dấu để tránh biến positive của câu tương đương thành hard negative.

  ## Output của bước này
  * documents.parquet
  * chunks.parquet
  * doc_meta.json
  * train_queries.parquet
  * duplicate_groups.json
  * empty_context_ids.json

  ## Data examples từ source BTC

  ### 1. Train record — single-label-looking nhưng task vẫn multi-label
  ```JSON
{
  "146300": {
    "question": "Dự án đầu tư là gì?",
    "answer": ["2113"]
  }
}
  ```

  ### 2. Train record — multi-positive
  ```JSON
{
  "66688": {
    "question": "Khi công ty muốn tăng vốn điều lệ thì các thành viên góp vốn trước khi làm thủ tục tăng có trái quy định không?",
    "answer": ["200355", "21398"]
  }
}
  ```
  Điều này cho thấy answer phải được coi là set of positives, không phải một class duy nhất.

  ### 3. Public record
  ```JSON
{
  "76264": {
    "question": "Phạt nguội là gì?",
    "answer": null
  }
}
  ```
  Không có gold ở inference time.

  ### 4. Context record — schema chính thức
  ```JSON
{
  "link": "https://thuvienphapluat.vn/...",
  "name": "Quyet-dinh-5868-QD-BYT-2018-...",
  "passage": "BỘ Y TẾ ... Số: 5868/QĐ-BYT ...",
  "id": 740
}
  ```
  Ở bước indexing, id=740 phải được preserve nguyên vẹn vì đây là ID được trả trong submission.

  ## Data lineage
  ```Mermaid
flowchart TD
    Z["selected-contexts.zip"] --> J["context_*.json"]
    J --> A["Raw document table"]
    A --> B["Normalize metadata"]
    B --> C["Detect empty / duplicate passages"]
    C --> D["Extract title, legal number, year, headings"]
    D --> E["Legal-aware chunks"]
    E --> F1["BM25 index"]
    E --> F2["Dense vectors"]
    E --> F3["Sparse / multi-vector features"]
    T["train.json"] --> G["query → positive document IDs"]
    G --> H["query → positive chunks via weak localization"]
    H --> M["training pairs/groups"]
  ```

  ## Ví dụ về legal-aware chunking
  Giả sử một passage có dạng:
  ```Plain Text
LUẬT ...
Chương III ...
Điều 112. Nghỉ lễ, tết
1. Người lao động được nghỉ làm, hưởng nguyên lương...
2. Lao động là người nước ngoài...
Điều 113. Nghỉ hằng năm
...
  ```
  Không nên tạo một vector duy nhất cho cả luật. Nên tạo tối thiểu:
  * Chunk A: metadata + Điều 112 + toàn bộ các khoản của Điều 112.
  * Chunk B: metadata + Điều 113 + toàn bộ các khoản của Điều 113.
  ```Mermaid
flowchart LR
    D["Document rất dài"] --> H1["Heading / Chương"]
    H1 --> A1["Điều 112"]
    H1 --> A2["Điều 113"]
    A1 --> C1["Chunk 112 + metadata"]
    A2 --> C2["Chunk 113 + metadata"]
    C1 --> ID["document_id giữ nguyên"]
    C2 --> ID
  ```

  ## Ví dụ normalization tốt vs xấu
    | Normalization nên làm | Raw text | Không nên làm |
    | lowercase copy + giữ raw field | Nghị định 44/2023/NĐ-CP | xóa /, số, năm |
    | giữ 112, 1, token Điều/khoản | Điều 112, khoản 1 | lọc số như stopword |
    | chuẩn hóa khoảng trắng/dấu phân cách nhưng giữ value | 500.000.000 đồng | chỉ giữ từ đồng |
    | giữ acronym và có thể thêm expanded form nếu có trong chính corpus | BHYT, BHXH, ABTC | tự thêm external dictionary ngoài BTC |

  ## Noise handling — điều BTC đã xác nhận
  rules.txt xác nhận train có một số passage rỗng hoặc trùng, còn public/private gold không có hiện tượng context trùng/rỗng. Do đó:
  * Train empty passage: không ép neural model học một positive body rỗng; giữ mapping label và tận dụng metadata nếu có.
  * Train duplicate passage: tránh coi duplicate tương đương là hard negative của nhau.
  * Public/private: vẫn build index đầy đủ, nhưng không cần thiết kế special-case để dự đoán empty gold.

  ## Audit table nên tạo trước khi train
    | Mục đích | Audit | Hành động khi phát hiện |
    | tránh positive không có text | empty passage | metadata-only / exclude span supervision |
    | tránh false negative | duplicate passage | group duplicates, preserve all doc IDs |
    | giảm mất metadata | missing name | extract từ URL slug |
    | tránh truncation | very long passage | legal-aware chunking |
    | đúng objective | multiple gold IDs | all positives in group loss/ranking |
    | memory + leakage control | near-duplicate questions | group/flag khi split validation |


========================================
# 03 — Hybrid Candidate Retrieval
========================================
03 — Hybrid Candidate Retrieval

  ## Mục tiêu
  Đạt candidate recall cực cao trước khi rerank. Stage này chấp nhận nhiều false positives; nhiệm vụ là đảm bảo gold document hiếm khi rơi khỏi top 100–180 candidate.

  ## Retriever A — BM25 trên chunks
  BM25 rất quan trọng cho:
  * exact legal number;
  * năm;
  * Điều/Khoản;
  * tên biểu mẫu;
  * tên chức danh;
  * thuật ngữ hiếm;
  * tỉnh/thành.

  ### Field boosts khởi điểm
  ```Plain Text
legal_number: 5.0
title:        3.0
article:      2.0
body:         1.0
url_slug:     1.5
  ```
  Tune bằng validation, không coi đây là giá trị cố định.

  ### Document aggregation
  Lấy top chunks rồi aggregate theo document:
  ```Python
doc_score = max(chunk_scores) + 0.1 * second_best
  ```
  Giữ best chunk IDs để đưa sang reranker.

  ## Retriever B — BGE-M3
  Model đề xuất: BAAI/bge-m3 (~568M).
  Lý do chọn:
  * multilingual, phù hợp tiếng Việt;
  * hỗ trợ long-input tốt hơn retriever nhỏ truyền thống;
  * có dense, sparse/lexical và multi-vector capabilities trong cùng backbone;
  * tổng parameter budget vẫn rất thấp so với 4B.

  ### Index strategy
  Encode từng chunk, không encode whole document.
  Document candidate score = aggregate top chunk scores.

  ## Exact signal retriever
  Parse query để tìm:
  * số hiệu văn bản;
  * Điều/Khoản;
  * năm;
  * mã biểu mẫu;
  * tên luật/văn bản rõ ràng.
  Nếu exact identifier match document metadata, candidate đó được boost mạnh.

  ## Candidate union
  Khởi điểm:
  ```Plain Text
BM25 docs            top 50
BGE-M3 dense docs     top 50
BGE-M3 lexical/docs   top 50
multi-vector docs     top 50
exact matcher         all high-confidence matches
  ```
  Sau dedup thường giữ ~100–180 docs.

  ## Fusion baseline
  Trước khi có learned fusion, dùng Reciprocal Rank Fusion:
  ```Python
RRF(d) = Σ 1 / (k + rank_r(d))
  ```
  Tune k trên validation.

  ## Metric của stage này
  Không nhìn Precision@5 đầu tiên. Theo dõi:
  * Gold candidate Recall@20
  * Gold candidate Recall@50
  * Gold candidate Recall@100
  * Gold candidate Recall@150
  Mục tiêu: Recall@100/150 càng gần 1 càng tốt trước khi chuyển sang reranker.

  ## Failure analysis
  Mỗi miss phải gắn tag:
  * exact identifier miss;
  * semantic paraphrase miss;
  * long-document chunking miss;
  * same-law wrong-article;
  * temporal/year mismatch;
  * unseen-document issue;
  * noisy/missing train context.

  ## Concrete retrieval example
  Query giả định:
  > Mẫu Tờ khai hải quan mới nhất 2023 là mẫu nào?
  Một neural retriever có thể hiểu ngữ nghĩa tờ khai hải quan, nhưng sparse retrieval có lợi thế lớn với các token hiếm/chính xác như 2023, Mẫu, số hiệu văn bản hoặc mã mẫu nếu xuất hiện. Vì vậy không nên chọn dense-only.
  ```Mermaid
flowchart LR
    Q["Query"] --> X["Extract signals"]
    X --> K1["Rare lexical tokens<br>2023, mã văn bản, Điều"]
    X --> K2["Semantic intent<br>mẫu tờ khai hải quan"]
    K1 --> B["BM25 / sparse"]
    K2 --> D["Dense embedding"]
    K2 --> M["Multi-vector / ColBERT-style"]
    B --> U["Union candidates"]
    D --> U
    M --> U
  ```

  ## Candidate retrieval options
    | Context length | Vai trò đề xuất | Retriever | Khi nào mạnh | Option |
    | không phải neural limit | bắt buộc baseline + ensemble branch | BM25 | số hiệu, năm, tên riêng, exact terms | Lucene / Pyserini / Elasticsearch-compatible local implementation |
    | 8192 | default option A | Dense + sparse + multi-vector | hybrid multilingual retrieval | BAAI/bge-m3 |
    | 8192 | option B / ablation | Dense + sparse | compute thấp, long context | Alibaba-NLP/gte-multilingual-base |
    | 512 | option C; bắt buộc chunk nhỏ hợp lý | Dense | semantic retrieval mạnh | intfloat/multilingual-e5-large-instruct |
    | long-context architecture | option D; research/non-commercial license | Dense | Vietnamese multilingual option | jinaai/jina-embeddings-v3 |

  ## Recommended candidate quotas
  Đây là starting point để tune, không phải constant cố định:
  * BM25: top 50 documents sau chunk→doc aggregation.
  * Dense: top 50.
  * Sparse neural / lexical BGE-M3: top 30–50.
  * Multi-vector: top 30–50 nếu compute cho phép.
  * Train-question memory: top document candidates từ 10–20 nearest questions.
  * Exact identifier/title matcher: inject tất cả high-confidence exact hits.
  Sau dedup, mục tiêu candidate pool khoảng 100–180 documents/query.
  ```Mermaid
flowchart TD
    B["BM25 top 50"] --> U["Deduplicated union"]
    D["Dense top 50"] --> U
    S["Sparse neural top 40"] --> U
    C["Multi-vector top 40"] --> U
    M["Question-memory docs"] --> U
    E["Exact-match docs"] --> U
    U --> CR["Candidate Recall@100-ish"]
    CR --> RR["Reranker"]
  ```

  ## Chunk→document aggregation options
  Giả sử doc A có 3 chunks với scores 0.91, 0.77, 0.35, doc B có 0.86, 0.85.
    | Aggregation | Ý nghĩa | Doc B | Doc A |
    | Max | tốt khi chỉ một Điều chứa answer | 0.86 | 0.91 |
    | Max + α·second | thưởng doc có nhiều evidence chunks | 0.86 + α·0.85 | 0.91 + α·0.77 |
    | Reciprocal ranks | dễ fusion giữa score scales khác nhau | theo rank chunks | theo rank chunks |
  Khuyến nghị: bắt đầu bằng max cho candidate recall, sau đó benchmark max + small second-best bonus ở fusion stage.

  ## Exact-match branch
  Các pattern nên detect trực tiếp từ chính query và corpus:
  * \d+/\d{4}/... hoặc số hiệu văn bản tương tự.
  * Điều N, Khoản N, Điểm a/b/c.
  * năm 2022, 2023, ...
  * mã biểu mẫu, acronym như ABTC, BHYT, BHXH, VSD.
  * tỉnh/thành hoặc tên cơ quan xuất hiện nguyên văn.
  Không cần external legal dictionary; exact matcher chỉ dùng text/metadata BTC cung cấp.

  ## Success criterion của Stage 1
  Không đánh giá Stage 1 bằng Precision@5. Đánh giá:
  * Candidate Recall@20.
  * Candidate Recall@50.
  * Candidate Recall@100.
  Nếu gold chưa lọt candidate pool thì reranker mạnh đến đâu cũng không cứu được. Do đó Stage 1 phải ưu tiên coverage trước.


========================================
# 04 — Train-Question Memory
========================================
04 — Train-Question Memory

  > 💡 
    Đây là một nhánh retrieval riêng, không phải classifier thay thế corpus retrieval. Nó khai thác việc nhiều câu test có cách hỏi gần giống train và nhiều legal documents được reuse rất nhiều lần.

  ## Ý tưởng
  Index toàn bộ question trong train. Với query mới, retrieve những train questions gần nhất rồi transfer các document IDs đã biết của chúng thành candidate documents.

  ## Memory A — char n-gram TF-IDF
  Đề xuất:
  ```Plain Text
analyzer = char_wb
ngram_range = (3, 5) hoặc (3, 6)
min_df = 1
sublinear_tf = true
  ```
  Char n-gram rất mạnh cho câu hỏi tiếng Việt có wording tương tự nhưng khác một ít dấu câu/từ nối.

  ## Memory B — BGE-M3 question embedding
  Dùng chính retriever backbone đã có, không tăng parameter budget.

  ## Exact normalized-question table
  Trước retrieval:
  ```Python
if norm(q_test) in train_question_map:
    exact_docs = union(all train answers for same normalized question)
  ```
  Các doc này được boost cực mạnh nhưng vẫn giữ thêm các slot khác để bảo vệ Recall nếu annotation train không exhaustive.

  ## Transfer score
  Ví dụ:
  ```Python
score_mem[doc] += similarity(q, q_train) ** 3
  ```
  Nếu nhiều nearest questions cùng vote cho một doc, cộng vote.

  ## Safety against annotation noise
  Không giả định train answers luôn exhaustive. Nếu hai câu hỏi giống hệt/near-duplicate có gold khác nhau:
  * union positives cho memory;
  * không coi doc positive của câu kia là hard negative;
  * gắn ambiguous_query_group_id để dùng khi train reranker.

  ## Tại sao không làm classification
  Một classifier chỉ predict trong các document đã từng positive ở train sẽ fail với document chưa xuất hiện trong train labels. Vì vậy memory chỉ là một retriever branch, luôn chạy song song full-corpus retrieval.

  ## Features đưa sang meta-ranker
  ```Plain Text
best_train_q_similarity
mean_top3_train_q_similarity
num_neighbor_votes
exact_question_match
memory_rank
memory_score
train_doc_frequency
  ```

  ## Tune
  Ablation bắt buộc:
  * BM25 + dense
  * 
    * char TF-IDF memory
  * 
    * neural memory
  * 
    * exact-question override
  Nếu memory làm random split tăng mạnh nhưng document-disjoint giảm, giảm weight để tránh overfit private.

  ## Ví dụ trực quan: query-memory hoạt động như thế nào
  Giả sử public query là:
  > Công dân bị xóa đăng ký tạm trú trong những trường hợp nào?
  Nếu train có một query gần như tương đương:
  > Xóa đăng ký tạm trú trong trường hợp nào?
  và train answer của query đó là document 72265, thì 72265 phải được đưa vào candidate pool với một memory score cao. Đây không phải external knowledge; nó chỉ tái sử dụng official train labels.
  ```Mermaid
flowchart LR
    Q["Test question"] --> T1["Nearest train Q1<br>sim 0.92"]
    Q --> T2["Nearest train Q2<br>sim 0.81"]
    Q --> T3["Nearest train Q3<br>sim 0.72"]
    T1 --> D1["gold docs of Q1"]
    T2 --> D2["gold docs of Q2"]
    T3 --> D3["gold docs of Q3"]
    D1 --> F["Aggregate memory score by doc"]
    D2 --> F
    D3 --> F
    F --> U["Inject into candidate union"]
  ```

  ## Hai memory retrievers nên chạy song song

  ### Option A — char n-gram TF-IDF
  Mạnh với:
  * viết lại câu nhưng giữ phần lớn từ khóa;
  * lỗi chính tả nhẹ;
  * biến thể được quy định như thế nào? ↔ quy định ra sao?;
  * câu có tên cơ quan/chức danh rất giống nhau.

  ### Option B — neural question embedding
  Có thể dùng cùng retriever backbone đang dùng cho corpus, ví dụ BGE-M3/GTE/E5, để tránh cộng thêm model params. Mạnh hơn khi wording khác nhiều nhưng intent tương đương.

  ## Exact normalized match
  Nếu normalized test question trùng hoàn toàn một train question, dùng train gold docs như high-confidence seed, nhưng vẫn giữ các slot còn lại cho full-corpus retrieval.
  Ví dụ:
  ```Plain Text
train:  "Công ty cổ phần là gì?" → [21398]
public: "Công ty cổ phần là gì?"
  ```
  Chiến lược tốt hơn answer=[21398] ngay lập tức là:
  * khóa 21398 vào candidate set với feature exact_train_match = 1;
  * tiếp tục rerank/fill tối đa 4 slots còn lại để phòng annotation khác hoặc multi-document gold.

  ## Memory score options
  Không cần chốt một công thức duy nhất từ đầu. Benchmark:
  * max(similarity_of_neighbor_supporting_doc)
  * sum(similarity^p) với p=2 hoặc 3
  * reciprocal-rank weighted vote
  * learned feature trong LightGBM

  ### Ví dụ voting
  Giả sử:
  * Q1 sim 0.90 → doc A
  * Q2 sim 0.85 → doc A, doc B
  * Q3 sim 0.78 → doc C
  Doc A có support từ 2 hàng xóm nên thường đáng tin hơn doc C chỉ có 1 support.

  ## False-memory guard
  Query-memory dễ gây lỗi khi hai câu trông giống nhưng khác điều kiện pháp lý nhỏ.
  Ví dụ kiểu lỗi:
  * thời hạn cấp ... lần đầu vs thời hạn cấp lại ...;
  * cấp tỉnh vs cấp huyện;
  * 2022 vs 2023;
  * người lao động vs viên chức.
  Do đó memory score phải đi cùng exact features như year/entity/legal-number overlap và không được override full-corpus evidence một cách mù quáng.
  ```Mermaid
flowchart TD
    M["High memory similarity"] --> C{"Critical modifiers match?"}
    C -->|"year/entity/procedure match"| H["High-confidence memory feature"]
    C -->|"modifier conflict"| L["Downweight memory"]
    H --> F["Fusion"]
    L --> F
    R["Corpus retrieval evidence"] --> F
  ```

  ## Validation bắt buộc cho memory branch
  Tách ít nhất 3 nhóm query:
  1. exact/near-duplicate train questions;
  1. related-topic nhưng wording khác;
  1. unseen-document validation.
  Memory branch phải tăng nhóm 1–2 nhưng không làm tụt mạnh nhóm 3. Nếu tụt, fusion đang phụ thuộc train-frequency quá nhiều.


========================================
# 05 — Retriever Training & Hard Negatives
========================================
05 — Retriever Training & Hard Negatives

  ## Mục tiêu
  Fine-tune BGE-M3 để query gần relevant legal chunks hơn và xa các legal chunks rất giống nhưng sai văn bản/Điều.

  ## Weak localization từ document labels
  Train chỉ cho question -> document_id, không có relevant span. Với mỗi gold document:
  1. chunk document;
  1. dùng BM25 + pretrained BGE-M3 để lấy top chunks bên trong gold doc;
  1. xem các top chunks đó là weak positive spans;
  1. có thể giữ nhiều positive chunks nếu score gần nhau.
  Không tạo dữ liệu mới ngoài BTC; chỉ suy ra span từ official document.

  ## Positive pairs
  ```Plain Text
(question, gold chunk)
  ```
  Với multi-label query, tất cả gold docs đều là positives.

  ## Hard negative mining
  Sau baseline hybrid retrieval:
  1. retrieve top 50–100 docs;
  1. remove known positives;
  1. lấy top false candidates làm negatives.

  ### Negative tốt
  * cùng luật nhưng sai Điều;
  * cùng thủ tục nhưng sai đối tượng;
  * cùng chức danh nhưng sai cơ quan;
  * văn bản cũ/sửa đổi gần giống;
  * cùng tỉnh nhưng sai quyết định;
  * keyword rất gần nhưng answer requirement khác.

  ### False-negative guard
  Nếu candidate doc là positive của:
  * exact same normalized question khác;
  * hoặc near-duplicate question group;
  thì không dùng làm negative mạnh.

  ## Objective
  Khởi điểm dùng contrastive / multiple-negative ranking loss:
  ```Plain Text
positive: localized gold chunks
negative: in-batch + mined hard negatives
  ```
  Có thể curriculum:
  1. random/in-batch negatives;
  1. BM25 hard negatives;
  1. hybrid retriever hard negatives;
  1. reranker-hard negatives.

  ## Training split
  Luôn có 2 validation regime:
  * Random query split: đo seen-law/query reuse.
  * Document-disjoint split: đo generalization sang unseen documents.

  ## Checkpoints
  Chọn checkpoint theo Recall@5 sau full retrieval pipeline, không chỉ loss hoặc MRR của bi-encoder.

  ## Stop condition
  Fine-tuning chỉ được giữ nếu:
  * candidate Recall@100 tăng hoặc giữ;
  * Recall@5 full pipeline tăng;
  * document-disjoint không collapse.

  ## Từ document-level labels sang retriever training examples
  Train chỉ cho question → document_id, không chỉ ra đoạn nào trong passage là relevant. Vì vậy cần weak positive localization bên trong gold document.
  Ví dụ train:
  ```JSON
{
  "146300": {
    "question": "Dự án đầu tư là gì?",
    "answer": ["2113"]
  }
}
  ```
  Giả sử doc 2113 đã được chia thành 30 chunks. Ta không gán cả 30 chunks là positive mạnh. Ta chọn một nhóm positive candidate bằng lexical/dense matching nội bộ trong chính doc 2113.
  ```Mermaid
flowchart TD
    Q["Train question"] --> G["Gold document"]
    G --> C1["chunk 1"]
    G --> C2["chunk 2"]
    G --> C3["..."]
    G --> CN["chunk N"]
    Q --> L["Localize top chunks within gold doc"]
    C1 --> L
    C2 --> L
    CN --> L
    L --> P["Weak positive chunk set"]
    P --> T["Retriever training"]
  ```

  ## Positive construction options
    | Cách dùng | Strategy | Rủi ro | Ưu |
    | top BM25 chunks trong gold doc | Lexical localization | miss paraphrase | ổn định với exact legal terms |
    | top pretrained embedding chunks | Dense localization | baseline model có thể sai | bắt semantic match |
    | lấy top chunks của cả hai | Union lexical+dense | nhiều weak positives hơn | recall cao hơn |
    | coi document là positive bag, model tự chọn chunk | Multi-instance learning | training phức tạp hơn | giảm phụ thuộc heuristic |
  Khuyến nghị thực dụng: bắt đầu bằng union lexical+dense localization, sau đó ablate.

  ## Hard-negative mining — ví dụ đúng và sai
  Với query về thời hạn cấp lại giấy phép, negative tốt là:
  * văn bản về cùng loại giấy phép nhưng nói cấp mới;
  * văn bản nói đúng cấp lại nhưng khác cơ quan/lĩnh vực;
  * cùng luật nhưng sai Điều.
  Negative kém:
  * văn bản hoàn toàn không liên quan như luật thủy sản cho query lao động.
  ```Mermaid
flowchart LR
    Q["Query"] --> R["Baseline hybrid retrieve top 50"]
    R --> G{"Is gold doc?"}
    G -->|Yes| P["Positive"]
    G -->|No| F{"Potential false negative?"}
    F -->|Yes: equivalent query/doc evidence| SKIP["Do not use as hard negative"]
    F -->|No| H["Hard negative"]
  ```

  ## False-negative guard bằng train graph
  Tạo graph đơn giản:
  * node query;
  * node document;
  * edge = official positive label.
  Nếu hai query gần như trùng và doc X là positive của query kia, đừng vội dùng X làm negative cho query hiện tại. Train có thể không exhaustively annotate mọi document tương đương.

  ## Training objective options
  Không cần khóa vào một loss từ đầu.
    | Objective | Khi dùng | Ghi chú |
    | Multiple-negative ranking / InfoNCE | baseline bi-encoder | đơn giản, hiệu quả |
    | Multi-positive contrastive | query có nhiều gold docs/chunks | phù hợp task hơn single-positive |
    | Hard-negative contrastive | sau baseline retrieval | thường tạo bước nhảy chất lượng lớn |
    | Distillation from reranker scores | chỉ nếu teacher nằm trong hệ thống/quy tắc và không dùng external data | cần kiểm tra parameter/rules cẩn thận |

  ## Model-training options
  * BGE-M3: ưu tiên fine-tune dense trước; nếu compute cho phép benchmark unified dense+sparse+multi-vector training.
  * GTE multilingual base: lựa chọn nhẹ hơn cho dense/sparse domain adaptation.
  * multilingual-E5-large-instruct: strong dense baseline; do 512-token limit, chunk supervision phải tốt.
  * Jina embeddings v3: alternative multilingual option; giữ license evidence trong README.

  ## Curriculum đề xuất
  ```Mermaid
flowchart TD
    A["Stage 0 pretrained"] --> B["Stage 1 easy positives + in-batch negatives"]
    B --> C["Mine top false candidates"]
    C --> D["Stage 2 hard-negative training"]
    D --> E["Re-index corpus"]
    E --> F["Measure candidate Recall@100"]
    F --> G{"Improved?"}
    G -->|Yes| H["Freeze best retriever"]
    G -->|No| I["Revisit chunking/negatives"]
  ```

  ## Chỉ số để quyết định retriever thắng
  * Candidate Recall@20/50/100 trên random split.
  * Candidate Recall@100 trên document-disjoint split.
  * Recall breakdown theo query có exact identifier vs semantic paraphrase.
  * Latency/index size chỉ là tie-breaker sau retrieval quality, trừ khi hardware giới hạn nghiêm trọng.


========================================
# 06 — Cross-Encoder Reranking
========================================
06 — Cross-Encoder Reranking

  ## Model
  Đề xuất: BAAI/bge-reranker-v2-m3 (~568M).
  Cùng với BGE-M3 retriever, tổng neural params khoảng 1.14B.

  ## Input format
  Không đưa nguyên document quá dài. Với mỗi candidate document:
  1. lấy top 2–3 chunks theo hybrid retriever;
  1. prepend document title / legal number;
  1. score từng (question, chunk).
  Ví dụ:
  ```Plain Text
Question: ...
Document: Bộ luật Lao động 2019 — 45/2019/QH14
Section: Điều 112. Nghỉ lễ, tết
Passage: ...
  ```

  ## Document score
  Baseline:
  ```Python
rerank_doc = max(chunk_scores) + 0.1 * second_best
  ```
  Sau đó tune weight bằng validation.

  ## Training data

  ### Positive
  * localized chunk từ gold document.
  * nếu nhiều relevant-looking chunks trong cùng gold doc, giữ multi-positive.

  ### Negative
  * top hybrid retrieval false positives.
  * ưu tiên same-topic/same-law negatives.
  * tránh false negatives từ duplicate/near-duplicate train questions.

  ## Loss
  Thử theo thứ tự:
  1. binary cross-entropy / pointwise baseline;
  1. pairwise margin ranking;
  1. listwise/grouped objective nếu implementation ổn định.
  Thước đo quyết định vẫn là Recall@5 end-to-end.

  ## Candidate budget cho reranker
  Không rerank cả corpus. Khởi điểm:
  ```Plain Text
100–180 documents/query
2–3 chunks/document
  ```
  Nếu compute hạn chế, shortlist còn 60–100 docs bằng RRF trước rerank.

  ## Efficiency
  * Cache chunk tokenization.
  * Batch theo length bucket.
  * FP16/BF16 inference nếu hardware hỗ trợ.
  * Rerank offline public/private questions; không có latency online requirement.

  ## Ablation cần chạy
  * max 1 chunk/doc
  * max 2 chunks/doc
  * max 3 chunks/doc
  * title only + best chunk
  * title + legal number + best chunk
  Giữ cấu hình Recall@5 cao nhất, không chọn chỉ vì nhanh hơn.

  ## Reranker input — ví dụ cụ thể
  Reranker không nên nhận toàn bộ legal document nếu document rất dài. Với mỗi candidate doc, lấy vài chunks tốt nhất từ Stage 1.
  Ví dụ:
  ```Plain Text
Question:
"Người lao động đóng từ bao nhiêu tháng thì sẽ được hưởng trợ cấp thất nghiệp?"

Candidate document A:
- title / legal metadata
- best chunk #1: điều kiện về thời gian đóng
- best chunk #2: điều kiện hưởng trợ cấp

Candidate document B:
- title / legal metadata
- best chunk #1: quy định BHXH nhưng không phải trợ cấp thất nghiệp
  ```
  Cross-encoder cần học rằng A relevant hơn B dù cả hai cùng chứa nhiều từ người lao động, đóng, bảo hiểm.
  ```Mermaid
flowchart TD
    D["Candidate document"] --> C1["Best chunk 1"]
    D --> C2["Best chunk 2"]
    D --> C3["Best chunk 3"]
    Q["Question"] --> R1["Cross-encoder score q,c1"]
    Q --> R2["Cross-encoder score q,c2"]
    Q --> R3["Cross-encoder score q,c3"]
    C1 --> R1
    C2 --> R2
    C3 --> R3
    R1 --> A["Aggregate to doc score"]
    R2 --> A
    R3 --> A
  ```

  ## Reranker model options
    | Model | Language/context | Why test it | License note | Approx params |
    | BAAI/bge-reranker-v2-m3 | multilingual, XLM-R base family, long input config | default strong multilingual reranker, natural pair with BGE-M3 | Apache-2.0 | ~568M |
    | jinaai/jina-reranker-v2-base-multilingual | multilingual, up to 1024 tokens | much lighter; useful speed/quality trade-off | CC-BY-NC-4.0 research/evaluation | 278M |
    | Lightweight classical ranker only | N/A | ablation / fallback if GPU budget is tight | N/A | negligible |

  ## Document-score aggregation options
  Suppose a candidate doc yields chunk reranker scores [4.2, 2.8, -0.6].
  Possible aggregations:
  * max: 4.2 — best when one Điều is enough.
  * max + α·second: 4.2 + α×2.8 — rewards multiple supporting chunks.
  * log-sum-exp: smoother multi-evidence aggregation.
  * learn all chunk features in LightGBM: defer aggregation decision to meta-ranker.
  Recommended starting point: keep at least best_score, second_score, count_score_above_threshold as separate meta-features instead of collapsing too early.

  ## Pair construction examples

  ### Positive pair
  question + localized gold chunk.

  ### Hard negative pair
  question + top retrieved non-gold chunk where:
  * topic is very close;
  * same institution/law family;
  * different procedure/year/article causes it to be wrong.

  ### Do not use blindly as negative
  A document positive for an almost identical train question, especially when annotations appear non-exhaustive.
  ```Mermaid
flowchart LR
    T["Train labels"] --> P["Positive docs"]
    H["Hybrid top candidates"] --> N["Hard-negative pool"]
    P --> L["Localize best chunks"]
    N --> G["False-negative guard"]
    L --> B["Query-group training batch"]
    G --> B
    B --> R["Fine-tuned reranker"]
  ```

  ## How many candidates should reranker see?
  Benchmark at least:
  * top 20: cheap, may miss retrieval errors;
  * top 50: practical default;
  * top 100: higher ceiling if GPU allows.
  Decision should be based on gold candidate coverage. If Candidate Recall@50 is already nearly saturated, reranking 100 may waste compute.

  ## Success criterion
  Reranker stage is successful if:
  * candidate Recall@K stays unchanged by definition;
  * Recall@5 increases materially;
  * gains persist on document-disjoint validation;
  * it fixes semantic confusions rather than only memorizing frequent laws.


========================================
# 07 — Learned Fusion & Top-5 Selection
========================================
07 — Learned Fusion & Top-5 Selection

  ## Mục tiêu
  Kết hợp các retriever/reranker không cùng thang điểm bằng một meta-ranker nhỏ, sau đó chọn 1–5 documents theo chiến lược tối ưu Recall trước.

  ## Feature set
  Mỗi (query, candidate_doc) tạo features:
  ```Plain Text
bm25_rank / score
dense_rank / score
lexical_rank / score
multi_vector_rank / score
rrf_score
reranker_max
reranker_second
memory_rank
memory_score
exact_question_match
exact_legal_number_match
title_token_overlap
rare_token_overlap
year_match
province_match
article_match
log1p(train_positive_frequency)
  ```

  ## Model fusion
  Khởi điểm:
  * LightGBM ranker / LambdaMART;
  * hoặc logistic regression nếu muốn baseline dễ debug.
  Parameter count không đáng kể.

  ## Training labels
  ```Plain Text
gold document = 1
other candidate = 0
  ```
  Group theo query.

  ## Output policy

  ### Default
  Chọn top 5 unique docs theo meta-score.
  Lý do: scorer ưu tiên Recall và false positive trong top 5 không làm giảm Recall.

  ### Adaptive K
  Chỉ bật nếu validation cho thấy:
  * Recall@5/Recall thực tế không giảm;
  * Precision tăng đủ để có lợi khi tie.
  Ví dụ rule có thể học:
  * exact question + reranker gap cực lớn → K=1–2;
  * uncertain/disagreement giữa retrievers → K=5.
  Nhưng không dùng heuristic này trước khi có evidence.

  ## Diversity protection
  Thử một alternative:
  ```Plain Text
3 docs tốt nhất theo meta-ranker
+ 1 sparse-strong doc chưa có
+ 1 memory/dense-strong doc chưa có
  ```
  So sánh với pure top5. Nếu diverse selection tăng Recall@5 thì giữ.

  ## Calibration
  Không cần probability calibration hoàn hảo nếu luôn top5. Chỉ cần calibration nếu dùng adaptive K.

  ## Submission post-processing
  ```Python
answer = dedupe_preserve_order(answer)
answer = answer[:5]
assert 1 <= len(answer) <= 5
  ```

  ## Guardrail
  Nếu một query vì bug không có candidate:
  * fallback sang global hybrid top docs cho chính query;
  * không để list rỗng vì scorer cho 0.

  ## Why learned fusion instead of one magic score
  Different retrievers produce scores on incompatible scales. BM25 score 18.2 and cosine 0.73 cannot be meaningfully added without calibration. Rank-based fusion is robust; learned fusion can exploit more signals once OOF features exist.
  ```Mermaid
flowchart LR
    B["BM25 rank/score"] --> F["Feature vector per query-doc"]
    D["Dense rank/score"] --> F
    S["Sparse neural rank/score"] --> F
    M["Memory sim/support"] --> F
    R["Reranker chunk scores"] --> F
    E["Exact year/entity/legal-id matches"] --> F
    P["Train doc frequency prior"] --> F
    F --> L["LightGBM / logistic ranker"]
    L --> T["Final document score"]
  ```

  ## Feature example for one candidate
  Suppose document A has:
    | Example | Interpretation | Feature |
    | 2 | strong lexical evidence | BM25 rank |
    | 8 | moderate semantic evidence | dense rank |
    | 0.94 | very similar train question supports A | memory similarity |
    | 5.1 | strong pair relevance | reranker best score |
    | 1 | query/document year consistent | year exact match |
    | 0 | no explicit number match | legal-number exact |
    | high | weak prior only | log train frequency |

  ## Fusion options to benchmark

  ### Option 1 — Reciprocal Rank Fusion
  Best first baseline because it does not need score calibration.

  ### Option 2 — weighted normalized scores
  Useful for quick experiments, but weights can overfit.

  ### Option 3 — LightGBM ranking/classification
  Recommended final option when features are generated out-of-fold. It can learn interactions such as:
  * memory high + year match → trust strongly;
  * memory high + year conflict → downweight;
  * BM25 + reranker both high → strong evidence;
  * frequent-law prior alone → insufficient.

  ## OOF requirement
  Never train fusion features from a base model that already trained on the same validation example without controlling leakage.
  ```Mermaid
flowchart TD
    D["Train queries"] --> S["K folds"]
    S --> F1["Train base models on folds 2..K<br>generate features for fold 1"]
    S --> F2["Train base models excluding fold 2<br>generate features for fold 2"]
    S --> FN["Repeat"]
    F1 --> OOF["OOF feature table"]
    F2 --> OOF
    FN --> OOF
    OOF --> L["Train fusion model"]
  ```

  ## Top-5 selection strategies

  ### Strategy A — pure top 5
  Take five highest fused scores. Strong default.

  ### Strategy B — diversity-protected
  Example:
  * 3 best fused candidates;
  * 1 best BM25-supported candidate not already selected;
  * 1 best memory/dense candidate not already selected.
  This can increase Recall if model errors are correlated, but may reduce Precision. Only keep it if validation Recall@5 improves.

  ### Strategy C — adaptive 1–5 outputs
  Can improve Precision when confidence is very high, but Recall is the primary ranking metric. Do not use unless Recall remains identical or improves.

  ## Concrete top-5 example
  Gold unknown at test time. Fused candidates:
  A=0.96, B=0.91, C=0.82, D=0.79, E=0.75, F=0.74.
  If BM25 strongly supports F while E appears only due to a frequent-document prior, diversity selection may choose [A,B,C,D,F]. This decision must come from validation evidence, not intuition.

  ## Hard rules before serialization
  * Deduplicate IDs.
  * Never output >5 IDs.
  * Preserve ID representation consistently with official files.
  * Ensure every query key exists exactly once.
  * Log final feature contributions for error analysis.


========================================
# 08 — Validation, Ablation & Error Analysis
========================================
08 — Validation, Ablation & Error Analysis

  ## Hai validation regimes bắt buộc

  ### A. Random query split
  Mô phỏng trường hợp test hỏi lại cùng luật/chủ đề và đo giá trị của train-question memory.

  ### B. Document-disjoint split
  Gold documents của validation không được xuất hiện làm positive trong training fold.
  Mục tiêu: bảo vệ private test khỏi overfit memorization.

  ## Metrics
    | Metric | Dùng để làm gì |
    | Recall@1 | Độ chính xác top candidate |
    | Recall@3 | Intermediate ranking |
    | Recall@5 | Metric tối ưu chính |
    | Precision@5 | Tie-break awareness |
    | Candidate Recall@50/100/150 | Đánh giá retriever stage |

  ## Experiment ladder
  1. E0 — BM25 whole-doc baseline.
  1. E1 — BM25 chunk.
  1. E2 — pretrained BGE-M3 chunk.
  1. E3 — BM25 + BGE-M3 RRF.
  1. E4 — + char TF-IDF question memory.
  1. E5 — + neural question memory + exact matcher.
  1. E6 — fine-tuned retriever.
  1. E7 — retriever + hard negatives round 2.
  1. E8 — cross-encoder reranker.
  1. E9 — learned fusion.
  1. E10 — top-5 diversity/adaptive-K tuning.

  ## Experiment log schema
  Mỗi run lưu:
  ```Plain Text
run_id
git_commit
model_versions
chunking config
retrieval topK
training seed
random_R@5
doc_disjoint_R@5
candidate_R@100
precision@5
public_score
notes
  ```

  ## Error taxonomy
  Mỗi failed validation query gắn một tag:
  * lexical exact miss;
  * paraphrase miss;
  * chunk boundary miss;
  * same-law wrong article;
  * old/new regulation ambiguity;
  * multi-document gold incomplete;
  * train annotation conflict;
  * unseen document;
  * location/year mismatch;
  * reranker inversion;
  * fusion mistake.

  ## Quy tắc quyết định
  Không accept một component chỉ vì Public tăng. Một change nên:
  * tăng/giữ Random Recall@5;
  * tăng/giữ Document-disjoint Recall@5;
  * hoặc có lý do rõ ràng dựa trên error class.
  Nếu Public tăng nhưng document-disjoint giảm mạnh, coi đó là dấu hiệu overfit query memory/seen documents.

  ## Validation phải trả lời 2 câu hỏi khác nhau

  ### A. Random-query split
  Đo khả năng khai thác:
  * luật/document đã từng xuất hiện ở train;
  * train-question memory;
  * wording tương tự.

  ### B. Document-disjoint split
  Gold documents của validation không được xuất hiện làm positive trong training fold. Đo khả năng generalize sang document chưa thấy.
  ```Mermaid
flowchart TD
    T["Official train"] --> R["Random query split"]
    T --> D["Document-disjoint split"]
    R --> R1["Measures seen-doc + memory performance"]
    D --> D1["Measures full-corpus generalization"]
    R1 --> C["Model selection"]
    D1 --> C
  ```

  ## Ví dụ leakage
  Giả sử document 129823 là positive cho rất nhiều câu hỏi lao động.
  Bad split:
  * train fold có 20 queries → 129823;
  * validation fold có 3 queries → 129823.
  Model có thể học strong prior cho 129823 và validation trông rất tốt dù khả năng tìm unseen law yếu.
  Document-disjoint split buộc toàn bộ positive edges tới 129823 nằm một phía, giúp kiểm tra generalization thật.

  ## Metrics dashboard nên có
    | Metric | Stage | Ý nghĩa |
    | Recall@1 | final | top document quality |
    | Recall@3 | final | early-list quality |
    | Recall@5 | final | metric chính để chọn hệ thống |
    | Precision@5 | final | tie-break / secondary |
    | Candidate Recall@20 | retrieval | reranker headroom |
    | Candidate Recall@50 | retrieval | practical rerank pool |
    | Candidate Recall@100 | retrieval | retrieval ceiling |

  ## Query buckets for error analysis
  Mỗi validation query nên được tag tự động theo signal có trong query:
  * definition: là gì, thế nào là;
  * time/amount: bao lâu, bao nhiêu, %, năm;
  * procedure: trình tự, thủ tục, hồ sơ;
  * authority: ai, cơ quan nào, thẩm quyền;
  * exact legal identifier present;
  * geographic entity present;
  * multi-positive gold;
  * high train-question similarity;
  * low train-question similarity.
  ```Mermaid
flowchart LR
    E["Errors"] --> B1["BM25 miss"]
    E --> B2["Dense miss"]
    E --> B3["Candidate found but reranker drops it"]
    E --> B4["Memory false friend"]
    E --> B5["Chunking loses relevant Điều"]
    E --> B6["Multi-positive partial recall"]
    B1 --> A["Actionable fixes"]
    B2 --> A
    B3 --> A
    B4 --> A
    B5 --> A
    B6 --> A
  ```

  ## Ablation ladder
  Run changes one at a time:
    | System | Exp | What it answers |
    | BM25 document-level | E0 | weak baseline |
    | BM25 chunk-level | E1 | value of chunking |
    | pretrained dense chunks | E2 | semantic baseline |
    | BM25 + dense | E3 | hybrid gain |
    |   • train-question memory | E4 | reuse gain |
    |   • exact metadata matcher | E5 | legal identifier gain |
    | fine-tuned retriever | E6 | domain adaptation gain |
    | hard-negative retraining | E7 | confusion reduction |
    | cross-encoder reranker | E8 | top-5 ranking gain |
    | learned fusion | E9 | multi-signal gain |
    | top-5 diversity tuning | E10 | metric-specific gain |

  ## Error record example
  For every miss, store:
  ```Plain Text
query_id
question
gold_doc_ids
final_top5
BM25 ranks of gold
dense ranks of gold
memory rank/similarity of gold
reranker score of gold
best gold chunk text preview
reason tag
  ```
  Không cần lưu để submission; đây là artifact debugging để biết stage nào làm mất gold.

  ## Decision rule
  Một model chỉ được promote nếu:
  1. Recall@5 tăng hoặc giữ nguyên trên random split;
  1. document-disjoint Recall không tụt bất thường;
  1. gain lặp lại qua nhiều seeds/folds nếu training stochastic;
  1. public score chỉ đóng vai trò confirmation, không phải source duy nhất cho quyết định.


========================================
# 09 — Implementation & Repository Layout
========================================
09 — Implementation & Repository Layout

  ## Repository đề xuất
  ```Plain Text
legalir/
├── configs/
│   ├── preprocess.yaml
│   ├── retrieval.yaml
│   ├── train_retriever.yaml
│   └── train_reranker.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── indices/
│   └── folds/
├── src/
│   ├── preprocess.py
│   ├── legal_chunker.py
│   ├── build_bm25.py
│   ├── build_dense_index.py
│   ├── question_memory.py
│   ├── retrieve.py
│   ├── mine_negatives.py
│   ├── train_retriever.py
│   ├── train_reranker.py
│   ├── rerank.py
│   ├── train_fusion.py
│   ├── predict.py
│   └── make_submission.py
├── scripts/
│   ├── 01_preprocess.sh
│   ├── 02_build_indices.sh
│   ├── 03_train_retriever.sh
│   ├── 04_mine_hard_negatives.sh
│   ├── 05_train_reranker.sh
│   ├── 06_train_fusion.sh
│   └── 07_predict.sh
├── outputs/
│   ├── experiments/
│   └── submissions/
├── requirements.txt
└── README.md
  ```

  ## Core data flow
  ```Mermaid
flowchart LR
    A["raw contexts"] --> B["legal chunker"]
    B --> C["BM25 index"]
    B --> D["BGE-M3 index"]
    E["train questions"] --> F["question-memory index"]
    C --> G["retrieve candidates"]
    D --> G
    F --> G
    G --> H["reranker"]
    H --> I["fusion"]
    I --> J["submission.json"]
  ```

  ## Cache strategy
  Cache mọi thứ có thể:
  * processed chunks;
  * dense embeddings;
  * BM25 index;
  * train question embeddings;
  * candidate lists cho train folds;
  * reranker logits cho meta-ranker.
  Điều này giúp ablation nhanh mà không encode lại corpus.

  ## Reproducibility
  * Fix random seeds.
  * Pin package/model revisions.
  * Save config YAML theo mỗi run.
  * Save Git commit hash.
  * Không hardcode local absolute paths.
  * Có script từ raw BTC files → submission.

  ## Inference sequence
  1. Load processed metadata.
  1. Load BM25 + BGE-M3 indices.
  1. Load train-question memory.
  1. Retrieve candidate union.
  1. Rerank top candidates.
  1. Apply fusion model.
  1. Dedup + output top 5.
  1. Validate JSON schema.
  1. Zip.

  ## Minimum viable baseline trước khi train
  Phải có một script chạy được:
  ```Plain Text
preprocess -> BM25 chunks -> top5 -> submission
  ```
  Sau đó mới thêm neural components để mọi improvement đo được rõ ràng.

  ## Recommended repository map — with artifact contracts
  Không cần bắt đầu bằng code phức tạp. Quan trọng là mỗi stage có input/output rõ ràng để có thể ablate và reproduce.
  ```Plain Text
project/
├── data/
│   ├── raw/                 # official files only
│   ├── processed/           # normalized documents/chunks
│   └── folds/               # random + doc-disjoint split definitions
├── indexes/
│   ├── bm25/
│   ├── dense/
│   └── memory/
├── models/
│   ├── retriever/
│   ├── reranker/
│   └── fusion/
├── artifacts/
│   ├── candidate_runs/
│   ├── rerank_runs/
│   ├── oof_features/
│   └── error_analysis/
├── configs/
├── scripts/
├── submission/
└── README.md
  ```

  ## Stage contracts
    | Input | What must be preserved | Stage | Output |
    | context JSONs | document_id, raw text, metadata | Preprocess | documents + chunks |
    | chunks | chunk→document mapping | Index | BM25/dense/sparse indexes |
    | question | all component ranks for analysis | Retrieve | ranked candidate docs + component scores |
    | question + candidate best chunks | best/second-best chunk evidence | Rerank | cross-encoder scores |
    | OOF/candidate features | feature names/version | Fusion | final doc score |
    | top docs | 1–5 unique IDs/query | Serialize | submission.json |

  ## Artifact-flow diagram
  ```Mermaid
flowchart TD
    RAW["Official raw files"] --> PROC["processed documents/chunks"]
    PROC --> IDX["versioned indexes"]
    Q["queries"] --> RET["candidate run"]
    IDX --> RET
    RET --> RR["rerank run"]
    RR --> FEAT["feature table"]
    FEAT --> FUSE["fusion model"]
    FUSE --> FINAL["final run"]
    FINAL --> SUB["submission.json"]
    RET --> ERR["error analysis"]
    RR --> ERR
    FINAL --> ERR
  ```

  ## Run naming convention
  Mỗi experiment nên có một run ID thể hiện pipeline config, ví dụ:
  e08_bgem3_bm25_memory_bgereranker_fold3_seed42
  Một run directory nên lưu:
  * config snapshot;
  * git commit hash;
  * model name + exact revision nếu có;
  * data/fold checksum;
  * candidate metrics;
  * final metrics;
  * prediction file;
  * logs.
  Mục tiêu: 2 tuần sau vẫn biết chính xác score đến từ đâu.

  ## Configuration categories
  Không hardcode hyperparameters rải rác. Config nên nhóm:
  * chunking: method, max tokens, overlap;
  * bm25: analyzer, field boosts, top_k;
  * dense: model, max_length, embedding dim, top_k;
  * memory: char-ngram range, neural top_k, similarity threshold;
  * reranker: model, candidate_k, chunks_per_doc;
  * fusion: algorithm, features, fold source;
  * selection: pure-top5 / diversity strategy;
  * runtime: batch size, fp16/bf16, device.

  ## Model registry — multiple options but one final stack
    | Use case | Retriever | Alias | Reranker |
    | recommended quality baseline | BGE-M3 | stack_A | BGE-reranker-v2-m3 |
    | lighter retriever | GTE-multilingual-base | stack_B | BGE-reranker-v2-m3 |
    | semantic comparison | multilingual-E5-large-instruct | stack_C | BGE-reranker-v2-m3 |
    | research/non-commercial alternative | Jina embeddings v3 | stack_D | Jina reranker v2 multilingual |
  Final submission repo chỉ cần chứa stack thắng; không cần đóng gói mọi experimental model nếu BTC không yêu cầu.

  ## Data provenance guard
  ```Mermaid
flowchart LR
    O["Official BTC data"] --> OK["Allowed training/evaluation"]
    P["Pretrained model weights"] --> OK2["Allowed per BTC clarification"]
    E["External legal corpus"] --> NO["Do not use"]
    S["Synthetic questions / augmentation"] --> NO
    API["External API inference"] --> NO
  ```
  README nên có mục Data provenance ghi rõ:
  * chỉ sử dụng train.json, selected-contexts.zip và official test inputs cho task 1;
  * không dùng dữ liệu task 2;
  * không dùng synthetic augmentation;
  * pretrained models và licenses được liệt kê riêng.

  ## Minimal reproducibility checklist
  * deterministic fold files checked in;
  * seed được log;
  * model revisions được pin nếu có thể;
  * package versions được freeze;
  * command order từ raw data → submission được ghi trong README;
  * không cần download external dataset ở bất kỳ bước nào;
  * submission validator chạy cuối cùng trước khi zip.


========================================
# 10 — Final Training, Packaging & Submission
========================================
10 — Final Training, Packaging & Submission

  ## Final model stack
  * Retriever: BAAI/bge-m3 ~568M.
  * Reranker: BAAI/bge-reranker-v2-m3 ~568M.
  * BM25 + TF-IDF + LightGBM: negligible neural params.
  * Tổng neural: ~1.14B, dưới 4B với khoảng an toàn lớn.

  > 💡 
    Trước submission chính thức, lưu model card/license/revision và tự kiểm tra lại tổng parameter count. BTC là bên quyết định cuối cùng về eligibility.

  ## Final training recipe
  1. Chốt preprocessing/chunking bằng ablation.
  1. Build full official train corpus index.
  1. Fine-tune retriever bằng weak-localized positives.
  1. Mine hard negatives bằng hybrid retriever.
  1. Fine-tune retriever round 2 nếu có lợi.
  1. Generate candidate pool cho toàn train bằng out-of-fold procedure để tránh leakage khi train reranker/fusion.
  1. Train reranker trên OOF candidates.
  1. Generate OOF reranker scores.
  1. Train fusion/meta-ranker.
  1. Refit retriever/reranker trên full train sau khi hyperparameter đã freeze.
  1. Build final full-corpus indices.
  1. Predict public/private input.

  ## Tránh leakage
  Meta-ranker và reranker validation phải dùng candidate/scores sinh từ model không được train trực tiếp trên chính query validation đó nếu có thể. Dùng folds/OOF để estimate thật hơn.

  ## Sanity checks
  - [ ] Candidate Recall@100 không giảm so với best experiment.
  - [ ] Random Recall@5 đạt best hoặc gần best.
  - [ ] Document-disjoint Recall@5 không collapse.
  - [ ] Exact identifier queries được retrieve đúng.
  - [ ] Multi-label queries không bị collapse về một doc duy nhất.
  - [ ] Output mỗi query 1–5 unique IDs.
  - [ ] Không ID ngoài corpus.
  - [ ] Số query output = số query test.

  ## Submission validator
  Script nên fail fast:
  ```Python
for qid, obj in pred.items():
    ans = obj["answer"]
    assert isinstance(ans, list)
    assert 1 <= len(ans) <= 5
    assert len(ans) == len(set(ans))
    assert all(isinstance(x, str) for x in ans)
  ```

  ## Package cho BTC
  README phải ghi:
  * OS/CUDA/Python.
  * Cách tải model weights hợp lệ.
  * Model names + exact revisions.
  * Parameter count.
  * Preprocessing command.
  * Training commands.
  * Inference command.
  * Expected output path.
  * Thời gian/RAM/VRAM ước tính.

  ## Freeze rule trước private
  Khi pipeline đã đạt best validation + public hợp lý:
  * freeze model family;
  * chỉ sửa bug/reproducibility;
  * không chase leaderboard bằng heuristic không có validation support.

  ## Definition of Done
  Task hoàn tất khi từ bộ file BTC nguyên bản có thể chạy một command/script chuẩn và tạo đúng submission.zip, không cần API hay dữ liệu ngoài.

  ## Final model-selection gate
  Đừng chọn stack theo cảm giác. Trước khi final-train, lập bảng như sau và chỉ chọn model sau khi có đủ số liệu:
    | Doc-disjoint Recall@5 | Latency | Stack | Decision | Random Recall@5 | Total params | Candidate R@100 |
    | TBD | TBD | A: BGE-M3 + BGE reranker | benchmark | TBD | ~1.14B | TBD |
    | TBD | TBD | B: GTE multilingual + BGE reranker | benchmark | TBD | ~0.87B | TBD |
    | TBD | TBD | C: E5 multilingual + BGE reranker | benchmark | TBD | ~1.17B | TBD |
    | TBD | TBD | D: Jina embed + Jina reranker | benchmark + license check | TBD | ~0.88B | TBD |

  > 💡 
    Recommended starting favorite: Stack A. Recommended final decision rule: choose the stack with the best robust Recall@5 across validation protocols, not necessarily the largest model.

  ## Full final-training flow
  ```Mermaid
flowchart TD
    V["Validation complete"] --> S["Select winning stack + hyperparameters"]
    S --> T1["Train retriever on all allowed train data"]
    T1 --> I["Rebuild full corpus indexes"]
    I --> H["Mine final candidates / hard negatives if training schedule requires"]
    H --> T2["Train final reranker"]
    T2 --> O["Generate OOF-compatible/final fusion artifacts"]
    O --> T3["Train final fusion model"]
    T3 --> P["Run inference on official test"]
    P --> V2["Submission validation"]
    V2 --> Z["submission.zip"]
  ```

  ## What exactly gets frozen before final training
  Freeze:
  * preprocessing rules;
  * chunking strategy;
  * model family/revision;
  * retrieval top-k values;
  * memory strategy;
  * reranker candidate_k and chunks/doc;
  * fusion features;
  * top-5 selection policy.
  Sau khi freeze, không dùng public leaderboard để liên tục sửa heuristic nhỏ nếu không có validation evidence.

  ## Final inference example
  Public input:
  ```JSON
{
  "106660": {
    "question": "Công ty cổ phần là gì?",
    "answer": null
  }
}
  ```
  Inference stages conceptually produce:
  ```Plain Text
BM25 candidates      → docs A, B, C, ...
Dense candidates     → docs A, D, E, ...
Question memory      → docs X, A, ...
Union                → ~100–180 docs
Reranker             → top-scored candidate docs
Fusion               → A > X > D > B > E > ...
Final                → 5 unique document IDs
  ```
  Không điền answer text; output cuối chỉ là IDs.

  ## Submission validator — conceptual checks
  ```Mermaid
flowchart TD
    S["submission.json"] --> K{"Same number/query keys as test?"}
    K -->|No| X1["STOP"]
    K -->|Yes| L{"Each answer is a list?"}
    L -->|No| X2["STOP"]
    L -->|Yes| N{"1 <= len(answer) <= 5?"}
    N -->|No| X3["STOP: scorer may give 0"]
    N -->|Yes| U{"IDs unique?"}
    U -->|No| X4["Deduplicate/fix"]
    U -->|Yes| C{"All IDs exist in context corpus?"}
    C -->|No| X5["Fix invalid IDs"]
    C -->|Yes| OK["Ready to zip"]
  ```

  ## Compliance checklist before handing model to BTC

  ### Parameter constraint
  - [ ] Sum parameters of all neural models < 4B.
  - [ ] Parameter count is measured on original architecture, not reduced because of quantization.
  - [ ] Exact model names/revisions documented.

  ### Data constraint
  - [ ] No external legal corpus.
  - [ ] No synthetic/augmented questions.
  - [ ] No task-2 data used for task 1.
  - [ ] Fine-tuning examples derive only from official BTC train/context data.

  ### API/model constraint
  - [ ] No API is required for training or inference.
  - [ ] Model weights are directly accessible/controllable.
  - [ ] Model licenses are recorded.
  - [ ] Any CC-BY-NC/research-only option is explicitly documented and checked against BTC rules before final submission.

  ### Reproduction
  - [ ] README has exact steps from raw files to prediction.
  - [ ] Environment/package versions recorded.
  - [ ] Model download instructions or packaged weights documented.
  - [ ] Random seeds/config files included.
  - [ ] Final scoring can be reproduced offline after weights are available.

  ## Packaging options BTC allows
  Theo rules.txt, BTC không bắt buộc duy nhất Docker; có thể cung cấp code qua GitHub, zip source/weights hoặc hình thức khác miễn có thể tái lập. Vì vậy chọn packaging theo độ ổn định: