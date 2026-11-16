# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `day6_adapter.py` | Đọc trực tiếp index Chroma B và cấu hình embedding Day 6 |
| `retrievers.py` | Dense Day 6, TF-IDF lexical và hybrid RRF |
| `model_controller.py` | Model thật qua OpenAI-compatible tool calling |
| `compare_retrieval.py` | So sánh ba phương pháp trên cùng split |
| `run_agent_benchmark.py` | Chạy agent thật và lưu trace/metadata |
| `prepare_day7_benchmark.py` | Tạo benchmark Day 7 từ Day 6 và câu hỏi mới |

Starter hiện nối trực tiếp index Chroma Day 6 trên máy này. Ba phương pháp là dense retrieval trên collection B, TF-IDF lexical, và hybrid reciprocal-rank fusion (RRF). Model agent phải hỗ trợ OpenAI-compatible Chat Completions và tool calling. Mock chỉ dùng để kiểm tra code, không dùng để báo benchmark chất lượng.

## 1. Dữ liệu Day 6 đã nối

Mặc định adapter đọc `D:\VinUni\K5-L3B-DAY06-LeHuyAnh-2A202603786-DataIngestion\starter\results\chroma`, collection `day6_strategy_b`, không sao chép index hay tạo lại embeddings. Nó đọc khóa/model từ `.env` Day 6 khi cần gọi embedding hoặc chat API; giá trị secret không được ghi ra log. Nếu thư mục khác, đặt `DAY6_ROOT`; có thể đổi collection bằng `DAY6_COLLECTION` và đường dẫn index bằng `DAY6_CHROMA_PATH`.

Sinh bộ câu hỏi:

```powershell
python starter/prepare_day7_benchmark.py
```

Benchmark giữ 15 câu dev có đáp án từ Day 6, viết 5 câu test mới trên trang 28–34 để thay bộ test Day 6 đã được xem, và thêm 4 câu thiếu evidence. Các lý do thiếu evidence của U01–U04 đã được đối chiếu với toàn bộ corpus đã index. Lỗi retrieval đơn lẻ không được dùng làm bằng chứng cho nhãn thiếu.

## 2. So sánh retrieval

Chạy từ repo Day 7; mỗi retrieval embedding query có thể phát sinh API cost:

```powershell
python starter/compare_retrieval.py --split dev --top-k 10 --budget 2400 --output starter/results/dev-k10-b2400
```

Script lưu `traces.json` và `summary.json` cho cả ba chiến lược. Chỉ tính câu có đáp án vào Hit@1, Evidence Hit@3 và Context Evidence. Thử nghiệm cải thiện có thể đổi `--top-k` hoặc `--budget`, hoặc chọn hybrid; lưu mỗi cấu hình vào thư mục riêng. Chọn bằng dev trước khi chạy test.

## 3. Chạy agent thật

Chat completion và query embedding dùng cùng khóa/provider ở Day 6 `.env`, hoặc cấu hình `.env` riêng trong `starter/`. `CHAT_MODEL` được dùng mặc định; có thể đặt `DAY7_CHAT_MODEL` và `DAY7_CHAT_BASE_URL` để override. Ví dụ chạy dev:

```powershell
python starter/run_agent_benchmark.py --split dev --strategy dense --top-k 3 --max-searches 3 --budget 2400 --output starter/results/agent-dev-dense
```

Chạy lexical hoặc hybrid bằng `--strategy lexical` hoặc `--strategy hybrid`. Output gồm case labels tách khỏi prompt, trace từng câu và metadata số lượt gọi/token model. Mỗi câu gọi model thật và có thể bị tính phí; script không tự gửi request khi chỉ sinh benchmark hay chỉ đọc cấu hình. Chấm citation và grounded answer thủ công, sau đó dùng `agent_metrics.py` để tổng hợp. Chỉ chạy test sau khi chốt cấu hình trên dev.

## 4. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 5. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 6. Dùng dữ liệu khác / xuất JSON

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 7. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
