# Test Set & Scoring Design — Multi-Agent Data Analyst

## 1. Vấn đề với cách đo hiện tại

`metrics.py` cho ra **Reliability Score** dựa trên 4 trọng số:
`language_purity (30%) + format_success (15%) + tool_success (30%) + task_completed (25%)`.

Đây là các chỉ số **hành vi** (behavioral): agent có nói tiếng Việt sạch không, có viết
đúng cú pháp ReAct không, tool có báo lỗi không, agent có "chốt" Final Answer không bị
timeout/bluff.

Điều nó **không đo được**: agent có tính **đúng số** không, có vẽ **đúng cột** không, có
load **đúng file** không. Một agent trả lời rất mượt, tiếng Việt sạch 100%, đúng format,
gọi tool "thành công" (không có exception) — nhưng vẽ nhầm histogram của cột `quantity`
thay vì `revenue` — vẫn được ~100 điểm Reliability Score.

=> Bộ test set này thêm **tầng "Task Correctness"** dựa trên side-effect thật (state của
`DataStore`, file ảnh sinh ra, giá trị trả về của tool) làm ground truth độc lập với
những gì model "nói" nó đã làm — đúng tinh thần `store.debug()` đã có sẵn trong code gốc.

## 2. Kim tự tháp test (test pyramid)

```
        ▲  4. Regression / trend theo thời gian (so sánh Qwen-local vs Gemma-API,
        │     so sánh khi đổi prompt/model) — chạy theo lịch, không chạy mỗi commit
        │
        │  3. Agent-level eval set (cần gọi LLM thật/API)
        │     - Golden test cases theo từng specialist + multi-agent pipeline
        │     - Adversarial / negative cases (lỗi cố ý, câu hỏi mơ hồ, ép trôi ngôn ngữ)
        │     - Chấm bằng: assertion trên DataStore (tự động) + Reliability Score (tự động)
        │       + rubric người đọc cho phần "insight"/"report" tự nhiên ngôn ngữ (thủ công)
        │
        │  2. Prompt & parsing tests (không cần LLM thật, dùng LLM giả lập - FakeListLLM)
        │     - react_prompt render đúng biến, đúng thứ tự Thought/Action/Observation
        │     - _truncate_at_stop cắt đúng tại "\nObservation"
        │     - handle_parsing_errors trả đúng PARSING_ERROR_MESSAGE
        │
        ▼  1. Unit tests thuần Python (nhanh, chạy mỗi commit, KHÔNG cần mạng/API key)
           - metrics.py: language_purity, format_success, tool_success, completed,
             reliability_score, bluff_detected — test bằng input giả lập (không LLM)
           - data_store.py: add/get/debug, singleton behavior
           - llm_setup.py: RotatingGeminiClient rotation logic (mock genai.Client),
             contains_cjk regex
           - agents._truncate_at_stop
```

File đính kèm hiện thực tầng 1 (`test_metrics_unit.py` cho `metrics.py`,
`test_tools_unit.py` cho `tools/`) và tầng 3 (`eval_harness.py` + `test_cases.jsonl`) —
vì đây là các tầng có giá trị cao nhất / chi phí thấp nhất để bắt đầu, và tầng 1 đã bắt
được 1 bug thật ngay lần chạy đầu (xem mục 6). Tầng 2 và 4 mô tả cách mở rộng bên dưới.

## 3. Test set (`test_cases.jsonl`) — cách phân loại

Mỗi dòng JSON là 1 test case với các trường:

| field | ý nghĩa |
|---|---|
| `id` | mã định danh, để trace lỗi |
| `category` | nhóm chức năng (xem bảng dưới) |
| `difficulty` | `easy` (1 tool call) / `medium` (2-3 bước, 1 specialist) / `hard` (nhiều specialist phối hợp hoặc cần nhớ context từ lượt trước) |
| `setup` | tên 1 hàm setup có sẵn trong harness (vd tạo sẵn dataset trước khi hỏi), hoặc `null` |
| `query` | câu hỏi tiếng Việt gửi vào `analyze()` |
| `expect` | dict mô tả điều kiện PASS, harness tự assert (xem mục 4) |
| `notes` | ghi chú cho người review |

### Các nhóm category và lý do có mặt

- **`data_loader_*`** — load CSV/Excel/sample/custom column. Test riêng vì đây là bước
  nền, lỗi ở đây sẽ lan ra toàn bộ pipeline phía sau.
- **`statistician_*`** — describe/correlation/hypothesis/outlier/insight. Cần oracle số
  học (tính sẵn bằng pandas trong harness) để so khớp, không chỉ tin lời agent.
- **`visualizer_*`** — kiểm tra **file ảnh có thực sự được tạo** tại `CHART_OUTPUT_PATH`
  (đúng tinh thần comment trong `config.py` về lỗi `savefig()` từng gặp), không chỉ tin
  Final Answer nói "đã vẽ xong".
- **`transformer_*`** — filter/aggregate/cột tính toán: so khớp shape/giá trị dataset kết
  quả trong `store`.
- **`reporter_*`** — tổng hợp báo cáo, dùng `history` trong `DataStore`.
- **`multi_agent_pipeline`** — 1 câu hỏi buộc orchestrator gọi ≥2 specialist tool (vd
  "describe rồi vẽ histogram") — test khả năng điều phối, không phải khả năng 1 specialist.
- **`context_memory`** — 2 lượt `analyze()` liên tiếp, lượt 2 dựa vào dataset đã load ở
  lượt 1 (test đúng cơ chế "context injection" trong `make_analyze_fn`).
- **`error_handling_*`** — dataset không tồn tại / tên cột sai / thao tác không hợp lệ.
  PASS = agent **báo lỗi rõ ràng**, KHÔNG bịa kết quả (đây chính là mục đích của
  `BLUFF_MARKERS` + đoạn "QUAN TRỌNG: Nếu Observation... THẤT BẠI THẬT SỰ" trong prompt).
- **`adversarial_language_drift`** — chèn tiếng Anh/Trung/yêu cầu "trả lời bằng tiếng Anh"
  vào query để kiểm tra 3 lớp phòng thủ ngôn ngữ (system_instruction + prompt nhắc lại +
  `language_purity` score) có thực sự giữ được tiếng Việt không.
- **`adversarial_fabrication_bait`** — cố tình hỏi về dataset/cột không tồn tại theo cách
  "gài" model trả lời như thể nó tồn tại — bắt lỗi hallucination sớm.
- **`ambiguous_request`** — câu hỏi thiếu thông tin (vd "vẽ biểu đồ" không nói cột nào) —
  PASS chấp nhận cả 2 hướng: agent hỏi lại làm rõ, HOẶC agent chọn hợp lý + nói rõ đã chọn
  gì (không PASS nếu im lặng chọn bừa).

## 4. Cách chấm điểm — 3 tầng độc lập

### Tầng A — Tự động, khách quan (không cần đọc văn bản)
Assertion trực tiếp trên **side effect**, không tin lời model:
- `store.datasets` có đúng tên/đúng shape/đúng cột sau khi chạy không
- File ảnh tồn tại tại đúng path, mtime mới hơn thời điểm bắt đầu test
- Với `statistician_*`: so số agent báo (parse từ `intermediate_steps` observation của
  tool, KHÔNG parse từ Final Answer tiếng Việt) với giá trị pandas tính trực tiếp, sai số
  cho phép `1e-3`
- Với `error_handling_*`: `trace.completed == False` HOẶC final answer chứa từ khóa lỗi,
  VÀ `trace.bluff_detected == False`

=> Đây là **Task Correctness Score** (đề xuất thêm, không có sẵn trong `metrics.py`):
`task_correctness = số case Tầng-A PASS / tổng số case`, báo cáo **tách riêng** khỏi
Reliability Score, không gộp chung — vì hai chỉ số đo hai thứ khác nhau (hành vi vs đúng).

### Tầng B — Tự động, dùng `QueryTrace` có sẵn
Lấy trực tiếp từ `monitor.build_trace()` sau mỗi lần `analyze()`:
`reliability_score`, `purity`, `format_success`, `tool_success`, `completed`. Ngưỡng PASS
theo đúng badge có sẵn trong `config.py` (`SCORE_GOOD_THRESHOLD=85`,
`SCORE_OK_THRESHOLD=60`) — case nào rơi vào 🔴 thì tự động fail, 🟡 thì warn.

### Tầng C — Thủ công, rubric người đọc (chỉ áp dụng cho các category sinh văn bản tự do:
`statistician_insight`, `reporter_summary`)
Thang 1-5 cho từng tiêu chí, người review chấm tay theo mẫu:

| Tiêu chí | 1 | 3 | 5 |
|---|---|---|---|
| Đúng số liệu | Sai số/bịa số | Đúng nhưng thiếu ngữ cảnh | Đúng + diễn giải hợp lý |
| Hữu ích | Chung chung, không actionable | Có gợi ý nhưng mơ hồ | Cụ thể, người không rành kỹ thuật hiểu được |
| Trung thực | Nói "thành công" dù tool lỗi | Không rõ ràng | Nêu đúng giới hạn/lỗi nếu có |

Case nào có bất kỳ tiêu chí nào = 1 → tự động coi là FAIL toàn bộ case (không cho trung
bình cộng che lỗi nghiêm trọng).

## 5. Báo cáo tổng hợp

`eval_harness.py` in ra:
1. Bảng theo `category` × `difficulty`: % PASS tầng A, Reliability Score trung bình tầng B
2. Danh sách case FAIL kèm lý do (để không phải đọc lại toàn bộ log)
3. `monitor.dashboard()` gốc (giữ nguyên, không thay thế)

Lưu kết quả mỗi lần chạy ra `eval_runs/<timestamp>.json` để tầng 4 (regression theo thời
gian, so sánh giữa các model/backend) có dữ liệu so sánh — trả lời đúng câu hỏi README đặt
ra: "đổi từ Qwen local sang Gemma API có thực sự tốt hơn không", bằng số chứ không chỉ bằng
cảm nhận.

## 6. Findings — bug thật đã tìm thấy khi dựng bộ test này

Chạy `test_tools_unit.py` trực tiếp trên `tools/` (không cần LLM) phát hiện:

**🔴 Bug hệ thống: 10/10 tool thao tác trên dataset có sẵn crash thay vì báo lỗi, khi
dataset không tồn tại.**

`describe_dataset`, `correlation_analysis`, `hypothesis_test`, `outlier_detection`,
`generate_summary_report`, `filter_data`, `aggregate_data`, `add_calculated_column`,
`generate_insights`, `create_visualization`, `create_distribution_report` đều gọi
`df = store.get(dataset_name)` **ngoài** khối `try/except` của chúng. `DataStore.get()`
raise `ValueError` khi dataset không tồn tại - exception này không được tool nào trong
danh sách trên bắt lại, nên nó bay thẳng qua `AgentExecutor` (cả specialist lẫn
orchestrator) tới `agents.make_analyze_fn()`, bị chặn đứng bởi khối `try/except` NGOÀI
CÙNG:

```python
try:
    result = master_executor.invoke({"input": context + query})
    record_agent_result(result)          # <-- KHÔNG BAO GIỜ chạy tới đây
    trace = monitor.build_trace(query, result)   # <-- KHÔNG BAO GIỜ chạy tới đây
    ...
except Exception as e:
    return f"Lỗi: {str(e)}"
```

Hệ quả:
1. Toàn bộ ReAct loop bị cắt ngang - agent không có cơ hội đọc Observation lỗi và tự sửa
   (đúng cơ chế mà prompt `agents.py` đã cố dạy: "sửa lại Action Input... hoặc Final
   Answer phải nói rõ THẤT BẠI"). Người dùng nhận 1 thông báo lỗi kỹ thuật thô thay vì
   câu trả lời tiếng Việt tự nhiên.
2. **`monitor.traces` không có entry nào cho lượt hỏi này** - lượt hỏi thất bại nặng nhất
   lại là lượt hoàn toàn vô hình với Reliability Score / `monitor.dashboard()`, có thể
   tạo ảo giác hệ thống đang chạy tốt hơn thực tế (survivorship bias trong chính công cụ
   đo lường).

Đã tái hiện bằng code thật (không phải suy đoán) - xem `test_tools_unit.py::test_KNOWN_BUG_tool_crashes_instead_of_returning_error_string` (9 case parametrize, tất cả đang PASS = bug vẫn còn).

**Đề xuất fix** (không tự áp dụng vào code của bạn, chỉ gợi ý hướng): thêm 1 decorator
hoặc đổi ngay trong `DataStore.get()`:

```python
# cách 1 - sửa từng tool: bọc luôn store.get() vào trong try/except đã có sẵn
try:
    df = store.get(dataset_name)
    column = resolve_column(df, column)
    ...
except (ValueError, ColumnNotFoundError) as e:
    return f"Lỗi: {str(e)}"

# cách 2 - tập trung 1 chỗ: decorator dùng chung cho mọi @tool thao tác trên dataset có sẵn
def catches_dataset_errors(fn):
    @wraps(fn)
    def wrapper(tool_input):
        try:
            return fn(tool_input)
        except (ValueError, ColumnNotFoundError) as e:
            return f"Lỗi: {str(e)}"
    return wrapper
```

Cách 2 an toàn hơn vì không phụ thuộc việc mỗi tool tự nhớ bọc try/except đúng chỗ -
đúng lỗi gốc gây ra bug này.

Sau khi bạn sửa, đổi 9 test `test_KNOWN_BUG_*` từ `pytest.raises(ValueError)` sang
`assert tool_fn.func(arg).startswith("Lỗi")` - test sẽ báo đỏ ngay nếu ai đó vô tình làm
regression lại.

## 7. Cách chạy

```bash
# Tầng 1 — không cần API key, chạy trong vài giây
pytest eval/test_metrics_unit.py -v

# Tầng 3 — cần GOOGLE_API_KEYS thật (tốn quota/tiền, chạy trước khi release hoặc theo lịch)
python eval/eval_harness.py --cases eval/test_cases.jsonl
python eval/eval_harness.py --cases eval/test_cases.jsonl --only easy   # chạy nhanh subset
```
