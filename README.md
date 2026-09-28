# 🤖 Multi-Agent Data Analyst

### A Vietnamese Multi-Agent Data Analysis System powered by Gemma 4 26B-A4B-IT

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Gemma](https://img.shields.io/badge/Model-Gemma%204%2026B--A4B--IT-orange)](https://ai.google.dev/gemma)
[![Google AI](https://img.shields.io/badge/API-Google%20AI%20Studio-4285F4?logo=google\&logoColor=white)](https://aistudio.google.com/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle\&logoColor=white)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A modular **multi-agent data analysis system** designed to let users analyze datasets using natural-language instructions in Vietnamese. The system combines specialist agents, data analysis tools, reliability monitoring, and a rate-limit-aware LLM backend.

---

## ✨ Features

* 🇻🇳 **Vietnamese-first interaction**
  Designed around Vietnamese prompts and responses for data analysis tasks.

* 🤖 **Multi-agent architecture**
  Uses multiple specialist agents coordinated by an orchestrator to handle different stages of data analysis.

* 🧠 **Gemma 4 26B-A4B-IT API**
  Uses the Gemma API instead of local inference to improve stability on Kaggle/Colab environments.

* 🔄 **Automatic API key rotation**
  Supports multiple Google API keys and automatically switches keys when rate limits are reached.

* 📊 **End-to-end data analysis**

  * CSV loading
  * Excel loading
  * Dataset creation
  * Descriptive statistics
  * Correlation analysis
  * Hypothesis testing
  * Outlier detection
  * Filtering and aggregation
  * Calculated columns
  * Visualization
  * Distribution reports
  * Analysis history

* 🛡️ **Reliability Score**
  Provides a single 0–100 score that summarizes the quality of each analysis run.

* 🔍 **Column resolution**
  Helps agents correctly identify dataset columns even when users refer to them using natural-language descriptions.

* 🧪 **Mock-tested key rotation**
  The API rotation mechanism has been unit-tested using mocked rate-limit failures.

* 📈 **Session monitoring**
  Tracks analysis quality across multiple requests through `monitor.dashboard()`.

---

# 🏗️ Architecture

The system follows a modular **orchestrator + specialist agents + tools** architecture.

```text
                        ┌──────────────────────┐
                        │      User Query      │
                        │   Vietnamese Input   │
                        └──────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │       Orchestrator       │
                    │    ReAct-based Agent     │
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Data Agent   │ │ Stats Agent  │ │  Viz Agent   │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               └────────────────┼────────────────┘
                                │
                                ▼
                    ┌──────────────────────────┐
                    │          Tools           │
                    │                          │
                    │ Data / Stats / Transform │
                    │ Visualization / Reports │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       DataStore           │
                    │  Session Dataset State    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Reliability Monitor    │
                    │        0 — 100            │
                    └──────────────────────────┘
```

### LLM Backend

The agents interact with Gemma through a centralized LLM setup layer:

```text
                 ┌───────────────────────┐
                 │    Multi-Agent System  │
                 └───────────┬───────────┘
                             │
                             ▼
                   build_gemini_llm()
                             │
                             ▼
              ┌─────────────────────────────┐
              │ RotatingGeminiClient        │
              └──────────────┬──────────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
            API Key 1     API Key 2     API Key N
               │             │             │
               └─────────────┼─────────────┘
                             │
                             ▼
                  Gemma 4 26B-A4B-IT API
```

The client keeps using the current API key until it encounters a genuine rate-limit or quota error. It then switches to the next available key.

---

# 📁 Project Structure

```text
analyst_agent/
│
├── config.py
│   └── Global configuration parameters
│       (model, thresholds, weights, etc.)
│
├── llm_setup.py
│   └── Gemma API setup and rotating API-key client
│
├── data_store.py
│   └── DataStore singleton for session-level datasets
│
├── metrics.py
│   └── Reliability Score and monitoring
│
├── agents.py
│   └── ReAct prompts, specialist agents, orchestrator
│
├── main.py
│   └── System initialization and entry point
│
└── tools/
    │
    ├── parsing_utils.py
    │   └── Shared key=value parser
    │
    ├── data_tools.py
    │   └── CSV / Excel / sample dataset loading
    │
    ├── stats_tools.py
    │   └── Statistics, correlation, hypothesis testing,
    │       and outlier detection
    │
    ├── viz_tools.py
    │   └── Visualization and distribution reports
    │
    ├── transform_tools.py
    │   └── Filtering, aggregation, calculated columns
    │
    └── report_tools.py
        └── Summary reports and analysis history
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

For Excel support:

```bash
pip install excel-parser
```

---

# 🔐 API Key Configuration

This project is designed to run with the **Gemma 4 26B-A4B-IT API**.

Because the project may be run on public Kaggle notebooks, API keys should **never be hard-coded** into the notebook or source code.

## Kaggle

Go to:

**Add-ons → Secrets**

Create the following secrets:

```text
GOOGLE_API_KEY_1
GOOGLE_API_KEY_2
GOOGLE_API_KEY_3
GOOGLE_API_KEY_4
GOOGLE_API_KEY_5
```

You can obtain API keys from [Google AI Studio](https://aistudio.google.com/apikey).

Then load the keys before calling `setup()`:

```python
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()

keys = [
    secrets.get_secret(f"GOOGLE_API_KEY_{i}")
    for i in range(1, 6)
]

os.environ["GOOGLE_API_KEYS"] = ",".join(keys)

from main import setup

analyze = setup()
```

### Google Colab

Use `google.colab.userdata.get(...)` instead of `UserSecretsClient`:

```python
from google.colab import userdata
import os

keys = [
    userdata.get(f"GOOGLE_API_KEY_{i}")
    for i in range(1, 6)
]

os.environ["GOOGLE_API_KEYS"] = ",".join(keys)
```

---

# 🔄 API Key Rotation

The project uses:

```python
llm_setup.RotatingGeminiClient
```

The client follows a **rate-limit-aware rotation strategy**.

### Key behavior

1. Start with the current API key.
2. Keep using that key normally.
3. If the API returns a rate-limit/quota error:

   * `429`
   * `RESOURCE_EXHAUSTED`
   * quota-related errors

   switch to the next key.
4. Continue using the new key until it is also rate-limited.
5. If all keys are exhausted, raise a clear error.

The system **does not randomly round-robin** across keys. This allows each key to use its available quota efficiently.

### Non-rate-limit errors

Errors such as:

* Invalid API key
* Invalid model name
* Invalid request
* Other configuration errors

are raised immediately rather than retrying with every API key.

### Testing

The rotation logic has been unit-tested using mocked API failures:

```text
API Key 1 → Rate limited
API Key 2 → Rate limited
API Key 3 → Success
```

---

# 💻 Usage

## Basic Usage

```python
from main import setup

analyze = setup()

result = analyze(
    "Create a sample sales dataset named sales_data"
)

print(result)
```

## Analyze an Existing Dataset

```python
result = analyze(
    "Describe 'sales_data' and plot a histogram of revenue"
)

print(result)
```

## Load an Excel File

```python
result = analyze(
    "Load the Excel file at "
    "/kaggle/input/data.xlsx "
    "and name it 'bao_cao_q4'"
)

print(result)
```

The Excel loader uses `excel-parser` instead of directly relying on `pandas.read_excel`.

This can preserve additional spreadsheet information such as formulas, tables, and source-cell references such as:

```text
Sheet!A1:F18
```

These references can be useful when the agent needs to explain where a value originated from in an Excel file.

> **Compatibility note:** `excel-parser` APIs such as `to_dataframe()` and `to_records()` may vary across versions. If the Excel loader fails, check the installed version with:
>
> ```bash
> pip show excel-parser
> ```

---

# 📊 Reliability Score

The system provides a single **Reliability Score from 0 to 100**.

| Score | Status          | Meaning                  |
| ----: | --------------- | ------------------------ |
|  ≥ 85 | 🟢 Good         | Reliable output          |
| 60–84 | 🟡 Acceptable   | Monitor the result       |
|  < 60 | 🔴 Needs Review | Output should be checked |

The score combines four behavioral components:

| Component | Weight | What it measures |
| --- | ---: | --- |
| Vietnamese Purity | 30% | Whether generated text stays in the expected Vietnamese language style |
| Format Compliance | 15% | Whether agent steps follow the expected syntax and output format |
| Tool Success | 30% | Whether the requested tools execute successfully |
| Task Completion | 25% | Whether the agent reaches a completed result rather than stopping or timing out |

> **Important:** Reliability Score measures agent behavior and execution quality. It does **not** by itself prove that the numerical result, selected column, loaded file, or generated visualization is correct.

---

# 📈 Monitoring

The monitor can be used to inspect individual requests:

```python
from metrics import monitor

print(
    analyze(
        "Describe 'sales_data' and "
        "plot a histogram of revenue"
    )
)
```

Example output:

```text
🟢 Good — Reliability Score: 92.5/100

  • Vietnamese Purity:       100.0%
  • Format Compliance:        80.0%
  • Question Completion:      Yes
```

For multiple requests:

```python
print(monitor.dashboard())
```

This provides an aggregated view of system performance over the current session and can help identify quality degradation over time.

---

# 🧪 Example Workflow

A typical analysis session looks like this:

```python
from main import setup
from data_store import store
from metrics import monitor

# Initialize the system
analyze = setup()

# Create a dataset
print(
    analyze(
        "Create a sample sales dataset "
        "named sales_data"
    )
)

# Inspect the dataset
print(
    analyze(
        "Describe sales_data"
    )
)

# Analyze the data
print(
    analyze(
        "Find the correlation between revenue "
        "and quantity"
    )
)

# Visualize the result
print(
    analyze(
        "Plot the distribution of revenue"
    )
)

# Debug current session state
store.debug()

# Check overall reliability
print(monitor.dashboard())
```

---

# 🧪 Benchmark Results: Gemma vs Qwen

The benchmark evaluates two dimensions separately:

- **Task Correctness (`task_correctness_pass`)** is the primary metric. It checks the actual task outcome using dataset state, generated files, tool-returned values, and other observable side effects.
- **Reliability Score** is used as a supporting behavioral metric. It measures language quality, format compliance, tool success, and task completion; it should not be interpreted as proof that the task result is numerically or semantically correct.

This separation follows the testing design: Task Correctness is independent from Reliability Score because the two metrics measure different things.

## Overall results

| Difficulty | Gemma | Qwen | Comparison |
| --- | ---: | ---: | --- |
| **Easy** | **2/3 = 66.7%** | **1/3 = 33.3%** | Gemma +33.4 percentage points |
| **Medium** | **4/4 = 100%** | **5/10 = 50.0%** | Gemma +50.0 percentage points |
| **Hard** | **5/7 = 71.4%** | **5/7 = 71.4%** | Same pass rate |
| **Overall** | **11/14 = 78.6%** | **11/20 = 55.0%** | Gemma +23.6 percentage points |

> The totals use the available test cases for each model. The benchmark should therefore be read as a result for this specific test set, not as an absolute accuracy measure for the models.

## Analysis by difficulty

### Easy

Gemma passes **2/3** cases, while Qwen passes **1/3**.

Gemma performs well on dataset creation and basic statistical tasks. For cases where a Reliability Score is available, both models can achieve strong reliability. One Gemma failure involves the visualizer being rate-limited, preventing the task from completing.

Qwen's main issue is dataset state consistency. The `sales_data` dataset was not present when expected, causing downstream statistics and visualization tasks that depended on it to fail as well.

### Medium

This is the largest difference in the benchmark.

Gemma passes **4/4** medium cases, including custom columns, correlation, and statistical insight tasks.

Qwen passes **5/10**. Several failures are associated with the expected dataset state not being available — including missing `diem_hai_long` and cases where `sales_data` could not be found — which then affected dependent tasks.

### Hard

Both models pass **5/7 = 71.4%**.

Gemma handles several difficult workflows successfully, including multi-agent pipelines, context memory, and error-handling scenarios. Its failures include adversarial/bait cases involving requests over columns that do not exist.

Qwen also passes 5/7. One context-memory failure has a Reliability Score of **47.5**, while some other successful cases rely on detecting that the expected dataset does not exist.

## Key findings

Across this benchmark:

- **Gemma:** 78.6% overall Task Correctness pass rate.
- **Qwen:** 55.0% overall Task Correctness pass rate.
- The largest gap appears in **Medium**, where Gemma reaches 100% and Qwen 50%.
- The models have the **same Hard pass rate (5/7)**.
- Qwen shows repeated **dataset/state consistency** issues in the tested workflows.
- Gemma has its own failure modes, including a **visualizer rate-limit** failure and some adversarial/bait failures.
- A pass rate such as 78.6% should **not** be interpreted as absolute model accuracy; it describes performance on this benchmark.

## Gemma API vs local Qwen architecture

The current project uses the Gemma API primarily for stability and easier deployment. The previous local Qwen implementation is preserved as:

```text
llm_setup_local_qwen.py.bak
```

The benchmark above provides an empirical comparison of the two backends on the supplied test set, while the following architectural differences remain relevant:

| Aspect | Gemma API | Local Qwen |
| --- | --- | --- |
| GPU requirement | None | Required |
| Kaggle/Colab OOM risk | Low | Higher |
| Deployment | Simple | More setup |
| API key rotation | Yes | N/A |
| Offline inference | No | Yes |

The benchmark does not establish a universal model preference outside this test set.

---

# 🛡️ Design Principles

The project follows several design principles:

### Modularity

The LLM backend is separated from the agent and tool layers.

Changing the underlying model does not require rewriting the entire agent system.

### Reliability

The system does not rely only on the final natural-language answer. It also monitors:

* Language quality
* Output format
* Task completion

### Stateful Analysis

Datasets are maintained through a session-level `DataStore`, allowing multiple natural-language instructions to operate on the same dataset.

### Tool-Based Analysis

The agents rely on deterministic tools for operations such as:

* Loading data
* Computing statistics
* Transforming datasets
* Creating visualizations
* Generating reports

This reduces the risk of the model fabricating numerical results.

---

# 🔧 Architecture Stability

The following core modules remain unchanged across the LLM backend migration:

```text
agents.py
tools/
metrics.py
data_store.py
```

Only the LLM backend is replaced through:

```python
llm_setup.build_gemini_llm()
```

Existing v2 improvements remain active, including:

* Reliability Score
* `resolve_column`
* `create_custom_dataset`
* Protection against fabricated analysis results

This allows the system to evolve its model backend without changing the overall analysis architecture.

---

# 📋 Requirements

Recommended environment:

```text
Python >= 3.10
```

The project is designed primarily for:

* Kaggle Notebooks
* Google Colab
* Standard Python environments with internet access

For Kaggle/Colab, API-based inference avoids depending on the available GPU memory of the current session.

---

# 📌 Notes

### API Keys

Never commit API keys to Git:

```text
❌ GOOGLE_API_KEY_1="AIza..."
❌ Hard-coded keys in notebooks
❌ API keys inside source files
```

Use Kaggle Secrets, Colab User Secrets, or environment variables instead.

### Excel Parser

`excel-parser` is a third-party dependency. Its API may vary between versions.

If you encounter compatibility issues:

```bash
pip show excel-parser
```

and verify the installed version and available conversion methods.

### Public Data

This project is intended to work with datasets that are appropriate for the deployment environment. Avoid uploading private or sensitive datasets to public notebooks or repositories.

---


# 🧪 Testing & Evaluation Design


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

---

# 📜 License

This project is released under the [MIT License](LICENSE).

---

# 🙌 Acknowledgements

This project is built around a modular multi-agent approach and uses **Gemma 4 26B-A4B-IT** as the primary LLM backend.

Special thanks to the open-source AI and data-science community for the tools and libraries that make this project possible.
