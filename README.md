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

The score combines three measurable components:

### 1. Vietnamese Purity — 50%

Measures the percentage of generated characters that are not foreign-language characters.

This helps detect language-quality problems that can occur when models are primarily trained on other languages or when using smaller models.

### 2. Format Compliance — 30%

Measures how many agent steps follow the expected syntax and output format.

### 3. Question Completion — 20%

Checks whether the agent successfully completes the requested task rather than stopping or failing midway.

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

# 🧠 Why Gemma API Instead of Local Qwen?

The project previously included a local Qwen-based implementation.

The current version uses the Gemma API primarily for **stability and easier deployment**.

### Advantages

| Aspect                | Gemma API | Local Qwen                  |
| --------------------- | --------- | --------------------------- |
| GPU requirement       | None      | Required                    |
| Kaggle/Colab OOM risk | Low       | Higher                      |
| Deployment            | Simple    | More setup                  |
| Vietnamese generation | Strong    | More tuning may be required |
| API key rotation      | ✅         | N/A                         |
| Offline inference     | ❌         | ✅                           |

The previous local implementation is preserved as:

```text
llm_setup_local_qwen.py.bak
```

It can be used as a reference for offline/local inference experiments.

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

# 📜 License

This project is released under the [MIT License](LICENSE).

---

# 🙌 Acknowledgements

This project is built around a modular multi-agent approach and uses **Gemma 4 26B-A4B-IT** as the primary LLM backend.

Special thanks to the open-source AI and data-science community for the tools and libraries that make this project possible.
