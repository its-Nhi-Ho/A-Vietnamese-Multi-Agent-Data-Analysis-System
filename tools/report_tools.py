"""report_tools.py - Báo cáo tổng hợp và lịch sử thao tác."""
from langchain_core.tools import tool

from data_store import store
from tools.parsing_utils import parse_kv_string


@tool
def generate_summary_report(tool_input: str) -> str:
    """Tạo báo cáo tổng hợp đầy đủ cho 1 dataset: shape, dtypes, missing values, describe.
    Action Input dạng: dataset_name=ten_dataset"""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", tool_input.strip())
    df = store.get(dataset_name)
    report = [
        f"=== REPORT: {dataset_name} ===",
        f"Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột",
        f"Dtypes:\n{df.dtypes.to_string()}",
        f"Missing values:\n{df.isnull().sum().to_string()}",
        f"Describe:\n{df.describe().to_string()}",
    ]
    return "\n\n".join(report)


@tool
def get_analysis_history() -> str:
    """Xem lịch sử các hành động đã thực hiện trong session."""
    if not store.history:
        return "Chưa có lịch sử nào."
    return "\n".join(f"{i+1}. {h}" for i, h in enumerate(store.history))
