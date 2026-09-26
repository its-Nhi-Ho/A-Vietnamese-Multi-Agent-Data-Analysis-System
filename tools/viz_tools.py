"""viz_tools.py - Vẽ biểu đồ và báo cáo phân bố."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from scipy import stats as scipy_stats

from config import CHART_OUTPUT_PATH
from data_store import store
from tools.parsing_utils import parse_kv_string, resolve_column, ColumnNotFoundError
from tools.insight_tools import _numeric_insights, _categorical_insights


def _resolve_or_none(df, name):
    return resolve_column(df, name) if name else name


@tool
def create_visualization(tool_input: str) -> str:
    """Tạo biểu đồ. Action Input dạng: dataset_name=ten, chart_type=histogram|scatter|bar|line|box|heatmap|pie,
    x=ten_cot, y=ten_cot, hue=ten_cot (tùy chọn, để phân nhóm màu)."""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    chart_type = args.get("chart_type", "")
    x = args.get("x") or args.get("column")
    y = args.get("y")
    hue = args.get("hue")
    if not chart_type:
        return "Lỗi: thiếu chart_type. Action Input cần dạng: dataset_name=..., chart_type=histogram|scatter|bar|line|box|heatmap|pie, x=..."
    df = store.get(dataset_name)

    # QUAN TRỌNG: kiểm tra & khớp tên cột (không phân biệt hoa/thường) TRƯỚC khi tạo
    # figure. Trước đây model hay gõ sai case (vd 'duration' thay vì cột thật 'Duration')
    # -> KeyError xảy ra SAU khi đã tạo fig -> tool trả lỗi đúng nhưng model phớt lờ,
    # tự bịa "đã vẽ thành công", để lại 1 figure rỗng gây hiểu lầm.
    try:
        if x and chart_type != "heatmap":
            x = resolve_column(df, x)
        if y:
            y = resolve_column(df, y)
        if hue:
            hue = resolve_column(df, hue)
    except ColumnNotFoundError as e:
        return f"Lỗi tạo biểu đồ: {str(e)}"

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if chart_type == "histogram":
            sns.histplot(df[x], ax=ax, kde=True)
        elif chart_type == "scatter":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
        elif chart_type == "bar":
            df.groupby(x)[y].mean().plot(kind="bar", ax=ax)
        elif chart_type == "line":
            df.plot(x=x, y=y, ax=ax)
        elif chart_type == "box":
            sns.boxplot(data=df, x=x, y=y, ax=ax)
        elif chart_type == "heatmap":
            sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, ax=ax)
        elif chart_type == "pie":
            df[x].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%")
        else:
            return f"chart_type '{chart_type}' không hợp lệ."
        ax.set_title(f"{chart_type} - {dataset_name}")
        fig.tight_layout()
        fig.savefig(CHART_OUTPUT_PATH, dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # Tự tính insight THẬT cho cột chính của biểu đồ (không để model tự bịa nhận
        # xét) - ưu tiên cột y nếu có (vd bar/line/scatter thường insight nằm ở y),
        # ngược lại dùng cột x (vd histogram/pie chỉ có 1 cột).
        insight_column = y or x
        insight_lines = []
        if insight_column and chart_type != "heatmap":
            try:
                if pd.api.types.is_numeric_dtype(df[insight_column]):
                    insight_lines = _numeric_insights(df, insight_column)
                else:
                    insight_lines = _categorical_insights(df, insight_column)
            except Exception:
                pass  # insight là phần "thêm" - lỗi ở đây không được làm hỏng kết quả vẽ biểu đồ chính

        result = f"Đã tạo biểu đồ {chart_type}, lưu tại {CHART_OUTPUT_PATH}"
        if insight_lines:
            result += f"\n\nInsight cho cột '{insight_column}':\n- " + "\n- ".join(insight_lines)
        return result
    except Exception as e:
        plt.close(fig)
        return f"Lỗi tạo biểu đồ: {str(e)}"


@tool
def create_distribution_report(tool_input: str) -> str:
    """Tạo report phân tích phân bố 4-plot (histogram, box, QQ-plot, violin) cho 1 cột.
    Action Input dạng: dataset_name=ten, column=ten_cot"""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    column = args.get("column", "")
    df = store.get(dataset_name)
    try:
        column = resolve_column(df, column)
    except ColumnNotFoundError as e:
        return f"Lỗi tạo report: {str(e)}"
    col = df[column].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sns.histplot(col, kde=True, ax=axes[0, 0]); axes[0, 0].set_title("Histogram")
    sns.boxplot(x=col, ax=axes[0, 1]); axes[0, 1].set_title("Boxplot")
    scipy_stats.probplot(col, plot=axes[1, 0]); axes[1, 0].set_title("Q-Q Plot")
    sns.violinplot(x=col, ax=axes[1, 1]); axes[1, 1].set_title("Violin")
    fig.tight_layout()
    fig.savefig(CHART_OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return f"Đã tạo distribution report cho '{column}'. Mean={col.mean():.2f}, Std={col.std():.2f}, Skew={col.skew():.2f}"
