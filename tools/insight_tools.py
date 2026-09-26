"""
insight_tools.py - Sinh insight (nhận xét) TỪ SỐ LIỆU THẬT, không để LLM tự đoán/bịa.

Ý tưởng: mọi con số trong insight (mean, skew, % outlier, top category, xu hướng...)
đều do Python tính trực tiếp trên DataFrame rồi trả về dạng text cho model đọc lại và
diễn giải bằng tiếng Việt tự nhiên - model KHÔNG được tự nghĩ ra số, chỉ được tường
thuật lại số đã có trong Observation. Cách này ăn khớp với triết lý chống hallucination
đã có sẵn trong codebase (xem parsing_utils.resolve_column, metrics.BLUFF_MARKERS).
"""
import numpy as np
import pandas as pd
from langchain_core.tools import tool

from data_store import store
from tools.parsing_utils import parse_kv_string, resolve_column, ColumnNotFoundError

# Ngưỡng |skewness| để coi là "lệch đáng kể" khi mô tả phân bố bằng lời.
SKEW_THRESHOLD = 1.0


def _find_date_column(df: pd.DataFrame):
    """Tìm 1 cột kiểu ngày tháng trong df (ưu tiên dtype datetime, sau đó thử parse tên
    cột có chứa 'date'/'ngay') - dùng để tính xu hướng theo thời gian cho cột số."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        lname = col.lower()
        if "date" in lname or "ngay" in lname or "ngày" in lname:
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                continue
    return None


def _numeric_insights(df: pd.DataFrame, column: str) -> list:
    col = df[column].dropna()
    lines = []
    if col.empty:
        return [f"Cột '{column}' không có giá trị nào (toàn bộ là missing)."]

    mean, median, std = col.mean(), col.median(), col.std()
    lines.append(f"Trung bình={mean:.2f}, Trung vị={median:.2f}, Độ lệch chuẩn={std:.2f}, "
                 f"Min={col.min():.2f}, Max={col.max():.2f}.")

    skew = col.skew()
    if abs(skew) >= SKEW_THRESHOLD:
        direction = "lệch phải (đuôi dài về phía giá trị lớn)" if skew > 0 else "lệch trái (đuôi dài về phía giá trị nhỏ)"
        lines.append(f"Phân bố {direction} (skewness={skew:.2f}) - trung bình có thể bị kéo lệch bởi vài giá trị cực trị.")
    else:
        lines.append(f"Phân bố tương đối đối xứng (skewness={skew:.2f}).")

    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = col[(col < lower) | (col > upper)]
    pct = 100.0 * len(outliers) / len(col)
    if len(outliers) > 0:
        lines.append(f"Có {len(outliers)} outlier ({pct:.1f}% số dòng), nằm ngoài khoảng [{lower:.2f}, {upper:.2f}] (method=IQR).")
    else:
        lines.append("Không phát hiện outlier đáng kể (method=IQR).")

    date_col = _find_date_column(df)
    if date_col and date_col != column:
        try:
            tmp = df[[date_col, column]].dropna()
            tmp = tmp.sort_values(date_col)
            x = np.arange(len(tmp))
            if len(tmp) >= 3:
                slope = np.polyfit(x, tmp[column].values, 1)[0]
                rel_slope = slope * len(tmp) / (abs(mean) if mean != 0 else 1)
                if abs(rel_slope) < 0.05:
                    lines.append(f"Theo '{date_col}': không có xu hướng tăng/giảm rõ rệt.")
                else:
                    trend = "TĂNG" if slope > 0 else "GIẢM"
                    lines.append(f"Theo '{date_col}': có xu hướng {trend} dần theo thời gian (độ dốc={slope:.4f}/đơn vị thời gian).")
        except Exception:
            pass  # xu hướng theo thời gian là phần "thêm", lỗi ở đây không nên chặn insight chính

    return lines


def _categorical_insights(df: pd.DataFrame, column: str) -> list:
    col = df[column].dropna()
    lines = []
    if col.empty:
        return [f"Cột '{column}' không có giá trị nào (toàn bộ là missing)."]

    n_unique = col.nunique()
    top = col.value_counts().head(3)
    top_str = ", ".join(f"'{idx}' ({cnt} dòng, {100*cnt/len(col):.1f}%)" for idx, cnt in top.items())
    lines.append(f"Có {n_unique} giá trị khác nhau. Top phổ biến nhất: {top_str}.")

    top_pct = 100.0 * top.iloc[0] / len(col)
    if top_pct >= 50:
        lines.append(f"Dữ liệu bị LỆCH mạnh về nhóm '{top.index[0]}' (chiếm {top_pct:.1f}%) - cẩn thận khi so sánh giữa các nhóm.")

    return lines


def _datetime_insights(df: pd.DataFrame, column: str) -> list:
    col = pd.to_datetime(df[column], errors="coerce").dropna()
    if col.empty:
        return [f"Cột '{column}' không parse được thành ngày tháng hợp lệ."]
    span_days = (col.max() - col.min()).days
    return [f"Khoảng thời gian: từ {col.min().date()} đến {col.max().date()} ({span_days} ngày)."]


@tool
def generate_insights(tool_input: str) -> str:
    """Sinh nhận xét (insight) THẬT dựa trên số liệu tính toán cho 1 cột cụ thể - dùng
    khi người dùng hỏi 'insight là gì', 'nhận xét cột X', hoặc sau khi đã vẽ biểu đồ
    muốn giải thích ý nghĩa. Action Input dạng: dataset_name=ten, column=ten_cot"""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    column = args.get("column", "")
    if not dataset_name or not column:
        return "Lỗi: thiếu dataset_name hoặc column. Action Input cần dạng: dataset_name=..., column=..."
    df = store.get(dataset_name)
    try:
        column = resolve_column(df, column)
    except ColumnNotFoundError as e:
        return f"Lỗi: {str(e)}"

    if pd.api.types.is_datetime64_any_dtype(df[column]):
        lines = _datetime_insights(df, column)
    elif pd.api.types.is_numeric_dtype(df[column]):
        lines = _numeric_insights(df, column)
    else:
        lines = _categorical_insights(df, column)

    header = f"=== INSIGHT cho cột '{column}' (dataset '{dataset_name}') ==="
    return header + "\n- " + "\n- ".join(lines)
