"""stats_tools.py - Thống kê mô tả, tương quan, kiểm định giả thuyết, outlier."""
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy import stats as scipy_stats

from data_store import store
from tools.parsing_utils import parse_kv_string, resolve_column, ColumnNotFoundError


@tool
def describe_dataset(tool_input: str) -> str:
    """Thống kê mô tả đầy đủ (describe) cho 1 dataset.
    Action Input dạng: dataset_name=ten_dataset"""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", tool_input.strip())
    df = store.get(dataset_name)
    return df.describe(include="all").to_string()


@tool
def correlation_analysis(tool_input: str) -> str:
    """Tính ma trận tương quan giữa các cột số. Action Input dạng:
    dataset_name=ten_dataset, method=pearson (hoặc spearman, mặc định pearson)."""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", tool_input.strip())
    method = args.get("method", "pearson")
    df = store.get(dataset_name)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return "Không đủ cột số để tính tương quan."
    corr = numeric_df.corr(method=method)
    return corr.to_string()


@tool
def hypothesis_test(tool_input: str) -> str:
    """Chạy kiểm định thống kê. Action Input dạng:
    dataset_name=ten, test_type=normality|ttest|anova|chi2, column=ten_cot, group_column=ten_cot_nhom (cần cho ttest/anova/chi2)."""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    test_type = args.get("test_type", "")
    column = args.get("column", "")
    group_column = args.get("group_column")
    df = store.get(dataset_name)
    try:
        column = resolve_column(df, column)
        if group_column:
            group_column = resolve_column(df, group_column)
    except ColumnNotFoundError as e:
        return f"Lỗi: {str(e)}"
    try:
        if test_type == "normality":
            stat, p = scipy_stats.shapiro(df[column].dropna())
            return f"Shapiro-Wilk: stat={stat:.4f}, p={p:.4f} -> {'Có vẻ chuẩn' if p > 0.05 else 'Không chuẩn'} (alpha=0.05)"
        elif test_type == "ttest":
            groups = df[group_column].unique()
            if len(groups) != 2:
                return f"T-test cần đúng 2 nhóm, '{group_column}' có {len(groups)} nhóm."
            g1 = df[df[group_column] == groups[0]][column].dropna()
            g2 = df[df[group_column] == groups[1]][column].dropna()
            stat, p = scipy_stats.ttest_ind(g1, g2)
            return f"T-test: stat={stat:.4f}, p={p:.4f} -> {'Khác biệt có ý nghĩa' if p < 0.05 else 'Không khác biệt có ý nghĩa'}"
        elif test_type == "anova":
            groups = [g[column].dropna() for _, g in df.groupby(group_column)]
            stat, p = scipy_stats.f_oneway(*groups)
            return f"ANOVA: stat={stat:.4f}, p={p:.4f} -> {'Khác biệt có ý nghĩa' if p < 0.05 else 'Không khác biệt có ý nghĩa'}"
        elif test_type == "chi2":
            contingency = pd.crosstab(df[column], df[group_column])
            stat, p, dof, _ = scipy_stats.chi2_contingency(contingency)
            return f"Chi-square: stat={stat:.4f}, p={p:.4f}, dof={dof} -> {'Có liên hệ' if p < 0.05 else 'Không có liên hệ'}"
        else:
            return f"test_type '{test_type}' không hợp lệ."
    except Exception as e:
        return f"Lỗi khi test: {str(e)}"


@tool
def outlier_detection(tool_input: str) -> str:
    """Tìm outlier trong 1 cột số. Action Input dạng:
    dataset_name=ten, column=ten_cot, method=iqr (hoặc zscore, mặc định iqr)."""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    column = args.get("column", "")
    method = args.get("method", "iqr")
    df = store.get(dataset_name)
    try:
        column = resolve_column(df, column)
    except ColumnNotFoundError as e:
        return f"Lỗi: {str(e)}"
    col = df[column].dropna()
    if method == "iqr":
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = col[(col < lower) | (col > upper)]
    else:
        z = np.abs(scipy_stats.zscore(col))
        outliers = col[z > 3]
    return f"Tìm thấy {len(outliers)} outlier trong '{column}' (method={method}): {outliers.tolist()[:20]}"
