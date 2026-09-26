"""
test_tools_unit.py - Tầng 1 (unit, deterministic) cho tools/. KHÔNG cần LLM/API key -
gọi thẳng hàm Python đứng sau @tool (`.func(...)`). Chạy trong CI mỗi commit, độc lập
hoàn toàn với chất lượng model.

Mỗi tool ở đây là 1 hàm THUẦN, xác định trước cả input lẫn output đúng -> có thể chấm
đúng/sai tuyệt đối, không cần Reliability Score hay rubric người đọc. Đây là lý do các
tool này tồn tại (insight_tools.py nói rõ: "model KHÔNG được tự nghĩ ra số"), nên bản
thân chúng CŨNG phải được test tuyệt đối, không phải chỉ dựa vào eval agent-level.

Chạy: pytest eval/test_tools_unit.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from data_store import DataStore
from tools.data_tools import create_sample_dataset, create_custom_dataset
from tools.stats_tools import describe_dataset, correlation_analysis, outlier_detection, hypothesis_test
from tools.transform_tools import filter_data, aggregate_data, add_calculated_column
from tools.insight_tools import generate_insights
from tools.report_tools import generate_summary_report, get_analysis_history
from tools.parsing_utils import parse_kv_string, resolve_column, ColumnNotFoundError


@pytest.fixture(autouse=True)
def clean_store():
    """DataStore là singleton toàn cục -> reset trước MỖI test để test không rò rỉ state
    sang nhau (nếu không, thứ tự chạy test sẽ ảnh hưởng tới kết quả - rất khó debug)."""
    store = DataStore()
    store.datasets.clear()
    store.history.clear()
    yield store


# ---------------------------------------------------------------------------
# parsing_utils - nền tảng cho MỌI tool khác, lỗi ở đây sẽ lan ra toàn bộ hệ thống
# ---------------------------------------------------------------------------

def test_parse_kv_string_basic():
    assert parse_kv_string("a=1, b=2") == {"a": "1", "b": "2"}


def test_parse_kv_string_value_contains_comparison_operator():
    # Case quan trọng nhất theo đúng comment trong code: condition="age >= 30" chứa
    # dấu phẩy/nhưng KHÔNG chứa dấu phẩy ở đây - test giá trị có chứa toán tử so sánh
    out = parse_kv_string("dataset_name=x, condition=age >= 30, new_dataset_name=y")
    assert out == {"dataset_name": "x", "condition": "age >= 30", "new_dataset_name": "y"}


def test_parse_kv_string_empty_input():
    assert parse_kv_string("") == {}
    assert parse_kv_string(None) == {}


def test_resolve_column_exact_match():
    df = pd.DataFrame({"Revenue": [1, 2]})
    assert resolve_column(df, "Revenue") == "Revenue"


def test_resolve_column_case_insensitive():
    df = pd.DataFrame({"Revenue": [1, 2]})
    assert resolve_column(df, "revenue") == "Revenue"


def test_resolve_column_ignores_underscore_and_space():
    df = pd.DataFrame({"units_sold": [1, 2]})
    assert resolve_column(df, "units sold") == "units_sold"


def test_resolve_column_not_found_lists_real_columns():
    df = pd.DataFrame({"revenue": [1], "region": ["A"]})
    with pytest.raises(ColumnNotFoundError) as exc_info:
        resolve_column(df, "loi_nhuan")
    assert "revenue" in str(exc_info.value) and "region" in str(exc_info.value)


# ---------------------------------------------------------------------------
# data_tools - schema thật của dataset mẫu (dùng để đồng bộ với eval_harness/test_cases)
# ---------------------------------------------------------------------------

def test_create_sample_dataset_sales_schema():
    create_sample_dataset.func("dataset_type=sales, dataset_name=sales_data")
    store = DataStore()
    df = store.get("sales_data")
    # CHỐT LẠI schema thật: cột số lượng tên là 'units_sold', KHÔNG PHẢI 'quantity'.
    assert set(df.columns) == {"date", "region", "product", "revenue", "units_sold"}
    assert len(df) == 200


def test_create_sample_dataset_missing_name_returns_error_string():
    out = create_sample_dataset.func("dataset_type=sales")
    assert out.startswith("Lỗi")


def test_create_sample_dataset_invalid_type_returns_error_string():
    out = create_sample_dataset.func("dataset_type=invalid_xyz, dataset_name=x")
    assert out.startswith("Lỗi")


def test_create_custom_dataset_respects_requested_columns():
    create_custom_dataset.func(
        "dataset_name=khao_sat, n_rows=15, columns=ten_khach_hang:category,diem:int"
    )
    df = DataStore().get("khao_sat")
    assert list(df.columns) == ["ten_khach_hang", "diem"]
    assert len(df) == 15


def test_create_custom_dataset_invalid_type_returns_error_string():
    out = create_custom_dataset.func("dataset_name=x, columns=col1:kieu_khong_ton_tai")
    assert out.startswith("Lỗi")


# ---------------------------------------------------------------------------
# stats_tools / transform_tools - so khớp SỐ THẬT với oracle pandas độc lập
# ---------------------------------------------------------------------------

def test_correlation_analysis_matches_pandas_oracle():
    create_sample_dataset.func("dataset_type=sales, dataset_name=sales_data")
    df = DataStore().get("sales_data")
    expected = df[["revenue", "units_sold"]].corr().loc["revenue", "units_sold"]

    output = correlation_analysis.func("dataset_name=sales_data")
    # Tool trả về bảng dạng text (DataFrame.to_string()), không phải JSON có cấu trúc.
    # Test này CHỐT LẠI limitation đó: parse lại số từ text để so oracle là giòn (brittle)
    # -> khuyến nghị (xem README) đổi tool trả kèm 1 dict/JSON số liệu, không chỉ text.
    assert "revenue" in output and "units_sold" in output
    parsed = pd.read_csv(pd.io.common.StringIO(output), sep=r"\s+")
    actual = parsed.loc["revenue", "units_sold"]
    assert abs(actual - expected) < 1e-2


def test_outlier_detection_iqr_matches_oracle():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})  # 100 là outlier rõ ràng theo IQR
    DataStore().add("d", df)
    out = outlier_detection.func("dataset_name=d, column=x, method=iqr")
    assert "1 outlier" in out
    assert "100" in out


def test_outlier_detection_wrong_column_returns_error_not_crash():
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    DataStore().add("d", df)
    out = outlier_detection.func("dataset_name=d, column=cot_khong_ton_tai")
    assert out.startswith("Lỗi")


def test_filter_data_matches_oracle():
    df = pd.DataFrame({"revenue": [100, 2000, 3000, 50]})
    DataStore().add("d", df)
    filter_data.func("dataset_name=d, condition=revenue > 1000, new_dataset_name=d_high")
    result = DataStore().get("d_high")
    expected_rows = (df["revenue"] > 1000).sum()
    assert len(result) == expected_rows == 2


def test_filter_data_invalid_condition_returns_error_not_crash():
    df = pd.DataFrame({"revenue": [1, 2]})
    DataStore().add("d", df)
    out = filter_data.func("dataset_name=d, condition=cot_khong_ton_tai > 5, new_dataset_name=x")
    assert out.startswith("Lỗi")


def test_aggregate_data_matches_oracle():
    df = pd.DataFrame({"region": ["A", "A", "B"], "revenue": [10, 20, 5]})
    DataStore().add("d", df)
    aggregate_data.func("dataset_name=d, group_by=region, aggregations=revenue:sum, new_dataset_name=agg")
    result = DataStore().get("agg").set_index("region")["revenue"]
    expected = df.groupby("region")["revenue"].sum()
    assert result["A"] == expected["A"] == 30
    assert result["B"] == expected["B"] == 5


def test_add_calculated_column_matches_oracle():
    df = pd.DataFrame({"revenue": [100, 200]})
    DataStore().add("d", df)
    add_calculated_column.func("dataset_name=d, new_column=revenue_x2, expression=revenue * 2")
    result = DataStore().get("d")
    assert list(result["revenue_x2"]) == [200, 400]


# ---------------------------------------------------------------------------
# insight_tools - insight phải phản ánh ĐÚNG số liệu Python tính, không lệch
# ---------------------------------------------------------------------------

def test_generate_insights_numeric_reports_correct_mean():
    df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0]})
    DataStore().add("d", df)
    out = generate_insights.func("dataset_name=d, column=revenue")
    assert "Trung bình=200.00" in out


def test_generate_insights_wrong_column_returns_error_not_crash():
    df = pd.DataFrame({"revenue": [1, 2]})
    DataStore().add("d", df)
    out = generate_insights.func("dataset_name=d, column=cot_khong_ton_tai")
    assert out.startswith("Lỗi")


# ---------------------------------------------------------------------------
# report_tools
# ---------------------------------------------------------------------------

def test_get_analysis_history_reflects_real_actions():
    # get_analysis_history không nhận tham số -> gọi .func() không đối số (không phải
    # .func("") như các tool key=value khác).
    store = DataStore()
    create_sample_dataset.func("dataset_type=sales, dataset_name=sales_data")
    out = get_analysis_history.func()
    assert "sales_data" in out
    assert len(store.history) == 1


def test_get_analysis_history_empty_says_so():
    out = get_analysis_history.func()
    assert "Chưa có" in out


# ---------------------------------------------------------------------------
# BUG THẬT ĐÃ PHÁT HIỆN: mọi tool thao tác trên dataset có sẵn gọi store.get() NGOÀI
# khối try/except -> ValueError bị ném thẳng ra ngoài thay vì trả "Lỗi: ..." như quy
# ước toàn hệ thống. Hệ quả: exception bay thẳng lên tới agents.make_analyze_fn(), bị
# bắt bởi khối try/except NGOÀI CÙNG -> record_agent_result()/monitor.build_trace()
# KHÔNG được gọi -> lượt hỏi đó BIẾN MẤT khỏi Reliability Score thay vì được tính là 1
# lượt thất bại. Các test dưới đây CHỦ ĐỘNG document hành vi lỗi hiện tại bằng
# pytest.raises, để: (1) làm rõ đây là bug đã biết, không phải flaky test; (2) khi sửa
# xong (bọc thêm try/except quanh store.get(), hoặc chuyển logic đó vào DataStore.get()
# với 1 lớp adapter trả string), CHỈ CẦN đổi pytest.raises(ValueError) thành
# assert out.startswith("Lỗi") ở dưới, test sẽ tự báo đỏ cho tới khi sửa đúng.
# ---------------------------------------------------------------------------

NONEXISTENT_DATASET_CASES = [
    ("describe_dataset", describe_dataset, "dataset_name=khong_ton_tai"),
    ("correlation_analysis", correlation_analysis, "dataset_name=khong_ton_tai"),
    ("outlier_detection", outlier_detection, "dataset_name=khong_ton_tai, column=x"),
    ("hypothesis_test", hypothesis_test, "dataset_name=khong_ton_tai, test_type=normality, column=x"),
    ("generate_summary_report", generate_summary_report, "dataset_name=khong_ton_tai"),
    ("filter_data", filter_data, "dataset_name=khong_ton_tai, condition=x>1, new_dataset_name=y"),
    ("aggregate_data", aggregate_data, "dataset_name=khong_ton_tai, group_by=x, aggregations=y:sum, new_dataset_name=z"),
    ("add_calculated_column", add_calculated_column, "dataset_name=khong_ton_tai, new_column=y, expression=x*2"),
    ("generate_insights", generate_insights, "dataset_name=khong_ton_tai, column=x"),
]


@pytest.mark.parametrize("name,tool_fn,arg", NONEXISTENT_DATASET_CASES, ids=[c[0] for c in NONEXISTENT_DATASET_CASES])
def test_KNOWN_BUG_tool_crashes_instead_of_returning_error_string(name, tool_fn, arg):
    """🔴 KNOWN BUG (xem comment ở trên). Test này PASS nghĩa là bug VẪN CÒN.
    Nếu test này bắt đầu FAIL (không raise ValueError nữa), đó là tín hiệu tốt: tool đã
    được sửa để trả 'Lỗi: ...' như quy ước -> hãy update test thành
    `assert tool_fn.func(arg).startswith("Lỗi")` và xoá pytest.raises."""
    with pytest.raises(ValueError, match="không tồn tại"):
        tool_fn.func(arg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
