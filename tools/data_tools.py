"""
data_tools.py - Nạp dữ liệu vào hệ thống: dataset mẫu, CSV, và Excel (dùng excel-parser).
"""
import numpy as np
import pandas as pd
from langchain_core.tools import tool

from data_store import store
from tools.parsing_utils import parse_kv_string


@tool
def create_sample_dataset(tool_input: str) -> str:
    """Tạo dataset mẫu để test. Action Input dạng: dataset_type=sales, dataset_name=ten_dataset
    dataset_type phải là 1 trong: 'sales', 'customers', 'timeseries', 'survey'.
    dataset_name là tên để lưu dataset đó, dùng cho các tool khác sau này."""
    args = parse_kv_string(tool_input)
    dataset_type = args.get("dataset_type", "")
    dataset_name = args.get("dataset_name", "")
    if not dataset_name:
        return "Lỗi: thiếu dataset_name. Action Input cần dạng: dataset_type=sales, dataset_name=ten_dataset"

    np.random.seed(42)
    n = 200
    if dataset_type == "sales":
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "region": np.random.choice(["North", "South", "East", "West"], n),
                "product": np.random.choice(["A", "B", "C"], n),
                "revenue": np.random.normal(1000, 300, n).round(2),
                "units_sold": np.random.poisson(20, n),
            }
        )
    elif dataset_type == "customers":
        df = pd.DataFrame(
            {
                "customer_id": range(1, n + 1),
                "age": np.random.randint(18, 70, n),
                "spend": np.random.exponential(200, n).round(2),
                "satisfaction": np.random.randint(1, 6, n),
            }
        )
    elif dataset_type == "timeseries":
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "value": np.cumsum(np.random.randn(n)) + 100,
            }
        )
    elif dataset_type == "survey":
        df = pd.DataFrame(
            {
                "respondent_id": range(1, n + 1),
                "score": np.random.randint(1, 11, n),
                "group": np.random.choice(["control", "treatment"], n),
            }
        )
    else:
        return f"Lỗi: dataset_type '{dataset_type}' không hợp lệ. Dùng: sales, customers, timeseries, survey"

    store.add(dataset_name, df)
    store.log(f"Created sample dataset '{dataset_name}' (type={dataset_type})")
    return f"Đã tạo dataset '{dataset_name}' với {len(df)} dòng, cột: {list(df.columns)}"


@tool
def create_custom_dataset(tool_input: str) -> str:
    """Tạo dataset mẫu với TÊN CỘT TÙY CHỌN (dùng khi người dùng yêu cầu cấu trúc cụ thể,
    không khớp với các template có sẵn của create_sample_dataset).
    Action Input dạng: dataset_name=ten, n_rows=200, columns=ten_cot1:kieu1,ten_cot2:kieu2,...
    kieu phải là 1 trong: id (số thứ tự tăng dần), int (số nguyên ngẫu nhiên 1-1000),
    float (số thực ngẫu nhiên), date (ngày ngẫu nhiên trong 2024), category (chọn ngẫu
    nhiên trong vài nhóm mẫu: A/B/C).
    Ví dụ: dataset_name=sales, n_rows=100, columns=ID_san_pham:id,ngay_ban:date,gia:float,so_luong:int"""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    n_rows = int(args.get("n_rows", 200))
    columns_spec = args.get("columns", "")
    if not dataset_name or not columns_spec:
        return "Lỗi: thiếu dataset_name hoặc columns. Action Input cần dạng: dataset_name=..., columns=ten_cot:kieu,..."

    np.random.seed(42)
    data = {}
    for pair in columns_spec.split(","):
        if ":" not in pair:
            return f"Lỗi: '{pair}' không đúng format ten_cot:kieu"
        col_name, col_type = pair.split(":", 1)
        col_name, col_type = col_name.strip(), col_type.strip().lower()
        if col_type == "id":
            data[col_name] = range(1, n_rows + 1)
        elif col_type == "int":
            data[col_name] = np.random.randint(1, 1000, n_rows)
        elif col_type == "float":
            data[col_name] = np.random.uniform(10, 1000, n_rows).round(2)
        elif col_type == "date":
            data[col_name] = pd.date_range("2024-01-01", periods=n_rows, freq="D")
        elif col_type == "category":
            data[col_name] = np.random.choice(["A", "B", "C"], n_rows)
        else:
            return f"Lỗi: kiểu '{col_type}' không hợp lệ. Dùng: id, int, float, date, category"

    df = pd.DataFrame(data)
    store.add(dataset_name, df)
    store.log(f"Created custom dataset '{dataset_name}' (columns={list(df.columns)})")
    return f"Đã tạo dataset '{dataset_name}' với {len(df)} dòng, ĐÚNG cột yêu cầu: {list(df.columns)}"


@tool
def load_csv(tool_input: str) -> str:
    """Load 1 file CSV từ đường dẫn (path) vào hệ thống. Action Input dạng:
    file_path=duong/dan/file.csv, dataset_name=ten_dataset"""
    args = parse_kv_string(tool_input)
    file_path = args.get("file_path", "")
    dataset_name = args.get("dataset_name", "")
    if not file_path or not dataset_name:
        return "Lỗi: thiếu file_path hoặc dataset_name. Action Input cần dạng: file_path=..., dataset_name=..."
    try:
        df = pd.read_csv(file_path)
        store.add(dataset_name, df)
        store.log(f"Loaded CSV '{file_path}' as '{dataset_name}'")
        return f"Đã load '{dataset_name}': {len(df)} dòng, cột: {list(df.columns)}"
    except Exception as e:
        return f"Lỗi load file: {str(e)}"


@tool
def load_excel(tool_input: str) -> str:
    """Load 1 file Excel (.xlsx) vào hệ thống, dùng thư viện excel-parser để giữ lại
    công thức, bảng, và trích dẫn ô nguồn (cell citation) thay vì chỉ đọc giá trị thô.
    Action Input dạng: file_path=duong/dan/file.xlsx, dataset_name=ten_dataset,
    sheet_name=ten_sheet (tùy chọn, mặc định lấy sheet có nhiều dữ liệu nhất)."""
    args = parse_kv_string(tool_input)
    file_path = args.get("file_path", "")
    dataset_name = args.get("dataset_name", "")
    sheet_name = args.get("sheet_name")
    if not file_path or not dataset_name:
        return "Lỗi: thiếu file_path hoặc dataset_name. Action Input cần dạng: file_path=..., dataset_name=..."

    try:
        from excel_parser import parse_workbook
    except ImportError:
        return (
            "Lỗi: chưa cài thư viện excel-parser. Chạy: pip install excel-parser "
            "(xem https://github.com/knowledgestack/excel-parser)"
        )

    try:
        result = parse_workbook(path=file_path)
    except Exception as e:
        return f"Lỗi parse Excel: {str(e)}"

    # excel-parser trả về workbook graph (sheet, bảng, công thức, chart...) chứ không
    # phải 1 DataFrame phẳng như pandas. Ta lấy bảng lớn nhất (hoặc bảng theo sheet_name
    # nếu người dùng chỉ định) rồi convert sang DataFrame để dùng chung với các tool
    # thống kê/biểu đồ hiện có (describe_dataset, correlation_analysis, v.v.).
    tables = getattr(result, "tables", None) or []
    if sheet_name:
        tables = [t for t in tables if getattr(t, "sheet_name", None) == sheet_name] or tables
    if not tables:
        return (
            f"Đã parse '{file_path}' nhưng không tìm thấy bảng dữ liệu dạng hàng/cột nào. "
            f"File có {len(getattr(result, 'chunks', []))} chunk nội dung, có thể là dạng "
            "báo cáo/tự do thay vì bảng."
        )
    best_table = max(tables, key=lambda t: getattr(t, "row_count", 0) * getattr(t, "col_count", 0))
    # API cụ thể để lấy bảng ra pandas có thể khác nhau giữa các version của excel-parser
    # (to_dataframe() / to_records() / rows thô) - thử lần lượt, cái nào có thì dùng.
    if hasattr(best_table, "to_dataframe"):
        df = best_table.to_dataframe()
    elif hasattr(best_table, "to_records"):
        df = pd.DataFrame(best_table.to_records())
    else:
        df = pd.DataFrame(getattr(best_table, "rows", []))

    store.add(dataset_name, df)
    citation = getattr(best_table, "source_uri", f"{file_path}#{sheet_name or 'unknown'}")
    store.log(f"Loaded Excel table '{citation}' as '{dataset_name}'")
    return (
        f"Đã load '{dataset_name}' từ Excel: {len(df)} dòng, cột: {list(df.columns)}. "
        f"Nguồn (citation): {citation}"
    )


@tool
def list_available_datasets() -> str:
    """Liệt kê tất cả dataset đang có trong hệ thống."""
    if not store.datasets:
        return "Chưa có dataset nào được load."
    return "\n".join(
        f"- {name}: {len(df)} dòng, cột {list(df.columns)}" for name, df in store.datasets.items()
    )
