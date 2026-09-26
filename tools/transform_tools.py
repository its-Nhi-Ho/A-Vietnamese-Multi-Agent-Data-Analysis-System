"""transform_tools.py - Filter, aggregate, thêm cột tính toán."""
from langchain_core.tools import tool

from data_store import store
from tools.parsing_utils import parse_kv_string


@tool
def filter_data(tool_input: str) -> str:
    """Filter dữ liệu theo điều kiện pandas query. Action Input dạng:
    dataset_name=ten, condition=age > 30, new_dataset_name=ten_moi"""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    condition = args.get("condition", "")
    new_dataset_name = args.get("new_dataset_name", "")
    df = store.get(dataset_name)
    try:
        filtered = df.query(condition)
        store.add(new_dataset_name, filtered)
        return f"Đã filter '{dataset_name}' -> '{new_dataset_name}': {len(filtered)}/{len(df)} dòng còn lại"
    except Exception as e:
        return f"Lỗi filter: {str(e)}"


@tool
def aggregate_data(tool_input: str) -> str:
    """Group & aggregate. Action Input dạng: dataset_name=ten, group_by=ten_cot,
    aggregations=col:func,col2:func2 (vd revenue:sum,profit:mean), new_dataset_name=ten_moi"""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    group_by = args.get("group_by", "")
    aggregations = args.get("aggregations", "")
    new_dataset_name = args.get("new_dataset_name", "")
    df = store.get(dataset_name)
    try:
        agg_dict = {}
        for pair in aggregations.split(","):
            col, func = pair.split(":")
            agg_dict[col.strip()] = func.strip()
        result = df.groupby(group_by).agg(agg_dict).reset_index()
        store.add(new_dataset_name, result)
        return f"Đã aggregate -> '{new_dataset_name}':\n{result.to_string()}"
    except Exception as e:
        return f"Lỗi aggregate: {str(e)}"


@tool
def add_calculated_column(tool_input: str) -> str:
    """Thêm cột mới tính từ expression pandas. Action Input dạng:
    dataset_name=ten, new_column=ten_cot_moi, expression=revenue * 0.1"""
    args = parse_kv_string(tool_input)
    dataset_name = args.get("dataset_name", "")
    new_column = args.get("new_column", "")
    expression = args.get("expression", "")
    df = store.get(dataset_name)
    try:
        df[new_column] = df.eval(expression)
        store.add(dataset_name, df)
        return f"Đã thêm cột '{new_column}' vào '{dataset_name}'"
    except Exception as e:
        return f"Lỗi tính cột: {str(e)}"
