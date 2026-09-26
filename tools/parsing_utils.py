"""
parsing_utils.py

QUAN TRỌNG: prompt ReAct dạy model viết Action Input dạng "key=value, key2=value2"
(1 string). Nhưng @tool với NHIỀU tham số sẽ tạo ra StructuredTool cần 1 dict nhiều
field. Khi LangChain nhận 1 string cho tool nhiều field, nó KHÔNG tự parse "key=value"
-> nó nhồi nguyên string vào field ĐẦU TIÊN, các field còn lại bị thiếu -> ValidationError
"Field required". Nên mọi tool nhiều tham số nhận 1 string duy nhất (tool_input), rồi tự
parse bằng hàm dưới đây - khớp đúng format model đang generate.
"""
import re


class ColumnNotFoundError(Exception):
    pass


def resolve_column(df, name: str) -> str:
    """Tìm tên cột THẬT trong df khớp với `name`, không phân biệt hoa/thường/khoảng trắng.
    Model hay gõ sai case (vd 'duration' thay vì 'Duration') -> đây là nguyên nhân phổ biến
    nhất gây lỗi "vẽ biểu đồ trống"/KeyError. Raise lỗi RÕ RÀNG liệt kê cột thật có sẵn,
    thay vì để KeyError mù mờ lọt ra ngoài."""
    if name in df.columns:
        return name
    target = name.strip().lower().replace(" ", "").replace("_", "")
    for col in df.columns:
        if col.strip().lower().replace(" ", "").replace("_", "") == target:
            return col
    raise ColumnNotFoundError(
        f"Không tìm thấy cột '{name}'. Các cột thực sự có trong dataset: {list(df.columns)}"
    )


def parse_kv_string(s: str) -> dict:
    """Parse 'key1=value1, key2=value2' -> dict. Dùng regex tìm các vị trí 'key=' để
    tách value chính xác (value có thể chứa dấu phẩy hoặc dấu '=', ví dụ condition='age >= 30')."""
    s = (s or "").strip()
    if not s:
        return {}
    matches = list(re.finditer(r"(\w+)\s*=\s*", s))
    if not matches:
        return {}
    out = {}
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        value = s[start:end].rstrip(", ").strip().strip('"').strip("'")
        out[key] = value
    return out
