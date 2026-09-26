"""
config.py - Cấu hình trung tâm.

Đổi model / đường dẫn / tham số ở ĐÂY, không cần sửa rải rác trong các module khác.
"""
import os


def _load_google_api_keys() -> list:
    """Đọc danh sách API key (Google AI Studio) từ biến môi trường GOOGLE_API_KEYS,
    cách nhau bởi dấu phẩy. KHÔNG hardcode key trực tiếp vào code/notebook - xem
    README.md phần 'Cấu hình API key an toàn trên Kaggle' để set biến này qua Kaggle
    Secrets trước khi chạy."""
    raw = os.environ.get("GOOGLE_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


GOOGLE_API_KEYS = _load_google_api_keys()

# Đổi từ chạy local Qwen2.5-3B sang gọi API Gemma 4 26B-A4B-IT (Google AI Studio) vì:
# - Instruction-tuning mới hơn, bám ngôn ngữ Việt tốt hơn hẳn, không cần vá CJK-blocking
# - Native function calling, MoE 26B tổng/~4B active -> suy luận nhiều bước tốt hơn 3B
# - Không phụ thuộc GPU của Kaggle/Colab -> ổn định hơn khi session bị giới hạn tài nguyên
GEMMA_MODEL_NAME = "gemma-4-26b-a4b-it"


# ĐÃ ĐỔI: base model -> Instruct model.
# Qwen/Qwen2.5-3B (base) chỉ được train để "đoán chữ kế tiếp", không được train để
# bám sát ngôn ngữ / yêu cầu của người dùng -> dễ "trôi" sang tiếng Trung giữa chừng.
# Qwen/Qwen2.5-3B-Instruct được fine-tune theo hội thoại (chat template), bám ngôn ngữ
# đầu vào tốt hơn NHIỀU. Đây là fix quan trọng nhất cho vấn đề "nhảy qua tiếng Trung".
# (Giữ lại constant này để tương thích ngược nếu bạn muốn quay lại chạy local Qwen -
# xem llm_setup_local_qwen.py.bak nếu cần.)
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.15
MAX_ITERATIONS_SPECIALIST = 4
MAX_ITERATIONS_ORCHESTRATOR = 8

def _detect_output_dir() -> str:
    """Tự chọn thư mục lưu ảnh (biểu đồ) tuỳ theo môi trường đang chạy, để không bị
    hardcode cứng /kaggle/working (chỉ tồn tại trên Kaggle, KHÔNG có trên Colab/local
    -> savefig() báo lỗi 'không tìm thấy đường dẫn' như bạn đang gặp).

    Thứ tự ưu tiên:
    1. Biến môi trường CHART_OUTPUT_DIR (nếu bạn tự set - ưu tiên cao nhất, override
       hết mọi auto-detect bên dưới).
    2. /kaggle/working nếu đang chạy trên Kaggle (thư mục này Kaggle tự tạo sẵn).
    3. /content nếu đang chạy trên Colab (thư mục gốc Colab tự tạo sẵn).
    4. Ngược lại (local, VSCode, terminal...): dùng ./outputs cạnh nơi chạy code, và
       TỰ TẠO thư mục này nếu chưa tồn tại (local thường không có sẵn thư mục nào).
    """
    env_override = os.environ.get("CHART_OUTPUT_DIR")
    if env_override:
        chosen = env_override
    elif os.path.isdir("/kaggle/working"):
        chosen = "/kaggle/working"
    elif os.path.isdir("/content"):
        chosen = "/content"
    else:
        chosen = os.path.join(os.getcwd(), "outputs")

    os.makedirs(chosen, exist_ok=True)  # phòng trường hợp local chưa có sẵn thư mục
    return chosen


CHART_OUTPUT_PATH = os.path.join(_detect_output_dir(), "last_chart.png")

# Ngưỡng hiển thị đèn giao thông cho Reliability Score (xem metrics.py)
SCORE_GOOD_THRESHOLD = 85
SCORE_OK_THRESHOLD = 60

# Trọng số 4 thành phần của Reliability Score (phải cộng lại = 1.0)
# ĐÃ THÊM: WEIGHT_TOOL_SUCCESS - trước đây thiếu thành phần này nên model có thể tự bịa
# "đã vẽ biểu đồ thành công" dù tool thực sự trả về lỗi, mà điểm vẫn ra 100/100.
WEIGHT_LANGUAGE_PURITY = 0.30
WEIGHT_FORMAT_SUCCESS = 0.15
WEIGHT_TOOL_SUCCESS = 0.30
WEIGHT_TASK_COMPLETED = 0.25
