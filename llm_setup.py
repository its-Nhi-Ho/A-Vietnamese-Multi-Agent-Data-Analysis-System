"""
llm_setup.py - Gọi Gemma 4 26B-A4B-IT qua Google AI Studio (Gemini API), có XOAY VÒNG
nhiều API key khi bị rate limit (429 / RESOURCE_EXHAUSTED), để hệ thống chạy ổn định
liên tục thay vì dừng ngang khi 1 key hết quota.

=== VÌ SAO ĐỔI TỪ QWEN LOCAL SANG GEMMA API ===
1. ỔN ĐỊNH HƠN: không phụ thuộc GPU/RAM của Kaggle/Colab session, không bị crash vì OOM,
   không cần lo session hết giờ giữa chừng khi model đang load.
2. BÁM TIẾNG VIỆT TỐT HƠN: Gemma 4 là bản Instruct đời mới hơn nhiều, không cần vá cứng
   ở tầng token như BlockCJKLogitsProcessor (vốn chỉ khả thi khi chạy model local, có
   quyền truy cập logits - qua API không có quyền này).
3. NATIVE FUNCTION CALLING: Gemma 4 hỗ trợ gọi tool có cấu trúc (không bắt buộc dùng ở
   bản này để giữ tối thiểu thay đổi so với kiến trúc ReAct hiện tại của bạn - nhưng đây
   là hướng nâng cấp tiếp theo nếu muốn giảm hẳn lỗi "Lỗi format").

VIETNAMESE_ONLY_INSTRUCTION vẫn được giữ lại làm lớp phòng thủ nhẹ (system_instruction),
dù Gemma 4 đã bám ngôn ngữ tốt hơn Qwen base rất nhiều.
"""
import re
import time

from google import genai
from google.genai import types
from langchain_core.runnables import RunnableLambda

from config import GOOGLE_API_KEYS, GEMMA_MODEL_NAME, MAX_NEW_TOKENS

CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7a3]"
)

VIETNAMESE_ONLY_INSTRUCTION = (
    "Bạn LUÔN LUÔN trả lời bằng tiếng Việt. Tuyệt đối KHÔNG được dùng tiếng Trung, "
    "tiếng Nhật, tiếng Hàn hay bất kỳ ngôn ngữ nào khác, dù chỉ một từ, trong bất kỳ "
    "phần nào của câu trả lời (kể cả Thought, Action, Final Answer)."
)

# Các cụm từ / mã lỗi thường gặp khi bị rate limit / hết quota trên Gemini API.
# Chỉ rotate key khi khớp 1 trong các cụm này - lỗi khác (vd sai API key, model không
# tồn tại) thì raise ngay, rotate sẽ chỉ lãng phí 5 lần retry vô ích.
RATE_LIMIT_HINTS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota", "RATE_LIMIT_EXCEEDED")


def contains_cjk(text: str) -> bool:
    """Kiểm tra nhanh 1 đoạn text có lẫn ký tự Hán/Nhật/Hàn không - vẫn hữu ích để
    QC nhanh output, dù Gemma hiếm khi bị lỗi này như Qwen base."""
    return bool(CJK_RE.search(text or ""))


class RotatingGeminiClient:
    """Xoay vòng qua N API key khi bị rate limit.

    Chiến lược: GIỮ NGUYÊN key hiện tại cho tới khi nó thật sự lỗi (không đổi key mỗi
    request), để tận dụng hết quota của từng key trước khi bỏ qua - tối đa hoá tổng số
    request có thể xử lý trong 1 khoảng thời gian, thay vì round-robin ngẫu nhiên khiến
    5 key cùng cạn quota gần như đồng thời.
    """

    def __init__(self, api_keys: list, model_name: str = GEMMA_MODEL_NAME):
        if not api_keys:
            raise ValueError(
                "Chưa cấu hình GOOGLE_API_KEYS. Set biến môi trường GOOGLE_API_KEYS "
                "(5 key cách nhau bởi dấu phẩy) TRƯỚC khi setup() - xem README.md."
            )
        self.api_keys = api_keys
        self.model_name = model_name
        self._key_index = 0
        self._client = genai.Client(api_key=self.api_keys[self._key_index])
        print(f"✅ RotatingGeminiClient sẵn sàng với {len(api_keys)} key, model={model_name}")

    def _rotate_key(self):
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        self._client = genai.Client(api_key=self.api_keys[self._key_index])
        print(f"🔄 Rate limit - chuyển sang key #{self._key_index + 1}/{len(self.api_keys)}")

    def generate(self, prompt: str) -> str:
        last_error = None
        for _attempt in range(len(self.api_keys)):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=VIETNAMESE_ONLY_INSTRUCTION,
                        max_output_tokens=MAX_NEW_TOKENS,
                        temperature=0.0,
                    ),
                )
                return response.text or ""
            except Exception as e:
                last_error = e
                err_str = str(e)
                if any(hint in err_str for hint in RATE_LIMIT_HINTS):
                    self._rotate_key()
                    time.sleep(1)  # nghỉ ngắn, tránh spam ngay lập tức key mới
                    continue
                raise  # lỗi không phải rate limit (vd sai key, model sai tên) -> raise ngay
        raise RuntimeError(
            f"Đã thử hết {len(self.api_keys)} key, TẤT CẢ đều bị rate limit. "
            f"Lỗi cuối cùng: {last_error}"
        )


def build_gemini_llm(api_keys: list = None, model_name: str = GEMMA_MODEL_NAME):
    """Trả về 1 Runnable tương thích LangChain (dùng được với `|`, `.invoke()`) để cắm
    thẳng vào build_react_llm() trong agents.py mà không cần sửa gì thêm ở đó."""
    client = RotatingGeminiClient(api_keys or GOOGLE_API_KEYS, model_name)

    def _call(prompt, **kwargs) -> str:
        # create_react_agent truyền vào 1 PromptValue (có .to_string()) hoặc string thô.
        # QUAN TRỌNG: create_react_agent tự động .bind(stop=[...]) lên llm trước khi
        # invoke -> RunnableLambda sẽ gọi _call(prompt, stop=[...]), nên hàm PHẢI nhận
        # thêm **kwargs (dù không dùng tới) để không bị "unexpected keyword argument
        # 'stop'". Không cần xử lý stop ở đây vì đã tự cắt chuỗi ở
        # agents._truncate_at_stop() ngay sau khi llm trả về.
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        return client.generate(text)

    return RunnableLambda(_call)
