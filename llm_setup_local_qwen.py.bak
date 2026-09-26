"""
llm_setup.py - Load Qwen2.5 và CHẶN CỨNG việc model sinh ra tiếng Trung/Nhật/Hàn.

=== VÌ SAO QWEN HAY "NHẢY" QUA TIẾNG TRUNG? (3 nguyên nhân, đã fix cả 3) ===

1. DÙNG NHẦM BASE MODEL, KHÔNG PHẢI INSTRUCT MODEL
   "Qwen/Qwen2.5-3B" chỉ được train kiểu "đoán chữ tiếp theo" trên dữ liệu đa ngôn ngữ
   (phần lớn là tiếng Trung + tiếng Anh). Nó KHÔNG được dạy "phải trả lời đúng ngôn ngữ
   người dùng đang dùng". Bản Instruct ("Qwen/Qwen2.5-3B-Instruct") được huấn luyện thêm
   bằng hội thoại có hướng dẫn, bám ngôn ngữ đầu vào tốt hơn hẳn. -> Đổi model (config.py).

2. PROMPT GHÉP THÔ (completion) THAY VÌ CHAT TEMPLATE CHUẨN
   Cách cũ nhồi thẳng text ReAct vào model như một đoạn văn để "viết tiếp" -> model không
   phân biệt được đâu là hệ thống, đâu là câu hỏi, dễ trôi ngôn ngữ giữa chừng.

3. KHÔNG CÓ RÀO CHẮN Ở TẦNG TOKEN
   Dù prompt bằng tiếng Việt, model vẫn CÓ THỂ chọn sinh ra token tiếng Trung nếu xác suất
   nội bộ của nó nghiêng về hướng đó (đặc biệt khi model nhỏ, 3B tham số, dễ "lú").
   -> Thêm BlockCJKLogitsProcessor: quét toàn bộ vocab 1 lần, CẤM TUYỆT ĐỐI mọi token mà
   khi giải mã ra có chứa ký tự Hán/Nhật/Hàn. Đây là rào chắn CỨNG, không phụ thuộc việc
   model có "nghe lời" hay không - kể cả khi model 'muốn' sinh tiếng Trung, nó không sinh
   được nữa vì các token đó đã bị gán xác suất = 0.
"""
import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    LogitsProcessor,
    LogitsProcessorList,
)

from config import MODEL_NAME, MAX_NEW_TOKENS, REPETITION_PENALTY

# Dải Unicode của chữ Hán (CJK), Hiragana/Katakana (Nhật), Hangul (Hàn)
CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7a3]"
)

VIETNAMESE_ONLY_INSTRUCTION = (
    "Bạn LUÔN LUÔN trả lời bằng tiếng Việt. Tuyệt đối KHÔNG được dùng tiếng Trung, "
    "tiếng Nhật, tiếng Hàn hay bất kỳ ngôn ngữ nào khác, dù chỉ một từ, trong bất kỳ "
    "phần nào của câu trả lời (kể cả Thought, Action, Final Answer)."
)


def contains_cjk(text: str) -> bool:
    """Kiểm tra nhanh 1 đoạn text có lẫn ký tự Hán/Nhật/Hàn không."""
    return bool(CJK_RE.search(text or ""))


class BlockCJKLogitsProcessor(LogitsProcessor):
    """Cấm cứng mọi token mà khi decode ra có chứa ký tự CJK.

    Build danh sách token bị cấm 1 LẦN DUY NHẤT lúc khởi tạo (quét ~150k token trong
    vocab, mất vài giây), sau đó áp dụng cho MỌI lượt sinh chữ mà không tốn thêm chi phí
    đáng kể.
    """

    def __init__(self, tokenizer):
        self.blocked_ids = self._build_blocklist(tokenizer)

    def _build_blocklist(self, tokenizer) -> torch.Tensor:
        vocab = tokenizer.get_vocab()
        blocked = []
        for token_str, token_id in vocab.items():
            try:
                decoded = tokenizer.convert_tokens_to_string([token_str])
            except Exception:
                decoded = token_str
            if CJK_RE.search(decoded):
                blocked.append(token_id)
        if not blocked:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(blocked, dtype=torch.long)

    def __call__(self, input_ids, scores):
        if self.blocked_ids.numel() > 0:
            scores[:, self.blocked_ids.to(scores.device)] = -float("inf")
        return scores


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"✅ Đã load {model_name} lên {model.device}")
    return model, tokenizer


def build_pipeline(model, tokenizer, max_new_tokens: int = MAX_NEW_TOKENS):
    print("⏳ Đang dựng bộ lọc chặn ký tự Trung/Nhật/Hàn (chạy 1 lần, vài giây)...")
    cjk_blocker = BlockCJKLogitsProcessor(tokenizer)
    logits_processor = LogitsProcessorList([cjk_blocker])

    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=REPETITION_PENALTY,
        return_full_text=False,
        logits_processor=logits_processor,
    )
    print(
        f"✅ Pipeline sẵn sàng - đã chặn cứng {cjk_blocker.blocked_ids.numel()} "
        "token tiếng Trung/Nhật/Hàn khỏi vocab sinh chữ."
    )
    return hf_pipe


def build_langchain_llm(hf_pipe):
    """Bọc pipeline HF thành LLM cho LangChain."""
    from langchain_huggingface import HuggingFacePipeline

    llm = HuggingFacePipeline(pipeline=hf_pipe)
    print("✅ LangChain LLM wrapper sẵn sàng")
    return llm
