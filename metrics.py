"""
metrics.py - MỘT chỉ số duy nhất, dễ hiểu cho người không rành kỹ thuật:

        ĐIỂM TIN CẬY (Reliability Score) - thang 0-100, kèm đèn giao thông 🟢🟡🔴

Gồm 3 thành phần đo được, gộp lại theo trọng số:
  1. Độ thuần tiếng Việt (50%) - % ký tự SINH RA không phải chữ Trung/Nhật/Hàn.
     Đây là chỉ số trực tiếp giải quyết vấn đề "Qwen nhảy qua tiếng Trung".
  2. Tỉ lệ đúng format (30%) - agent viết đúng cú pháp Action/Action Input bao nhiêu %
     số bước, không cần LangChain tự sửa lỗi parsing giùm.
  3. Hoàn thành câu hỏi (20%) - agent có ra được Final Answer trong giới hạn số bước
     cho phép hay bị "đuối" (loop, timeout).

Không cần biết code, người dùng chỉ cần nhìn 1 con số + 1 màu đèn để biết agent
đang chạy tốt hay có vấn đề.
"""
import re
from dataclasses import dataclass, field
from typing import List

from config import (
    SCORE_GOOD_THRESHOLD,
    SCORE_OK_THRESHOLD,
    WEIGHT_LANGUAGE_PURITY,
    WEIGHT_FORMAT_SUCCESS,
    WEIGHT_TOOL_SUCCESS,
    WEIGHT_TASK_COMPLETED,
)

CJK_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7a3]"
)

PARSING_ERROR_MARKER = "Lỗi format:"  # phải khớp với message trong agents.py
STOPPED_MARKER = "Agent stopped due to"  # message mặc định của LangChain khi hết max_iterations
TOOL_ERROR_PREFIX = "Lỗi"  # quy ước: MỌI tool trong hệ thống trả lỗi bằng cách bắt đầu bằng "Lỗi"

# Model hay viết Final Answer "nghe có vẻ thành công" dù thực tế tool đã báo lỗi
# (vd tự bịa link ảnh giả, nói "giả định rằng cột X tồn tại"...). Đây là các cụm từ
# thường xuất hiện khi model đang "chống chế" - dùng làm tín hiệu phụ trợ.
BLUFF_MARKERS = [
    "không hiển thị do lỗi",
    "kết quả thực tế không hiển thị",
    "do lỗi kỹ thuật",
    "giả định rằng",
    "giả định ở trên",
    "giả sử",
]


def _looks_like_bluff(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in BLUFF_MARKERS)


# ==== Bộ đếm CHIA SẺ giữa các tầng agent (orchestrator gọi specialist gọi tool) ====
# Vấn đề đã fix: AgentExecutor.invoke() của orchestrator CHỈ trả về intermediate_steps
# của CHÍNH orchestrator (các lượt gọi "data_loader_agent", "visualizer_agent"...),
# KHÔNG thấy được lỗi xảy ra BÊN TRONG mỗi specialist. Kết quả cũ: dù specialist lặp
# lỗi 5-6 lần bên trong, orchestrator vẫn coi là "1 bước thành công" -> điểm luôn 100.
# Nay mỗi lần 1 specialist chạy xong (dù ở tầng nào), nó tự cộng dồn số liệu vào đây.
_run_stats = {"total_steps": 0, "parsing_errors": 0, "tool_calls": 0, "tool_errors": 0}


def reset_run_stats() -> None:
    global _run_stats
    _run_stats = {"total_steps": 0, "parsing_errors": 0, "tool_calls": 0, "tool_errors": 0}


def record_agent_result(result: dict) -> None:
    """Gọi sau MỖI lần 1 AgentExecutor (specialist hoặc orchestrator) chạy xong, để cộng
    dồn số liệu thật về các bước bên trong nó vào bộ đếm chung."""
    steps = result.get("intermediate_steps", [])
    _run_stats["total_steps"] += len(steps)
    for _action, observation in steps:
        obs_str = str(observation)
        if PARSING_ERROR_MARKER in obs_str:
            _run_stats["parsing_errors"] += 1
        else:
            _run_stats["tool_calls"] += 1
            if obs_str.strip().startswith(TOOL_ERROR_PREFIX):
                _run_stats["tool_errors"] += 1


def language_purity(text: str) -> float:
    """% ký tự KHÔNG phải chữ Trung/Nhật/Hàn trong đoạn text. 100 = hoàn toàn sạch."""
    if not text:
        return 100.0
    cjk_count = len(CJK_PATTERN.findall(text))
    return round(100.0 * (1 - cjk_count / max(len(text), 1)), 1)


def badge_for_score(score: float) -> str:
    if score >= SCORE_GOOD_THRESHOLD:
        return "🟢 Tốt"
    elif score >= SCORE_OK_THRESHOLD:
        return "🟡 Tạm ổn, nên theo dõi"
    return "🔴 Cần xem lại"


@dataclass
class QueryTrace:
    """Kết quả đo cho MỘT lượt hỏi (1 lần gọi analyze())."""

    query: str
    final_answer: str = ""
    total_steps: int = 0
    parsing_errors: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    stopped_by_limit: bool = False
    bluff_detected: bool = False

    @property
    def format_success(self) -> float:
        if self.total_steps == 0:
            return 100.0
        return round(100.0 * (1 - self.parsing_errors / self.total_steps), 1)

    @property
    def tool_success(self) -> float:
        """% lượt gọi tool THẬT SỰ chạy thành công (observation không bắt đầu bằng 'Lỗi').
        Đây là chỉ số trực tiếp bắt được ca 'model bịa thành công dù tool báo lỗi'."""
        if self.tool_calls == 0:
            return 100.0
        return round(100.0 * (1 - self.tool_errors / self.tool_calls), 1)

    @property
    def purity(self) -> float:
        return language_purity(self.final_answer)

    @property
    def completed(self) -> bool:
        """Hoàn thành THẬT SỰ = không bị timeout/hết lượt, VÀ không có dấu hiệu model
        đang chống chế/bịa kết quả, VÀ không phải MỌI tool call đều thất bại."""
        if self.stopped_by_limit:
            return False
        if self.bluff_detected:
            return False
        if self.tool_calls > 0 and self.tool_errors == self.tool_calls:
            return False
        return True

    @property
    def reliability_score(self) -> float:
        task_score = 100.0 if self.completed else 0.0
        score = (
            WEIGHT_LANGUAGE_PURITY * self.purity
            + WEIGHT_FORMAT_SUCCESS * self.format_success
            + WEIGHT_TOOL_SUCCESS * self.tool_success
            + WEIGHT_TASK_COMPLETED * task_score
        )
        return round(score, 1)

    @property
    def badge(self) -> str:
        return badge_for_score(self.reliability_score)

    def summary(self) -> str:
        flags = []
        if self.stopped_by_limit:
            flags.append("⏱️ Bị timeout/hết lượt")
        if self.bluff_detected:
            flags.append("⚠️ Nghi ngờ model bịa kết quả (có cụm 'giả định', 'lỗi kỹ thuật'...)")
        flags_str = ("\n  " + "\n  ".join(flags)) if flags else ""
        return (
            f"{self.badge} — Điểm tin cậy: {self.reliability_score}/100\n"
            f"  • Độ thuần tiếng Việt:          {self.purity}%\n"
            f"  • Tỉ lệ agent viết đúng format: {self.format_success}%\n"
            f"  • Tỉ lệ tool chạy thành công:   {self.tool_success}% ({self.tool_calls - self.tool_errors}/{self.tool_calls} lượt gọi tool)\n"
            f"  • Hoàn thành câu hỏi thật sự:   {'Có' if self.completed else 'Không'}"
            f"{flags_str}"
        )


class QualityMonitor:
    """Theo dõi Điểm tin cậy qua nhiều lượt hỏi để xem xu hướng theo thời gian."""

    def __init__(self):
        self.traces: List[QueryTrace] = []

    def build_trace(self, query: str, agent_result: dict) -> QueryTrace:
        """Dựng QueryTrace từ kết quả trả về của AgentExecutor orchestrator, GỘP CHUNG
        với _run_stats (đã được các specialist con cộng dồn suốt lượt hỏi này qua
        record_agent_result). Gọi reset_run_stats() TRƯỚC khi orchestrator.invoke()
        chạy, để _run_stats chỉ chứa dữ liệu của đúng 1 lượt hỏi này."""
        final_answer = agent_result.get("output", "")
        # Cộng luôn steps của chính orchestrator (record_agent_result đã được gọi cho nó
        # từ agents.make_analyze_fn, nên _run_stats ở đây đã là tổng hợp đầy đủ)
        trace = QueryTrace(
            query=query,
            final_answer=final_answer,
            total_steps=_run_stats["total_steps"],
            parsing_errors=_run_stats["parsing_errors"],
            tool_calls=_run_stats["tool_calls"],
            tool_errors=_run_stats["tool_errors"],
            stopped_by_limit=STOPPED_MARKER in final_answer,
            bluff_detected=_looks_like_bluff(final_answer),
        )
        self.traces.append(trace)
        return trace

    @property
    def average_score(self) -> float:
        if not self.traces:
            return 0.0
        return round(sum(t.reliability_score for t in self.traces) / len(self.traces), 1)

    def dashboard(self) -> str:
        if not self.traces:
            return "Chưa có dữ liệu."
        lines = [f"📊 TRUNG BÌNH {len(self.traces)} LƯỢT: {self.average_score}/100 {badge_for_score(self.average_score)}\n"]
        for i, t in enumerate(self.traces, 1):
            short_q = t.query if len(t.query) <= 50 else t.query[:50] + "..."
            lines.append(f"{i}. {t.badge} ({t.reliability_score}) - \"{short_q}\"")
        return "\n".join(lines)


monitor = QualityMonitor()
