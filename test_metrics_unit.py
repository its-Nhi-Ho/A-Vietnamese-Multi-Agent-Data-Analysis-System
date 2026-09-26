"""
test_metrics_unit.py - Tầng 1 (unit, deterministic). KHÔNG cần GOOGLE_API_KEYS, KHÔNG
gọi LLM, KHÔNG gọi mạng. Chạy trong CI mỗi commit.

Mục tiêu: chốt lại hành vi ĐÚNG của công thức Reliability Score bằng input giả lập, để
sau này ai sửa metrics.py (đổi trọng số, đổi công thức) sẽ biết ngay có phá vỡ hành vi
cũ hay không.

Chạy: pytest eval/test_metrics_unit.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import metrics
from metrics import (
    QueryTrace,
    language_purity,
    badge_for_score,
    reset_run_stats,
    record_agent_result,
)

# NB: KHÔNG "from metrics import _run_stats" - reset_run_stats() rebind biến module-level
# này sang 1 dict MỚI (global _run_stats = {...}), nên 1 reference đã import trước đó sẽ
# trỏ vào dict CŨ, stale. Phải đọc qua metrics._run_stats mỗi lần để luôn thấy dict hiện tại.


# ---------------------------------------------------------------------------
# language_purity
# ---------------------------------------------------------------------------

def test_language_purity_full_vietnamese():
    text = "Doanh thu trung bình của sales_data là 120 triệu đồng."
    assert language_purity(text) == 100.0


def test_language_purity_empty_string_is_perfect():
    # Quy ước trong code gốc: text rỗng -> 100.0 (không phạt khi không có gì để đánh giá)
    assert language_purity("") == 100.0


def test_language_purity_detects_cjk():
    # Chèn 3 ký tự Hán vào 1 câu tiếng Việt
    text = "Kết quả là 中文字"
    score = language_purity(text)
    assert score < 100.0, "phải phát hiện được ký tự CJK và trừ điểm"


def test_language_purity_all_cjk_is_zero():
    assert language_purity("中文中文") == 0.0


# ---------------------------------------------------------------------------
# QueryTrace - format_success / tool_success / completed / reliability_score
# ---------------------------------------------------------------------------

def _trace(**kwargs):
    defaults = dict(
        query="q",
        final_answer="Đã xong.",
        total_steps=0,
        parsing_errors=0,
        tool_calls=0,
        tool_errors=0,
        stopped_by_limit=False,
        bluff_detected=False,
    )
    defaults.update(kwargs)
    return QueryTrace(**defaults)


def test_format_success_no_steps_is_perfect():
    t = _trace(total_steps=0, parsing_errors=0)
    assert t.format_success == 100.0


def test_format_success_half_parsing_errors():
    t = _trace(total_steps=4, parsing_errors=2)
    assert t.format_success == 50.0


def test_tool_success_no_calls_is_perfect():
    t = _trace(tool_calls=0, tool_errors=0)
    assert t.tool_success == 100.0


def test_tool_success_partial_errors():
    t = _trace(tool_calls=5, tool_errors=1)
    assert t.tool_success == 80.0


def test_completed_false_when_stopped_by_limit():
    t = _trace(stopped_by_limit=True)
    assert t.completed is False


def test_completed_false_when_bluff_detected():
    t = _trace(bluff_detected=True)
    assert t.completed is False


def test_completed_false_when_all_tool_calls_failed():
    # Đây là case quan trọng nhất mà comment trong config.py nhắc tới: agent có thể "bịa"
    # thành công dù MỌI tool call đều lỗi. completed phải là False trong trường hợp này
    # kể cả khi final_answer nghe rất tự tin.
    t = _trace(tool_calls=3, tool_errors=3, final_answer="Đã hoàn thành xuất sắc!")
    assert t.completed is False


def test_completed_true_when_some_tool_errors_but_not_all():
    t = _trace(tool_calls=3, tool_errors=1)
    assert t.completed is True


def test_reliability_score_perfect_run():
    t = _trace(
        final_answer="Kết quả hoàn toàn bằng tiếng Việt.",
        total_steps=3,
        parsing_errors=0,
        tool_calls=3,
        tool_errors=0,
    )
    assert t.reliability_score == 100.0
    assert t.badge == badge_for_score(100.0)


def test_reliability_score_bluff_run_is_penalized_hard():
    """Agent nói tiếng Việt sạch, đúng format, NHƯNG bịa kết quả (bluff) -> completed=False
    -> mất trọn WEIGHT_TASK_COMPLETED (25 điểm theo config hiện tại), điểm phải < 85
    (ngưỡng 'Tốt') để không bị lọt lưới thành 🟢."""
    t = _trace(
        final_answer="Đã vẽ biểu đồ thành công, giả định rằng cột revenue tồn tại.",
        total_steps=2,
        parsing_errors=0,
        tool_calls=1,
        tool_errors=1,
        bluff_detected=True,
    )
    assert t.completed is False
    assert t.reliability_score < 85.0, (
        "bluff run không được phép lọt vào badge 🟢 Tốt dù văn phong tự tin/mượt"
    )


def test_badge_thresholds_match_config():
    from config import SCORE_GOOD_THRESHOLD, SCORE_OK_THRESHOLD

    assert "🟢" in badge_for_score(SCORE_GOOD_THRESHOLD)
    assert "🟡" in badge_for_score(SCORE_GOOD_THRESHOLD - 0.1)
    assert "🟡" in badge_for_score(SCORE_OK_THRESHOLD)
    assert "🔴" in badge_for_score(SCORE_OK_THRESHOLD - 0.1)


# ---------------------------------------------------------------------------
# record_agent_result / reset_run_stats - bộ đếm chia sẻ giữa orchestrator & specialist
# ---------------------------------------------------------------------------

def test_record_agent_result_accumulates_across_calls():
    """Mô phỏng đúng kịch bản comment trong metrics.py: orchestrator gọi 1 specialist,
    specialist tự lỗi 2 lần bên trong nó -> cả 2 lỗi đó phải được cộng dồn vào _run_stats
    chung, không bị orchestrator 'che' đi."""
    reset_run_stats()

    specialist_result = {
        "intermediate_steps": [
            ("action1", "Lỗi: cột 'revenu' không tồn tại. Hiện có: ['revenue']"),
            ("action2", "OK: đã tính xong"),
        ]
    }
    record_agent_result(specialist_result)

    orchestrator_result = {
        "intermediate_steps": [
            ("delegate_to_statistician", "Đã hoàn thành"),
        ]
    }
    record_agent_result(orchestrator_result)

    assert metrics._run_stats["total_steps"] == 3
    assert metrics._run_stats["tool_calls"] == 3
    assert metrics._run_stats["tool_errors"] == 1


def test_record_agent_result_counts_parsing_errors_separately_from_tool_errors():
    reset_run_stats()
    result = {
        "intermediate_steps": [
            ("a", "Lỗi format: bạn chỉ được viết MỘT Action..."),
        ]
    }
    record_agent_result(result)
    assert metrics._run_stats["parsing_errors"] == 1
    assert metrics._run_stats["tool_calls"] == 0, (
        "lỗi PARSING phải tách khỏi tool_calls, không được tính vào tool_success"
    )


def test_reset_run_stats_clears_between_queries():
    reset_run_stats()
    record_agent_result({"intermediate_steps": [("a", "OK")]})
    assert metrics._run_stats["total_steps"] == 1
    reset_run_stats()
    assert metrics._run_stats["total_steps"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
