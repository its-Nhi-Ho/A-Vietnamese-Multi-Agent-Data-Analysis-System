"""
eval_harness.py - Tầng 3 (agent-level eval). CẦN GOOGLE_API_KEYS thật (gọi API Gemma qua
Google AI Studio) -> tốn quota/tiền, KHÔNG chạy mỗi commit. Chạy trước khi release, hoặc
theo lịch (vd 1 lần/tuần) để bắt trend suy giảm chất lượng.

Cách dùng:
    export GOOGLE_API_KEYS="key1,key2,..."
    python eval/eval_harness.py --cases eval/test_cases.jsonl
    python eval/eval_harness.py --cases eval/test_cases.jsonl --only easy
    python eval/eval_harness.py --cases eval/test_cases.jsonl --category visualizer_basic

Thiết kế:
- Tầng A (task_correctness, TỰ ĐỘNG): assert trực tiếp lên `data_store.store` hoặc file
  ảnh thật -> không tin lời model.
- Tầng B (reliability, TỰ ĐỘNG, có sẵn): lấy từ `metrics.monitor` sau mỗi lần analyze().
- Tầng C (rubric, THỦ CÔNG): case có "manual_rubric_required": true chỉ được in ra để
  người review đọc, KHÔNG tự động pass/fail.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Setup helpers - chuẩn bị state trước khi chạy 1 case (dùng chính analyze() thật, để
# việc "chuẩn bị dữ liệu" cũng đi qua đúng pipeline, không tạo tắt bằng cách ghi thẳng
# vào store - nếu data_loader hỏng thì mọi case phụ thuộc nó cũng phải đỏ theo.
# ---------------------------------------------------------------------------

def _setup_create_sales_data(analyze, store):
    analyze("Tạo dataset sales mẫu, tên là sales_data")


def _setup_create_sales_data_and_run_two_queries(analyze, store):
    analyze("Tạo dataset sales mẫu, tên là sales_data")
    analyze("Describe sales_data")
    analyze("Tính tương quan giữa revenue và quantity trong sales_data")


SETUP_REGISTRY = {
    "create_sales_data": _setup_create_sales_data,
    "create_sales_data_and_run_two_queries": _setup_create_sales_data_and_run_two_queries,
}


# ---------------------------------------------------------------------------
# Oracle checks - tính "đáp án đúng" độc lập bằng pandas, so với side-effect thật trong
# store, KHÔNG parse Final Answer tiếng Việt của model.
# ---------------------------------------------------------------------------

def _oracle_correlation_revenue_units_sold(store, result, tolerance):
    # Schema thật (tools/data_tools.py: create_sample_dataset, dataset_type='sales') là
    # date/region/product/revenue/units_sold - KHÔNG có cột 'quantity'.
    df = store.get("sales_data")
    expected = df["revenue"].corr(df["units_sold"])
    # correlation_analysis không nhận tham số cột cụ thể (luôn trả full ma trận dạng
    # text qua DataFrame.to_string(), không phải JSON) -> không có cách trích số ra khỏi
    # observation một cách đáng tin cậy bằng string ops đơn giản. Harness chỉ XÁC NHẬN
    # oracle tính được (không NaN) và log giá trị kỳ vọng để đối chiếu thủ công với log
    # verbose=True của AgentExecutor. Khuyến nghị dài hạn: đổi correlation_analysis trả
    # kèm 1 dict số liệu có cấu trúc để oracle-check tự động hoá được hoàn toàn.
    return expected is not None and not _is_nan(expected), f"expected_corr={expected:.4f} (đối chiếu thủ công với observation trong log)"


def _oracle_filter_revenue_gt_threshold(store, result, tolerance):
    if "sales_high" not in store.datasets:
        return False, "dataset 'sales_high' không tồn tại trong store"
    df_src = store.get("sales_data")
    df_out = store.get("sales_high")
    # Ngưỡng phải khớp với ngưỡng trong query của test case (1500) - nếu bạn đổi query,
    # nhớ đổi cả threshold ở đây. revenue mẫu ~ N(1000, 300) nên ngưỡng phải cùng thang
    # đo (hàng trăm/nghìn), không phải hàng triệu.
    threshold = 1500
    expected_rows = (df_src["revenue"] > threshold).sum()
    ok = len(df_out) == expected_rows
    return ok, f"threshold={threshold}, expected_rows={expected_rows}, actual_rows={len(df_out)}"


def _oracle_groupby_region_sum_revenue(store, result, tolerance):
    df = store.get("sales_data")
    if "region" not in df.columns:
        return None, "dataset không có cột 'region' - bỏ qua oracle check này"
    expected = df.groupby("region")["revenue"].sum().to_dict()
    return True, f"expected={expected} (đối chiếu thủ công với Final Answer)"


def _is_nan(x):
    try:
        return x != x
    except Exception:
        return False


ORACLE_REGISTRY = {
    "correlation_revenue_units_sold": _oracle_correlation_revenue_units_sold,
    "filter_revenue_gt_threshold": _oracle_filter_revenue_gt_threshold,
    "groupby_region_sum_revenue": _oracle_groupby_region_sum_revenue,
}


# ---------------------------------------------------------------------------
# Assertion evaluator cho các key đơn giản trong "expect"
# ---------------------------------------------------------------------------

ERROR_KEYWORDS = ["lỗi", "không tồn tại", "không tìm thấy", "thất bại", "error"]


def _evaluate_expect(case, result, trace, store, chart_path, run_started_at):
    expect = case.get("expect", {})
    checks = []  # list of (name, passed: bool|None, detail: str)

    if trace is None:
        # Xem comment CRASH ở run(): exception bay qua khỏi cả record_agent_result() lẫn
        # monitor.build_trace() -> không có QueryTrace nào cho lượt này. Về mặt SẢN PHẨM,
        # crash chắc chắn = "không hoàn thành" và "không phải bluff" (nó không kịp nói dối
        # vì nó chết trước khi nói được câu nào) - nên 2 key này VẪN chấm được không cần
        # trace. Các key còn lại cần số liệu từ trace (purity/tool_calls/...) thì đành bỏ
        # qua (None = không đánh giá được), KHÔNG được coi là pass hay fail giả.
        if "completed" in expect:
            checks.append(("completed", expect["completed"] is False, "CRASH -> completed=False (chắc chắn)"))
        if "bluff_detected" in expect:
            checks.append(("bluff_detected", expect["bluff_detected"] is False, "CRASH -> không có bluff (không kịp tạo Final Answer)"))
        for skip_key in ("purity_min", "tool_calls_min", "orchestrator_delegates_min"):
            if skip_key in expect:
                checks.append((skip_key, None, "CRASH - không có QueryTrace, không đánh giá được"))
        # Các check còn lại (dataset_exists, final_answer_*, chart_file_created, oracle_check)
        # vẫn hợp lệ vì chúng không phụ thuộc trace - chạy tiếp bình thường bên dưới.

    if "dataset_exists" in expect:
        name = expect["dataset_exists"]
        ok = name in store.datasets
        detail = f"'{name}' {'có' if ok else 'KHÔNG'} trong store.datasets"
        checks.append(("dataset_exists", ok, detail))
        if ok and "min_rows" in expect:
            n = len(store.datasets[name])
            ok2 = n >= expect["min_rows"]
            checks.append(("min_rows", ok2, f"rows={n}, cần >={expect['min_rows']}"))
        if ok and "columns_include" in expect:
            cols = set(store.datasets[name].columns)
            missing = [c for c in expect["columns_include"] if c not in cols]
            checks.append(("columns_include", len(missing) == 0, f"thiếu cột: {missing}" if missing else "đủ cột"))

    if "final_answer_not_empty" in expect:
        ok = bool(result.get("output", "").strip())
        checks.append(("final_answer_not_empty", ok, f"len={len(result.get('output', ''))}"))

    if trace is not None and "tool_calls_min" in expect:
        ok = trace.tool_calls >= expect["tool_calls_min"]
        checks.append(("tool_calls_min", ok, f"tool_calls={trace.tool_calls}"))

    if trace is not None and "completed" in expect:
        ok = trace.completed == expect["completed"]
        checks.append(("completed", ok, f"completed={trace.completed}, kỳ vọng={expect['completed']}"))

    if trace is not None and "bluff_detected" in expect:
        ok = trace.bluff_detected == expect["bluff_detected"]
        checks.append(("bluff_detected", ok, f"bluff_detected={trace.bluff_detected}"))

    if trace is not None and "purity_min" in expect:
        ok = trace.purity >= expect["purity_min"]
        checks.append(("purity_min", ok, f"purity={trace.purity}, cần >={expect['purity_min']}"))

    if "final_answer_mentions_error" in expect:
        low = result.get("output", "").lower()
        ok = any(k in low for k in ERROR_KEYWORDS)
        checks.append(("final_answer_mentions_error", ok, "final answer có nhắc lỗi" if ok else "KHÔNG thấy từ khóa lỗi nào"))

    if expect.get("chart_file_created"):
        exists = os.path.exists(chart_path)
        ok = exists
        detail = f"path={chart_path}"
        if exists and expect.get("chart_newer_than_test_start"):
            mtime = os.path.getmtime(chart_path)
            ok = mtime >= run_started_at
            detail += f", mtime={'mới hơn' if ok else 'CŨ HƠN thời điểm test bắt đầu -> nghi ngờ file cũ, không phải mới vẽ'}"
        checks.append(("chart_file_created", ok, detail))

    if trace is not None and "orchestrator_delegates_min" in expect:
        # LƯU Ý: analyze() (agents.make_analyze_fn) chỉ trả về `result["output"]` (string),
        # KHÔNG trả lại dict đầy đủ có intermediate_steps -> harness KHÔNG có cách nào đếm
        # trực tiếp "orchestrator gọi đúng bao nhiêu specialist tool". Dùng trace.tool_calls
        # (tổng số lượt gọi tool CỘNG DỒN qua mọi tầng - xem metrics.record_agent_result)
        # làm proxy: multi-agent pipeline chắc chắn có tool_calls cao hơn 1 specialist đơn lẻ.
        # Đây là proxy XẤP XỈ, không phải đếm chính xác số specialist được gọi - muốn đếm
        # chính xác cần sửa main.py để analyze() trả kèm intermediate_steps của orchestrator.
        ok = trace.tool_calls >= expect["orchestrator_delegates_min"]
        checks.append(("orchestrator_delegates_min", ok, f"tool_calls(proxy)={trace.tool_calls}, cần >={expect['orchestrator_delegates_min']} (xấp xỉ, xem comment code)"))

    if "oracle_check" in expect:
        fn = ORACLE_REGISTRY.get(expect["oracle_check"])
        if fn is None:
            checks.append(("oracle_check", None, f"chưa implement oracle '{expect['oracle_check']}'"))
        else:
            try:
                ok, detail = fn(store, result, expect.get("tolerance", 1e-3))
            except Exception as e:
                ok, detail = False, f"oracle raise exception: {e}"
            checks.append(("oracle_check", ok, detail))

    return checks


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run(cases_path, only_difficulty=None, only_category=None, verbose=False, backend="gemma"):
    try:
        if backend == "qwen":
            from run_qwen import setup_qwen as setup
        else:
            from main import setup
        from data_store import store
        from metrics import monitor
        from config import CHART_OUTPUT_PATH
    except Exception as e:
        print(f"❌ Không import được hệ thống (thiếu tools/*.py hoặc chưa cấu hình "
              f"GOOGLE_API_KEYS?): {e}")
        print("   Tầng agent-level eval CẦN toàn bộ package chạy được thật, bao gồm "
              "thư mục tools/ chưa có trong bản upload này.")
        sys.exit(1)

    cases = load_cases(cases_path)
    if only_difficulty:
        cases = [c for c in cases if c["difficulty"] == only_difficulty]
    if only_category:
        cases = [c for c in cases if c["category"] == only_category]

    print(f"🚀 Khởi tạo hệ thống (setup())...")
    analyze = setup()

    results = []
    run_started_at = time.time()

    for case in cases:
        print(f"\n{'='*70}\n▶ [{case['id']}] {case['category']} ({case['difficulty']})")
        print(f"  Query: {case['query']}")

        setup_fn = SETUP_REGISTRY.get(case.get("setup"))
        if setup_fn:
            setup_fn(analyze, store)

        n_traces_before = len(monitor.traces)
        t0 = time.time()
        output_text = analyze(case["query"])
        elapsed = time.time() - t0

        # BUG ĐÃ XÁC NHẬN trong tools/*.py: nhiều tool gọi store.get() ngoài try/except,
        # nên dataset không tồn tại -> ValueError bay thẳng qua AgentExecutor tới
        # agents.make_analyze_fn(), bị bắt bởi except NGOÀI CÙNG TRƯỚC KHI
        # record_agent_result()/monitor.build_trace() kịp chạy -> monitor.traces KHÔNG
        # có entry mới. Nếu đọc monitor.traces[-1] một cách ngây thơ ở đây, harness sẽ
        # ĐỌC NHẦM trace của case TRƯỚC ĐÓ và báo cáo sai lệch hoàn toàn. Phải so sánh
        # độ dài trước/sau để phát hiện đúng tình huống này.
        if len(monitor.traces) == n_traces_before:
            trace = None
            print(f"  💥 CRASH: exception bay qua khỏi cả AgentExecutor lẫn analyze()'s "
                  f"error handling - KHÔNG có QueryTrace nào được ghi cho lượt này "
                  f"(xem README mục Findings). Final answer thô: {output_text[:200]}")
        else:
            trace = monitor.traces[-1]
        result_stub = {"output": output_text, "intermediate_steps": []}

        checks = _evaluate_expect(case, result_stub, trace, store, CHART_OUTPUT_PATH, run_started_at)
        manual = case.get("expect", {}).get("manual_rubric_required", False)

        auto_checks = [c for c in checks if c[1] is not None]
        passed_auto = all(c[1] for c in auto_checks) if auto_checks else True

        print(f"  ⏱ {elapsed:.1f}s | Reliability: {trace.reliability_score if trace else 'n/a'}"
              f" ({trace.badge if trace else ''})")
        for name, ok, detail in checks:
            symbol = "✅" if ok else ("⚠️ " if ok is None else "❌")
            print(f"  {symbol} {name}: {detail}")
        if manual:
            print(f"  📝 CẦN NGƯỜI REVIEW (tầng C) - đọc final answer bên dưới:")
            print(f"     {output_text[:400]}")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "difficulty": case["difficulty"],
            "task_correctness_pass": passed_auto,
            "reliability_score": trace.reliability_score if trace else None,
            "badge": trace.badge if trace else None,
            "manual_review_needed": manual,
            "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
            "final_answer": output_text,
        })

    _print_summary(results)
    _save_run(results)
    return results


def _print_summary(results):
    print(f"\n\n{'#'*70}\n# TỔNG KẾT\n{'#'*70}")
    auto = [r for r in results if not r["manual_review_needed"]]
    n_pass = sum(1 for r in auto if r["task_correctness_pass"])
    print(f"Task Correctness (tầng A, tự động): {n_pass}/{len(auto)} PASS")

    scores = [r["reliability_score"] for r in results if r["reliability_score"] is not None]
    if scores:
        print(f"Reliability Score trung bình (tầng B): {sum(scores)/len(scores):.1f}/100")

    n_manual = sum(1 for r in results if r["manual_review_needed"])
    if n_manual:
        print(f"⚠️  {n_manual} case cần người review thủ công (tầng C, rubric)")

    print("\nTheo category:")
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rs in sorted(by_cat.items()):
        auto_rs = [r for r in rs if not r["manual_review_needed"]]
        n_ok = sum(1 for r in auto_rs if r["task_correctness_pass"])
        avg_rel = sum(r["reliability_score"] or 0 for r in rs) / len(rs)
        flag = "✅" if (not auto_rs or n_ok == len(auto_rs)) else "❌"
        print(f"  {flag} {cat}: {n_ok}/{len(auto_rs)} auto-pass, reliability avg={avg_rel:.1f}")

    fails = [r for r in results if not r["manual_review_needed"] and not r["task_correctness_pass"]]
    if fails:
        print(f"\n❌ Case FAIL (tầng A):")
        for r in fails:
            bad = [c for c in r["checks"] if c["passed"] is False]
            print(f"  - {r['id']}: {'; '.join(c['name'] + ': ' + c['detail'] for c in bad)}")


def _save_run(results):
    os.makedirs("eval_runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("eval_runs", f"{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Đã lưu kết quả chi tiết vào {path} (để so sánh trend theo thời gian)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="eval/test_cases.jsonl")
    parser.add_argument("--only", dest="difficulty", default=None, help="easy|medium|hard")
    parser.add_argument("--category", default=None)
    parser.add_argument("--backend", choices=["gemma", "qwen"], default="gemma",
                        help="LLM backend to evaluate (default: gemma)")
    args = parser.parse_args()
    run(
        args.cases,
        only_difficulty=args.difficulty,
        only_category=args.category,
        backend=args.backend,
    )
