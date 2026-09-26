"""
main.py - Lắp ráp toàn bộ hệ thống. Chạy trực tiếp (python main.py) hoặc import từng
hàm vào notebook.

    from main import setup, analyze
    setup()
    print(analyze("Tạo dataset sales mẫu, tên là sales_data"))
"""
from pathlib import Path

from llm_setup import build_gemini_llm
from agents import build_react_llm, build_specialist, build_orchestrator, make_specialist_tool, make_analyze_fn
from metrics import monitor

from tools.data_tools import create_sample_dataset, create_custom_dataset, load_csv, load_excel, list_available_datasets
from tools.stats_tools import describe_dataset, correlation_analysis, hypothesis_test, outlier_detection
from tools.insight_tools import generate_insights
from tools.viz_tools import create_visualization, create_distribution_report
from tools.transform_tools import filter_data, aggregate_data, add_calculated_column
from tools.report_tools import generate_summary_report, get_analysis_history


def load_input_file():
    """Prompt for a CSV/Excel path and load it into the shared data store."""
    file_path = input("Đường dẫn CSV/XLSX (Enter để bỏ qua): ").strip().strip('"')
    if not file_path:
        return

    suffix = Path(file_path).suffix.lower()
    loader = load_csv if suffix == ".csv" else load_excel if suffix == ".xlsx" else None
    if loader is None:
        print("Chỉ hỗ trợ file .csv và .xlsx.")
        return

    default_name = Path(file_path).stem
    dataset_name = input(f"Tên dataset [{default_name}]: ").strip() or default_name
    result = loader.invoke(f"file_path={file_path}, dataset_name={dataset_name}")
    print(result)


def run_cli(analyze):
    """Run an interactive session that can load data and process multiple tasks."""
    print("\nNạp dữ liệu từ file để agent phân tích; Enter để bỏ qua (có thể dùng :load sau).")
    load_input_file()
    print(
        "Nhập task, mỗi dòng một yêu cầu; gõ :load để nạp thêm file, "
        "hoặc :quit để thoát.\n"
        "Ví dụ: liệt kê dataset; tạo dataset mẫu/custom; describe/insight/correlation/"
        "hypothesis/outlier; vẽ biểu đồ/phân phối; filter/aggregate/tính cột; "
        "tạo report/xem lịch sử."
    )
    while True:
        task = input("\nTask> ").strip()
        if task.lower() in {":quit", "quit", "exit", "q"}:
            break
        if task.lower() == ":load":
            load_input_file()
            continue
        if task:
            print(analyze(task))


def setup():
    """Khởi tạo kết nối tới Gemma 4 26B-A4B-IT (Google AI Studio), dựng toàn bộ 5
    specialist + orchestrator. Trả về hàm analyze()."""
    llm = build_gemini_llm()
    llm_react = build_react_llm(llm)

    data_loader_executor = build_specialist(llm_react, [load_csv, load_excel, create_sample_dataset, create_custom_dataset, list_available_datasets])
    statistician_executor = build_specialist(llm_react, [describe_dataset, correlation_analysis, hypothesis_test, outlier_detection, generate_insights])
    visualizer_executor = build_specialist(llm_react, [create_visualization, create_distribution_report])
    transformer_executor = build_specialist(llm_react, [filter_data, aggregate_data, add_calculated_column])
    reporter_executor = build_specialist(llm_react, [generate_summary_report, get_analysis_history])
    print("✅ 5 specialist agents đã sẵn sàng")

    data_loader_agent = make_specialist_tool(
        "data_loader_agent",
        "Giao việc liên quan tới load/tạo dataset (CSV, Excel, sample, hoặc dataset TÙY "
        "CHỈNH với tên cột cụ thể do người dùng chỉ định) cho chuyên gia data_loader. "
        "task: mô tả việc cần làm, NÊU RÕ tên cột nếu người dùng có yêu cầu cụ thể.",
        data_loader_executor,
    )
    statistician_agent = make_specialist_tool(
        "statistician_agent",
        "Giao việc phân tích thống kê (describe, correlation, hypothesis test, outlier) "
        "VÀ sinh insight/nhận xét cho 1 cột cụ thể (vd 'nhận xét cột revenue', 'insight "
        "của dataset X là gì') cho chuyên gia statistician.",
        statistician_executor,
    )
    visualizer_agent = make_specialist_tool(
        "visualizer_agent",
        "Giao việc vẽ biểu đồ cho chuyên gia visualizer. task: mô tả bằng ngôn ngữ tự nhiên, "
        "ví dụ 'Vẽ histogram cột weight_kg của dataset df_fifa'. KHÔNG dùng key=value trong task.",
        visualizer_executor,
    )
    transformer_agent = make_specialist_tool(
        "transformer_agent",
        "Giao việc biến đổi dữ liệu (filter, aggregate, tính cột mới) cho chuyên gia transformer.",
        transformer_executor,
    )
    reporter_agent = make_specialist_tool(
        "reporter_agent",
        "Giao việc tạo báo cáo tổng hợp cho chuyên gia reporter.",
        reporter_executor,
    )

    orchestrator_tools = [data_loader_agent, statistician_agent, visualizer_agent, transformer_agent, reporter_agent]
    master_executor = build_orchestrator(llm_react, orchestrator_tools)
    print("✅ Master orchestrator sẵn sàng - gọi analyze('câu hỏi') để dùng")

    return make_analyze_fn(master_executor)


if __name__ == "__main__":
    analyze = setup()
    run_cli(analyze)
    print("\n" + monitor.dashboard())
