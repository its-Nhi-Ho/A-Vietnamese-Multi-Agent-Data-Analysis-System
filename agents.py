"""
agents.py - Dựng các specialist agent, orchestrator, và hàm analyze() public API.

Prompt ReAct được thêm 1 dòng NHẮC LẠI bằng tiếng Việt ở ĐẦU và CUỐI (vị trí model
"chú ý" nhiều nhất khi sinh chữ) để giảm thêm khả năng trôi ngôn ngữ, cộng với rào chắn
cứng ở tầng token (llm_setup.BlockCJKLogitsProcessor) và việc đổi sang bản Instruct
(config.MODEL_NAME) - 3 lớp phòng thủ độc lập, không phụ thuộc lẫn nhau.
"""
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool as tool_decorator

from config import MAX_ITERATIONS_SPECIALIST, MAX_ITERATIONS_ORCHESTRATOR
from data_store import store
from llm_setup import VIETNAMESE_ONLY_INSTRUCTION
from metrics import monitor, reset_run_stats, record_agent_result

PARSING_ERROR_MESSAGE = (
    "Lỗi format: bạn chỉ được viết MỘT Action (kèm Action Input) "
    "HOẶC MỘT Final Answer trong 1 lượt, không được viết cả hai. "
    "Hãy thử lại đúng format, ngắn gọn."
)

REACT_TEMPLATE = f"""{VIETNAMESE_ONLY_INSTRUCTION}

Trả lời câu hỏi sau bằng cách dùng các tool có sẵn. Bạn có quyền dùng các tool sau:

{{tools}}

Dùng đúng format sau:

Question: câu hỏi cần trả lời
Thought: suy nghĩ về việc cần làm gì tiếp theo (bằng tiếng Việt)
Action: tên 1 tool trong [{{tool_names}}]
Action Input: tham số cho tool đó (dạng key=value, cách nhau bởi dấu phẩy)
Observation: kết quả của tool
... (lặp lại Thought/Action/Action Input/Observation nếu cần)
Thought: tôi đã có đủ thông tin để trả lời
Final Answer: câu trả lời cuối cùng cho người dùng (bằng tiếng Việt)

QUAN TRỌNG: Nếu Observation của 1 Action bắt đầu bằng chữ "Lỗi", nghĩa là hành động đó
THẤT BẠI THẬT SỰ. Bạn KHÔNG được viết Final Answer nói rằng đã thành công. Bạn phải:
sửa lại Action Input (vd đúng chính tả tên cột/tên dataset theo đúng như Observation liệt
kê) và thử lại, HOẶC nếu không sửa được thì Final Answer phải nói rõ THẤT BẠI và lý do,
KHÔNG được bịa ra kết quả, không được chèn link ảnh giả, không được nói "giả định rằng...".

Nhắc lại: {VIETNAMESE_ONLY_INSTRUCTION}

Bắt đầu!

Question: {{input}}
Thought:{{agent_scratchpad}}"""

react_prompt = PromptTemplate.from_template(REACT_TEMPLATE)

# QUAN TRỌNG: llm.bind(stop=[...]) KHÔNG có tác dụng thật với HuggingFacePipeline
# trong bản langchain_huggingface đang dùng (_call() không implement enforce_stop_tokens
# cho stop kwarg) -> model generate tới đâu cũng được, tự bịa luôn Observation + Final
# Answer trong 1 lần. Phải tự cắt chuỗi bằng Python NGAY SAU khi llm trả về, độc lập
# hoàn toàn với cơ chế stop của LangChain.
STOP_TOKENS = ["\nObservation:", "\nObservation"]


def _truncate_at_stop(text: str) -> str:
    """Cắt bỏ mọi thứ từ stop token đầu tiên trở đi (model hay tự bịa Observation/Final
    Answer giả sau đó). Lấy điểm cắt SỚM NHẤT trong số các stop token tìm được."""
    cut = len(text)
    for s in STOP_TOKENS:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def build_react_llm(llm):
    """Bọc llm gốc bằng bước cắt chuỗi ở stop token."""
    return llm | RunnableLambda(_truncate_at_stop)


def build_specialist(llm_react, tools, max_iterations: int = MAX_ITERATIONS_SPECIALIST):
    agent = create_react_agent(llm_react, tools, react_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        return_intermediate_steps=True,  # cần cho metrics.QualityMonitor
        handle_parsing_errors=PARSING_ERROR_MESSAGE,
    )


def build_orchestrator(llm_react, specialist_tools, max_iterations: int = MAX_ITERATIONS_ORCHESTRATOR):
    agent = create_react_agent(llm_react, specialist_tools, react_prompt)
    return AgentExecutor(
        agent=agent,
        tools=specialist_tools,
        verbose=True,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        handle_parsing_errors=PARSING_ERROR_MESSAGE,
    )


def make_specialist_tool(name: str, description: str, executor: AgentExecutor):
    """Bọc 1 AgentExecutor chuyên gia thành 1 @tool để orchestrator gọi được."""

    def _run(task: str) -> str:
        result = executor.invoke({"input": task})
        # Cộng dồn số liệu THẬT của specialist này (kể cả các lỗi xảy ra bên trong nó)
        # vào bộ đếm chung, để orchestrator/metrics thấy được toàn bộ bức tranh, không
        # chỉ thấy "specialist trả về 1 câu trả lời gọn gàng".
        record_agent_result(result)
        return result["output"]

    _run.__name__ = name
    _run.__doc__ = description
    return tool_decorator(_run)


def make_analyze_fn(master_executor: AgentExecutor):
    """Trả về hàm analyze(query) -> str, có tự động ghi Reliability Score vào
    metrics.monitor sau mỗi lượt gọi."""

    def analyze(query: str) -> str:
        # master_executor.invoke() mỗi lần là 1 conversation MỚI, không có chat_history.
        # Nếu không nhắc lại, orchestrator ở lần gọi sau sẽ không biết dataset nào đã
        # load từ lần gọi trước -> dễ tự bịa tên hoặc giao lại việc load từ đầu.
        if store.datasets:
            existing = ", ".join(f"'{name}'" for name in store.datasets.keys())
            context = f"(Các dataset đã có sẵn trong hệ thống: {existing}. Dùng lại tên này, KHÔNG cần load lại.) "
        else:
            context = "(Chưa có dataset nào trong hệ thống.) "

        reset_run_stats()  # bộ đếm _run_stats chỉ tính cho ĐÚNG 1 lượt hỏi này
        try:
            result = master_executor.invoke({"input": context + query})
            record_agent_result(result)  # cộng luôn các bước của CHÍNH orchestrator
            trace = monitor.build_trace(query, result)
            print("\n" + trace.summary())
            return result["output"]
        except Exception as e:
            return f"Lỗi: {str(e)}"

    return analyze
