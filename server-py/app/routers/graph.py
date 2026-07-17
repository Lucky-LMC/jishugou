import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.graphs.customer_graph import build_customer_graph

router = APIRouter()

# 使用惰性初始化：模块加载时先不编译图，第一次收到请求时才创建并缓存。
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        # 后续请求直接复用同一个编译结果，不重复调用 build_customer_graph。
        _graph = build_customer_graph()
    return _graph


class ChatMessage(BaseModel):
    # 工作流接口接收的历史消息结构。
    role: str
    content: str


class GraphRequest(BaseModel):
    # message 是当前问题；history 主要供 generalChat 节点形成对话上下文。
    message: str
    history: list[ChatMessage] = []


def _send(event_type, data):
    # 所有工作流事件统一为 {type, ...业务字段} 的 SSE JSON 格式。
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def graph_stream(req: GraphRequest):
    if not req.message:
        return JSONResponse(status_code=400, content={"error": "message 不能为空"})

    async def event_generator():
        try:
            graph = _get_graph()

            # 把前端历史记录转换为 GraphState.messages 接受的 LangChain 消息对象。
            history_messages = [
                HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
                for m in req.history
            ]

            # stream_mode="updates" 表示每完成一个节点，就返回该节点本次更新的状态字段，
            # 而不是每次都返回整个 GraphState。
            async for update in graph.astream(
                {
                    "user_input": req.message,
                    "messages": history_messages + [HumanMessage(content=req.message)],
                },
                stream_mode="updates",
            ):
                # 一次 update 的结构类似 {"intentRouter": {"intent": "order"}}。
                node_name, node_state = next(iter(update.items()))

                # node 事件让前端展示当前执行到了哪个工作流节点。
                yield _send("node", {"node": node_name, "intent": node_state.get("intent")})

                # 订单 Agent 节点会额外返回工具调用步骤，其他节点没有该字段。
                order_result = node_state.get("order_result")
                if order_result and order_result.get("steps"):
                    yield _send("steps", {"steps": order_result["steps"]})

                # 只有生成或透传最终答案的节点才会产生 answer 事件。
                if node_state.get("final_answer"):
                    yield _send("answer", {"content": node_state["final_answer"]})

            # 图运行到 END 后结束 SSE 流。
            yield _send("done", {})
        except Exception as err:
            # 工作流执行中任一节点出现未捕获异常，统一转换为 error 事件。
            print(f"[Graph Error] {err}")
            yield _send("error", {"content": "处理请求时出错，请重试"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
