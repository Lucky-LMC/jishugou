import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.agents.customer_agent import create_customer_agent

router = APIRouter()

# 独立客服 Agent 在应用导入路由时创建，后续请求复用同一个已编译 ReAct 图。
_agent_app = create_customer_agent()


class ChatMessage(BaseModel):
    # 前端历史消息的数据结构，例如 {"role": "user", "content": "查询订单"}。
    role: str
    content: str


class AgentRequest(BaseModel):
    # message 是本轮输入，history 是由前端随请求传回的历史对话。
    message: str
    history: list[ChatMessage] = []


def _send(event_type, data):
    # SSE 规定每条事件以 data: 开头、以两个换行结束；type 供前端区分步骤、答案和结束事件。
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def agent_stream(req: AgentRequest):
    # FastAPI 已通过 Pydantic 完成字段解析，这里继续校验内容不能为空字符串。
    if not req.message:
        return JSONResponse(status_code=400, content={"error": "message 不能为空"})

    async def event_generator():
        # 异步生成器每 yield 一次，StreamingResponse 就能向客户端发送一条 SSE 消息。
        try:
            # 将前端数据转为 LangChain 消息对象。
            # 排除 history 最后一条是为了避免前端已把本轮问题放入 history 时与 req.message 重复。
            history_messages = [
                HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
                for m in req.history[:-1]
            ]

            # ainvoke 会运行完整 ReAct 循环，结束后一次性得到该轮产生的全部消息。
            result = await _agent_app.ainvoke(
                {"messages": history_messages + [HumanMessage(content=req.message)]}
            )

            # 遍历消息，寻找模型发出的工具调用，并把紧随其后的工具结果作为 observation。
            msgs = result["messages"]
            for i, msg in enumerate(msgs):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        tool_result = msgs[i + 1] if i + 1 < len(msgs) else None
                        yield _send(
                            "step",
                            {
                                "tool": tc["name"],
                                "toolInput": tc["args"],
                                "observation": tool_result.content if tool_result else "",
                            },
                        )

            # ReAct 循环最后一条消息是模型面向用户生成的最终回答。
            final_msg = msgs[-1]
            yield _send("answer", {"content": final_msg.content})
            # done 告诉前端本轮事件流已经完整结束。
            yield _send("done", {})
        except Exception as err:
            # 流已经开始后不能再改成普通 HTTP 错误响应，因此异常也通过 SSE error 事件通知前端。
            print(f"[Agent Error] {err}")
            yield _send("error", {"content": "处理请求时出错，请重试"})

    # text/event-stream 让浏览器按 SSE 流处理响应，而不是等待完整 JSON。
    return StreamingResponse(event_generator(), media_type="text/event-stream")
