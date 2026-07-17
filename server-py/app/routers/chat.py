"""
第一章：FastAPI 路由
GET  /api/chat/health  - 健康检查
POST /api/chat         - 普通对话（一次性返回）
POST /api/chat/stream  - 流式对话（SSE）
"""
import json
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.chains.basic_chat import (
    customer_service_chain,
    customer_service_stream_chain,
    format_history,
)

router = APIRouter()


class ChatMessage(BaseModel):
    # 单条历史消息由角色和正文组成。
    role: str
    content: str


class ChatRequest(BaseModel):
    # 同一个请求结构同时供普通接口和流式接口使用。
    message: str
    history: list[ChatMessage] = []


def _now():
    # 当前时间会填入客服 Prompt，帮助模型回答带相对时间的问题。
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


# ─── 健康检查 ────────────────────────────────────────────────────
@router.get("/health")
async def health():
    # 健康检查不调用模型，只用于确认 FastAPI 服务能够正常响应。
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ─── 普通对话接口 ────────────────────────────────────────────────
@router.post("")
async def chat(req: ChatRequest):
    if not req.message:
        return JSONResponse(status_code=400, content={"error": "message 字段不能为空"})

    try:
        # ainvoke 等待 Chain 完整执行结束，再一次性返回最终字符串。
        response = await customer_service_chain.ainvoke(
            {
                "user_input": req.message,
                # Pydantic 对象先转为字典，再由 format_history 转成 Prompt 接受的角色元组。
                "chat_history": format_history([m.model_dump() for m in req.history]),
                "current_time": _now(),
            }
        )
        return {"content": response}
    except Exception as error:
        # 非流式接口尚未开始响应，可以直接返回 HTTP 500 和 JSON 错误信息。
        print(f"[Chat Error] {error}")
        return JSONResponse(status_code=500, content={"error": "服务暂时不可用，请稍后重试"})


# ─── 流式对话接口（SSE）─────────────────────────────────────────
@router.post("/stream")
async def chat_stream(req: ChatRequest):
    if not req.message:
        return JSONResponse(status_code=400, content={"error": "message 字段不能为空"})

    async def event_generator():
        try:
            # astream 每得到一段模型输出就进入一次循环，因此前端能逐字看到回答。
            async for chunk in customer_service_stream_chain.astream(
                {
                    "user_input": req.message,
                    "chat_history": format_history([m.model_dump() for m in req.history]),
                    "current_time": _now(),
                }
            ):
                if chunk:
                    # 每个文本片段都单独封装为一条 SSE data 事件。
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

            # 模型输出完成后发送明确的结束标记。
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as error:
            # 与 Agent 流式接口相同，流中的异常要通过 SSE 内容传递。
            print(f"[Stream Error] {error}")
            yield f"data: {json.dumps({'error': '生成回复时出错，请重试'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
