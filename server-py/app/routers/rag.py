import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.chains.rag_chain import rag_chain_with_sources

router = APIRouter()


class RagRequest(BaseModel):
    # 独立 RAG 接口只需要一个知识库问题，不接收聊天历史。
    question: str


def _send(event_type, data):
    # 独立 RAG 接口也使用与 Agent/Graph 相同的 SSE 事件包装格式。
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"


@router.post("/query")
async def rag_query(req: RagRequest):
    if not req.question:
        return JSONResponse(status_code=400, content={"error": "question 不能为空"})

    async def event_generator():
        try:
            # 这个包装链会返回 {answer, sources}，其中 sources 来自检索到的文档元数据。
            result = rag_chain_with_sources.invoke({"question": req.question})

            # 有来源时先发送来源，前端可以先展示检索依据，再接收答案。
            if result.get("sources"):
                yield _send("sources", {"sources": result["sources"]})

            yield _send("answer", {"content": result["answer"]})
            yield _send("done", {})
        except Exception as err:
            # 查询期间的向量库或模型异常通过流内 error 事件返回。
            print(f"[RAG Error] {err}")
            yield _send("error", {"content": "查询出错，请重试"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
