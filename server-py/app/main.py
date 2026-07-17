from dotenv import load_dotenv

# 先读取 server-py/.env 中的环境变量，再导入会读取 API Key 和数据库配置的其他模块。
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import agent, chat, graph, rag

# app 是 Uvicorn 启动时加载的 FastAPI 应用对象。
app = FastAPI(title="极速购 AI 客服系统")

# CORS 中间件允许浏览器中的前端页面跨域访问后端接口。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 每个业务模块只声明自己的相对路径，这里统一添加接口前缀并挂载到主应用。
app.include_router(chat.router, prefix="/api/chat")
app.include_router(agent.router, prefix="/api/agent")
app.include_router(rag.router, prefix="/api/rag")
app.include_router(graph.router, prefix="/api/graph")


@app.get("/")
async def root():
    # 根接口提供服务基本信息和主要入口，访问它不会触发模型或数据库调用。
    return {
        "service": "极速购 AI 客服系统",
        "version": "1.0.0",
        "routes": {
            "chat": "POST /api/chat/stream",
            "agent": "POST /api/agent/stream",
            "rag": "POST /api/rag/query",
            "graph": "POST /api/graph/stream",
        },
    }
