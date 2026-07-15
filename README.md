# 极速购 AI 客服系统

一个用于学习和演示 AI 应用开发的前后端分离项目。系统以虚构的电商平台“极速购”为业务场景，逐步展示四种常见能力：大模型基础对话、可调用工具的 Agent、检索增强生成（RAG），以及基于 LangGraph 的多节点工作流。

> 本项目中的订单、物流、商品和售后知识库均为模拟数据，仅用于学习与演示，不能直接用于生产环境。

## 功能一览

| 模块 | 访问页面 | 后端接口 | 说明 |
| --- | --- | --- | --- |
| 基础对话 | `/` | `POST /api/chat`、`POST /api/chat/stream` | 基于 DeepSeek 的普通与流式客服对话，支持传入历史消息。 |
| 订单 Agent | `/agent` | `POST /api/agent/stream` | 使用 LangGraph 预构建 ReAct Agent，根据问题调用订单、物流和用户订单工具。 |
| 知识库问答 | `/rag` | `POST /api/rag/query` | 将本地商品与售后文档切分后写入 PostgreSQL + pgvector，检索后生成回答及来源片段。 |
| 智能中枢 | `/graph` | `POST /api/graph/stream` | 先识别意图，再路由到订单 Agent、RAG 或通用对话节点，最后统一生成答案。 |

## 技术栈

- 前端：Vue 3、Vue Router、Vite
- 后端：Python、FastAPI、Uvicorn、Pydantic
- 大模型编排：LangChain、LangGraph、LangChain OpenAI 集成
- 对话模型：DeepSeek（OpenAI 兼容接口）
- 向量模型：智谱 AI `embedding-3`（默认）；代码中保留阿里云百炼切换示例
- 向量数据库：PostgreSQL、pgvector、`langchain-postgres`

## 项目结构

```text
jishugou/
├── client/                         # Vue 前端
│   ├── src/
│   │   ├── views/                   # 四个功能页面
│   │   ├── composables/             # SSE 请求和页面状态逻辑
│   │   └── router/                  # 前端路由
│   ├── package.json
│   └── pnpm-lock.yaml
├── server-py/                       # FastAPI 后端
│   ├── app/
│   │   ├── agents/                  # ReAct 客服 Agent
│   │   ├── chains/                  # 基础对话与 RAG 链
│   │   ├── data/                    # 模拟订单、物流和知识库文档
│   │   ├── db/                      # PostgreSQL 连接配置
│   │   ├── graphs/                  # LangGraph 工作流及节点
│   │   ├── models/                  # DeepSeek 与 Embedding 模型封装
│   │   ├── routers/                 # FastAPI 路由
│   │   ├── scripts/ingest.py        # 知识库入库脚本
│   │   └── tools/                   # Agent 可调用的订单、物流工具
│   ├── .env.example                 # 环境变量示例
│   └── requirements.txt
└── README.md
```

## 前置条件

- Python 3.10 或更高版本
- Node.js 18 或更高版本
- pnpm（推荐；项目已提交 `pnpm-lock.yaml`）
- PostgreSQL，并已安装 `pgvector` 扩展（当前后端启动时会导入并初始化 RAG 向量库）
- DeepSeek API Key（所有对话能力需要）
- 智谱 AI API Key（默认 RAG 向量模型需要）

## 快速开始

### 1. 启动 PostgreSQL 与 pgvector

创建数据库并启用向量扩展。以下以 `jisu_ai` 为例：

```sql
CREATE DATABASE jisu_ai;
\c jisu_ai
CREATE EXTENSION IF NOT EXISTS vector;
```

`langchain_postgres.PGVector` 会在首次连接或入库时创建所需集合和表，无需手动建表。

### 2. 配置后端环境

在 `server-py` 目录创建虚拟环境并安装依赖：

```powershell
cd server-py
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux 请将激活命令替换为：

```bash
source .venv/bin/activate
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

| 变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | DeepSeek 对话模型 API Key。 |
| `DEEPSEEK_BASE_URL` | 否 | 默认 `https://api.deepseek.com/v1`。 |
| `MODEL_NAME` | 否 | 默认 `deepseek-chat`。 |
| `ZHIPU_API_KEY` | RAG 必填 | 默认的智谱 AI Embedding API Key。 |
| `DASHSCOPE_API_KEY` | 可选 | 使用阿里云百炼 Embedding 时配置；还需在 `app/models/embedding.py` 切换对应代码块。 |
| `PG_HOST` | 后端启动必填 | PostgreSQL 主机，默认 `localhost`。 |
| `PG_PORT` | 后端启动必填 | PostgreSQL 端口，默认 `5432`。 |
| `PG_USER` | 后端启动必填 | PostgreSQL 用户名。 |
| `PG_PASSWORD` | 后端启动必填 | PostgreSQL 密码。 |
| `PG_DATABASE` | 后端启动必填 | PostgreSQL 数据库名，默认 `jisu_ai`。 |

> `.env` 包含 API Key 和数据库凭据，只能保留在本地，绝不能提交到 GitHub。

### 3. 导入知识库

RAG 使用 `server-py/app/data/knowledge/` 下的 Markdown 文档。首次使用 RAG、修改这些文档后，或希望重建向量数据时，执行：

```powershell
python -m app.scripts.ingest
```

该脚本会清空并重建 `knowledge_embeddings` 集合中的数据。

### 4. 启动后端

仍在 `server-py` 且虚拟环境已激活时运行：

```powershell
uvicorn app.main:app --reload --port 3000
```

访问 `http://localhost:3000/` 可查看服务信息 JSON；健康检查地址为 `http://localhost:3000/api/chat/health`。

### 5. 启动前端

另开一个终端：

```powershell
cd client
pnpm install
pnpm dev
```

按 Vite 的终端提示在浏览器中打开本地地址。前端当前固定请求 `http://localhost:3000/api`，因此本地开发时后端应运行在 3000 端口。

## 接口说明

除健康检查外，主要接口均接收 JSON 请求体。流式接口使用 Server-Sent Events（SSE）。

| 方法 | 路径 | 请求字段 | 返回方式 |
| --- | --- | --- | --- |
| `GET` | `/api/chat/health` | 无 | JSON 健康状态。 |
| `POST` | `/api/chat` | `message`，可选 `history` | 一次性 JSON 回答。 |
| `POST` | `/api/chat/stream` | `message`，可选 `history` | SSE 文本片段与完成事件。 |
| `POST` | `/api/agent/stream` | `message`，可选 `history` | SSE 工具调用步骤、回答与完成事件。 |
| `POST` | `/api/rag/query` | `question` | SSE 来源片段、回答与完成事件。 |
| `POST` | `/api/graph/stream` | `message`，可选 `history` | SSE 节点、工具步骤、回答与完成事件。 |

`history` 中的每项格式为：

```json
{
  "role": "user",
  "content": "你好"
}
```

### 调用示例

```bash
# 基础对话
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好"}'

# 订单 Agent：模拟订单 ORD-001
curl -N -X POST http://localhost:3000/api/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我查一下订单 ORD-001 的状态"}'

# RAG：查询售后知识库
curl -N -X POST http://localhost:3000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"退换货政策是什么？"}'

# LangGraph 工作流：由系统识别订单、知识库或闲聊意图
curl -N -X POST http://localhost:3000/api/graph/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"我的快递到哪了，单号 SF1234567890"}'
```

对 SSE 接口使用 `curl` 时，请保留 `-N`，以禁用输出缓冲并实时查看事件。

## 运行流程

```text
浏览器（Vue）
    │ HTTP / SSE
    ▼
FastAPI
    ├── 基础对话 ───────────► DeepSeek
    ├── 订单 Agent ────────► ReAct Agent ─► 模拟订单/物流工具
    ├── RAG ───────────────► pgvector 检索 ─► DeepSeek
    └── 智能中枢 ──────────► 意图识别 ─► Agent / RAG / 闲聊 ─► 答案合成
```

## 常见问题

### 后端启动时连接 PostgreSQL 失败

RAG 链在导入时会初始化 `PGVector`。确认 PostgreSQL 服务已经启动、`.env` 中的连接信息正确，并且目标数据库已启用 `vector` 扩展。

### RAG 没有检索结果或数据过期

确认已经执行 `python -m app.scripts.ingest`。每次修改 `app/data/knowledge/` 中的文档后都需要重新入库。

### 模型请求返回 401 或 403

检查当前启用的对话与 Embedding 服务 API Key（`DEEPSEEK_API_KEY`、`ZHIPU_API_KEY` 或 `DASHSCOPE_API_KEY`）是否有效，并确认对应服务的账户额度与模型权限。

### 前端无法请求后端

确认后端正在 `http://localhost:3000` 运行。前端 API 地址目前写在 `client/src/composables/` 的各个请求模块中，部署到其他环境前需要改为可配置地址。

## GitHub 发布清单

可以提交：

- `client/src/`、`client/package.json`、`client/pnpm-lock.yaml`
- `server-py/app/`、`server-py/requirements.txt`
- `server-py/.env.example`、项目说明文档
- 模拟订单、物流和知识库 Markdown 文档

不要提交：

- `.env` 及任何真实 API Key、数据库密码
- `.venv/`、`node_modules/`、`dist/`、`__pycache__/`、`*.pyc`
- 本地数据库导出、日志、证书与私钥

建议在 `jishugou` 目录单独初始化 Git 仓库，避免将外层课程目录一并提交：

```powershell
cd jishugou
git init
git add .
git status
```

在首次提交前，请再次检查 `git status` 和 `git diff --cached`，确认没有真实凭据。项目当前的前后端子目录已有各自的忽略规则；发布前建议补充项目根目录的统一 `.gitignore`。

## 当前限制与生产注意事项

- 订单、用户和物流工具读取的是内存中的模拟数据，不连接真实业务系统。
- 前端 API 地址固定为本机 `localhost:3000`，尚未通过环境变量配置。
- 后端 CORS 当前允许任意来源，仅适用于本地教学演示；上线前应限制为受信任的前端域名。
- API Key 只应通过环境变量注入，不能写入源码、README、Issue 或提交历史。
- 项目目前未附带许可证；公开发布前请根据你的开源意图补充合适的 `LICENSE` 文件。
