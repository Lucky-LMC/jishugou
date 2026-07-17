"""
Embedding 模型封装
方式一（推荐）：智谱 AI — 注册地址 https://open.bigmodel.cn
方式二：阿里云百炼 — 注册地址 https://bailian.console.aliyun.com
两种方式只有 model / api_key / base_url 三个字段不同，其余代码一样
"""
import os
from langchain_openai import OpenAIEmbeddings

# Embedding 模型把文本转换为数值向量；入库和查询必须使用语义空间兼容的模型。
# 当前启用方式一：智谱 AI。
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
    check_embedding_ctx_length=False,
)

# 方式二：阿里云百炼的 OpenAI 兼容接口示例（当前未启用）。
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-v3",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     check_embedding_ctx_length=False,
# )
