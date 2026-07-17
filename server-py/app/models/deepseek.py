"""
模型封装
将 DeepSeek 封装为 LangChain ChatModel
DeepSeek 兼容 OpenAI 协议，使用 ChatOpenAI 并替换 base_url 即可
"""
import os
from langchain_openai import ChatOpenAI

# 模型配置统一从环境变量读取；未提供可选配置时使用项目默认值。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


def create_model(**overrides) -> ChatOpenAI:
    """创建 DeepSeek 模型实例，可通过 overrides 覆盖默认参数，如 temperature=0, streaming=True"""
    # 先建立所有调用共享的默认参数。
    params = {
        "model": MODEL_NAME,
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "temperature": 0.7,
        "streaming": False,
    }
    # 调用方传入的同名参数会覆盖默认值，因此不同场景可以调整温度或开启流式输出。
    params.update(overrides)
    # DeepSeek 提供 OpenAI 兼容接口，所以可以直接使用 LangChain 的 ChatOpenAI 适配器。
    return ChatOpenAI(**params)


# 默认导出一个标准实例（非流式）
model = create_model()

# 流式实例，用于 SSE 接口
streaming_model = create_model(streaming=True)
