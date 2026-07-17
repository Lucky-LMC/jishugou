"""
Chain 链式调用
使用 LCEL（LangChain Expression Language）管道语法
将 Prompt → Model → OutputParser 串联
"""
from langchain_core.output_parsers import StrOutputParser

from app.models.deepseek import create_model
from app.prompts.customer_service import customer_service_prompt, general_chat_prompt

# 非流式和流式 Chain 共用同一套 Prompt 与字符串解析器，区别在于模型是否逐块返回内容。
_model = create_model(temperature=0.5)
_parser = StrOutputParser()

# LCEL 中的 | 表示把左侧输出作为右侧输入：Prompt → Model → 普通字符串。
customer_service_chain = customer_service_prompt | _model | _parser

# streaming=True 让调用方可以使用 stream() 逐块消费模型输出。
_streaming_model = create_model(temperature=0.5, streaming=True)

customer_service_stream_chain = customer_service_prompt | _streaming_model | _parser

# 通用对话只更换 Prompt，继续复用非流式模型和输出解析器。
general_chat_chain = general_chat_prompt | _model | _parser


def format_history(history=None):
    """将前端传来的 { role, content } 数组转换为 LangChain 消息格式"""
    # 前端没有传历史记录时，用空列表保证后续循环可以正常执行。
    history = history or []
    result = []
    for msg in history:
        # ChatPromptTemplate 使用 human/assistant 作为对话角色名称。
        # 这里只接受用户和助手消息，其他未知角色不会进入模型上下文。
        if msg.get("role") == "user":
            result.append(("human", msg.get("content")))
        elif msg.get("role") == "assistant":
            result.append(("assistant", msg.get("content")))
    return result
