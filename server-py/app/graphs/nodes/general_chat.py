from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.models.deepseek import create_model

# 普通对话 Prompt 由客服身份、历史消息占位符和本轮输入组成。
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '你是极速购电商平台的客服助手小购。语气友好，称呼用户为"亲"，回复简洁。'),
        ("placeholder", "{chat_history}"),
        ("human", "{user_input}"),
    ]
)

# 普通聊天允许一定表达变化，因此温度高于意图分类和订单查询。
_chain = prompt | create_model(temperature=0.7) | StrOutputParser()


def general_chat_node(state):
    user_input = state["user_input"]
    # messages 可能不存在，因此用空列表兜底；这里只保留最近 8 条，作为普通对话上下文。
    messages = state.get("messages") or []

    # ChatPromptTemplate 接受 (角色, 内容) 元组；把 LangChain 消息转换为它需要的格式。
    chat_history = [
        ("human" if m.type == "human" else "assistant", m.content) for m in messages[-8:]
    ]

    # general 分支已经直接生成可展示答案，所以写入 final_answer。
    result = _chain.invoke({"user_input": user_input, "chat_history": chat_history})
    return {"final_answer": result}
