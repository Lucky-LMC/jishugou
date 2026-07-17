"""
Prompt 模板
极速购客服系统 System Prompt
"""
from langchain_core.prompts import ChatPromptTemplate

# 极速购客服 Prompt 由三部分组成：
# system 定义固定角色、规则和边界；placeholder 插入本次请求携带的历史消息；
# human 放入当前问题。这里的“记忆”来自前端回传 history，不是后端持久化记忆。
customer_service_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是极速购电商平台的专业客服助手小购。

规则：
1. 只回答与购物、订单、物流、商品、售后相关的问题
2. 语气友好、专业，称呼用户为"亲"
3. 回复简洁，不超过 150 字
4. 遇到需要人工处理的复杂问题，引导用户拨打 400-888-8888
5. 不要编造订单信息，没有工具查询时如实告知用户

当前时间：{current_time}""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{user_input}"),
    ]
)

# 通用对话 Prompt 没有电商业务约束，用于演示同一模型搭配不同 Prompt 的效果。
general_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的 AI 助手，用中文回答问题。"),
        ("placeholder", "{chat_history}"),
        ("human", "{user_input}"),
    ]
)
