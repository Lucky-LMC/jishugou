from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.models.deepseek import create_model

# 意图识别只负责分流，不负责回答问题。
# system 消息给模型分类规则，human 消息中的占位符会在调用时替换为真实用户输入。
intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个意图分类器。

根据用户的问题，返回以下三个分类之一，只返回分类词，不要有任何其他内容：

- order：用户询问订单状态、物流信息、退款进度等需要查询订单数据的问题
- knowledge：用户询问商品介绍、规格参数、售后政策、退换货规则等可从知识库获取的问题
- general：其他类型的对话、闲聊、无法归类的问题

只输出一个词：order 或 knowledge 或 general""",
        ),
        ("human", "{user_input}"),
    ]
)

# LCEL 管道的执行顺序：填充 Prompt → 调用模型 → 把模型消息解析成普通字符串。
# 分类任务需要稳定输出，因此 temperature 设为 0。
_chain = intent_prompt | create_model(temperature=0) | StrOutputParser()

# 白名单用于校验模型输出，防止模型返回解释文字或未定义分类。
VALID_INTENTS = ["order", "knowledge", "general"]


def intent_router_node(state):
    # LangGraph 调用节点时会传入当前完整状态；这里只需要读取本轮输入。
    user_input = state["user_input"]
    raw = _chain.invoke({"user_input": user_input})

    # 去除空格并统一小写，再判断结果是否属于允许的三种意图。
    intent = raw.strip().lower()
    # 无法识别的结果降级为 general，保证条件路由一定能找到后续节点。
    final = intent if intent in VALID_INTENTS else "general"
    print(f'[intentRouter] "{user_input}" → {final}')

    # 节点只返回自己负责更新的字段，LangGraph 会把它合并回 GraphState。
    return {"intent": final}


def route_by_intent(state):
    # 条件边读取 intent，并返回 customer_graph.py 中已经注册的节点名称。
    mapping = {"order": "orderAgent", "knowledge": "ragNode", "general": "generalChat"}
    return mapping.get(state["intent"], "generalChat")
