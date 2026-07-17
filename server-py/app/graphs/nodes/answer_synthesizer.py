import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.models.deepseek import create_model

# 这个节点不是把多条分支结果合并在一起。
# 每次请求只会带来订单结果、知识库结果或普通聊天答案中的一种，它负责统一最终输出方式。
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是极速购电商平台的客服助手小购。

根据以下查询结果，为用户生成一个清晰、友好的回答。
称呼用户为"亲"，语气专业，内容简洁准确。

订单查询结果（如有）：{order_result}
知识库查询结果（如有）：{rag_result}""",
        ),
        ("human", "{user_input}"),
    ]
)

# 订单或知识库已经给出事实内容，这里的模型主要负责把结果整理成友好的客服措辞。
_chain = prompt | create_model(temperature=0.5) | StrOutputParser()


def answer_synthesizer_node(state):
    # 同时读取三条分支可能写入的字段，但一次工作流实际上只会命中其中一条分支。
    user_input = state["user_input"]
    order_result = state.get("order_result")
    rag_result = state.get("rag_result")
    final_answer = state.get("final_answer")
    intent = state.get("intent")

    # generalChat 已经生成完整回答，无需再次调用模型，直接透传可减少一次模型调用。
    if intent == "general" and final_answer:
        return {"final_answer": final_answer}

    # order 分支传入订单 Agent 的结论，knowledge 分支传入 RAG 已生成的答案。
    # 未执行分支用“无”占位，让 Prompt 的输入结构保持固定。
    result = _chain.invoke(
        {
            "user_input": user_input,
            "order_result": json.dumps(order_result["answer"], ensure_ascii=False)
            if order_result
            else "无",
            "rag_result": rag_result or "无",
        }
    )
    # 路由层最终只需要读取 final_answer，不必关心之前走的是哪条业务分支。
    return {"final_answer": result}
