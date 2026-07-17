"""LangGraph 状态定义"""
from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    # messages 保存对话消息。Annotated 中的 add_messages 是 LangGraph 的“状态合并器”：
    # 不同 id 的新消息通常会追加到原列表，同 id 的消息则会更新原消息，而不是直接覆盖整个列表。
    messages: Annotated[list, add_messages]

    # 本轮用户原始输入。各个业务节点都从这里读取当前问题。
    user_input: str

    # 意图分类结果，只会使用 order、knowledge、general 三个值之一。
    intent: str

    # 三条分支各自写入不同字段，避免节点之间互相覆盖数据。
    # Optional 表示订单分支尚未执行时，这个字段可以没有值或为 None。
    order_result: Optional[dict]
    # 知识库分支生成的回答文本。
    rag_result: str
    # 工作流最终返回给前端的客服回答。
    final_answer: str
