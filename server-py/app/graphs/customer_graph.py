from langgraph.graph import END, START, StateGraph

from app.graphs.nodes.answer_synthesizer import answer_synthesizer_node
from app.graphs.nodes.general_chat import general_chat_node
from app.graphs.nodes.intent_router import intent_router_node, route_by_intent
from app.graphs.nodes.order_agent import order_agent_node
from app.graphs.nodes.rag_node import rag_node
from app.graphs.state import GraphState


def build_customer_graph():
    # StateGraph 规定整个工作流中的节点都通过 GraphState 读取和更新共享状态。
    graph = StateGraph(GraphState)

    # 注册节点时，左边的字符串是图中的节点名称，右边是节点真正执行的函数。
    graph.add_node("intentRouter", intent_router_node)
    graph.add_node("orderAgent", order_agent_node)
    graph.add_node("ragNode", rag_node)
    graph.add_node("generalChat", general_chat_node)
    graph.add_node("answerSynthesizer", answer_synthesizer_node)

    # 每次请求都先进入意图识别节点。
    graph.add_edge(START, "intentRouter")

    # route_by_intent 会返回下一个节点的名称。
    # 因为这是条件边，所以一次请求只会进入三个业务分支中的一个，并不会并行执行三条分支。
    graph.add_conditional_edges(
        "intentRouter",
        route_by_intent,
        {
            "orderAgent": "orderAgent",
            "ragNode": "ragNode",
            "generalChat": "generalChat",
        },
    )

    # 三条分支虽然都连接到 answerSynthesizer，但每次只会有被选中的那条边真正到达这里。
    graph.add_edge("orderAgent", "answerSynthesizer")
    graph.add_edge("ragNode", "answerSynthesizer")
    graph.add_edge("generalChat", "answerSynthesizer")
    graph.add_edge("answerSynthesizer", END)

    # compile 把“节点 + 边”的声明编译为可通过 invoke/stream 调用的 LangGraph 应用。
    return graph.compile()
