from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.models.deepseek import create_model
from app.tools.order_tools import all_tools

# 模型和 Agent 在模块加载时创建一次，避免每次请求都重新构造对象。
_model = create_model(temperature=0)

# create_react_agent 会在内部构建 ReAct 子图：模型判断是否调用工具，
# tools 节点执行工具，再把工具结果交回模型；模型可继续调用工具，也可以结束循环。
_agent_app = create_react_agent(
    model=_model,
    tools=all_tools,
    prompt="""你是极速购的订单查询助手。
根据用户的问题，调用相应工具查询订单或物流信息。
只查询数据，不需要生成最终的客服回答。""",
)


def order_agent_node(state):
    user_input = state["user_input"]
    try:
        # 此处只把当前问题传给订单 Agent，没有传入 GraphState 中的历史 messages。
        # invoke 返回的 messages 包含本次 ReAct 循环里的用户消息、模型工具调用、工具结果和最终消息。
        result = _agent_app.invoke({"messages": [HumanMessage(content=user_input)]})

        # 从消息列表提取工具调用过程，供 SSE 接口把“调用了什么工具、得到什么结果”展示给前端。
        msgs = result["messages"]
        steps = []
        for i, msg in enumerate(msgs):
            # 普通消息没有 tool_calls；模型决定调用工具时，AIMessage 才会带有该属性。
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    # create_react_agent 通常会把对应 ToolMessage 放在工具调用消息之后。
                    obs = msgs[i + 1].content if i + 1 < len(msgs) else ""
                    steps.append({"tool": tc["name"], "input": tc["args"], "obs": obs})

        # 最后一条消息是 Agent 根据工具结果得出的本轮结论。
        # 节点把结论和步骤写入 order_result，最终措辞交给后续节点处理。
        final_msg = msgs[-1]
        return {"order_result": {"answer": final_msg.content, "steps": steps}}
    except Exception as err:
        # Agent 或工具出现未处理异常时，仍返回符合 GraphState 约定的结构，让主图可以继续执行。
        print(f"[orderAgentNode] {err}")
        return {"order_result": {"answer": "查询订单信息时出错", "steps": []}}
