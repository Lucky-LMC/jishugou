from datetime import datetime

from langgraph.prebuilt import create_react_agent

from app.models.deepseek import create_model
from app.tools.order_tools import all_tools

# 独立 Agent 使用确定性较高的模型配置，便于稳定地选择订单工具和回答事实问题。
_model = create_model(temperature=0)


def _system_prompt():
    # 创建 Agent 时读取当前时间；应用通常会复用已创建的 Agent，因此提示中的时间也是创建时的时间。
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    return f"""你是极速购电商平台的智能客服助手小购。

回答规则：
1. 需要查询数据时，先调用对应工具获取真实数据，不要猜测或编造
2. 语气友好，称呼用户为"亲"
3. 拿到数据后用自然语言组织回答，不要直接粘贴 JSON
4. 如果用户没有提供订单号但需要查询，先询问订单号

当前时间：{now}"""


def create_customer_agent():
    # create_react_agent 自动搭建“模型判断 → 工具执行 → 模型继续判断”的 ReAct 循环。
    # 模型依据工具的名称、description 和参数模型选择是否调用，以及调用哪个工具。
    return create_react_agent(model=_model, tools=all_tools, prompt=_system_prompt())
