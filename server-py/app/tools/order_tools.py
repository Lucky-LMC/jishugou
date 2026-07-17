import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.data.mock import orders, logistics


# 下面三个 Pydantic 模型描述工具参数的名称、类型和含义。
# create_react_agent 会把这些结构提供给模型，模型据此生成格式正确的工具参数。
class OrderIdInput(BaseModel):
    orderId: str = Field(description="订单号，格式为 ORD-xxx，例如 ORD-001")


class TrackingNoInput(BaseModel):
    trackingNo: str = Field(description="快递单号，例如 SF1234567890")


class UserIdInput(BaseModel):
    userId: str = Field(description="用户 ID，格式为 U-xxx，例如 U-100")


@tool(
    "getOrderInfo",
    args_schema=OrderIdInput,
    description="根据订单号查询订单详情，包括订单状态、商品列表、金额、快递信息。当用户询问订单状态、订单内容时调用。",
)
def get_order_info_tool(orderId: str) -> str:
    # 当前项目用内存中的模拟数据代替真实订单数据库，并按订单号精确查找。
    order = orders.get(orderId)
    if not order:
        # “查不到订单”属于正常业务结果，因此返回结构化错误信息，而不是抛出异常。
        return json.dumps({"error": f"订单 {orderId} 不存在"}, ensure_ascii=False)
    # ToolMessage 的内容通常使用字符串；JSON 既方便模型理解，也保留了字段结构。
    return json.dumps(order, ensure_ascii=False)


@tool(
    "getLogisticsInfo",
    args_schema=TrackingNoInput,
    description="根据快递单号查询物流轨迹，包括各节点时间、地点、状态。当用户询问快递到哪了、物流状态时调用。",
)
def get_logistics_tool(trackingNo: str) -> str:
    # 使用快递单号查询对应的物流轨迹列表。
    records = logistics.get(trackingNo)
    if not records:
        return json.dumps({"error": f"快递单号 {trackingNo} 暂无物流信息"}, ensure_ascii=False)
    return json.dumps({"trackingNo": trackingNo, "records": records}, ensure_ascii=False)


@tool(
    "getUserOrders",
    args_schema=UserIdInput,
    description='根据用户 ID 查询该用户的所有订单列表摘要。当用户询问"我有哪些订单"、"最近的订单"时调用。',
)
def get_user_orders_tool(userId: str) -> str:
    # 遍历模拟订单，只保留属于指定用户的记录。
    user_orders = [o for o in orders.values() if o["userId"] == userId]
    if not user_orders:
        return json.dumps({"error": f"用户 {userId} 暂无订单"}, ensure_ascii=False)
    # 列表查询只返回摘要字段，避免把每个订单的全部明细都交给模型。
    summary = [
        {
            "orderId": o["orderId"],
            "status": o["status"],
            "amount": o["amount"],
            "createTime": o["createTime"],
        }
        for o in user_orders
    ]
    return json.dumps(summary, ensure_ascii=False)


# Agent 注册的是经过 @tool 装饰后的工具对象，而不是普通业务函数列表。
all_tools = [get_order_info_tool, get_logistics_tool, get_user_orders_tool]
