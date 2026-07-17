from app.chains.rag_chain import rag_chain


def rag_node(state):
    user_input = state["user_input"]
    try:
        # rag_chain 内部会先检索知识库，再把检索到的文档交给模型生成答案。
        # 与订单 Agent 不同，RAG 节点不会让模型自主选择工具，也没有 ReAct 循环。
        result = rag_chain.invoke({"question": user_input})

        # 只更新知识库分支的结果字段，之后由 answerSynthesizer 继续处理。
        return {"rag_result": result}
    except Exception as err:
        # 检索服务、向量数据库或模型调用异常时，为后续节点提供可读的降级结果。
        print(f"[ragNode] {err}")
        return {"rag_result": "查询知识库时出错"}
