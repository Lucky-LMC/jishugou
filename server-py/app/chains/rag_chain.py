from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector

from app.db.postgres import PG_CONNECTION_STRING
from app.models.deepseek import create_model
from app.models.embedding import embeddings

# collection 相当于向量数据库中的逻辑知识库名称，入库和查询必须使用同一个名称。
COLLECTION_NAME = "knowledge_embeddings"

# PGVector 负责把文本向量及其元数据保存到 PostgreSQL，并提供相似度检索能力。
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=PG_CONNECTION_STRING,
    use_jsonb=True,
)

# retriever 把向量库封装为 LangChain 检索器；每次取与问题最相似的 4 个文档块。
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Prompt 要求模型尽量依据检索上下文作答，缺少依据时明确说明未查到，降低编造风险。
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是极速购电商平台的专业客服助手小购。

请根据以下知识库内容回答用户的问题。
如果知识库中没有相关内容，请如实告知用户，不要编造信息。
回答语气友好，称呼用户为"亲"，回复简洁清晰。

知识库内容：
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    # 把多个 Document 的正文拼成一个 context，分隔线帮助模型区分不同文档块。
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


_model = create_model(temperature=0)

# 标准 RAG 链的输入是 {"question": ...}。
# 第一个字典步骤会并行构造 Prompt 所需的 context 和 question：
# context 先检索再拼接，question 则原样透传。
rag_chain = (
    {
        "context": lambda input: format_docs(retriever.invoke(input["question"])),
        "question": lambda input: input["question"],
    }
    | rag_prompt
    | _model
    | StrOutputParser()
)


def _build_answer_chain():
    # 这个内部 Chain 接收已经检索好的 docs，只负责整理上下文并调用模型生成回答。
    # 它供“需要同时返回答案和来源”的查询流程复用，避免重复检索。
    return (
        (lambda input: {"context": format_docs(input["docs"]), "question": input["question"]})
        | rag_prompt
        | _model
        | StrOutputParser()
    )


_answer_chain = _build_answer_chain()


def _with_sources(input):
    # 先只检索一次，同一批 docs 同时用于生成答案和整理来源信息。
    docs = retriever.invoke(input["question"])
    answer = _answer_chain.invoke({"docs": docs, "question": input["question"]})

    # 来源返回正文前 100 个字符并加省略号，同时返回入库时写入 metadata 的 source 字段。
    sources = [
        {"content": doc.page_content[:100] + "...", "source": doc.metadata.get("source")}
        for doc in docs
    ]
    return {"answer": answer, "sources": sources}


class _RagChainWithSources:
    """带来源信息的 RAG Chain"""

    def invoke(self, input: dict) -> dict:
        # 提供与普通 LangChain Runnable 相似的 invoke 调用形式，方便路由层统一使用。
        return _with_sources(input)


# 保留驼峰和下划线两种变量名，它们指向的是同一个包装对象。
ragChainWithSources = _RagChainWithSources()
rag_chain_with_sources = ragChainWithSources
