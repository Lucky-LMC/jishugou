"""
文档入库脚本，执行一次即可，知识库更新时重新执行
运行：python -m app.scripts.ingest
"""
import os

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db.postgres import PG_CONNECTION_STRING
from app.models.embedding import embeddings

# 必须与 rag_chain.py 使用相同 collection，查询时才能检索到这里写入的数据。
COLLECTION_NAME = "knowledge_embeddings"

# 根据脚本文件位置计算知识库目录，不依赖执行命令时所在的当前目录。
_KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge")


def _load_docs():
    # 当前知识库由商品说明和售后政策两个 Markdown 文件组成。
    files = ["products.md", "policies.md"]
    docs = []
    for file in files:
        with open(os.path.join(_KNOWLEDGE_DIR, file), "r", encoding="utf-8") as f:
            content = f.read()
        # metadata.source 会随文档块写入向量库，查询接口用它展示答案来源。
        docs.append(Document(page_content=content, metadata={"source": file}))
    return docs


def ingest():
    print("开始处理文档...")

    docs = _load_docs()
    # 把长文档切成约 500 字符的片段，相邻片段保留 50 字符重叠以减少语义断裂。
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"切分完成，共 {len(chunks)} 个片段")

    # pre_delete_collection=True 会先删除同名 collection，再执行本次全量入库。
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONNECTION_STRING,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    # add_documents 内部会调用 Embedding 模型生成向量，再把向量、正文和 metadata 写入 PostgreSQL。
    vector_store.add_documents(chunks)

    print("入库完成")


if __name__ == "__main__":
    # 只有通过 python -m app.scripts.ingest 直接运行本模块时才执行入库；被 import 时不会自动执行。
    ingest()
