"""PostgreSQL 连接配置"""
import os

# 这里只读取进程环境变量；通过 main.py 的 load_dotenv() 提前加载 .env 后即可使用文件配置，
# 直接运行入库脚本时也需要先保证这些环境变量已经加载，否则使用本地开发默认值。
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "jisu_ai")

# 按 SQLAlchemy/psycopg3 的格式拼接连接串，供 langchain_postgres.PGVector 使用。
PG_CONNECTION_STRING = (
    f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)
