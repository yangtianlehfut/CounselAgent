import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    LAW_DB_PATH = Path(os.getenv("LAW_DB_PATH", BASE_DIR / "data" / "db" / "laws.db"))
    CRIME_DB_PATH = Path(os.getenv("CRIME_DB_PATH", BASE_DIR / "data" / "db" / "crime.db"))

    # 向量模型配置（目前只支持 ollama，连接地址等用 ollama 默认配置）
    EMBEDDING_PROVIDER = 'ollama'
    EMBEDDING_MODEL_NAME = 'qwen3-embedding:0.6b'
    EMBEDDING_BATCH_SIZE = 16

    # 向量库默认位置
    VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", BASE_DIR / "data" / "vector_db"))
    COLLECTION_NAME = 'judgments'

    # 类案检索召回数量
    VECTOR_RECALL_SIZE = 5   # 语义召回条数
    TAG_RECALL_SIZE = 5      # 标签匹配召回条数
    HYBRID_RESULT_SIZE = 3   # 混合检索 RRF 融合后保留条数
    VECTOR_RECALL_WEIGHT = 1   # 语义路在 RRF 融合中的权重
    TAG_RECALL_WEIGHT = 1      # 标签路在 RRF 融合中的权重

config = Config()
