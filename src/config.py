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

config = Config()
