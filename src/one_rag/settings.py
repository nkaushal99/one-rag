from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "One RAG"
    environment: str = "development"
    postgres_dsn: str = "postgresql://one_rag:one_rag_dev_password@postgres:5432/one_rag"
    redis_url: str = "redis://redis:6379/0"
    qdrant_url: str = "http://qdrant:6333"
    kafka_bootstrap_servers: str = "kafka:9092"
    collection: str = "documents"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_provider: str = ""
    llm_model: str = ""
    reranker_model: str = ""
    chunking_strategy: str = "parent_child"
    fixed_chunk_size: int = 500
    fixed_chunk_overlap: int = 100
    sentence_window_size: int = 2
    sentence_window_overlap: int = 1
    parent_child_window_size: int = 2
    parent_child_window_overlap: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
