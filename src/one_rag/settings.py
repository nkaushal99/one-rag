from functools import lru_cache
from pydantic import SecretStr
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
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0
    google_api_key: SecretStr | None = None
    ragas_judge_model: str = "gemini-2.5-flash-lite"
    ragas_judge_temperature: float = 0
    ragas_embedding_model: str = "BAAI/bge-small-en-v1.5"
    ragas_embedding_revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
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
