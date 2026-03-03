from functools import lru_cache
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "test", "production"] = "development"
    api_port: int = 8000
    log_level: str = "INFO"
    secret_key: str = "change-me"

    # JWT
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "eip"
    postgres_user: str = "eip"
    postgres_password: str = "eip_secret"

    # Allow explicit override via DATABASE_URL env var
    database_url: str = ""
    database_url_sync: str = ""

    @computed_field
    @property
    def async_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def sync_database_url(self) -> str:
        if self.database_url_sync:
            return self.database_url_sync
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_raw: str = "eip-raw"
    minio_bucket_derived: str = "eip-derived"

    # Embedding models
    text_embedding_model: str = "BAAI/bge-large-en-v1.5"
    text_embedding_dim: int = 1024
    image_embedding_model: str = "ViT-B-32"
    image_embedding_pretrained: str = "openai"
    image_embedding_dim: int = 512

    # LLM provider (openai | ollama | mock)
    # Controls which LLM backend is used for Cognee and future LLM features.
    llm_provider: str = "openai"
    openai_api_key: str = ""

    # Ollama (local VLM for schematics + Cognee when llm_provider=ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_vlm_model: str = "llava"
    ollama_llm_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # Cognee storage (networkx/lancedb require no extra services — air-gapped safe)
    cognee_graph_engine: str = "networkx"   # networkx | neo4j
    cognee_vector_engine: str = "lancedb"   # lancedb | pgvector
    cognee_data_dir: str = "/app/data/cognee"

    # OCR thresholds
    ocr_tesseract_confidence_threshold: float = 0.75
    ocr_easyocr_confidence_threshold: float = 0.60

    # Security / ABAC
    default_classification: str = "UNCLASSIFIED"
    abac_policy_path: str = "/app/policy/abac.yaml"

    # Directory watcher
    watch_dir_poll_interval_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
