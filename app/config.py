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
    sql_echo: bool = False
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

    # LLM provider — system-wide control (openai | ollama | mock)
    # Controls which LLM backend is used for ALL LLM-dependent features:
    # docling-graph entity extraction, Cognee memory, and future LLM features.
    llm_provider: str = "ollama"
    openai_api_key: str = ""

    # Ollama connection (shared by all features when llm_provider=ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_vlm_model: str = "llava"  # DEPRECATED: replaced by Docling service
    ollama_embedding_model: str = "nomic-embed-text"

    # Per-feature model selection — each feature can use a different model
    docling_graph_model: str = "llama3.2"  # Model for graph entity/relationship extraction
    cognee_model: str = "llama3.2"         # Model for Cognee memory operations

    # docling-graph extraction settings
    docling_graph_timeout: float = 120.0

    # Neo4j (knowledge graph — replaces Apache AGE)
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "eip_neo4j_secret"

    # Qdrant (vector search — replaces pgvector)
    qdrant_url: str = "http://qdrant:6333"
    qdrant_text_collection: str = "eip_text_chunks"
    qdrant_image_collection: str = "eip_image_chunks"

    # GraphRAG (community detection + reports)
    graphrag_model: str = "llama3.2"
    graphrag_indexing_enabled: bool = True
    graphrag_indexing_interval_minutes: int = 60
    graphrag_max_cluster_size: int = 10

    # Docling document conversion service (granite-docling-258M VLM)
    docling_service_url: str = "http://docling:8001"
    docling_timeout_seconds: float = 300.0
    docling_fallback_enabled: bool = False  # fall back to legacy extraction if Docling is down

    # Cognee storage (networkx/lancedb require no extra services — air-gapped safe)
    cognee_graph_engine: str = "networkx"   # networkx | neo4j
    cognee_vector_engine: str = "lancedb"   # lancedb | pgvector
    cognee_data_dir: str = "/app/data/cognee"

    # Memory layer
    memory_enabled: bool = True

    # OCR thresholds
    ocr_tesseract_confidence_threshold: float = 0.75
    ocr_easyocr_confidence_threshold: float = 0.60

    # Security / ABAC
    default_classification: str = "UNCLASSIFIED"
    abac_policy_path: str = "/app/policy/abac.yaml"

    # Pipeline retry & time-limit settings (env-var configurable)
    prepare_max_retries: int = 3
    prepare_retry_delay: int = 30
    prepare_soft_time_limit: int = 600
    prepare_time_limit: int = 660
    embed_max_retries: int = 2
    embed_retry_delay: int = 60
    embed_soft_time_limit: int = 300
    embed_time_limit: int = 360
    graph_max_retries: int = 2
    graph_retry_delay: int = 60
    graph_soft_time_limit: int = 600
    graph_time_limit: int = 660
    finalize_max_retries: int = 1
    finalize_retry_delay: int = 30
    finalize_soft_time_limit: int = 120
    finalize_time_limit: int = 180
    # Docling concurrency: max concurrent Docling conversions (Redis semaphore)
    docling_concurrency: int = 1
    # Lock timeout (auto-release if worker crashes)
    docling_lock_timeout: int = 600
    # Health probe timeout (seconds)
    docling_health_timeout: float = 10.0

    # Directory watcher
    watch_dir_poll_interval_seconds: int = 30

    # --- Retrieval scoring — new ontology relation weights ---
    retrieval_onto_weight_associated_with: float = 0.95
    retrieval_onto_weight_installed_on: float = 0.92
    retrieval_onto_weight_deployed_on: float = 0.92
    retrieval_onto_weight_uses_waveform: float = 0.92
    retrieval_onto_weight_operates_in_band: float = 0.92
    retrieval_onto_weight_has_antenna: float = 0.90
    retrieval_onto_weight_has_receiver: float = 0.90
    retrieval_onto_weight_has_transmitter: float = 0.90
    retrieval_onto_weight_cues: float = 0.90
    retrieval_onto_weight_guides: float = 0.90
    retrieval_onto_weight_tracks: float = 0.90
    retrieval_onto_weight_emits: float = 0.88
    retrieval_onto_weight_has_signature: float = 0.88
    retrieval_onto_weight_has_performance: float = 0.85
    retrieval_onto_weight_has_processing_chain: float = 0.85
    retrieval_onto_weight_supported_by: float = 0.80
    retrieval_onto_weight_mentioned_in: float = 0.80
    retrieval_onto_weight_alias_of: float = 0.95

    # --- Retrieval scoring (env-var configurable) ---
    # Expansion limits
    retrieval_doc_expand_k: int = 5
    retrieval_doc_max_hops: int = 2
    retrieval_ontology_expand_k: int = 5

    # Document-structure link weights
    retrieval_weight_next_chunk: float = 0.90
    retrieval_weight_same_section: float = 0.88
    retrieval_weight_same_artifact: float = 0.82
    retrieval_weight_same_page: float = 0.78

    # Score fusion weights (should sum to 1.0)
    retrieval_semantic_weight: float = 0.65
    retrieval_doc_structure_weight: float = 0.20
    retrieval_ontology_weight: float = 0.15

    # Ontology relation-specific weights
    retrieval_onto_weight_is_variant_of: float = 0.95
    retrieval_onto_weight_uses_component: float = 0.92
    retrieval_onto_weight_is_subsystem_of: float = 0.90
    retrieval_onto_weight_contains: float = 0.90
    retrieval_onto_weight_part_of: float = 0.90
    retrieval_onto_weight_interfaces_with: float = 0.85
    retrieval_onto_weight_operates_on: float = 0.85
    retrieval_onto_weight_meets_standard: float = 0.80
    retrieval_onto_weight_related_to: float = 0.75
    retrieval_onto_weight_default: float = 0.70

    # Hop penalty and bonuses
    retrieval_hop_penalty_base: float = 0.92
    retrieval_mil_id_bonus: float = 0.03

    # Legacy decay (fallback when chunk_links unavailable)
    retrieval_cross_modal_decay: float = 0.85
    retrieval_ontology_decay: float = 0.75


@lru_cache
def get_settings() -> Settings:
    return Settings()
