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

    # LLM provider — controls which LLM backend is used for GraphRAG reports
    # and other LLM-dependent features (openai | ollama | mock).
    llm_provider: str = "ollama"

    # Ollama connection
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_base_url: str = ""   # chat/reasoning models — falls back to ollama_base_url
    ollama_vlm_base_url: str = ""   # vision/image models — falls back to ollama_base_url
    ollama_embedding_base_url: str = ""  # embedding models — falls back to ollama_base_url
    ollama_num_ctx: int = 16384
    ollama_think: str = ""  # "low", "medium", "high" for gpt-oss thinking level

    def get_ollama_llm_url(self) -> str:
        return self.ollama_llm_base_url or self.ollama_base_url

    def get_ollama_vlm_url(self) -> str:
        return self.ollama_vlm_base_url or self.ollama_base_url

    def get_ollama_embedding_url(self) -> str:
        return self.ollama_embedding_base_url or self.ollama_base_url
    llm_max_tokens: int = 64000

    # --- Docling-Graph service (entity/relationship extraction) ---
    docling_graph_base_url: str = "http://docling-graph:8002"
    docling_graph_concurrency: int = 2
    docling_graph_timeout: int = 300

    # Docling human-review confidence threshold
    docling_review_confidence_threshold: float = 0.60

    # Graph confidence quality gates
    graph_node_min_confidence: float = 0.60
    graph_rel_min_confidence: float = 0.55

    # Neo4j (knowledge graph — replaces Apache AGE)
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "eip_neo4j_secret"

    # Qdrant (vector search — replaces pgvector)
    qdrant_url: str = "http://qdrant:6333"
    qdrant_text_collection: str = "eip_text_chunks"
    qdrant_image_collection: str = "eip_image_chunks"
    qdrant_trusted_text_collection: str = "eip_trusted_text"
    qdrant_upsert_batch_size: int = 128
    qdrant_timeout_seconds: float = 60.0

    # Reranker (cross-encoder for retrieval re-scoring)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"  # cpu | cuda
    reranker_enabled: bool = True
    reranker_top_n: int = 20

    # Minimum cosine similarity threshold (below this, results are dropped)
    retrieval_min_score_threshold: float = 0.25

    # Embedding batching
    embed_text_batch_size: int = 128

    # GraphRAG (Microsoft GraphRAG — community detection + reports + search)
    graphrag_indexing_enabled: bool = True
    graphrag_indexing_interval_minutes: int = 60
    graphrag_max_cluster_size: int = 10
    graphrag_community_level: int = 2
    graphrag_data_dir: str = "/app/graphrag_data"
    # LLM provider for GraphRAG (ollama | openai)
    graphrag_llm_provider: str = "ollama"
    graphrag_llm_model: str = "llama3.2"
    graphrag_llm_api_base: str = "http://localhost:11434/v1"
    graphrag_api_key: str = ""
    # Embedding model for GraphRAG's LanceDB store
    graphrag_embedding_model: str = "nomic-embed-text"
    # LLM request timeout for GraphRAG search/indexing (seconds, default 3h)
    graphrag_llm_timeout: int = 10800
    # Auto-tuning schedule (minutes, default 24h)
    graphrag_tune_interval_minutes: int = 1440

    # Document Analysis (LLM metadata extraction)
    doc_analysis_enabled: bool = True
    doc_analysis_llm_model: str = "gpt-oss:120b"
    doc_analysis_timeout: int = 300
    doc_analysis_summary_prompt: str = "Summarize this document in 3-5 sentences for a technical reader. Focus on the main subject, scope, and notable findings. Do not include source endnote markings such as [1]."
    doc_analysis_date_prompt: str = "Extract the most relevant date of information (publication date, report date, or coverage window). If only month/year appears, return that. If there is a range, return it exactly. If unsure, return Unknown. Provide ONLY the date or range."
    doc_analysis_source_prompt: str = "Characterize the source: 1) Organization or author (if unknown return UNKNOWN) 2) Type of information (website, journal, etc; if unknown return UNKNOWN) 3) Reliability score 1-10. Format: Organization: <name>\\nType: <type>\\nReliability: <score>/10"
    doc_analysis_classification_prompt: str = "Identify the document classification marking if present (UNCLASSIFIED, CUI, FOUO, SECRET, TOP SECRET). If none, reply UNCLASSIFIED. Provide ONLY the marking."

    # Picture Description (post-conversion enrichment via Ollama)
    picture_description_model: str = "gemma3:27b"
    picture_description_timeout: int = 300
    picture_description_prompt: str = "Analyze this image from a multi-modal PDF using the required narrative sections and the missile/radar/S&T emphasis. Return sections 1-8 exactly as specified. Use the PDF Summary for context but rely on visual evidence.\\n\\n- PDF Summary: {document_summary}\\n\\n- Image:"

    # Translation (foreign language detection + LLM translation)
    translation_enabled: bool = True
    translation_model: str = "gpt-oss:120b"
    translation_timeout: int = 300
    translation_prompt: str = "Translate the following text to English. If the text is already in English, return it unchanged. Preserve all markdown formatting including headings (#), bullet points, tables, and code blocks. Preserve technical designators, model numbers, NATO reporting names, and military identifiers verbatim — do not transliterate or translate them (e.g., keep С-75, ЗРК, 9М38 as-is). Preserve all numbers, units, and acronyms. Preserve ---ELEMENT_BOUNDARY--- markers exactly as they appear. Return only the translated text with no commentary."
    translation_min_detect_length: int = 5
    translation_detect_threshold: float = 0.5
    translation_soft_time_limit: int = 3600
    translation_time_limit: int = 3660

    # Docling OCR language
    docling_ocr_lang: str = "en"

    # Docling document conversion service (granite-docling-258M VLM)
    docling_service_url: str = "http://docling:8001"
    docling_timeout_seconds: float = 3600.0
    docling_fallback_enabled: bool = False  # fall back to legacy extraction if Docling is down

    # Security / ABAC
    default_classification: str = "UNCLASSIFIED"

    # Pipeline retry & time-limit settings (env-var configurable)
    prepare_max_retries: int = 3
    prepare_retry_delay: int = 30
    prepare_soft_time_limit: int = 4200
    prepare_time_limit: int = 4260
    embed_max_retries: int = 10
    embed_retry_delay: int = 60
    embed_soft_time_limit: int = 1800
    embed_time_limit: int = 1860
    graph_max_retries: int = 2
    graph_retry_delay: int = 60
    graph_soft_time_limit: int = 1800
    graph_time_limit: int = 1860
    picture_desc_max_retries: int = 1
    picture_desc_retry_delay: int = 30
    picture_desc_soft_time_limit: int = 3600
    picture_desc_time_limit: int = 3660
    finalize_max_retries: int = 1
    finalize_retry_delay: int = 30
    finalize_soft_time_limit: int = 120
    finalize_time_limit: int = 180
    # Docling concurrency: max concurrent Docling conversions (Redis semaphore)
    docling_concurrency: int = 1
    # Lock timeout (auto-release if worker crashes)
    docling_lock_timeout: int = 4200
    # Health probe timeout (seconds)
    docling_health_timeout: float = 10.0
    # In-task 503 retry cap (Docling busy with prior conversion)
    docling_503_max_retries: int = 20

    # Celery Redis visibility timeout (seconds) — prevents redelivery of long tasks
    celery_visibility_timeout: int = 10800  # 3 hours

    # Singleflight lock timeout for prepare_document (seconds)
    prepare_singleflight_timeout: int = 5400  # 90 minutes

    # Directory watcher
    watch_dir_poll_interval_seconds: int = 30

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

    # Hop penalty and bonuses
    retrieval_hop_penalty_base: float = 0.92
    retrieval_mil_id_bonus: float = 0.03

    # Legacy decay (fallback when chunk_links unavailable)
    retrieval_cross_modal_decay: float = 0.85
    retrieval_ontology_decay: float = 0.75

    # Retrieval diversity (content-level dedup)
    retrieval_diversity_oversample_factor: int = 8
    retrieval_diversity_max_candidates: int = 800


@lru_cache
def get_settings() -> Settings:
    return Settings()
