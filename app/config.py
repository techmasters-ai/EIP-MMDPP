import json
import logging
from functools import lru_cache
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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

    # LLM provider — controls which LLM backend is used for community reports
    # and other LLM-dependent features (openai | ollama | mock).
    llm_provider: str = "ollama"

    # Ollama connection
    # Singular fallbacks (kept for back-compat with existing .env files).
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_base_url: str = ""   # chat/reasoning models — falls back to ollama_base_url
    ollama_vlm_base_url: str = ""   # vision/image models — falls back to ollama_base_url
    ollama_embedding_base_url: str = ""  # embedding models — falls back to ollama_base_url
    # Plural pools — JSON arrays as raw strings, e.g.
    # '["http://host1:11434","http://host2:11434"]'.
    # Stored as `str` (not `list[str]`) because pydantic-settings raises
    # SettingsError on blank-string values for list[str] before any
    # validator runs. Parsed in the get_ollama_*_urls() helpers below.
    # When set, these take precedence over the singular variants.
    ollama_llm_base_urls: str = ""
    ollama_vlm_base_urls: str = ""
    ollama_embedding_base_urls: str = ""
    # Per-function URL pools (Chunk 6, NEW). Each function gets its own
    # pool so the operator can route different LLM functions to different
    # banks of Ollama instances. When set, function-specific takes
    # precedence over role-level pools above. Cascade depth is 4-tier
    # for chat functions and embedding/vlm:
    #   DOC_ANALYSIS_LLM_BASE_URLS     > OLLAMA_LLM_BASE_URLS > OLLAMA_LLM_BASE_URL > OLLAMA_BASE_URL
    #   TRANSLATION_LLM_BASE_URLS      > OLLAMA_LLM_BASE_URLS > OLLAMA_LLM_BASE_URL > OLLAMA_BASE_URL
    #   COMMUNITY_REPORT_LLM_BASE_URLS > OLLAMA_LLM_BASE_URLS > OLLAMA_LLM_BASE_URL > OLLAMA_BASE_URL
    #   PICTURE_DESCRIPTION_BASE_URLS  > OLLAMA_VLM_BASE_URLS > OLLAMA_VLM_BASE_URL > OLLAMA_BASE_URL
    #   TEXT_EMBEDDING_BASE_URLS       > OLLAMA_EMBEDDING_BASE_URLS > OLLAMA_EMBEDDING_BASE_URL > OLLAMA_BASE_URL
    doc_analysis_llm_base_urls: str = ""
    translation_llm_base_urls: str = ""
    community_report_llm_base_urls: str = ""
    picture_description_base_urls: str = ""
    text_embedding_base_urls: str = ""
    ollama_num_ctx: int = 16384

    @staticmethod
    def _normalize_ollama_think(value: str) -> str | bool | None:
        normalized = (value or "").strip().lower()
        if not normalized:
            return None
        if normalized in {"false", "off", "disabled"}:
            return False
        if normalized in {"true", "on", "enabled"}:
            return True
        if normalized in {"low", "medium", "high"}:
            return normalized
        return None

    @classmethod
    def _resolve_ollama_think(cls, value: str, model: str) -> str | bool | None:
        normalized = cls._normalize_ollama_think(value)
        if normalized in {"low", "medium", "high"} and "gpt-oss" not in (model or "").lower():
            return None
        return normalized

    @staticmethod
    @lru_cache(maxsize=16)
    def _parse_url_pool(
        raw: str, field_name: str = "OLLAMA_*_BASE_URLS",
    ) -> list[str]:
        """Parse a JSON-array env value into list[str].

        Returns [] for blank/unset. Raises ValueError for malformed JSON
        or non-array values so misconfiguration fails loudly at startup
        rather than silently falling through to the singular fallback.
        Logs a warning on duplicate URLs (OllamaPool dedupes internally,
        but the operator should know their config is wrong).

        ``field_name`` is interpolated into error / log messages so the
        operator sees the actual offending env var (e.g.
        ``DOC_ANALYSIS_LLM_BASE_URLS``) rather than the wildcard.

        Cached via ``lru_cache(maxsize=16)`` keyed on ``(raw, field_name)``
        — pure function, side-effect is the duplicate-URL warning which
        we want emitted exactly once per misconfiguration. The cache is
        module-scoped, so it survives across ``Settings()`` instances and
        keeps the warning from firing twice when both function and role
        pools share the same JSON value.
        """
        s = (raw or "").strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{field_name} env value is not valid JSON: {exc}; "
                f"expected a JSON array like '[\"http://h1:11434\",...]', "
                f"got: {s!r}"
            ) from exc
        if not isinstance(parsed, list) or not all(
            isinstance(x, str) for x in parsed
        ):
            raise ValueError(
                f"{field_name} must be a JSON array of strings; got: {parsed!r}"
            )
        # Reject blank entries — they'd silently break the pool's
        # least-in-flight invariant (an empty URL would still get acquire'd
        # and posted to, raising httpx.UnsupportedProtocol on every request).
        if not all(x.strip() for x in parsed):
            raise ValueError(
                f"{field_name} contains blank entries; got: {parsed!r}"
            )
        # Warn on duplicates — OllamaPool dedupes silently, but the operator
        # almost certainly meant distinct URLs (defeats fan-out otherwise).
        if len(set(parsed)) != len(parsed):
            counts: dict[str, int] = {}
            for x in parsed:
                counts[x] = counts.get(x, 0) + 1
            dupes = sorted(x for x, c in counts.items() if c > 1)
            logger.warning(
                "%s contains duplicate URLs %s; "
                "OllamaPool will dedupe but you likely meant distinct URLs.",
                field_name, dupes,
            )
        return parsed

    def get_ollama_llm_urls(self) -> list[str]:
        """Return the LLM (chat/reasoning) URL pool.

        Priority: ollama_llm_base_urls (plural JSON) > ollama_llm_base_url
        (singular) > ollama_base_url. Always returns a non-empty list.
        """
        plural = self._parse_url_pool(
            self.ollama_llm_base_urls, "OLLAMA_LLM_BASE_URLS",
        )
        if plural:
            return plural
        if self.ollama_llm_base_url:
            return [self.ollama_llm_base_url]
        return [self.ollama_base_url]

    def get_ollama_vlm_urls(self) -> list[str]:
        plural = self._parse_url_pool(
            self.ollama_vlm_base_urls, "OLLAMA_VLM_BASE_URLS",
        )
        if plural:
            return plural
        if self.ollama_vlm_base_url:
            return [self.ollama_vlm_base_url]
        return [self.ollama_base_url]

    def get_ollama_embedding_urls(self) -> list[str]:
        plural = self._parse_url_pool(
            self.ollama_embedding_base_urls, "OLLAMA_EMBEDDING_BASE_URLS",
        )
        if plural:
            return plural
        if self.ollama_embedding_base_url:
            return [self.ollama_embedding_base_url]
        return [self.ollama_base_url]

    # Back-compat singular getters: return urls[0] from the new pools.
    def get_ollama_llm_url(self) -> str:
        return self.get_ollama_llm_urls()[0]

    def get_ollama_vlm_url(self) -> str:
        return self.get_ollama_vlm_urls()[0]

    def get_ollama_embedding_url(self) -> str:
        return self.get_ollama_embedding_urls()[0]

    # ----- Per-function pool helpers (Chunk 6) -----
    # Each falls through to its role-level helper via composition; the 4-tier
    # cascade (function → role → singular → base) is realized by chaining
    # rather than re-implementing the singular/base fallback per function.
    def get_doc_analysis_llm_urls(self) -> list[str]:
        plural = self._parse_url_pool(
            self.doc_analysis_llm_base_urls, "DOC_ANALYSIS_LLM_BASE_URLS",
        )
        if plural:
            return plural
        return self.get_ollama_llm_urls()

    def get_translation_llm_urls(self) -> list[str]:
        plural = self._parse_url_pool(
            self.translation_llm_base_urls, "TRANSLATION_LLM_BASE_URLS",
        )
        if plural:
            return plural
        return self.get_ollama_llm_urls()

    def get_community_report_llm_urls(self) -> list[str]:
        plural = self._parse_url_pool(
            self.community_report_llm_base_urls, "COMMUNITY_REPORT_LLM_BASE_URLS",
        )
        if plural:
            return plural
        return self.get_ollama_llm_urls()

    def get_picture_description_urls(self) -> list[str]:
        plural = self._parse_url_pool(
            self.picture_description_base_urls, "PICTURE_DESCRIPTION_BASE_URLS",
        )
        if plural:
            return plural
        return self.get_ollama_vlm_urls()

    def get_text_embedding_urls(self) -> list[str]:
        plural = self._parse_url_pool(
            self.text_embedding_base_urls, "TEXT_EMBEDDING_BASE_URLS",
        )
        if plural:
            return plural
        return self.get_ollama_embedding_urls()

    llm_max_tokens: int = 64000

    # --- Docling-Graph service (entity/relationship extraction) ---
    docling_graph_base_url: str = "http://docling-graph:8002"
    docling_graph_concurrency: int = 2
    # Per-call HTTP timeout for docling-graph /extract-pass. Each pass fans
    # out to many small LLM calls inside docling-graph; total wall-time scales
    # with doc size. 72000s (20h) accommodates field-rich passes on 90+ page
    # PDFs where schema-grammar constrained decoding can run multi-hour per
    # pass on mid-size models like gemma4:31b.
    # Note: DOCLING_GRAPH_LLM_TIMEOUT is consumed by the separate docling-graph
    # service — update via docker-compose.yml fallback and
    # docker/docling-graph/app/config_builder.py.
    docling_graph_timeout: int = 72000
    # Max attempts per extraction pass (spec §6.5). Default 3 means 1 initial
    # call + 2 retries before giving up.
    pass_max_retries: int = 3
    # Max transport-level retries per pass attempt (connection drops, name
    # resolution failures, mid-stream disconnects from docling-graph). Infra
    # failures must NOT consume the business ``pass_max_retries`` budget.
    # Default 3 (30+60+120s backoff = 3.5min per pass) covers a typical
    # docker-compose restart of docling-graph. Higher caps risk blowing the
    # 8h ``graph_soft_time_limit`` when 12 passes each retry the maximum.
    pass_max_transport_retries: int = 3
    # Per-pass Celery task configuration (Task 5 — per-pass fan-in architecture).
    # Soft time limit per individual pass task (seconds). Passed directly to the
    # Celery task decorator — SoftTimeLimitExceeded fires at this threshold.
    pass_soft_time_limit: int = 3600
    # Max concurrent entity passes per document. _try_advance_phase enforces this:
    # it only dispatches the next entity pass when in_flight < this cap.
    pass_concurrency_per_document: int = 2
    # Stale phase reclaim threshold in seconds. A phase stuck in 'claimed' or
    # 'dispatched' longer than this is eligible for reconciler reclaim (Task 9).
    phase_claim_stale_seconds: int = 30
    # Reconciler beat period in seconds. The Task-9 celery beat schedules the
    # reconciler to run at this interval.
    reconciler_period_seconds: int = 60
    # Quality-gate floor shared with the docling-graph service. Mirrors
    # DoclingGraphSettings.docling_graph_quality_min_instances so compose
    # can propagate a single DOCLING_GRAPH_QUALITY_MIN_INSTANCES env var
    # to both worker and service. The service applies per-pass overrides
    # (system_links=1) internally; see docker/docling-graph/app/config_builder.py.
    docling_graph_quality_min_instances: int = 3

    # Docling human-review confidence threshold
    docling_review_confidence_threshold: float = 0.60

    # Graph confidence quality gates
    graph_node_min_confidence: float = 0.60
    graph_rel_min_confidence: float = 0.55
    # Validation matrix enforcement: when true, relationships whose
    # (source_type, rel_type, target_type) triple is not in the active
    # ontology's validation_matrix are rejected (raise ValueError).  When
    # false (default), invalid triples are logged and skipped silently.
    graph_reject_invalid_relationships: bool = False

    # Default bundle for brand-new documents when neither the caller nor
    # the Source overrides. Spec §4.5 bundle selection threading + §7.4.
    # Kept as a Settings field (not a module-level constant) so ops can
    # override via env var DOCLING_GRAPH_DEFAULT_ONTOLOGY_BUNDLE_KEY in
    # a different deployment profile.
    default_ontology_bundle_key: str = "air_defense_v3"

    # Governance: when true, graph mutation patches require two distinct
    # curator approvals before they can be applied.  When false (default),
    # a single approval is sufficient.
    governance_dual_approval_required: bool = False

    # ArcadeDB
    arcadedb_url: str = "http://arcadedb:2480"
    arcadedb_user: str = "root"
    arcadedb_password: str = "eip_arcadedb_secret"
    arcadedb_database: str = "eip_knowledge_graph"
    arcadedb_ready_cache_seconds: float = 60.0
    arcadedb_slow_query_threshold_seconds: float = 1.0
    # efSearch controls vector search recall vs latency (ArcadeDB Manual §4.14.7).
    # Higher = more accurate, slower. 0 = use ArcadeDB's adaptive default.
    # Set explicitly only for consistently high recall or strict latency.
    arcadedb_vector_ef_search: int = 0

    # Community detection
    community_detection_enabled: bool = True
    community_detection_interval_minutes: int = 60
    community_detection_post_ingest_enabled: bool = True
    community_detection_post_ingest_threshold: int = 5
    # algo.leiden in ArcadeDB 26.5.x silently returns NULL communityId for
    # every node — it doesn't actually assign communities. louvain works.
    community_detection_algorithm: str = "louvain"
    community_detection_resolution: float = 1.0
    community_detection_max_iterations: int = 20
    community_report_llm_model: str = "llama3.2"
    community_report_llm_think: str = ""  # true|false for most models; low|medium|high only for gpt-oss
    community_report_llm_prompt: str = ""
    community_global_synthesis_prompt: str = ""

    # Reranker (cross-encoder for retrieval re-scoring)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"  # cpu | cuda
    reranker_enabled: bool = True
    reranker_top_n: int = 20

    # Minimum cosine similarity threshold (below this, results are dropped)
    retrieval_min_score_threshold: float = 0.25

    # Query defaults (exposed via /v1/settings/retrieval for frontend)
    query_default_top_k: int = 20
    query_default_min_confidence: float = 0.1

    # Embedding-pipeline chunking (Pass A — worker-embed → bge-m3).
    # Independent of the extraction-pipeline chunking, which lives in
    # docling-graph and is configured via DOCLING_GRAPH_CHUNK_MAX_TOKENS.
    embedding_chunk_max_tokens: int = 512
    embedding_chunk_overlap_tokens: int = 64
    embedding_chunk_tokenizer_model: str = "BAAI/bge-m3"  # HuggingFace repo for HybridChunker tokenizer

    # Embedding batching
    embed_text_batch_size: int = 128

    # Document Analysis (LLM metadata extraction)
    doc_analysis_enabled: bool = True
    doc_analysis_llm_model: str = "gpt-oss:120b"
    doc_analysis_llm_think: str = ""  # true|false for most models; low|medium|high only for gpt-oss
    doc_analysis_timeout: int = 1500  # HTTP client timeout for individual LLM calls
    # Per-call HTTP timeout for community-report + global-synthesis chats
    # (built by get_community_report_client). Defaults to the same value
    # as doc_analysis_timeout to preserve historical behavior; set
    # COMMUNITY_REPORT_TIMEOUT to decouple from doc-analysis tuning.
    community_report_timeout: int = 1500
    # Celery soft/hard time limits for derive_document_metadata. Previously derived
    # inline as doc_analysis_timeout+60/+120; now explicit so env controls directly.
    doc_analysis_soft_time_limit: int = 1800
    doc_analysis_time_limit: int = 2400
    doc_analysis_summary_prompt: str = "Summarize this document in 3-5 sentences for a technical reader. Focus on the main subject, scope, and notable findings. Do not include source endnote markings such as [1]."
    doc_analysis_date_prompt: str = "Extract the most relevant date of information (publication date, report date, or coverage window). If only month/year appears, return that. If there is a range, return it exactly. If unsure, return Unknown. Provide ONLY the date or range."
    doc_analysis_source_prompt: str = "Characterize the source: 1) Organization or author (if unknown return UNKNOWN) 2) Type of information (website, journal, etc; if unknown return UNKNOWN) 3) Reliability score 1-10. Format: Organization: <name>\\nType: <type>\\nReliability: <score>/10"
    doc_analysis_classification_prompt: str = "Identify the document classification marking if present (UNCLASSIFIED, CUI, FOUO, SECRET, TOP SECRET). If none, reply UNCLASSIFIED. Provide ONLY the marking."

    # Picture Description (post-conversion enrichment via Ollama)
    picture_description_model: str = "gemma3:27b"
    picture_description_think: str = ""  # true|false for most models; low|medium|high only for gpt-oss
    picture_description_timeout: int = 300
    picture_description_prompt: str = "Analyze this image from a multi-modal PDF using the required narrative sections and the missile/radar/S&T emphasis. Return sections 1-8 exactly as specified. Use the PDF Summary for context but rely on visual evidence.\\n\\n- PDF Summary: {document_summary}\\n\\n- Image:"

    # Translation (foreign language detection + LLM translation)
    translation_enabled: bool = True
    translation_model: str = "llama3.3:70b"
    translation_think: str = ""  # true|false for most models; low|medium|high only for gpt-oss
    translation_timeout: int = 180
    translation_prompt: str = "Translate ALL non-English text to English. The input may contain a mix of English and non-English text — translate every non-English word or phrase to English while keeping English portions unchanged. Do NOT skip non-English text even if it appears alongside English. Preserve all markdown formatting including headings (#), bullet points, tables, and code blocks. Keep technical designators like model numbers (С-75, 9М38) in their original form, but translate descriptive words around them (e.g., 'Зенитный Ракетный Комплекс С-75' → 'Anti-aircraft Missile System С-75'). Preserve all numbers, units, and acronyms. Preserve ---ELEMENT_BOUNDARY--- markers exactly as they appear. Return only the translated text with no commentary."
    translation_min_detect_length: int = 5
    translation_soft_time_limit: int = 7200
    translation_time_limit: int = 8100

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
    prepare_soft_time_limit: int = 3600
    prepare_time_limit: int = 4500
    embed_max_retries: int = 10
    embed_retry_delay: int = 60
    embed_soft_time_limit: int = 3600
    embed_time_limit: int = 4500
    graph_max_retries: int = 2
    graph_retry_delay: int = 60
    # Graph extraction is the slowest stage. 8h soft covers the 6h+ observed on
    # one doc in the 2026-04-24 run; 9h hard SIGKILL ceiling.
    graph_soft_time_limit: int = 28800
    graph_time_limit: int = 32400
    picture_desc_max_retries: int = 1
    picture_desc_retry_delay: int = 30
    picture_desc_soft_time_limit: int = 18000
    picture_desc_time_limit: int = 21600
    # Max concurrent VLM calls per document during derive_picture_descriptions.
    # Should not exceed OLLAMA_NUM_PARALLEL on the model server.
    picture_desc_concurrency: int = 5
    finalize_max_retries: int = 1
    finalize_retry_delay: int = 30
    finalize_soft_time_limit: int = 600
    finalize_time_limit: int = 900
    # Max age (seconds) a stage_run row can sit at status='RUNNING' before the
    # periodic sweeper marks it FAILED. Must be larger than max(*_time_limit)
    # so the sweeper only fires after Celery's own timeout should already have
    # killed + retried. max time_limit = graph_time_limit = 32400, so 34200.
    stale_stage_run_threshold_seconds: int = 34200
    # Max chain-level sweeper-triggered retries per document. Beyond this, the
    # document is permanently FAILED for operator triage.
    max_doc_retry_count: int = 3
    # Docling concurrency: max concurrent Docling conversions (Redis semaphore)
    docling_concurrency: int = 1
    # Lock timeout (auto-release if worker crashes)
    docling_lock_timeout: int = 4200
    # Health probe timeout (seconds)
    docling_health_timeout: float = 10.0
    # In-task 503 retry cap (Docling busy with prior conversion)
    docling_503_max_retries: int = 20

    # Celery Redis visibility timeout (seconds) — prevents redelivery of long tasks
    celery_visibility_timeout: int = 36000  # 10h — must exceed longest Celery time_limit

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

    def get_doc_analysis_llm_think(self) -> str | bool | None:
        return self._resolve_ollama_think(self.doc_analysis_llm_think, self.doc_analysis_llm_model)

    def get_picture_description_think(self) -> str | bool | None:
        return self._resolve_ollama_think(self.picture_description_think, self.picture_description_model)

    def get_translation_think(self) -> str | bool | None:
        return self._resolve_ollama_think(self.translation_think, self.translation_model)

    def get_community_report_llm_think(self) -> str | bool | None:
        return self._resolve_ollama_think(
            self.community_report_llm_think,
            self.community_report_llm_model,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
