"""Unit tests for application configuration.

Tests defaults, computed URLs, retrieval weight constraints, and caching.
"""

import pytest

pytestmark = pytest.mark.unit


class TestSettingsDefaults:
    def test_app_env_default(self):
        from app.config import Settings
        # In test env, APP_ENV is set to 'test' via conftest
        s = Settings()
        assert s.app_env in ("development", "test", "production")

    def test_arcadedb_url_default(self):
        from app.config import Settings
        s = Settings()
        assert s.arcadedb_url == "http://arcadedb:2480"

    def test_community_report_model_default(self, monkeypatch):
        from app.config import Settings
        s = Settings(_env_file=None)
        assert isinstance(s.community_report_llm_model, str)


class TestComputedUrls:
    def test_async_database_url_computed_from_parts(self):
        from app.config import Settings
        s = Settings(database_url="", postgres_user="u", postgres_password="p",
                     postgres_host="h", postgres_port=5432, postgres_db="d")
        assert s.async_database_url == "postgresql+asyncpg://u:p@h:5432/d"

    def test_async_database_url_explicit_override(self):
        from app.config import Settings
        s = Settings(database_url="postgresql+asyncpg://explicit:url@host/db")
        assert s.async_database_url == "postgresql+asyncpg://explicit:url@host/db"

    def test_sync_database_url_computed_from_parts(self):
        from app.config import Settings
        s = Settings(database_url_sync="", postgres_user="u", postgres_password="p",
                     postgres_host="h", postgres_port=5432, postgres_db="d")
        assert s.sync_database_url == "postgresql+psycopg2://u:p@h:5432/d"

    def test_sync_database_url_explicit_override(self):
        from app.config import Settings
        s = Settings(database_url_sync="postgresql+psycopg2://explicit:url@host/db")
        assert s.sync_database_url == "postgresql+psycopg2://explicit:url@host/db"


class TestRetrievalWeights:
    def test_fusion_weights_sum_to_one(self):
        from app.config import Settings
        s = Settings()
        total = s.retrieval_semantic_weight + s.retrieval_doc_structure_weight + s.retrieval_ontology_weight
        assert abs(total - 1.0) < 0.001

    def test_hop_penalty_between_zero_and_one(self):
        from app.config import Settings
        s = Settings()
        assert 0 < s.retrieval_hop_penalty_base < 1


class TestRerankerConfig:
    def test_new_reranker_config_defaults(self):
        """Reranker config fields should exist with correct defaults."""
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
        assert s.reranker_device in ("cpu", "cuda")
        assert s.reranker_enabled is True
        assert s.reranker_top_n == 20
        assert s.retrieval_min_score_threshold == 0.25

    def test_docling_graph_client_defaults(self):
        """Docling-Graph client settings should have correct defaults."""
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.docling_graph_base_url == "http://docling-graph:8002"
        assert s.docling_graph_concurrency == 2
        assert s.docling_graph_timeout == 1800  # 30 min per LLM call — Celery soft-limits already cap the full extraction run

    def test_ollama_settings_present(self):
        """Ollama settings should be present."""
        from app.config import Settings
        s = Settings(ollama_num_ctx=16384, ollama_base_url="http://localhost:11434")
        assert s.ollama_num_ctx == 16384
        assert s.ollama_base_url == "http://localhost:11434"


class TestArcadeDBConfig:
    def test_arcadedb_defaults(self):
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.arcadedb_url == "http://arcadedb:2480"
        assert s.arcadedb_database == "eip_knowledge_graph"

    def test_community_detection_defaults(self):
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.community_detection_enabled is True
        # algo.leiden in ArcadeDB 26.5.x returns null communityId — louvain
        # is the working default until upstream fixes leiden.
        assert s.community_detection_algorithm == "louvain"


class TestGetSettingsCaching:
    def test_returns_same_instance(self):
        from app.config import get_settings
        a = get_settings()
        b = get_settings()
        assert a is b


# --- Task 5.2: graph_extraction_engine removed (spec §7.5 cleanup) ---------

class TestGraphExtractionEngineRemoved:
    def test_field_does_not_exist(self):
        """graph_extraction_engine was deleted in Task 5.2; Settings.extra=ignore
        means env values for this key are silently dropped."""
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert not hasattr(s, "graph_extraction_engine")

    def test_default_ontology_bundle_key(self):
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.default_ontology_bundle_key == "air_defense_v3"


# --- Task 3.6: IngestDispatchResult dataclass (spec §5.2) ---------------

class TestIngestDispatchResult:
    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError
        from app.workers.dispatch_types import IngestDispatchResult

        r = IngestDispatchResult(pipeline_run_id="run-1", celery_task_id="task-1")
        assert r.pipeline_run_id == "run-1"
        assert r.celery_task_id == "task-1"
        with pytest.raises(FrozenInstanceError):
            r.pipeline_run_id = "run-2"  # type: ignore[misc]

    def test_requires_both_fields(self):
        from app.workers.dispatch_types import IngestDispatchResult

        # Both fields are required
        with pytest.raises(TypeError):
            IngestDispatchResult(pipeline_run_id="run-1")  # type: ignore[call-arg]


# --- Task 3: dispatch ledger v1 settings (spec 2026-05-10) ---------------

class TestDispatchLedgerSettings:
    def test_max_stage_dispatches_default(self):
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.max_stage_dispatches == 5

    def test_stale_dispatched_threshold_default(self):
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.stale_dispatched_threshold_seconds == 600
