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

    def test_neo4j_uri_default(self):
        from app.config import Settings
        s = Settings()
        assert s.neo4j_uri == "bolt://neo4j:7687"

    def test_qdrant_url_default(self):
        from app.config import Settings
        s = Settings()
        assert s.qdrant_url == "http://qdrant:6333"

    def test_graphrag_defaults(self):
        from app.config import Settings
        s = Settings()
        assert s.graphrag_indexing_enabled is True
        assert s.graphrag_max_cluster_size == 10
        assert s.graphrag_model == "llama3.2"


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


class TestGetSettingsCaching:
    def test_returns_same_instance(self):
        from app.config import get_settings
        a = get_settings()
        b = get_settings()
        assert a is b
