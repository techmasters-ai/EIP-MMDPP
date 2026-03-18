"""Unit tests for GraphRAG configuration builder."""

import pytest

pytestmark = pytest.mark.unit


class TestGraphRAGSettings:
    def test_default_graphrag_settings(self):
        """New GraphRAG settings have correct defaults."""
        from app.config import Settings

        s = Settings(
            _env_file=None,
            postgres_password="test",
            neo4j_password="test",
        )
        assert s.graphrag_llm_provider == "ollama"
        assert s.graphrag_llm_model == "llama3.2"
        assert s.graphrag_llm_api_base == "http://localhost:11434/v1"
        assert s.graphrag_api_key == ""
        assert s.graphrag_embedding_model == "nomic-embed-text"
        assert s.graphrag_data_dir == "/app/graphrag_data"
        assert s.graphrag_community_level == 2
        assert s.graphrag_tune_interval_minutes == 1440
        assert s.graphrag_indexing_enabled is True
        assert s.graphrag_indexing_interval_minutes == 60
        assert s.graphrag_max_cluster_size == 10

    def test_openai_provider_config(self):
        """Settings accept OpenAI provider configuration."""
        from app.config import Settings

        s = Settings(
            _env_file=None,
            postgres_password="test",
            neo4j_password="test",
            graphrag_llm_provider="openai",
            graphrag_llm_model="gpt-4o",
            graphrag_api_key="sk-test",
            graphrag_embedding_model="text-embedding-3-small",
        )
        assert s.graphrag_llm_provider == "openai"
        assert s.graphrag_api_key == "sk-test"
