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
        assert s.graphrag_llm_api_base == ""
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


class TestBuildGraphRAGConfig:
    def test_builds_config_with_ollama(self, tmp_path):
        """Config builder creates valid GraphRAG config for Ollama."""
        from unittest.mock import MagicMock

        from app.services.graphrag_config import build_graphrag_config

        settings = MagicMock()
        settings.graphrag_llm_provider = "ollama"
        settings.graphrag_llm_model = "llama3.2"
        settings.graphrag_llm_api_base = "http://ollama:11434/v1"
        settings.graphrag_api_key = ""
        settings.graphrag_embedding_model = "nomic-embed-text"
        settings.graphrag_data_dir = str(tmp_path / "graphrag_data")
        settings.graphrag_community_level = 2
        settings.graphrag_max_cluster_size = 10

        config = build_graphrag_config(settings)
        assert config is not None

    def test_builds_config_with_openai(self, tmp_path):
        """Config builder creates valid GraphRAG config for OpenAI."""
        from unittest.mock import MagicMock

        from app.services.graphrag_config import build_graphrag_config

        settings = MagicMock()
        settings.graphrag_llm_provider = "openai"
        settings.graphrag_llm_model = "gpt-4o"
        settings.graphrag_llm_api_base = ""
        settings.graphrag_api_key = "sk-test"
        settings.graphrag_embedding_model = "text-embedding-3-small"
        settings.graphrag_data_dir = str(tmp_path / "graphrag_data")
        settings.graphrag_community_level = 2
        settings.graphrag_max_cluster_size = 10

        config = build_graphrag_config(settings)
        assert config is not None

    def test_prompts_dir_created(self, tmp_path):
        """Config builder creates prompts directory."""
        from unittest.mock import MagicMock

        from app.services.graphrag_config import build_graphrag_config

        settings = MagicMock()
        settings.graphrag_llm_provider = "ollama"
        settings.graphrag_llm_model = "llama3.2"
        settings.graphrag_llm_api_base = "http://ollama:11434/v1"
        settings.graphrag_api_key = ""
        settings.graphrag_embedding_model = "nomic-embed-text"
        settings.graphrag_data_dir = str(tmp_path / "graphrag_data")
        settings.graphrag_community_level = 2
        settings.graphrag_max_cluster_size = 10

        build_graphrag_config(settings)
        assert (tmp_path / "graphrag_data" / "prompts").is_dir()
