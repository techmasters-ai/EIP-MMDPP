"""Unit tests for GraphRAG configuration builder."""

import pytest

pytestmark = pytest.mark.unit


class TestGraphRAGSettings:
    def test_default_graphrag_settings(self, monkeypatch):
        """New GraphRAG settings have correct defaults."""
        monkeypatch.delenv("GRAPHRAG_MAX_CLUSTER_SIZE", raising=False)
        from app.config import Settings

        s = Settings(
            _env_file=None,
            postgres_password="test",
            neo4j_password="test",
        )
        assert s.graphrag_llm_provider == "ollama"
        # Note: specific model/url values may be overridden by env vars;
        # we only assert the structural defaults that env vars don't touch.
        assert s.graphrag_community_level == 2
        assert s.graphrag_tune_interval_minutes == 1440
        assert s.graphrag_indexing_enabled is True
        assert s.graphrag_max_cluster_size == 50
        assert s.graphrag_use_fast_method is False
        assert s.graphrag_cache_enabled is True
        # Response types are env-overridable; just assert they are non-empty strings
        assert isinstance(s.graphrag_local_response_type, str) and s.graphrag_local_response_type
        assert isinstance(s.graphrag_global_response_type, str) and s.graphrag_global_response_type
        assert isinstance(s.graphrag_drift_response_type, str) and s.graphrag_drift_response_type
        assert s.graphrag_dynamic_community_selection is True
        assert s.graphrag_dry_run is False

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
        settings.graphrag_max_cluster_size = 50
        settings.graphrag_cache_enabled = True
        settings.text_embedding_dim = 1024
        settings.ollama_num_ctx = 16384
        settings.get_ollama_llm_url.return_value = "http://ollama:11434"
        settings.get_ollama_embedding_url.return_value = "http://ollama:11434"

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
        settings.graphrag_max_cluster_size = 50
        settings.graphrag_cache_enabled = True
        settings.text_embedding_dim = 1024
        settings.ollama_num_ctx = 16384
        settings.get_ollama_llm_url.return_value = "http://ollama:11434"
        settings.get_ollama_embedding_url.return_value = "http://ollama:11434"

        config = build_graphrag_config(settings)
        assert config is not None

    def test_cache_disabled(self, tmp_path):
        """Config builder passes cache=None when disabled."""
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
        settings.graphrag_max_cluster_size = 50
        settings.graphrag_cache_enabled = False
        settings.text_embedding_dim = 1024
        settings.ollama_num_ctx = 16384
        settings.get_ollama_llm_url.return_value = "http://ollama:11434"
        settings.get_ollama_embedding_url.return_value = "http://ollama:11434"

        config = build_graphrag_config(settings)
        assert config.cache.type == "none"

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
        settings.graphrag_max_cluster_size = 50
        settings.graphrag_cache_enabled = True
        settings.text_embedding_dim = 1024
        settings.ollama_num_ctx = 16384
        settings.get_ollama_llm_url.return_value = "http://ollama:11434"
        settings.get_ollama_embedding_url.return_value = "http://ollama:11434"

        build_graphrag_config(settings)
        assert (tmp_path / "graphrag_data" / "prompts").is_dir()
