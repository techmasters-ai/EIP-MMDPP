"""Build Microsoft GraphRAG configuration from application settings."""

import logging
from pathlib import Path

from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.reporting_config import ReportingConfig
from graphrag_cache.cache_config import CacheConfig
from graphrag_llm.config.model_config import ModelConfig
from graphrag_storage.storage_config import StorageConfig
from graphrag_vectors.vector_store_config import VectorStoreConfig

logger = logging.getLogger(__name__)


def build_graphrag_config(settings) -> GraphRagConfig:
    """Build a GraphRagConfig from application settings.

    Supports both Ollama (via OpenAI-compatible API) and native OpenAI.
    GraphRAG v3 uses LiteLLM as the backend, with model_provider selecting
    the routing (openai for both real OpenAI and Ollama's compatible API).
    """
    data_dir = Path(settings.graphrag_data_dir)
    output_dir = data_dir / "output"
    prompts_dir = data_dir / "prompts"
    cache_dir = data_dir / "cache"

    for d in (output_dir, prompts_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Both Ollama and OpenAI use the "openai" model_provider in LiteLLM.
    # Ollama exposes an OpenAI-compatible API at /v1/chat/completions.
    if settings.graphrag_llm_provider == "ollama":
        api_key = settings.graphrag_api_key or "ollama"
        api_base = settings.graphrag_llm_api_base
    else:
        api_key = settings.graphrag_api_key
        api_base = settings.graphrag_llm_api_base or None

    chat_model = ModelConfig(
        model_provider="openai",
        model=settings.graphrag_llm_model,
        api_key=api_key,
        api_base=api_base,
    )

    embedding_model = ModelConfig(
        model_provider="openai",
        model=settings.graphrag_embedding_model,
        api_key=api_key,
        api_base=api_base,
    )

    config = GraphRagConfig(
        completion_models={"default_completion_model": chat_model},
        embedding_models={"default_embedding_model": embedding_model},
        output_storage=StorageConfig(base_dir=str(output_dir)),
        cache=CacheConfig(storage=StorageConfig(base_dir=str(cache_dir))),
        reporting=ReportingConfig(type="file", base_dir=str(data_dir / "logs")),
        vector_store=VectorStoreConfig(db_uri=str(output_dir / "lancedb")),
        cluster_graph=ClusterGraphConfig(
            max_cluster_size=settings.graphrag_max_cluster_size,
        ),
    )

    logger.info(
        "GraphRAG config built: provider=%s, model=%s, data_dir=%s",
        settings.graphrag_llm_provider,
        settings.graphrag_llm_model,
        data_dir,
    )
    return config
