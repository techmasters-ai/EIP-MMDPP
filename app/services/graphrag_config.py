"""Build Microsoft GraphRAG configuration from application settings."""

import logging
from pathlib import Path

from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.local_search_config import LocalSearchConfig
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

    # Pass num_ctx so Ollama allocates the correct context window.
    # Without this, Ollama uses the model's built-in default which may
    # be too small for GraphRAG's community report prompts.
    ollama_call_args = {}
    if settings.graphrag_llm_provider == "ollama":
        ollama_call_args["num_ctx"] = settings.ollama_num_ctx

    chat_model = ModelConfig(
        model_provider="openai",
        model=settings.graphrag_llm_model,
        api_key=api_key,
        api_base=api_base,
        call_args=ollama_call_args,
    )

    embedding_model = ModelConfig(
        model_provider="openai",
        model=settings.graphrag_embedding_model,
        api_key=api_key,
        api_base=api_base,
    )

    # Reserve tokens for the system prompt and LLM generation output;
    # the rest is the data-context budget GraphRAG can fill.
    ctx_tokens = int(settings.ollama_num_ctx * 0.75)

    config = GraphRagConfig(
        completion_models={"default_completion_model": chat_model},
        embedding_models={"default_embedding_model": embedding_model},
        output_storage=StorageConfig(base_dir=str(output_dir)),
        cache=CacheConfig(storage=StorageConfig(base_dir=str(cache_dir))),
        reporting=ReportingConfig(type="file", base_dir=str(data_dir / "logs")),
        vector_store=VectorStoreConfig(
            db_uri=str(output_dir / "lancedb"),
            vector_size=settings.text_embedding_dim,
        ),
        cluster_graph=ClusterGraphConfig(
            max_cluster_size=settings.graphrag_max_cluster_size,
        ),
        local_search=LocalSearchConfig(
            max_context_tokens=ctx_tokens,
        ),
        drift_search=DRIFTSearchConfig(
            data_max_tokens=ctx_tokens,
            primer_llm_max_tokens=ctx_tokens,
            local_search_max_data_tokens=ctx_tokens,
            # Default proportions (community=0.1 + text_unit=0.9 = 1.0) leave
            # zero budget for entity/relationship context, causing every
            # iteration to hit "Reached token limit". Rebalance to match
            # local search proportions.
            local_search_community_prop=0.15,
            local_search_text_unit_prop=0.55,
            # The primer sends all community reports split across N folds.
            # Default 5 folds with 162 reports = ~32 reports/fold = ~35K tokens,
            # which overflows the context window. More folds = smaller batches.
            primer_folds=20,
        ),
    )

    logger.info(
        "GraphRAG config built: provider=%s, model=%s, data_dir=%s",
        settings.graphrag_llm_provider,
        settings.graphrag_llm_model,
        data_dir,
    )
    return config
