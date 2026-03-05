"""Cognee knowledge graph integration service.

Wraps Cognee's async API to provide a unified search interface that
returns results in EIP's QueryResultItem format.

Configuration is driven by LLM_PROVIDER env var:
- openai: uses OpenAI API (requires OPENAI_API_KEY)
- ollama: uses local Ollama server (requires OLLAMA_BASE_URL + COGNEE_MODEL)
- mock: skips Cognee entirely, returns [] (for tests / no-LLM environments)

Storage backends (no extra services needed):
- Graph: NetworkX (in-memory) or neo4j
- Vector: LanceDB (local file) or pgvector
"""

import asyncio
import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_configure_lock = asyncio.Lock()
_configured = False


async def _configure_cognee() -> None:
    """One-time Cognee configuration. Thread-safe via asyncio.Lock."""
    global _configured
    async with _configure_lock:
        if _configured:
            return

        settings = get_settings()

        if settings.llm_provider == "mock":
            _configured = True
            return

        try:
            import cognee

            # Configure vector and graph storage engines
            await cognee.config.set_vector_db_config(
                {"provider": settings.cognee_vector_engine,
                 "path": settings.cognee_data_dir}
            )
            await cognee.config.set_graph_db_config(
                {"provider": settings.cognee_graph_engine,
                 "path": settings.cognee_data_dir}
            )

            # Configure LLM provider
            if settings.llm_provider == "openai":
                await cognee.config.set_llm_config(
                    {
                        "provider": "openai",
                        "api_key": settings.openai_api_key,
                    }
                )
            elif settings.llm_provider == "ollama":
                await cognee.config.set_llm_config(
                    {
                        "provider": "ollama",
                        "endpoint": settings.ollama_base_url,
                        "model": settings.cognee_model,
                    }
                )

            _configured = True
            logger.info(
                "Cognee configured: llm_provider=%s graph=%s vector=%s",
                settings.llm_provider,
                settings.cognee_graph_engine,
                settings.cognee_vector_engine,
            )

        except Exception as exc:
            logger.warning("Cognee configuration failed: %s", exc)
            # Mark configured to prevent retry storms; service degrades to []
            _configured = True


def _result_to_text(result: Any) -> str | None:
    """Extract displayable text from a heterogeneous Cognee search result."""
    if result is None:
        return None
    if isinstance(result, str):
        return result.strip() or None
    # DataPoint / chunk objects expose .text or .payload / .chunk_text
    for attr in ("text", "chunk_text", "content", "summary", "answer"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str):
            return val.strip() or None
    # Some results are dicts
    if isinstance(result, dict):
        for key in ("text", "chunk_text", "content", "summary", "answer"):
            val = result.get(key)
            if val and isinstance(val, str):
                return val.strip() or None
    return None


def _result_to_score(result: Any, fallback: float = 0.5) -> float:
    """Extract similarity score from a Cognee result, defaulting to fallback."""
    for attr in ("score", "similarity", "distance"):
        val = getattr(result, attr, None)
        if val is not None:
            try:
                score = float(val)
                # distance → similarity (lower distance = higher similarity)
                if attr == "distance":
                    score = max(0.0, 1.0 - score)
                return min(1.0, max(0.0, score))
            except (TypeError, ValueError):
                pass
    if isinstance(result, dict):
        for key in ("score", "similarity"):
            val = result.get(key)
            if val is not None:
                try:
                    return min(1.0, max(0.0, float(val)))
                except (TypeError, ValueError):
                    pass
    return fallback


async def cognee_search(query: str, top_k: int = 10) -> list:
    """Run Cognee search and return results as QueryResultItem list.

    Runs GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, and SUMMARIES in parallel.
    Deduplicates by content text. Catches all exceptions — returns [] on failure.
    """
    from app.schemas.retrieval import QueryResultItem

    await _configure_cognee()

    if get_settings().llm_provider == "mock":
        return []

    try:
        import cognee
        from cognee.api.v1.search.search import SearchType
    except ImportError:
        logger.warning("Cognee package not installed — returning empty results")
        return []

    search_types = [
        (SearchType.GRAPH_COMPLETION, "graph_node", 0.75),
        (SearchType.RAG_COMPLETION, "text", 0.65),
        (SearchType.CHUNKS, "text", 0.55),
        (SearchType.SUMMARIES, "text", 0.50),
    ]

    async def _run_single(search_type, modality, score_fallback):
        try:
            raw = await cognee.search(query, query_type=search_type)
            return raw, modality, score_fallback
        except Exception as exc:
            logger.debug("Cognee %s search failed: %s", search_type.name, exc)
            return [], modality, score_fallback

    tasks = [_run_single(st, mod, sf) for st, mod, sf in search_types]
    search_results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_texts: set[str] = set()
    items: list[QueryResultItem] = []

    for outcome in search_results:
        if isinstance(outcome, Exception):
            continue
        raw_list, modality, score_fallback = outcome
        if not isinstance(raw_list, (list, tuple)):
            raw_list = [raw_list] if raw_list else []
        for result in raw_list:
            text = _result_to_text(result)
            if not text:
                continue
            # Deduplicate on exact content
            if text in seen_texts:
                continue
            seen_texts.add(text)
            items.append(
                QueryResultItem(
                    score=_result_to_score(result, score_fallback),
                    modality=modality,
                    content_text=text,
                    page_number=None,
                    classification="UNCLASSIFIED",
                    context=None,
                )
            )

    # Sort descending by score and cap at top_k
    items.sort(key=lambda x: x.score, reverse=True)
    return items[:top_k]


async def cognee_add(text: str, dataset_name: str) -> None:
    """Add a text artifact to Cognee's dataset for later cognify."""
    await _configure_cognee()

    if get_settings().llm_provider == "mock":
        return

    try:
        import cognee
        await cognee.add(text, dataset_name=dataset_name)
        logger.debug("cognee_add: added %d chars to dataset=%s", len(text), dataset_name)
    except Exception as exc:
        logger.warning("cognee_add failed (dataset=%s): %s", dataset_name, exc)


async def cognee_cognify(dataset_name: str) -> None:
    """Run Cognee's knowledge graph construction on a dataset."""
    await _configure_cognee()

    if get_settings().llm_provider == "mock":
        return

    try:
        import cognee
        await cognee.cognify(datasets=[dataset_name])
        logger.info("cognee_cognify: completed for dataset=%s", dataset_name)
    except Exception as exc:
        logger.warning("cognee_cognify failed (dataset=%s): %s", dataset_name, exc)
