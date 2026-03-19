"""Microsoft GraphRAG integration -- indexing, search, and prompt tuning.

Replaces the custom community detection/report implementation with
Microsoft's GraphRAG library. Uses the Neo4j ontology graph built by
Docling-Graph as input via the bridge layer.
"""

import asyncio
import logging
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.services.graphrag_bridge import export_all
from app.services.graphrag_config import build_graphrag_config
from app.services.graphrag_prompts import write_prompt_files

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def run_graphrag_indexing(neo4j_driver, db_session) -> dict:
    """Run the full GraphRAG indexing pipeline.

    1. Export Neo4j entities/relationships + Postgres text units to Parquet
    2. Run GraphRAG community detection + report generation + embeddings
    3. Returns stats dict with communities_created and reports_generated
    """
    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"communities_created": 0, "reports_generated": 0}

    try:
        data_dir = Path(settings.graphrag_data_dir)
        output_dir = data_dir / "output"

        # Step 1: Bridge -- export ontology graph to Parquet
        stats = export_all(neo4j_driver, db_session, output_dir)
        if stats.get("entities", 0) == 0:
            logger.info("No entities in graph -- skipping GraphRAG indexing")
            return {"communities_created": 0, "reports_generated": 0}

        # Ensure prompts are written
        prompts_dir = data_dir / "prompts"
        write_prompt_files(prompts_dir)

        # Step 2: Run GraphRAG pipeline
        result = _run_graphrag_pipeline(settings, data_dir, output_dir)
        return result

    except Exception:
        logger.exception("GraphRAG indexing failed")
        return {"communities_created": 0, "reports_generated": 0}


def _run_graphrag_pipeline(settings, data_dir: Path, output_dir: Path) -> dict:
    """Run Microsoft GraphRAG's indexing pipeline on pre-exported Parquet data."""
    from graphrag.api import build_index
    from graphrag.config.enums import IndexingMethod

    config = build_graphrag_config(settings)

    # Load pre-exported text as input documents for GraphRAG
    text_units_path = output_dir / "text_units.parquet"
    if text_units_path.exists():
        text_df = pd.read_parquet(text_units_path)
        input_docs = pd.DataFrame({
            "id": text_df["id"],
            "text": text_df["text"],
            "title": text_df["id"],
        })
    else:
        input_docs = None

    # Run indexing
    results = asyncio.run(build_index(
        config=config,
        method=IndexingMethod.Standard,
        is_update_run=False,
        verbose=True,
        input_documents=input_docs,
    ))

    communities_created = 0
    reports_generated = 0

    for result in results:
        if result.error:
            logger.warning(
                "GraphRAG workflow %s error: %s",
                result.workflow, result.error,
            )
        else:
            logger.info("GraphRAG workflow %s completed", result.workflow)

    # Count outputs
    communities_path = output_dir / "communities.parquet"
    reports_path = output_dir / "community_reports.parquet"
    if communities_path.exists():
        communities_created = len(pd.read_parquet(communities_path))
    if reports_path.exists():
        reports_generated = len(pd.read_parquet(reports_path))

    return {
        "communities_created": communities_created,
        "reports_generated": reports_generated,
    }


# ---------------------------------------------------------------------------
# Search -- all four methods
# ---------------------------------------------------------------------------


def _load_search_data(settings) -> dict:
    """Load indexed Parquet data for search queries."""
    output_dir = Path(settings.graphrag_data_dir) / "output"

    data = {}
    for name in (
        "entities", "communities", "community_reports",
        "text_units", "relationships",
    ):
        path = output_dir / f"{name}.parquet"
        if path.exists():
            data[name] = pd.read_parquet(path)
        else:
            data[name] = pd.DataFrame()

    return data


def _run_local_search(config, data: dict, query: str, community_level: int):
    """Run GraphRAG local search."""
    from graphrag.api import local_search as graphrag_local

    return asyncio.run(graphrag_local(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        text_units=data["text_units"],
        relationships=data["relationships"],
        covariates=None,
        community_level=community_level,
        response_type="Detailed explanation",
        query=query,
    ))


def _run_global_search(config, data: dict, query: str, community_level: int):
    """Run GraphRAG global search."""
    from graphrag.api import global_search as graphrag_global

    return asyncio.run(graphrag_global(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        community_level=community_level,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query,
    ))


def _run_drift_search(config, data: dict, query: str, community_level: int):
    """Run GraphRAG DRIFT search."""
    from graphrag.api import drift_search as graphrag_drift

    return asyncio.run(graphrag_drift(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        text_units=data["text_units"],
        relationships=data["relationships"],
        community_level=community_level,
        response_type="In-depth analysis",
        query=query,
    ))


def _run_basic_search(config, data: dict, query: str):
    """Run GraphRAG basic search."""
    from graphrag.api import basic_search as graphrag_basic

    return asyncio.run(graphrag_basic(
        config=config,
        text_units=data["text_units"],
        response_type="Concise answer",
        query=query,
    ))


def local_search(query: str) -> dict:
    """Entity-centric search with community context."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_local_search(
            config, data, query, settings.graphrag_community_level,
        )
        return {"response": response, "context": context}
    except Exception:
        logger.exception("GraphRAG local search failed")
        return {"response": "", "context": {}}


def global_search(query: str) -> dict:
    """Cross-community summarization for broad questions."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_global_search(
            config, data, query, settings.graphrag_community_level,
        )
        return {"response": response, "context": context}
    except Exception:
        logger.exception("GraphRAG global search failed")
        return {"response": "", "context": {}}


def drift_search(query: str) -> dict:
    """Community-informed expansion search."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_drift_search(
            config, data, query, settings.graphrag_community_level,
        )
        return {"response": response, "context": context}
    except Exception:
        logger.exception("GraphRAG DRIFT search failed")
        return {"response": "", "context": {}}


def basic_search(query: str) -> dict:
    """Vector search over text units."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_basic_search(config, data, query)
        return {"response": response, "context": context}
    except Exception:
        logger.exception("GraphRAG basic search failed")
        return {"response": "", "context": {}}


# ---------------------------------------------------------------------------
# Prompt tuning
# ---------------------------------------------------------------------------


def run_auto_tune() -> dict:
    """Run GraphRAG auto prompt tuning against the current corpus."""
    try:
        settings = get_settings()
        data_dir = Path(settings.graphrag_data_dir)
        config = build_graphrag_config(settings)

        from graphrag.api.prompt_tune import generate_indexing_prompts

        entity_prompt, entity_summary_prompt, community_prompt = asyncio.run(
            generate_indexing_prompts(
                config=config,
                domain="military equipment, radar systems, missile systems, "
                       "electronic warfare, and air defense",
                language="English",
                discover_entity_types=True,
            )
        )

        # Write tuned prompts to prompts dir
        prompts_dir = data_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "entity_extraction.txt").write_text(entity_prompt)
        (prompts_dir / "entity_summarization.txt").write_text(
            entity_summary_prompt
        )
        (prompts_dir / "community_report.txt").write_text(community_prompt)

        logger.info("GraphRAG auto-tuning complete: 3 prompts updated")
        return {"prompts_updated": 3}
    except Exception:
        logger.exception("GraphRAG auto-tuning failed")
        return {"prompts_updated": 0, "error": True}
