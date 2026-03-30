"""Microsoft GraphRAG integration -- indexing, search, and prompt tuning.

GraphRAG owns the full extraction pipeline: chunking, entity/relationship
extraction (LLM), community detection, report generation, and embeddings.
The bridge layer only exports documents from Postgres as input.
"""

import asyncio
import logging
import shutil
from pathlib import Path

import pandas as pd

# graphrag_llm calls nest_asyncio2.apply() at import time, which fails under
# uvloop (uvicorn's default loop).  Neutralise it before any graphrag import.
try:
    import nest_asyncio2
    nest_asyncio2.apply = lambda *a, **kw: None  # type: ignore[assignment]
except ImportError:
    pass

import litellm

from app.config import get_settings
from app.services.graphrag_bridge import export_all
from app.services.graphrag_config import build_graphrag_config
from app.services.graphrag_prompts import write_prompt_files

logger = logging.getLogger(__name__)

# Set LiteLLM global timeout from config (GraphRAG uses LiteLLM for all LLM calls)
litellm.request_timeout = float(get_settings().graphrag_llm_timeout)


def _run_async(coro):
    """Run an async coroutine from a sync context (threadpool thread).

    Creates a fresh event loop to avoid conflicts with uvloop in the main
    thread.  asyncio.run() fails under uvloop because it cannot patch the
    running loop type.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Monkey-patch graphrag_llm embeddings to sanitize Unicode before Ollama
# ---------------------------------------------------------------------------

_EMBED_SANITIZE = str.maketrans({
    "\u2011": "-", "\u2010": "-", "\u2012": "-",
    "\u2013": "-", "\u2014": "-", "\u202f": " ", "\u00a0": " ",
})


def _patch_embedding_sanitization():
    """Wrap LiteLLMEmbedding.embedding_async to strip chars that cause NaN.

    Primary fix: OLLAMA_FLASH_ATTENTION=false on the Ollama server
    (BGE-M3's BERT architecture auto-enables Flash Attention since Ollama
    v0.13.5, causing F32->F16 overflow -> NaN in embeddings).

    This patch is defense-in-depth: NFC normalize + replace known
    problematic Unicode characters before sending to Ollama.
    """
    try:
        import unicodedata
        from graphrag_llm.embedding.lite_llm_embedding import LiteLLMEmbedding

        _original = LiteLLMEmbedding.embedding_async

        def _sanitize(text: str) -> str:
            return unicodedata.normalize("NFC", text).translate(_EMBED_SANITIZE)

        async def _sanitized_embedding_async(self, /, **kwargs):
            if "input" in kwargs:
                inp = kwargs["input"]
                if isinstance(inp, list):
                    kwargs["input"] = [_sanitize(t) if isinstance(t, str) else t for t in inp]
                elif isinstance(inp, str):
                    kwargs["input"] = _sanitize(inp)
            return await _original(self, **kwargs)

        LiteLLMEmbedding.embedding_async = _sanitized_embedding_async
        logger.info("Patched LiteLLMEmbedding.embedding_async with Unicode sanitization")
    except Exception as e:
        logger.warning("Failed to patch LiteLLMEmbedding: %s", e)


_patch_embedding_sanitization()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def run_graphrag_indexing(neo4j_driver, db_session) -> dict:
    """Run the full GraphRAG indexing pipeline.

    1. Export documents from Postgres to input/
    2. Run GraphRAG: chunking, extraction, communities, reports, embeddings
    3. Returns stats dict with communities_created and reports_generated

    neo4j_driver is accepted for API compatibility but no longer used —
    GraphRAG owns entity/relationship extraction.
    """
    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"communities_created": 0, "reports_generated": 0}

    try:
        data_dir = Path(settings.graphrag_data_dir)
        output_dir = data_dir / "output"
        input_dir = data_dir / "input"

        # Step 1: Bridge -- export documents to input/ (separate from
        # GraphRAG's output/ which it owns for index artifacts)
        stats = export_all(db_session, input_dir)
        if stats.get("documents", 0) == 0:
            logger.info("No documents found -- skipping GraphRAG indexing")
            return {"communities_created": 0, "reports_generated": 0}

        # Ensure prompts are written
        prompts_dir = data_dir / "prompts"
        write_prompt_files(prompts_dir)

        # Step 2: Run GraphRAG pipeline
        result = _run_graphrag_pipeline(settings, data_dir, output_dir, input_dir)
        return result

    except Exception:
        logger.exception("GraphRAG indexing failed")
        return {"communities_created": 0, "reports_generated": 0}


def _run_graphrag_pipeline(settings, data_dir: Path, output_dir: Path, input_dir: Path) -> dict:
    """Run GraphRAG indexing pipeline with ontology-guided extraction."""
    from graphrag.api import build_index
    from graphrag.config.enums import IndexingMethod

    config = build_graphrag_config(settings)

    # Load documents exported by bridge as input
    docs_path = input_dir / "documents.parquet"
    if docs_path.exists():
        docs_df = pd.read_parquet(docs_path)
        input_docs = pd.DataFrame({
            "id": docs_df["id"],
            "text": docs_df["text"],
            "title": docs_df["title"],
        })
    else:
        input_docs = None

    # First run: no GraphRAG-produced index in output/ yet → full build
    # Subsequent runs: GraphRAG's own entities.parquet exists → incremental update
    # Note: output/ is exclusively owned by GraphRAG; input/ is the bridge export
    #
    # GraphRAG's _get_method() appends "-update" when is_update_run=True,
    # so we always pass IndexingMethod.Standard (or Fast) and let
    # is_update_run control whether it becomes "standard-update".
    has_existing_index = (output_dir / "entities.parquet").exists()

    use_fast = settings.graphrag_use_fast_method
    base_method = IndexingMethod.Fast if use_fast else IndexingMethod.Standard

    logger.info(
        "GraphRAG indexing mode: %s%s (entities.parquet exists=%s)",
        "fast" if use_fast else "standard",
        "-update" if has_existing_index else "",
        has_existing_index,
    )

    # Backup output before update run (#13: failure recovery)
    backup_dir = data_dir / "output_backup"
    if has_existing_index:
        try:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(output_dir, backup_dir)
            logger.info("Backed up output/ to output_backup/ before update")
        except Exception:
            logger.warning("Failed to backup output/ before update", exc_info=True)

    # Logging callback so we can see exactly which workflow is running
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks as _WFCallbacks

    class _LoggingCallbacks:
        def pipeline_start(self, names):
            logger.info("GraphRAG pipeline starting: %d workflows: %s", len(names), names)

        def pipeline_end(self, results):
            errors = [r for r in results if r.error]
            logger.info("GraphRAG pipeline finished: %d workflows, %d errors", len(results), len(errors))

        def workflow_start(self, name, instance):
            logger.info(">>> GraphRAG workflow STARTED: %s", name)

        def workflow_end(self, name, instance):
            logger.info("<<< GraphRAG workflow ENDED: %s", name)

        def progress(self, progress):
            if progress.total_items and progress.total_items > 0:
                logger.info("    GraphRAG progress: %s — %s/%s",
                            progress.description or "?",
                            progress.completed_items, progress.total_items)

        def pipeline_error(self, error):
            logger.error("GraphRAG pipeline error: %s", error)

    try:
        results = _run_async(build_index(
            config=config,
            method=base_method,
            is_update_run=has_existing_index,
            verbose=True,
            input_documents=input_docs,
            callbacks=[_LoggingCallbacks()],
        ))
    except Exception:
        # Restore backup on failure (#13)
        if has_existing_index and backup_dir.exists():
            logger.warning("Indexing failed, restoring output/ from backup")
            try:
                shutil.rmtree(output_dir)
                shutil.copytree(backup_dir, output_dir)
            except Exception:
                logger.exception("Failed to restore output/ from backup")
        raise

    has_errors = False
    for result in results:
        if result.error:
            logger.warning(
                "GraphRAG workflow %s error: %s",
                result.workflow, result.error,
            )
            has_errors = True
        else:
            logger.info("GraphRAG workflow %s completed", result.workflow)

    # If workflows had errors and this was an update, restore backup
    if has_errors and has_existing_index and backup_dir.exists():
        logger.warning("Update had errors, restoring output/ from backup")
        try:
            shutil.rmtree(output_dir)
            shutil.copytree(backup_dir, output_dir)
        except Exception:
            logger.exception("Failed to restore output/ from backup")

    # Sanitize all output parquets — bge-m3 produces NaN on non-breaking
    # hyphens (U+2011) and similar Unicode chars that gpt-oss generates.
    _sanitize_output_parquets(output_dir)

    # Count outputs
    communities_created = 0
    reports_generated = 0
    communities_path = output_dir / "communities.parquet"
    reports_path = output_dir / "community_reports.parquet"
    if communities_path.exists():
        communities_created = len(pd.read_parquet(communities_path))
    if reports_path.exists():
        reports_generated = len(pd.read_parquet(reports_path))

    # Clean up backup on success
    if not has_errors and backup_dir.exists():
        try:
            shutil.rmtree(backup_dir)
        except Exception:
            pass

    return {
        "communities_created": communities_created,
        "reports_generated": reports_generated,
    }


def _sanitize_output_parquets(output_dir: Path) -> None:
    """Sanitize Unicode chars in all output parquets that cause NaN in bge-m3."""
    replacements = str.maketrans({
        "\u2011": "-", "\u2010": "-", "\u2012": "-",
        "\u2013": "-", "\u2014": "-", "\u202f": " ", "\u00a0": " ",
    })

    for path in output_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(path)
            changed = False
            for col in df.columns:
                if df[col].dtype == object:
                    new_col = df[col].apply(
                        lambda v: v.translate(replacements) if isinstance(v, str) else v
                    )
                    if not new_col.equals(df[col]):
                        df[col] = new_col
                        changed = True
            if changed:
                df.to_parquet(path, index=False)
                logger.info("Sanitized Unicode in %s", path.name)
        except Exception as e:
            logger.warning("Failed to sanitize %s: %s", path.name, e)


# ---------------------------------------------------------------------------
# Search -- all four methods
# ---------------------------------------------------------------------------


_UNICODE_SANITIZE = _EMBED_SANITIZE


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Unicode chars that cause NaN in bge-m3 embeddings."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: v.translate(_UNICODE_SANITIZE) if isinstance(v, str) else v
            )
    return df


def _load_search_data(settings) -> dict:
    """Load indexed Parquet data for search queries."""
    output_dir = Path(settings.graphrag_data_dir) / "output"

    data = {}
    for name in (
        "entities", "communities", "community_reports",
        "text_units", "relationships", "covariates",
    ):
        path = output_dir / f"{name}.parquet"
        if path.exists():
            data[name] = _sanitize_df(pd.read_parquet(path))
        else:
            data[name] = pd.DataFrame()

    return data


def _run_local_search(config, data: dict, query: str, community_level: int,
                      response_type: str, dynamic_community_selection: bool):
    """Run GraphRAG local search."""
    from graphrag.api import local_search as graphrag_local

    covariates = data["covariates"] if not data["covariates"].empty else None

    return _run_async(graphrag_local(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        text_units=data["text_units"],
        relationships=data["relationships"],
        covariates=covariates,
        community_level=community_level,
        response_type=response_type,
        query=query,
    ))


def _run_global_search(config, data: dict, query: str, community_level: int,
                       response_type: str, dynamic_community_selection: bool):
    """Run GraphRAG global search."""
    from graphrag.api import global_search as graphrag_global

    return _run_async(graphrag_global(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        community_level=community_level,
        dynamic_community_selection=dynamic_community_selection,
        response_type=response_type,
        query=query,
    ))


def _run_drift_search(config, data: dict, query: str, community_level: int,
                      response_type: str):
    """Run GraphRAG DRIFT search."""
    from graphrag.api import drift_search as graphrag_drift

    return _run_async(graphrag_drift(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        text_units=data["text_units"],
        relationships=data["relationships"],
        community_level=community_level,
        response_type=response_type,
        query=query,
    ))


def _run_basic_search(config, data: dict, query: str, response_type: str):
    """Run GraphRAG basic search."""
    from graphrag.api import basic_search as graphrag_basic

    return _run_async(graphrag_basic(
        config=config,
        text_units=data["text_units"],
        response_type=response_type,
        query=query,
    ))


def _serialize_context(context):
    """Convert GraphRAG context (may contain DataFrames) to JSON-serializable form."""
    if isinstance(context, pd.DataFrame):
        return context.to_dict(orient="records")
    if isinstance(context, dict):
        return {k: _serialize_context(v) for k, v in context.items()}
    if isinstance(context, list):
        return [_serialize_context(item) for item in context]
    if context is None:
        return None
    return context


def local_search(query: str) -> dict:
    """Entity-centric search with community context."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        if data["communities"].empty or data["community_reports"].empty:
            return {"response": "", "context": {}, "error": "communities_not_indexed"}
        response, context = _run_local_search(
            config, data, query, settings.graphrag_community_level,
            settings.graphrag_local_response_type,
            settings.graphrag_dynamic_community_selection,
        )
        return {"response": response, "context": _serialize_context(context)}
    except Exception:
        logger.exception("GraphRAG local search failed")
        return {"response": "", "context": {}}


def global_search(query: str) -> dict:
    """Cross-community summarization for broad questions."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        if data["communities"].empty or data["community_reports"].empty:
            return {"response": "", "context": {}, "error": "communities_not_indexed"}
        response, context = _run_global_search(
            config, data, query, settings.graphrag_community_level,
            settings.graphrag_global_response_type,
            settings.graphrag_dynamic_community_selection,
        )
        return {"response": response, "context": _serialize_context(context)}
    except Exception:
        logger.exception("GraphRAG global search failed")
        return {"response": "", "context": {}}


def drift_search(query: str) -> dict:
    """Community-informed expansion search."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        if data["communities"].empty or data["community_reports"].empty:
            return {"response": "", "context": {}, "error": "communities_not_indexed"}
        response, context = _run_drift_search(
            config, data, query, settings.graphrag_community_level,
            settings.graphrag_drift_response_type,
        )
        return {"response": response, "context": _serialize_context(context)}
    except Exception:
        logger.exception("GraphRAG DRIFT search failed")
        return {"response": "", "context": {}}


def basic_search(query: str) -> dict:
    """Vector search over text units."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_basic_search(
            config, data, query, settings.graphrag_basic_response_type,
        )
        return {"response": response, "context": _serialize_context(context)}
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

        entity_prompt, entity_summary_prompt, community_prompt = _run_async(
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
        (prompts_dir / "extract_graph.txt").write_text(entity_prompt)
        (prompts_dir / "entity_summarization.txt").write_text(
            entity_summary_prompt
        )
        (prompts_dir / "community_report.txt").write_text(community_prompt)

        logger.info("GraphRAG auto-tuning complete: 3 prompts updated")
        return {"prompts_updated": 3}
    except Exception:
        logger.exception("GraphRAG auto-tuning failed")
        return {"prompts_updated": 0, "error": True}
