"""Microsoft GraphRAG integration -- indexing, search, and prompt tuning.

GraphRAG owns the full extraction pipeline: chunking, entity/relationship
extraction (LLM), community detection, report generation, and embeddings.
The bridge layer only exports documents from Postgres as input.
"""

import asyncio
import logging
import shutil
import threading
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
_graphrag_settings = get_settings()
litellm.request_timeout = float(_graphrag_settings.graphrag_llm_timeout)

# Override Ollama thinking level for GraphRAG (falls back to global ollama_think).
# "medium" is recommended: "high" causes intermittent empty-content responses
# on GraphRAG's structured JSON rating prompts.
_graphrag_think = _graphrag_settings.graphrag_ollama_think or _graphrag_settings.ollama_think
if _graphrag_think:
    try:
        from litellm.llms.ollama.common_utils import OllamaConfig as _OllamaConfig
        _OllamaConfig.think = _graphrag_think
        logger.info("GraphRAG: set OllamaConfig.think=%s", _graphrag_think)
    except ImportError:
        logger.debug("OllamaConfig not available — skipping think override")


_LOOP_LOCAL = threading.local()


def _get_graphrag_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for the current worker thread.

    Reusing a single loop per thread keeps LiteLLM's background logging
    coroutines alive instead of destroying them after every request.
    """
    loop = getattr(_LOOP_LOCAL, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _LOOP_LOCAL.loop = loop
    return loop


def _run_async(coro):
    """Run an async coroutine from a sync context (worker thread).

    Uses a persistent per-thread event loop to avoid destroying LiteLLM's
    background logging tasks on every call.
    """
    return _get_graphrag_loop().run_until_complete(coro)


def close_graphrag_loop():
    """Shut down the persistent event loop for the current thread.

    Call this from Celery worker_process_shutdown to cleanly drain
    LiteLLM's background logging tasks.
    """
    loop = getattr(_LOOP_LOCAL, "loop", None)
    if not loop or loop.is_closed():
        return
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    try:
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:
        pass
    loop.close()
    _LOOP_LOCAL.loop = None


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
# DRIFT primer: replace strict JSON parsing with tolerant multi-strategy
# ---------------------------------------------------------------------------

from app.services.graphrag_runtime_patches import install_graphrag_runtime_patches

install_graphrag_runtime_patches()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def run_graphrag_indexing(neo4j_driver, db_session, *, mode: str = "auto") -> dict:
    """Run the GraphRAG indexing pipeline.

    Modes (matching the GraphRAG CLI commands):
      - "index":  Full rebuild from scratch (like ``graphrag index``).
                  Replaces the entire existing index.
      - "update": Incremental update only (like ``graphrag update``).
                  Processes only new/modified documents via delta detection.
                  Halts immediately when there is nothing new.
      - "auto":   Auto-detect — uses "update" when a previous index exists,
                  otherwise falls back to "index".  Used by the scheduled task.

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
        result = _run_graphrag_pipeline(settings, data_dir, output_dir, input_dir, mode=mode)
        return result

    except Exception:
        logger.exception("GraphRAG indexing failed")
        return {"communities_created": 0, "reports_generated": 0}


def _patch_concat_dataframes():
    """Fix graphrag#update_final_documents 'arange: cannot compute length'.

    When old_df is empty or human_readable_id contains NaN,
    old_df["human_readable_id"].max() returns NaN, which makes
    np.arange(NaN, ...) fail.  Patch to default initial_id to 0.
    """
    import numpy as np
    from graphrag.index.update import incremental_index

    async def _safe_concat(name, previous_table_provider, delta_table_provider,
                           output_table_provider):
        old_df = await previous_table_provider.read_dataframe(name)
        delta_df = await delta_table_provider.read_dataframe(name)

        max_id = old_df["human_readable_id"].max() if len(old_df) > 0 else -1
        initial_id = int(max_id + 1) if not pd.isna(max_id) else 0
        delta_df["human_readable_id"] = np.arange(
            initial_id, initial_id + len(delta_df),
        )
        final_df = pd.concat([old_df, delta_df], ignore_index=True, copy=False)
        await output_table_provider.write_dataframe(name, final_df)
        return final_df

    incremental_index.concat_dataframes = _safe_concat


_patch_concat_dataframes()


def _run_graphrag_pipeline(settings, data_dir: Path, output_dir: Path, input_dir: Path,
                           *, mode: str = "auto") -> dict:
    """Run GraphRAG indexing pipeline with ontology-guided extraction.

    mode controls is_update_run, matching the GraphRAG CLI:
      - "index":  is_update_run=False  (like ``graphrag index``)
      - "update": is_update_run=True   (like ``graphrag update``)
      - "auto":   detect from existing index state
    """
    from graphrag.api import build_index
    from graphrag.config.enums import IndexingMethod

    config = build_graphrag_config(settings)

    has_existing_index = (output_dir / "entities.parquet").exists()

    # Resolve mode → is_update_run, matching the GraphRAG CLI:
    #   graphrag index  → is_update_run=False (full rebuild)
    #   graphrag update → is_update_run=True  (incremental)
    if mode == "index":
        is_update_run = False
    elif mode == "update":
        is_update_run = True
    else:  # "auto" — scheduled task path
        is_update_run = has_existing_index

    use_fast = settings.graphrag_use_fast_method
    base_method = IndexingMethod.Fast if use_fast else IndexingMethod.Standard

    logger.info(
        "GraphRAG indexing: mode=%s, method=%s%s (entities.parquet exists=%s)",
        mode,
        "fast" if use_fast else "standard",
        "-update" if is_update_run else "",
        has_existing_index,
    )

    # Backup output before update run (#13: failure recovery)
    backup_dir = data_dir / "output_backup"
    if is_update_run and has_existing_index:
        try:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(output_dir, backup_dir)
            logger.info("Backed up output/ to output_backup/ before update")
        except Exception:
            logger.warning("Failed to backup output/ before update", exc_info=True)

    # Logging callback so we can see exactly which workflow is running
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

    # Do NOT pass input_documents — let GraphRAG load from input_storage
    # (CSV files in input/) via its own load_input_documents or
    # load_update_documents workflow. This matches the CLI behavior:
    #   graphrag index  → load_input_documents (full build)
    #   graphrag update → load_update_documents → get_delta_docs() → stop=True if empty
    # Passing input_documents would bypass delta detection entirely.
    try:
        results = _run_async(build_index(
            config=config,
            method=base_method,
            is_update_run=is_update_run,
            verbose=True,
            callbacks=[_LoggingCallbacks()],
        ))
    except Exception:
        # Restore backup on failure (#13)
        if is_update_run and backup_dir.exists():
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
    if has_errors and is_update_run and backup_dir.exists():
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
        "documents",
    ):
        path = output_dir / f"{name}.parquet"
        if path.exists():
            data[name] = _sanitize_df(pd.read_parquet(path))
        else:
            data[name] = pd.DataFrame()

    return data


def _run_local_search(config, data: dict, query: str, community_level: int,
                      response_type: str):
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
        )
        return {
            "response": response,
            "context": _serialize_context(context),
        }
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
        return {
            "response": response,
            "context": _serialize_context(context),
        }
    except Exception:
        logger.exception("GraphRAG global search failed")
        return {"response": "", "context": {}}


def drift_search(query: str) -> dict:
    """Community-informed expansion search."""
    from app.services.graphrag_runtime_patches import DriftPrimerParseError

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
        return {
            "response": response,
            "context": _serialize_context(context),
        }
    except DriftPrimerParseError as exc:
        logger.exception("GraphRAG DRIFT primer parse failed")
        return {
            "response": "",
            "context": {},
            "error": "drift_primer_parse_failed",
            "error_detail": str(exc),
        }
    except Exception:
        logger.exception("GraphRAG DRIFT search failed")
        return {
            "response": "",
            "context": {},
            "error": "drift_search_failed",
            "error_detail": "Internal DRIFT search error; check worker logs.",
        }


def basic_search(query: str) -> dict:
    """Vector search over text units."""
    try:
        settings = get_settings()
        config = build_graphrag_config(settings)
        data = _load_search_data(settings)
        response, context = _run_basic_search(
            config, data, query, settings.graphrag_basic_response_type,
        )
        return {
            "response": response,
            "context": _serialize_context(context),
        }
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
