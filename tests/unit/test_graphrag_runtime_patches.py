"""Unit tests for GraphRAG runtime patches."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# Stub graphrag modules if not available (same pattern as test_graphrag_query_task.py)
class _AutoStubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock


def _try_import(name: str) -> bool:
    top = name.split(".")[0]
    try:
        __import__(top)
        return True
    except ImportError:
        return False


for mod_name in [
    "pandas", "graphrag", "graphrag.api", "graphrag.api.prompt_tune",
    "graphrag.config", "graphrag.config.enums",
    "graphrag.config.models", "graphrag.config.models.cluster_graph_config",
    "graphrag.config.models.extract_graph_config",
    "graphrag.config.models.drift_search_config",
    "graphrag.config.models.graph_rag_config",
    "graphrag.config.models.local_search_config",
    "graphrag.config.models.reporting_config",
    "graphrag.query", "graphrag.query.structured_search",
    "graphrag.query.structured_search.drift_search",
    "graphrag.query.structured_search.drift_search.primer",
    "graphrag.prompts", "graphrag.prompts.query",
    "graphrag.prompts.query.drift_search_system_prompt",
    "litellm", "nest_asyncio2",
    "graphrag.index", "graphrag.index.update",
    "graphrag.index.update.incremental_index",
    "graphrag_llm", "graphrag_llm.config", "graphrag_llm.config.model_config",
    "graphrag_llm.embedding", "graphrag_llm.embedding.lite_llm_embedding",
    "graphrag_llm.tokenizer",
    "graphrag_cache", "graphrag_cache.cache_config",
    "graphrag_storage", "graphrag_storage.storage_config",
    "graphrag_vectors", "graphrag_vectors.vector_store_config",
    "lancedb",
]:
    if not _try_import(mod_name):
        sys.modules.setdefault(mod_name, _AutoStubModule(mod_name))


class TestDriftPrimerPatch:
    def test_patch_is_applied(self):
        """Verify that install_graphrag_runtime_patches replaces decompose_query."""
        from graphrag.query.structured_search.drift_search.primer import (
            DRIFTPrimer,
        )

        # Reset the patch flag so we can re-apply
        import app.services.graphrag_runtime_patches as patches_mod
        patches_mod._PATCHES_INSTALLED = False

        original = DRIFTPrimer.decompose_query

        with patch.object(patches_mod, "_assert_patch_target_compatible", return_value=True):
            patches_mod.install_graphrag_runtime_patches()

        # decompose_query should now be the patched version
        assert DRIFTPrimer.decompose_query is not original
        assert DRIFTPrimer.decompose_query.__name__ == "_patched_decompose_query"

    def test_patch_is_idempotent(self):
        """Second call to install_graphrag_runtime_patches should be a no-op."""
        import app.services.graphrag_runtime_patches as patches_mod

        # Force the flag to True
        patches_mod._PATCHES_INSTALLED = True

        with patch.object(patches_mod, "_patch_drift_primer") as mock_patch:
            patches_mod.install_graphrag_runtime_patches()
            mock_patch.assert_not_called()

    def test_patch_skipped_on_incompatible_signature(self):
        """Patch should skip if DRIFTPrimer signature is unexpected."""
        import app.services.graphrag_runtime_patches as patches_mod
        patches_mod._PATCHES_INSTALLED = False

        with patch.object(patches_mod, "_assert_patch_target_compatible", return_value=False):
            from graphrag.query.structured_search.drift_search.primer import (
                DRIFTPrimer,
            )
            original = DRIFTPrimer.decompose_query
            patches_mod.install_graphrag_runtime_patches()
            # Should NOT have been replaced
            assert DRIFTPrimer.decompose_query is original


class TestDriftPrimerParseError:
    def test_is_exception(self):
        from app.services.graphrag_runtime_patches import DriftPrimerParseError
        assert issubclass(DriftPrimerParseError, Exception)

    def test_message_preserved(self):
        from app.services.graphrag_runtime_patches import DriftPrimerParseError
        err = DriftPrimerParseError("bad json")
        assert str(err) == "bad json"
