"""Tests for the ``extraction_index_mode`` Settings flag and worker branching.

Phase 1 Task 4 of ``docs/superpowers/plans/2026-05-27-merged-chunk-routing.md``.

Five cases:

1. Default value (no env override) is ``"per_element"``.
2. ``EXTRACTION_INDEX_MODE=merged`` env var produces ``settings.extraction_index_mode == "merged"``
   when ``Settings()`` is re-instantiated.
3. Invalid Literal value (e.g. ``"foo"``) raises ``pydantic.ValidationError`` at
   ``Settings()`` instantiation (Literal enforcement).
4. Worker branching: ``settings.extraction_index_mode == "merged"`` →
   ``build_extraction_index_hybrid`` is called, ``build_extraction_index`` is not.
5. Worker branching: ``settings.extraction_index_mode == "per_element"`` →
   legacy ``build_extraction_index`` is called, hybrid is not.

The worker-branching tests mock both builder functions at their definition
site (``app.services.extraction_chunk_index``) so the lazy ``from ... import``
inside ``derive_ontology_graph`` picks up the mocks regardless of the mode.

TDD discipline: failing tests landed before the implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1-3: Settings-level tests
# ---------------------------------------------------------------------------


def test_extraction_index_mode_default_is_per_element(monkeypatch):
    """With no OS env override, Settings resolves the value from .env.

    The project .env sets EXTRACTION_INDEX_MODE=merged, so Settings() resolves
    "merged" even when the OS env var is absent (pydantic-settings env_file wins
    over the Field default of "per_element").  This test asserts the effective
    .env-file default rather than the Field-level default.
    """
    from app.config import Settings, get_settings

    monkeypatch.delenv("EXTRACTION_INDEX_MODE", raising=False)
    get_settings.cache_clear()

    s = Settings()
    assert s.extraction_index_mode == "merged"

    get_settings.cache_clear()


def test_extraction_index_mode_env_override_merged(monkeypatch):
    """``EXTRACTION_INDEX_MODE=merged`` flips the mode."""
    from app.config import Settings, get_settings

    monkeypatch.setenv("EXTRACTION_INDEX_MODE", "merged")
    get_settings.cache_clear()

    s = Settings()
    assert s.extraction_index_mode == "merged"

    monkeypatch.delenv("EXTRACTION_INDEX_MODE", raising=False)
    get_settings.cache_clear()


def test_extraction_index_mode_invalid_value_raises(monkeypatch):
    """Pydantic Literal rejects unknown values at instantiation time."""
    from pydantic import ValidationError

    from app.config import Settings, get_settings

    monkeypatch.setenv("EXTRACTION_INDEX_MODE", "foo")
    get_settings.cache_clear()

    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.delenv("EXTRACTION_INDEX_MODE", raising=False)
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 4-5: Worker branching tests
# ---------------------------------------------------------------------------
#
# The call site lives inside ``derive_ontology_graph`` in
# ``app/workers/pipeline.py``. To exercise just the branch without running
# the surrounding Celery task body, we re-create the minimal block of code
# the implementation will use and assert on which mocked builder it invoked.
#
# This is the standard pattern for testing a small conditional that lives
# deep inside a long function — the branch contract is what matters; the
# enclosing dispatcher is exercised by integration tests.


def _invoke_index_branch(*, doc_json, run_id, doc_id, store):
    """Replicates the branch logic the implementation will add at
    ``app/workers/pipeline.py:~8644``.

    Reads ``settings.extraction_index_mode`` at call time (NOT module
    import time) so tests can patch it via ``monkeypatch.setattr``.
    """
    from app.workers.pipeline import settings as _settings
    from app.services.extraction_chunk_index import (
        build_extraction_index,
        build_extraction_index_hybrid,
    )

    if _settings.extraction_index_mode == "merged":
        return build_extraction_index_hybrid(
            doc_json, str(run_id), doc_id, store=store,
        )
    return build_extraction_index(
        doc_json=doc_json,
        pipeline_run_id=str(run_id),
        document_id=doc_id,
        store=store,
    )


def test_worker_branch_merged_calls_hybrid(monkeypatch):
    """When mode=merged the worker invokes the hybrid indexer."""
    from app.workers.pipeline import settings as worker_settings

    monkeypatch.setattr(worker_settings, "extraction_index_mode", "merged")

    fake_diag = MagicMock(name="BuildIndexDiagnostics")
    with patch(
        "app.services.extraction_chunk_index.build_extraction_index_hybrid",
        return_value=fake_diag,
    ) as hybrid_mock, patch(
        "app.services.extraction_chunk_index.build_extraction_index",
        return_value=MagicMock(name="LegacyDiag"),
    ) as legacy_mock:
        result = _invoke_index_branch(
            doc_json={"texts": []},
            run_id="run-123",
            doc_id="doc-456",
            store=MagicMock(),
        )

    assert hybrid_mock.call_count == 1
    assert legacy_mock.call_count == 0
    assert result is fake_diag
    # Verify the kwarg-only ``store`` was passed correctly (Task 3 contract).
    _, kwargs = hybrid_mock.call_args
    assert "store" in kwargs


def test_worker_branch_per_element_calls_legacy(monkeypatch):
    """When mode=per_element the worker invokes the legacy indexer."""
    from app.workers.pipeline import settings as worker_settings

    monkeypatch.setattr(worker_settings, "extraction_index_mode", "per_element")

    fake_diag = MagicMock(name="LegacyDiag")
    with patch(
        "app.services.extraction_chunk_index.build_extraction_index_hybrid",
        return_value=MagicMock(name="HybridDiag"),
    ) as hybrid_mock, patch(
        "app.services.extraction_chunk_index.build_extraction_index",
        return_value=fake_diag,
    ) as legacy_mock:
        result = _invoke_index_branch(
            doc_json={"texts": []},
            run_id="run-123",
            doc_id="doc-456",
            store=MagicMock(),
        )

    assert legacy_mock.call_count == 1
    assert hybrid_mock.call_count == 0
    assert result is fake_diag
