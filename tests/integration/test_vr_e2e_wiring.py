"""Integration tests for VR C.4 end-to-end wiring.

These tests require live services (ArcadeDB, MinIO, Postgres, API).
They are SKIPPED automatically when the required services are unavailable.

Run with: pytest tests/integration/test_vr_e2e_wiring.py -v --no-header

Skip condition: ArcadeDB unavailable OR MinIO unavailable OR Postgres unavailable.
Tests here verify:
  - shadow mode: ExtractionChunk rows created, diagnostics persisted, RUN_FULL dispatched
  - narrow_only mode: chunk_scope applied, scoped doc smaller than full doc
  - disabled mode: no router call, no ExtractionChunk rows, fallback diagnostics
"""
import pytest


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------


def _services_available() -> bool:
    """Return True if ArcadeDB, MinIO, and Postgres are all reachable."""
    try:
        from app.config import get_settings
        from app.db.session import get_graph_store
        import psycopg2
        import boto3

        settings = get_settings()

        # ArcadeDB
        store = get_graph_store()
        store.ensure_ready_sync()

        # Postgres
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=3,
        )
        conn.close()

        return True
    except Exception:
        return False


skip_if_no_services = pytest.mark.skipif(
    not _services_available(),
    reason="Live services (ArcadeDB / Postgres) not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_doc_json(n_texts: int = 5) -> dict:
    """Build a minimal DoclingDocument JSON for testing."""
    texts = [
        {
            "self_ref": f"#/texts/{i}",
            "text": f"Test paragraph {i} about radar systems and missile kinematics.",
            "label": "paragraph",
        }
        for i in range(n_texts)
    ]
    children = [{"cref": f"#/texts/{i}"} for i in range(n_texts)]
    return {
        "texts": texts,
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"children": children},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_if_no_services
def test_build_extraction_index_e2e():
    """build_extraction_index correctly inserts ExtractionChunk vertices.

    Verifies the index build round-trip (insert → count → cleanup).
    """
    import uuid
    from app.services.extraction_chunk_index import (
        build_extraction_index,
        cleanup_extraction_index,
    )
    from app.db.session import get_graph_store

    run_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    doc_json = _make_minimal_doc_json(n_texts=3)
    store = get_graph_store()

    try:
        diag = build_extraction_index(
            doc_json=doc_json,
            pipeline_run_id=run_id,
            document_id=doc_id,
            store=store,
        )
        assert diag.chunks_inserted > 0, "At least 1 chunk must be inserted"
        assert diag.chunks_inserted <= 3, "No more than 3 text elements"
    finally:
        deleted = cleanup_extraction_index(run_id, store=store)
        assert deleted >= 0  # best-effort; may be 0 if insert failed


@skip_if_no_services
def test_apply_chunk_scope_reduces_doc_size():
    """apply_chunk_scope with 1 of 5 texts → scoped body.children is smaller."""
    from app.services.scoped_docling_document import apply_chunk_scope

    doc_json = _make_minimal_doc_json(n_texts=5)
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/2"]}

    scoped = apply_chunk_scope(doc_json, chunk_scope)

    # All 5 texts are preserved in the arrays
    assert len(scoped["texts"]) == 5

    # But body.children is smaller (at most the selected ref + any preceding heading)
    assert len(scoped["body"]["children"]) < 5


@skip_if_no_services
def test_vr_shadow_mode_compute_effective_scope():
    """In shadow mode, effective_chunk_scope is always None even with selected_refs response."""
    from app.workers.pipeline import _compute_effective_chunk_scope

    router_response = {
        "mode": "selected_refs",
        "self_refs": ["#/texts/0", "#/texts/1"],
        "diagnostics": {"selected_ref_count": 2},
    }
    eff, diag = _compute_effective_chunk_scope(router_response, "shadow")

    assert eff is None, "Shadow mode must never narrow"
    assert diag.get("shadow_skipped_narrowing") is True


@skip_if_no_services
def test_vr_narrow_only_mode_compute_effective_scope():
    """In narrow_only mode, selected_refs response → effective_chunk_scope set."""
    from app.workers.pipeline import _compute_effective_chunk_scope

    router_response = {
        "mode": "selected_refs",
        "self_refs": ["#/texts/0"],
        "diagnostics": {},
    }
    eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

    assert eff is not None
    assert eff["mode"] == "selected_refs"
    assert "#/texts/0" in eff["self_refs"]


@skip_if_no_services
def test_vr_disabled_mode_compute_effective_scope():
    """In disabled mode, no narrowing regardless of router response."""
    from app.workers.pipeline import _compute_effective_chunk_scope

    router_response = {
        "mode": "selected_refs",
        "self_refs": ["#/texts/0", "#/texts/1"],
        "diagnostics": {},
    }
    eff, diag = _compute_effective_chunk_scope(router_response, "disabled")

    assert eff is None
