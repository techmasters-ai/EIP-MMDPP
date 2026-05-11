"""reingest_graph_only seeds derive_document_anchors PENDING; no chain."""
import uuid
from unittest.mock import patch, MagicMock
from sqlalchemy import text
from app.workers.pipeline import reingest_graph_only

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup_completed_doc(db_session):
    """Create a doc with a completed pipeline_run (typical graph_only target)."""
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, retry_count,
             uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',0,:u,'COMPLETE')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'COMPLETE')
    """), {"r": run, "d": doc})
    db_session.commit()
    return str(doc)


def test_reingest_graph_only_seeds_anchors_no_chain(db_session, patched_get_db):
    doc_id = _setup_completed_doc(db_session)
    # request is a Pydantic-ish object accessed via getattr(..., 'ontology_bundle_key', None)
    # and getattr(..., 'use_case_key', None). MagicMock with these attrs cleared works.
    request = MagicMock(ontology_bundle_key=None, use_case_key=None)

    with patch("app.workers.pipeline.celery_chain") as chain_mock:
        result = reingest_graph_only(doc_id, request)
        chain_mock.assert_not_called()

    new_run_id = result["pipeline_run_id"]
    row = db_session.execute(text("""
        SELECT status, task_name FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_document_anchors'
    """), {"r": new_run_id}).first()
    assert row is not None
    assert row.status == "PENDING"
    assert row.task_name == "app.workers.pipeline.derive_document_anchors"
    assert result["celery_task_id"] == ""
