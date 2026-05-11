"""Shared fixtures for `tests/workers/`.

The `patched_get_db` fixture re-routes `app.workers.pipeline._get_db()` to a
sub-Session that joins the test's outer Connection via
`join_transaction_mode="create_savepoint"`. Without this, helpers that open
their own session (Tx-3, Tx-4, ...) acquire a fresh pool connection that
cannot see rows the test wrote inside its uncommitted outer transaction
and deadlock on parent-row FK locks.
"""
from __future__ import annotations

import pytest
from sqlalchemy.orm import Session


@pytest.fixture
def patched_get_db(db_session, monkeypatch):
    """Make worker helpers' `_get_db()` share the test's Connection.

    The helper's `with db.begin():` then resolves to a SAVEPOINT, so callers
    still exercise real atomic-rollback semantics while the test's outer
    rollback cleans everything up.

    Patches both `app.workers.pipeline._get_db` and
    `app.workers.dispatcher._get_db` (each module imports the same underlying
    `get_worker_db` under its own `_get_db` name).
    """
    bind = db_session.get_bind()  # underlying Connection (transaction-joined)

    def fake_get_db():
        return Session(bind=bind, join_transaction_mode="create_savepoint")

    monkeypatch.setattr("app.workers.pipeline._get_db", fake_get_db)
    monkeypatch.setattr("app.workers.dispatcher._get_db", fake_get_db)
