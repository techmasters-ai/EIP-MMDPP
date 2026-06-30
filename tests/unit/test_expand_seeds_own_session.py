from __future__ import annotations

import uuid

from unittest.mock import AsyncMock

import app.api.v1.retrieval as R
from app.schemas.retrieval import QueryResultItem


def _item() -> QueryResultItem:
    return QueryResultItem(
        chunk_id=uuid.uuid4(),
        score=0.5,
        modality="text",
        content_text="SA-2 guidance radar",
    )


class _RecordingSession:
    """Stand-in async-context session that registers itself on creation."""

    def __init__(self, registry: list) -> None:
        registry.append(self)

    async def __aenter__(self) -> "_RecordingSession":
        return self

    async def __aexit__(self, *exc) -> bool:  # noqa: ANN002
        return False


async def test_expand_seeds_gives_each_expansion_its_own_session(monkeypatch):
    """Regression guard for the #88 session-race fix.

    Each per-seed expansion must run on its OWN DB session opened via
    AsyncSessionFactory — never the shared pipeline `db` passed into
    `_expand_seeds` (SQLAlchemy forbids concurrent ops on one session).
    """
    created_sessions: list[_RecordingSession] = []

    def _factory() -> _RecordingSession:
        return _RecordingSession(created_sessions)

    # The per-expansion session is created inside `_expand_one` via
    # `from app.db.session import AsyncSessionFactory`; patch it at the source.
    monkeypatch.setattr("app.db.session.AsyncSessionFactory", _factory)

    # Capture the `db` arg threaded into the expander that receives a session.
    doc_mock = AsyncMock(return_value=[_item()])
    monkeypatch.setattr(R, "_expand_via_doc_structure", doc_mock)
    monkeypatch.setattr(R, "_expand_via_cross_modal", AsyncMock(return_value=[]))
    monkeypatch.setattr(R, "_expand_via_ontology", AsyncMock(return_value=[]))
    monkeypatch.setattr(R, "_expand_via_domain_relations", AsyncMock(return_value=[]))

    pipeline_db_sentinel = object()
    seeds = [_item(), _item()]

    await R._expand_seeds(pipeline_db_sentinel, seeds, include_context=True, query_text="SA-2")

    # A fresh session was opened (at least one per seed).
    assert len(created_sessions) >= 1
    assert all(isinstance(s, _RecordingSession) for s in created_sessions)

    # _expand_via_doc_structure ran once per seed.
    assert doc_mock.call_count == len(seeds)

    # Every doc-structure call received a per-expansion session, NOT the
    # shared pipeline db sentinel.
    for call in doc_mock.call_args_list:
        db_arg = call.args[0]
        assert db_arg is not pipeline_db_sentinel
        assert isinstance(db_arg, _RecordingSession)
