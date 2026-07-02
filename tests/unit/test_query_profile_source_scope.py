"""Project-Source scope filtering for query-profile execution (Task 2).

Asserts the four source-scope guarantees of the reshaped service:
  (a) resolve applies the in-source filter to candidate sets BEFORE selection
      (never post-filter the single global winner — review finding 3);
  (b) the evidence SQL gains ``AND d.source_id = :source_id`` on BOTH UNION
      branches only when scoped (review finding 4);
  (c) associated systems are source-filtered when scoped (review finding 5);
  (d) NONE of it fires when ``source_id=None`` (Global = unfiltered, byte-identical).

The service (not the API) is exercised directly, so these run with no live DB /
graph — db and graph_store are mocked and their emitted SQL/params inspected.

Run standalone:
    python3 -m pytest tests/unit/test_query_profile_source_scope.py -v
"""

import types
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.graph_store import GraphEntityResult as SchemaEntity
from app.schemas.query_profiles import QueryProfileSearchRequest
from app.services.graph_store import GraphEntityResult as StoreEntity
from app.services.query_profiles import (
    _associated_systems,
    _fetch_chunk_evidence,
    _filter_candidates_in_source,
    resolve_root_entity,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_db(in_source_docs: list[str] | None = None) -> AsyncMock:
    """AsyncMock AsyncSession whose ``execute(...).fetchall()`` returns the
    given in-source document_ids as single-column rows."""
    db = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = [(d,) for d in (in_source_docs or [])]
    db.execute.return_value = result
    return db


def _evidence_side_effect(doc_by_node: dict[str, str]):
    """Build a get_entity_evidence_chunks side effect: node_id -> one chunk row
    carrying its document_id."""

    def _side(node_id, limit=25):
        doc = doc_by_node.get(node_id)
        return [{"document_id": doc}] if doc else []

    return _side


def _profile(**kw):
    defaults = dict(
        kind="section_properties",
        root_entity_types=["RADAR_SYSTEM"],
        definition={},
        source_id=None,
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# (b) Evidence SQL predicate on BOTH UNION branches, only when scoped
# ---------------------------------------------------------------------------


async def test_evidence_sql_adds_source_predicate_on_both_branches_when_scoped():
    source_id = uuid.uuid4()
    db = _mock_db([])

    await _fetch_chunk_evidence(db, ["chunk-1"], source_id=source_id)

    sql, params = db.execute.await_args.args
    sql_text = str(sql)
    # Present on the text_chunks branch AND the image_chunks branch.
    assert sql_text.count("AND d.source_id = :source_id") == 2
    # Aliases ingest.documents as d (NOT documents) on both branches.
    assert "JOIN ingest.documents d" in sql_text
    assert params["source_id"] == source_id


async def test_evidence_sql_has_no_source_predicate_when_global():
    db = _mock_db([])

    await _fetch_chunk_evidence(db, ["chunk-1"], source_id=None)

    sql, params = db.execute.await_args.args
    sql_text = str(sql)
    assert "source_id" not in sql_text
    assert "source_id" not in params


async def test_evidence_empty_chunk_ids_short_circuits():
    db = _mock_db([])
    out = await _fetch_chunk_evidence(db, [], source_id=uuid.uuid4())
    assert out == {}
    db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# (a) resolve filters candidate sets BEFORE selection
# ---------------------------------------------------------------------------


async def test_resolve_filters_candidates_before_selection():
    """The exact-name global winner (#1) is out-of-source; a weaker match (#2)
    is in-source. Pre-selection filtering must drop #1 and return #2. If the
    filter ran AFTER selection it would pick #1, drop it, and resolve to None."""
    source_id = uuid.uuid4()
    exact_out = StoreEntity(node_id="#1", name="SA-2", entity_type="RADAR_SYSTEM")
    weak_in = StoreEntity(node_id="#2", name="SA-2 Battery", entity_type="RADAR_SYSTEM")

    graph_store = AsyncMock()
    graph_store.search_by_alias = AsyncMock(return_value=[])
    graph_store.fulltext_search = AsyncMock(return_value=[exact_out, weak_in])
    graph_store.get_entity_evidence_chunks = AsyncMock(
        side_effect=_evidence_side_effect({"#1": "doc-out", "#2": "doc-in"})
    )
    db = _mock_db(["doc-in"])

    request = QueryProfileSearchRequest(
        profile_id="system_rf_parameters",
        query_text="SA-2",
        include_aliases=False,
        top_k=10,
    )

    resolved = await resolve_root_entity(
        graph_store, _profile(), request, db=db, source_id=source_id
    )

    assert resolved.node_id == "#2"  # in-source weaker match won → filtered before select
    graph_store.get_entity_evidence_chunks.assert_awaited()  # filter actually ran
    db.execute.assert_awaited()  # in-source lookup happened
    # Direct/co-extracted fallbacks must NOT be needed (a candidate survived).
    graph_store.resolve_root_entity.assert_not_awaited()


# ---------------------------------------------------------------------------
# (d) Global path: no filtering, no DB round-trips, no evidence-chunk calls
# ---------------------------------------------------------------------------


async def test_resolve_global_path_does_no_source_filtering():
    only = StoreEntity(node_id="#1", name="SA-2", entity_type="RADAR_SYSTEM")

    graph_store = AsyncMock()
    graph_store.search_by_alias = AsyncMock(return_value=[])
    graph_store.fulltext_search = AsyncMock(return_value=[only])
    graph_store.get_entity_evidence_chunks = AsyncMock()
    db = AsyncMock()

    request = QueryProfileSearchRequest(
        profile_id="system_rf_parameters",
        query_text="SA-2",
        include_aliases=False,
        top_k=10,
    )

    resolved = await resolve_root_entity(
        graph_store, _profile(), request, db=db, source_id=None
    )

    assert resolved.node_id == "#1"
    graph_store.get_entity_evidence_chunks.assert_not_called()
    db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# _filter_candidates_in_source unit behavior
# ---------------------------------------------------------------------------


async def test_filter_candidates_drops_out_of_source():
    source_id = uuid.uuid4()
    in_ent = StoreEntity(node_id="#2", name="B", entity_type="RADAR_SYSTEM")
    out_ent = StoreEntity(node_id="#1", name="A", entity_type="RADAR_SYSTEM")

    graph_store = AsyncMock()
    graph_store.get_entity_evidence_chunks = AsyncMock(
        side_effect=_evidence_side_effect({"#1": "doc-out", "#2": "doc-in"})
    )
    db = _mock_db(["doc-in"])

    kept = await _filter_candidates_in_source(
        [out_ent, in_ent], graph_store, db, source_id
    )

    assert [c.node_id for c in kept] == ["#2"]
    # Exactly one batched Postgres lookup for the whole candidate set.
    assert db.execute.await_count == 1


async def test_filter_candidates_noop_when_global():
    out_ent = StoreEntity(node_id="#1", name="A", entity_type="RADAR_SYSTEM")
    graph_store = AsyncMock()
    graph_store.get_entity_evidence_chunks = AsyncMock()
    db = AsyncMock()

    kept = await _filter_candidates_in_source(
        [out_ent], graph_store, db, None
    )

    assert kept == [out_ent]
    graph_store.get_entity_evidence_chunks.assert_not_called()
    db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# (c) Associated systems source-filtered when scoped; unfiltered when Global
# ---------------------------------------------------------------------------


async def test_associated_systems_source_filtered_when_scoped():
    source_id = uuid.uuid4()
    resolved = SchemaEntity(node_id="#5", name="SA-2", entity_type="MISSILE_SYSTEM")
    s_out = StoreEntity(node_id="#1", name="RelatedOut", entity_type="RADAR_SYSTEM")
    s_in = StoreEntity(node_id="#2", name="RelatedIn", entity_type="RADAR_SYSTEM")

    graph_store = AsyncMock()
    graph_store.get_associated_systems = AsyncMock(return_value=[s_out, s_in])
    graph_store.get_entity_evidence_chunks = AsyncMock(
        side_effect=_evidence_side_effect({"#1": "doc-out", "#2": "doc-in"})
    )
    db = _mock_db(["doc-in"])

    related = await _associated_systems(graph_store, db, resolved, source_id)

    assert [r.node_id for r in related] == ["#2"]


async def test_associated_systems_unfiltered_when_global():
    resolved = SchemaEntity(node_id="#5", name="SA-2", entity_type="MISSILE_SYSTEM")
    s_out = StoreEntity(node_id="#1", name="RelatedOut", entity_type="RADAR_SYSTEM")
    s_in = StoreEntity(node_id="#2", name="RelatedIn", entity_type="RADAR_SYSTEM")

    graph_store = AsyncMock()
    graph_store.get_associated_systems = AsyncMock(return_value=[s_out, s_in])
    graph_store.get_entity_evidence_chunks = AsyncMock()
    db = AsyncMock()

    related = await _associated_systems(graph_store, db, resolved, None)

    assert [r.node_id for r in related] == ["#1", "#2"]
    graph_store.get_entity_evidence_chunks.assert_not_called()
    db.execute.assert_not_called()
