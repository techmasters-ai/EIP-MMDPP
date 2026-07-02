"""Project-Source scope filtering + table-backed CRUD for query profiles (Task 2).

Asserts the source-scope guarantees of the reshaped service:
  (a) resolve applies the in-source filter to candidate sets BEFORE selection
      (never post-filter the single global winner — review finding 3), across
      the fulltext, direct-name, and co-extracted branches;
  (b) the evidence SQL gains ``AND d.source_id = :source_id`` on BOTH UNION
      branches only when scoped (review finding 4);
  (c) associated systems are source-filtered when scoped (review finding 5);
  (d) NONE of it fires when ``source_id=None`` (Global = unfiltered, byte-identical);
  (e) membership uses each entity's COMPLETE document set (not a first-N chunk
      sample), errors propagate instead of silently dropping, and the per-set
      graph lookup is batched + memoized.

Plus pure-logic CRUD tests (create/update partial/delete + dossier-reference guard).

The service (not the API) is exercised directly, so these run with no live DB /
graph — db and graph_store are mocked and their emitted SQL/params inspected.

Run standalone:
    python3 -m pytest tests/unit/test_query_profile_source_scope.py -v
"""

import types
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.query_profiles import QueryProfile
from app.schemas.graph_store import GraphEntityResult as SchemaEntity
from app.schemas.query_profiles import QueryProfileSearchRequest
from app.services.graph_store import GraphEntityResult as StoreEntity
from app.services.query_profiles import (
    QueryProfileNotFoundError,
    QueryProfileReferencedError,
    _associated_systems,
    _fetch_chunk_evidence,
    _filter_candidates_in_source,
    create_profile,
    delete_profile,
    resolve_root_entity,
    update_profile,
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


def _docs_side_effect(doc_by_node: dict[str, set[str]]):
    """Build a get_entity_source_document_ids side effect: returns the COMPLETE
    distinct document set per requested entity id."""

    def _side(entity_rids):
        return {rid: set(doc_by_node.get(rid, set())) for rid in entity_rids}

    return _side


def _ent(node_id: str, name: str = "X", etype: str = "RADAR_SYSTEM") -> StoreEntity:
    return StoreEntity(node_id=node_id, name=name, entity_type=etype)


def _profile(**kw):
    defaults = dict(
        kind="section_properties",
        root_entity_types=["RADAR_SYSTEM"],
        definition={},
        source_id=None,
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def _request(**kw):
    defaults = dict(
        profile_id="system_rf_parameters",
        query_text="SA-2",
        include_aliases=False,
        top_k=10,
    )
    defaults.update(kw)
    return QueryProfileSearchRequest(**defaults)


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
# (a) resolve filters candidate sets BEFORE selection — fulltext branch
# ---------------------------------------------------------------------------


async def test_resolve_filters_candidates_before_selection():
    """The exact-name global winner (#1) is out-of-source; a weaker match (#2)
    is in-source. Pre-selection filtering must drop #1 and return #2. If the
    filter ran AFTER selection it would pick #1, drop it, and resolve to None."""
    source_id = uuid.uuid4()
    exact_out = _ent("#1", "SA-2")
    weak_in = _ent("#2", "SA-2 Battery")

    graph_store = AsyncMock()
    graph_store.search_by_alias = AsyncMock(return_value=[])
    graph_store.fulltext_search = AsyncMock(return_value=[exact_out, weak_in])
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=_docs_side_effect({"#1": {"doc-out"}, "#2": {"doc-in"}})
    )
    db = _mock_db(["doc-in"])

    resolved = await resolve_root_entity(
        graph_store, _profile(), _request(), db=db, source_id=source_id
    )

    assert resolved.node_id == "#2"  # in-source weaker match won → filtered before select
    graph_store.get_entity_source_document_ids.assert_awaited()  # filter actually ran
    db.execute.assert_awaited()  # in-source lookup happened
    # Direct/co-extracted fallbacks must NOT be needed (a candidate survived).
    graph_store.resolve_root_entity.assert_not_awaited()


async def test_resolve_scoped_direct_name_branch():
    """alias + fulltext empty → direct name resolution; its result is
    source-filtered before being accepted."""
    source_id = uuid.uuid4()
    direct = _ent("#9", "SA-2")

    graph_store = AsyncMock()
    graph_store.search_by_alias = AsyncMock(return_value=[])
    graph_store.fulltext_search = AsyncMock(return_value=[])
    graph_store.resolve_root_entity = AsyncMock(return_value=direct)
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=_docs_side_effect({"#9": {"doc-in"}})
    )
    db = _mock_db(["doc-in"])

    resolved = await resolve_root_entity(
        graph_store, _profile(), _request(), db=db, source_id=source_id
    )

    assert resolved.node_id == "#9"
    graph_store.get_entity_source_document_ids.assert_awaited()


async def test_resolve_scoped_direct_name_out_of_source_falls_to_co_extracted():
    """A direct-name hit that is out-of-source must NOT be accepted; resolve
    falls through to the co-extracted branch, which is also source-filtered."""
    source_id = uuid.uuid4()
    direct_out = _ent("#9", "SA-2")
    co_in = _ent("#7", "SA-2 System")

    graph_store = AsyncMock()
    graph_store.search_by_alias = AsyncMock(return_value=[])
    graph_store.fulltext_search = AsyncMock(return_value=[])
    graph_store.resolve_root_entity = AsyncMock(return_value=direct_out)
    graph_store.get_co_extracted_entities = AsyncMock(return_value=[co_in])
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=_docs_side_effect({"#9": {"doc-out"}, "#7": {"doc-in"}})
    )
    db = _mock_db(["doc-in"])

    resolved = await resolve_root_entity(
        graph_store, _profile(), _request(), db=db, source_id=source_id
    )

    assert resolved.node_id == "#7"
    graph_store.get_co_extracted_entities.assert_awaited()


async def test_resolve_co_extracted_in_source_filter_error_propagates():
    """A graph error during the co-extracted branch's in-source filter must
    propagate — NOT be swallowed into a spurious QueryRootNotFoundError. The
    guard around the co-extracted fetch/selection does not cover the filter."""
    source_id = uuid.uuid4()
    direct_out = _ent("#9", "SA-2")
    co_ent = _ent("#7", "SA-2 System")

    def _side(entity_rids):
        if "#7" in entity_rids:  # co-extracted in-source filter
            raise RuntimeError("graph down")
        return {rid: {"doc-out"} for rid in entity_rids}  # direct → out of source

    graph_store = AsyncMock()
    graph_store.search_by_alias = AsyncMock(return_value=[])
    graph_store.fulltext_search = AsyncMock(return_value=[])
    graph_store.resolve_root_entity = AsyncMock(return_value=direct_out)
    graph_store.get_co_extracted_entities = AsyncMock(return_value=[co_ent])
    graph_store.get_entity_source_document_ids = AsyncMock(side_effect=_side)
    db = _mock_db(["doc-in"])

    with pytest.raises(RuntimeError, match="graph down"):
        await resolve_root_entity(
            graph_store, _profile(), _request(), db=db, source_id=source_id
        )


# ---------------------------------------------------------------------------
# (d) Global path: no filtering, no DB round-trips, no graph doc-id calls
# ---------------------------------------------------------------------------


async def test_resolve_global_path_does_no_source_filtering():
    only = _ent("#1", "SA-2")

    graph_store = AsyncMock()
    graph_store.search_by_alias = AsyncMock(return_value=[])
    graph_store.fulltext_search = AsyncMock(return_value=[only])
    graph_store.get_entity_source_document_ids = AsyncMock()
    db = AsyncMock()

    resolved = await resolve_root_entity(
        graph_store, _profile(), _request(), db=db, source_id=None
    )

    assert resolved.node_id == "#1"
    graph_store.get_entity_source_document_ids.assert_not_called()
    db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# (e) Exhaustive membership, error propagation, batching + memo
# ---------------------------------------------------------------------------


async def test_filter_candidates_drops_out_of_source():
    source_id = uuid.uuid4()
    graph_store = AsyncMock()
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=_docs_side_effect({"#1": {"doc-out"}, "#2": {"doc-in"}})
    )
    db = _mock_db(["doc-in"])

    kept = await _filter_candidates_in_source(
        [_ent("#1", "A"), _ent("#2", "B")], graph_store, db, source_id
    )

    assert [c.node_id for c in kept] == ["#2"]
    # Exactly one batched graph call + one batched Postgres lookup for the set.
    assert graph_store.get_entity_source_document_ids.await_count == 1
    assert db.execute.await_count == 1


async def test_membership_uses_complete_document_set_not_chunk_sample():
    """Regression guard for the first-25-chunks bug: the in-source document is
    only reachable via the entity's COMPLETE distinct document set (here it is
    one of 40 documents). Because the filter uses the exhaustive set — not a
    chunk sample — the candidate is still kept when scoped."""
    source_id = uuid.uuid4()
    complete = {f"doc-{i}" for i in range(40)} | {"doc-in"}
    graph_store = AsyncMock()
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=_docs_side_effect({"#1": complete})
    )
    db = _mock_db(["doc-in"])

    kept = await _filter_candidates_in_source([_ent("#1")], graph_store, db, source_id)

    assert [c.node_id for c in kept] == ["#1"]


async def test_entity_with_no_extracted_from_dropped_when_scoped():
    source_id = uuid.uuid4()
    graph_store = AsyncMock()
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=_docs_side_effect({"#1": set()})  # no EXTRACTED_FROM chunks
    )
    db = _mock_db(["doc-in"])

    kept = await _filter_candidates_in_source([_ent("#1")], graph_store, db, source_id)

    assert kept == []


async def test_filter_candidates_propagates_graph_error():
    """Graph errors must surface, not be swallowed into a silent scoped drop."""
    graph_store = AsyncMock()
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=RuntimeError("graph down")
    )
    db = _mock_db(["doc-in"])

    with pytest.raises(RuntimeError, match="graph down"):
        await _filter_candidates_in_source([_ent("#1")], graph_store, db, uuid.uuid4())


async def test_filter_candidates_noop_when_global():
    graph_store = AsyncMock()
    graph_store.get_entity_source_document_ids = AsyncMock()
    db = AsyncMock()

    ent = _ent("#1")
    kept = await _filter_candidates_in_source([ent], graph_store, db, None)

    assert kept == [ent]
    graph_store.get_entity_source_document_ids.assert_not_called()
    db.execute.assert_not_called()


async def test_resolve_memoizes_document_ids_across_candidate_sets():
    """An entity appearing in both alias and fulltext sets is queried for its
    document membership at most once within a single resolve pass."""
    source_id = uuid.uuid4()
    shared = _ent("#1", "SA-2")

    graph_store = AsyncMock()
    graph_store.search_by_alias = AsyncMock(return_value=[shared])
    graph_store.fulltext_search = AsyncMock(return_value=[shared])
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=_docs_side_effect({"#1": {"doc-in"}})
    )
    db = _mock_db(["doc-in"])

    resolved = await resolve_root_entity(
        graph_store, _profile(), _request(), db=db, source_id=source_id
    )

    assert resolved.node_id == "#1"
    assert graph_store.get_entity_source_document_ids.await_count == 1


# ---------------------------------------------------------------------------
# (c) Associated systems source-filtered when scoped; unfiltered when Global
# ---------------------------------------------------------------------------


async def test_associated_systems_source_filtered_when_scoped():
    source_id = uuid.uuid4()
    resolved = SchemaEntity(node_id="#5", name="SA-2", entity_type="MISSILE_SYSTEM")

    graph_store = AsyncMock()
    graph_store.get_associated_systems = AsyncMock(
        return_value=[_ent("#1", "RelatedOut"), _ent("#2", "RelatedIn")]
    )
    graph_store.get_entity_source_document_ids = AsyncMock(
        side_effect=_docs_side_effect({"#1": {"doc-out"}, "#2": {"doc-in"}})
    )
    db = _mock_db(["doc-in"])

    related = await _associated_systems(graph_store, db, resolved, source_id)

    assert [r.node_id for r in related] == ["#2"]


async def test_associated_systems_unfiltered_when_global():
    resolved = SchemaEntity(node_id="#5", name="SA-2", entity_type="MISSILE_SYSTEM")

    graph_store = AsyncMock()
    graph_store.get_associated_systems = AsyncMock(
        return_value=[_ent("#1", "RelatedOut"), _ent("#2", "RelatedIn")]
    )
    graph_store.get_entity_source_document_ids = AsyncMock()
    db = AsyncMock()

    related = await _associated_systems(graph_store, db, resolved, None)

    assert [r.node_id for r in related] == ["#1", "#2"]
    graph_store.get_entity_source_document_ids.assert_not_called()
    db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Table-backed CRUD (pure logic)
# ---------------------------------------------------------------------------


async def test_create_profile_persists_fields():
    db = AsyncMock()
    db.add = MagicMock()

    profile = await create_profile(
        db,
        profile_key="p1",
        label="Profile One",
        kind="section_properties",
        description="desc",
        root_entity_types=["RADAR_SYSTEM"],
        definition={"profile_sections": ["rf_parameters"]},
        source_id=None,
        enabled=True,
    )

    assert profile.profile_key == "p1"
    assert profile.label == "Profile One"
    assert profile.definition == {"profile_sections": ["rf_parameters"]}
    assert profile.root_entity_types == ["RADAR_SYSTEM"]
    db.add.assert_called_once_with(profile)
    db.flush.assert_awaited_once()


async def test_update_profile_partial_leaves_unset_fields_untouched():
    existing = QueryProfile(
        profile_key="p1",
        label="Old",
        kind="section_properties",
        description="keep me",
        root_entity_types=["RADAR_SYSTEM"],
        definition={"profile_sections": ["rf_parameters"]},
        source_id=None,
        enabled=True,
    )
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    db.execute.return_value = result

    updated = await update_profile(db, "p1", label="New", enabled=False)

    assert updated.label == "New"           # changed
    assert updated.enabled is False         # changed
    assert updated.description == "keep me"  # untouched (was _UNSET)
    assert updated.definition == {"profile_sections": ["rf_parameters"]}  # untouched
    db.flush.assert_awaited()


async def test_update_profile_missing_raises():
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute.return_value = result

    with pytest.raises(QueryProfileNotFoundError):
        await update_profile(db, "nope", label="X")


async def test_delete_profile_blocked_when_referenced_by_dossier():
    section = QueryProfile(
        profile_key="system_rf_parameters", label="RF",
        kind="section_properties", definition={},
    )
    dossier = QueryProfile(
        profile_key="system_dossier", label="Dossier", kind="dossier",
        definition={"section_profile_ids": ["system_rf_parameters"]},
    )
    db = AsyncMock()
    db.delete = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = section          # get_required_profile
    result.scalars.return_value.all.return_value = [section, dossier]  # list_profiles
    db.execute.return_value = result

    with pytest.raises(QueryProfileReferencedError, match="system_dossier"):
        await delete_profile(db, "system_rf_parameters")

    db.delete.assert_not_called()


async def test_delete_profile_succeeds_when_unreferenced():
    section = QueryProfile(
        profile_key="system_components", label="Components",
        kind="section_properties", definition={},
    )
    db = AsyncMock()
    db.delete = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = section
    result.scalars.return_value.all.return_value = [section]
    db.execute.return_value = result

    await delete_profile(db, "system_components")

    db.delete.assert_awaited_once_with(section)
    db.flush.assert_awaited()
