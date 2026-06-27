"""Regression tests for the silent zero-commit of merged entities (Task 3).

Background (Task 0 diagnostic, reports/collection/lineage_diagnostic_findings.md):
The SA-2 run merged to 22 entities, the merge phase was marked
``result='succeeded'``, yet 0 entities persisted to ArcadeDB — a SILENT
zero-commit. ``upsert_nodes_batch_sync`` returned RIDs (so the existing
``UpsertMissingRIDError`` RID-presence check passed) but the rows were not
durably queryable. The exact sub-stage (non-durable RID vs empty-merge vs
double-dispatch) was DEFERRED to a fresh run, so this layer adds a
*durability assertion*: after upsert returns N RIDs, re-query ArcadeDB and
assert the just-written records are actually queryable. If the committed
count is short, raise ``UpsertNotDurableError`` — turning silent data-loss
into the existing rollback/FAILED path in ``derive_ontology_graph_merge``.

These are unit tests: the ArcadeDB client is mocked, no live DB. The live
proof of the end-to-end conversion (succeeded → FAILED) is Task 5.
"""

import re

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

_RID_RE = re.compile(r"#\d+:\d+")


def _make_client(command_sync_result=None, durable_rids=None):
    """Mock ArcadeDBClient with configurable sync return values.

    ``command_sync`` drives the upsert sqlscript result (the returned RIDs).

    ``durable_rids`` drives the per-type durability re-query. The corrected
    assertion issues ``SELECT @rid FROM <Type> WHERE @rid IN [...]`` (one per
    vertex type) and counts the RIDs that come back — NOT a ``count(*)`` over a
    literal list, which against real ArcadeDB counts list entries and silently
    masks a non-durable RID. So the mock returns, for each query, the @rid rows
    for whichever of ``durable_rids`` are named in that query's RID-IN list.
    ``durable_rids=None`` means every RID mentioned is treated as durable.
    """
    client = MagicMock()
    client.command_sync = MagicMock(
        return_value=command_sync_result or [{"@rid": "#1:0"}],
    )

    def _query_sync(_db, _lang, sql, _params=None):
        rids_in_query = _RID_RE.findall(sql)
        if durable_rids is None:
            present = rids_in_query
        else:
            present = [r for r in rids_in_query if r in durable_rids]
        return [{"rid": r} for r in present]

    client.query_sync = MagicMock(side_effect=_query_sync)
    client.close_sync = MagicMock()
    return client


def _graph(client):
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    store = ArcadeDBGraphStore(client=client, database="testdb")
    store._validation_matrix = set()
    return store


def _node_record(name="APG-77", identity_fields=None, entity_type="RADAR_SYSTEM"):
    from app.services.graph_store import NodeRecord

    return NodeRecord(
        entity_type=entity_type,
        identity_fields=identity_fields or {"system_name": name},
        name=name,
    )


# ---------------------------------------------------------------------------
# FAILURE MODE: RIDs returned but 0 queryable -> MUST raise (silent->loud)
# ---------------------------------------------------------------------------

class TestDurabilityAssertionFailsLoudly:
    def test_rids_returned_but_zero_committed_raises(self):
        """Reproduces the SA-2 silent zero-commit.

        ``command_sync`` returns 2 valid RIDs (the existing RID-presence
        check passes — necessary but NOT sufficient), but the durability
        re-query (``query_sync``) shows count=0 committed. The upsert MUST
        raise rather than return RIDs for rows that were never persisted.
        """
        from app.services.arcadedb_graph import UpsertNotDurableError

        client = _make_client(
            command_sync_result=[{"@rid": "#10:0"}, {"@rid": "#10:1"}],
            durable_rids=set(),  # 0 of 2 actually queryable
        )
        store = _graph(client)
        records = [_node_record(name="A"), _node_record(name="B")]

        with pytest.raises(UpsertNotDurableError) as exc:
            store.upsert_nodes_batch_sync(records)

        # Error message must name the gap (2 written, 0 durable) so the
        # operator can act. Both records share one vertex type, so the per-type
        # re-query runs once; but a non-durable type retries up to 3× to ride
        # out async-flush lag, so allow the bounded retry budget.
        msg = str(exc.value)
        assert "0" in msg and "2" in msg
        assert 1 <= client.query_sync.call_count <= 3

    def test_partial_commit_raises(self):
        """N RIDs returned, only M<N queryable -> still a data-loss event."""
        from app.services.arcadedb_graph import UpsertNotDurableError

        client = _make_client(
            command_sync_result=[
                {"@rid": "#10:0"}, {"@rid": "#10:1"}, {"@rid": "#10:2"},
            ],
            durable_rids={"#10:0", "#10:1"},  # 2 of 3 committed; #10:2 missing
        )
        store = _graph(client)
        records = [
            _node_record(name="A"),
            _node_record(name="B"),
            _node_record(name="C"),
        ]

        with pytest.raises(UpsertNotDurableError) as exc:
            store.upsert_nodes_batch_sync(records)

        # Fixed query can name WHICH rid was non-durable.
        assert "#10:2" in str(exc.value)

    def test_durability_check_is_one_query_per_type_not_per_entity(self):
        """Cost guard: the re-query is ONE query PER VERTEX TYPE for the whole
        batch, not one per entity (the constraint forbids a per-entity re-query
        loop). A single-type batch → exactly one query naming all its RIDs.
        """
        client = _make_client(
            command_sync_result=[
                {"@rid": "#10:0"}, {"@rid": "#10:1"}, {"@rid": "#10:2"},
            ],
            durable_rids={"#10:0", "#10:1", "#10:2"},  # all durable
        )
        store = _graph(client)
        records = [
            _node_record(name="A"),
            _node_record(name="B"),
            _node_record(name="C"),
        ]

        store.upsert_nodes_batch_sync(records)

        # One vertex type (RADAR_SYSTEM) → exactly one durability re-query.
        assert client.query_sync.call_count == 1
        # It queries by vertex type with @rid IN [...], not count(*) over a
        # literal RID list (which would count list entries, not rows), and
        # names every written RID.
        rid_query = client.query_sync.call_args.args[2]
        assert "RADAR_SYSTEM" in rid_query
        assert "@rid in" in rid_query.lower()
        assert "count(*)" not in rid_query.lower()
        assert "#10:0" in rid_query
        assert "#10:1" in rid_query
        assert "#10:2" in rid_query

    def test_multi_type_batch_groups_rids_by_type(self):
        """A batch spanning multiple vertex types must group RIDs by type and
        sum per-type existence counts. A single ``FROM <Type>`` would
        under-count the other types → false UpsertNotDurableError. One query
        per type; all RIDs durable → passes.
        """
        client = _make_client(
            command_sync_result=[
                {"@rid": "#37:0"},  # RADAR_SYSTEM
                {"@rid": "#40:0"},  # MISSILE_SYSTEM
                {"@rid": "#37:1"},  # RADAR_SYSTEM
            ],
            durable_rids={"#37:0", "#40:0", "#37:1"},
        )
        store = _graph(client)
        records = [
            _node_record(name="A", entity_type="RADAR_SYSTEM"),
            _node_record(name="M", entity_type="MISSILE_SYSTEM"),
            _node_record(name="B", entity_type="RADAR_SYSTEM"),
        ]

        result = store.upsert_nodes_batch_sync(records)
        assert result == ["#37:0", "#40:0", "#37:1"]

        # Two distinct vertex types → two per-type re-queries (not three
        # per-entity, not one global list).
        assert client.query_sync.call_count == 2
        queried_types = {
            re.search(r"FROM (\w+) WHERE", call.args[2]).group(1)
            for call in client.query_sync.call_args_list
        }
        assert queried_types == {"RADAR_SYSTEM", "MISSILE_SYSTEM"}

    def test_multi_type_batch_one_type_non_durable_raises(self):
        """When ONE type's rows didn't commit, the per-type sum is short and
        the assertion must still raise — proving the grouping doesn't mask a
        partial commit hidden inside a multi-type batch.
        """
        from app.services.arcadedb_graph import UpsertNotDurableError

        client = _make_client(
            command_sync_result=[
                {"@rid": "#37:0"},  # RADAR_SYSTEM  (durable)
                {"@rid": "#40:0"},  # MISSILE_SYSTEM (NOT durable)
            ],
            durable_rids={"#37:0"},
        )
        store = _graph(client)
        records = [
            _node_record(name="A", entity_type="RADAR_SYSTEM"),
            _node_record(name="M", entity_type="MISSILE_SYSTEM"),
        ]

        with pytest.raises(UpsertNotDurableError) as exc:
            store.upsert_nodes_batch_sync(records)
        assert "#40:0" in str(exc.value)


# ---------------------------------------------------------------------------
# HAPPY PATH: all upserted + all queryable -> passes through, returns RIDs
# ---------------------------------------------------------------------------

class TestDurabilityAssertionHappyPath:
    def test_all_committed_passes_and_returns_rids(self):
        client = _make_client(
            command_sync_result=[{"@rid": "#10:0"}, {"@rid": "#10:1"}],
            durable_rids={"#10:0", "#10:1"},  # both durable
        )
        store = _graph(client)
        records = [_node_record(name="A"), _node_record(name="B")]

        result = store.upsert_nodes_batch_sync(records)

        assert result == ["#10:0", "#10:1"]

    def test_empty_records_short_circuits_no_query(self):
        """Empty batch returns [] without touching command_sync or query_sync
        (preserves the existing fast-path; nothing to verify).
        """
        client = _make_client()
        store = _graph(client)

        assert store.upsert_nodes_batch_sync([]) == []
        assert client.command_sync.call_count == 0
        assert client.query_sync.call_count == 0

    def test_existing_missing_rid_check_still_fires_first(self):
        """The new durability check is ADDITIVE — the pre-existing
        UpsertMissingRIDError (blank @rid) must still raise, and before any
        durability re-query (a blank RID can't be re-queried meaningfully).
        """
        from app.services.arcadedb_graph import UpsertMissingRIDError

        client = _make_client(
            command_sync_result=[{"@rid": "#10:0"}, {"@rid": ""}],
        )
        store = _graph(client)
        records = [_node_record(name="A"), _node_record(name="B")]

        with pytest.raises(UpsertMissingRIDError):
            store.upsert_nodes_batch_sync(records)
        # Missing-RID raised before we ever issued the durability re-query.
        assert client.query_sync.call_count == 0
