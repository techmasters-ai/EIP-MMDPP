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

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _make_client(command_sync_result=None, query_sync_result=None):
    """Mock ArcadeDBClient with configurable sync return values.

    ``command_sync`` drives the upsert sqlscript result (the returned RIDs);
    ``query_sync`` drives the durability re-query (the committed count).
    """
    client = MagicMock()
    client.command_sync = MagicMock(
        return_value=command_sync_result or [{"@rid": "#1:0"}],
    )
    client.query_sync = MagicMock(return_value=query_sync_result or [])
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
            query_sync_result=[{"count": 0}],  # 0 of 2 actually queryable
        )
        store = _graph(client)
        records = [_node_record(name="A"), _node_record(name="B")]

        with pytest.raises(UpsertNotDurableError) as exc:
            store.upsert_nodes_batch_sync(records)

        # Error message must name the gap (2 written, 0 durable) so the
        # operator can act, and must have run the re-query exactly once.
        msg = str(exc.value)
        assert "0" in msg and "2" in msg
        assert client.query_sync.call_count == 1

    def test_partial_commit_raises(self):
        """N RIDs returned, only M<N queryable -> still a data-loss event."""
        from app.services.arcadedb_graph import UpsertNotDurableError

        client = _make_client(
            command_sync_result=[
                {"@rid": "#10:0"}, {"@rid": "#10:1"}, {"@rid": "#10:2"},
            ],
            query_sync_result=[{"count": 2}],  # 2 of 3 committed
        )
        store = _graph(client)
        records = [
            _node_record(name="A"),
            _node_record(name="B"),
            _node_record(name="C"),
        ]

        with pytest.raises(UpsertNotDurableError):
            store.upsert_nodes_batch_sync(records)

    def test_durability_check_is_a_single_batch_query(self):
        """Cost guard: the re-query is ONE query for the whole batch, not
        one per entity (the constraint forbids a per-entity re-query loop).
        """
        client = _make_client(
            command_sync_result=[
                {"@rid": "#10:0"}, {"@rid": "#10:1"}, {"@rid": "#10:2"},
            ],
            query_sync_result=[{"count": 3}],
        )
        store = _graph(client)
        records = [
            _node_record(name="A"),
            _node_record(name="B"),
            _node_record(name="C"),
        ]

        store.upsert_nodes_batch_sync(records)

        # Exactly one durability re-query for the 3-record batch.
        assert client.query_sync.call_count == 1
        # And it queries the actual written RIDs (batch-by-RID), not a
        # per-record identity loop.
        rid_query = client.query_sync.call_args.args[2]
        assert "#10:0" in rid_query
        assert "#10:1" in rid_query
        assert "#10:2" in rid_query


# ---------------------------------------------------------------------------
# HAPPY PATH: all upserted + all queryable -> passes through, returns RIDs
# ---------------------------------------------------------------------------

class TestDurabilityAssertionHappyPath:
    def test_all_committed_passes_and_returns_rids(self):
        client = _make_client(
            command_sync_result=[{"@rid": "#10:0"}, {"@rid": "#10:1"}],
            query_sync_result=[{"count": 2}],  # both durable
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
