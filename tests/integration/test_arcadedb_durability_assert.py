"""Live-ArcadeDB integration test for the node durability assertion (Task 3).

This is the test that the all-mocked unit suite structurally CANNOT provide:
it exercises real ArcadeDB count/existence semantics, which is where the
original assertion was a silent no-op.

Background
----------
The first cut of ``_assert_nodes_durable_sync`` used
``SELECT count(*) FROM [<rid>, <rid>, ...]``. Proven against live ArcadeDB
26.5.1, that counts LIST ENTRIES, not existing rows — ``count(*) FROM
[#37:0, #37:999999]`` returns **2** even though ``#37:999999`` does not
exist. So a non-durable RID could never make the count short, and the
assertion never fired against the exact silent zero-commit it was built to
catch. The mocked unit tests passed only because they hand-fed ``query_sync``
a short count; they never touched real ArcadeDB count semantics.

The fix re-reads with a type-scoped existence query
(``SELECT @rid FROM <Type> WHERE @rid IN [...]``), which correctly returns
only rows that exist. These tests assert the corrected method against a live
DB:

* a real, just-upserted RID is reported DURABLE (passes);
* a real-bucket / non-existent-position RID (same type, position bumped to a
  huge number) is reported NON-durable -> raises ``UpsertNotDurableError``.

The fixture (``arcadedb_store``, tests/integration/conftest.py) skips when no
ArcadeDB is reachable, so the host unit run is unaffected.
"""
from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

pytestmark = pytest.mark.integration


def _mk_doc_id() -> str:
    return f"test-durable-{uuid.uuid4().hex[:12]}"


def _bump_position(rid: str, bump: int = 999_999_999) -> str:
    """Turn a real ``#bucket:pos`` RID into a same-bucket / non-existent
    position RID. The bucket is real (so the query does NOT error with
    "Bucket not found"); only the position is absent — which is exactly the
    silent failure mode the old ``count [list]`` query could not detect.
    """
    m = re.match(r"#(\d+):(\d+)", rid)
    assert m, f"unexpected RID shape: {rid!r}"
    bucket = int(m.group(1))
    return f"#{bucket}:{int(m.group(2)) + bump}"


@pytest.fixture(scope="function")
def upserted_radar(arcadedb_store: "ArcadeDBGraphStore"):
    """Upsert one real RADAR_SYSTEM node; yield (store, record, rid). Cleanup
    after by deleting the document graph (best effort)."""
    from app.services.graph_store import NodeRecord, ProvenanceMetadata

    store = arcadedb_store
    doc_id = _mk_doc_id()
    name = f"durable-radar-{uuid.uuid4().hex[:8]}"
    # identity_fields = {"system_name": ...}: the batch upsert script's WHERE
    # is ``<identity> AND entity_type = ...``, and RADAR_SYSTEM carries a
    # composite index on ``[system_name, entity_type]`` (verified via
    # schema:indexes). ArcadeDB's UPSERT requires the WHERE to be fully covered
    # by an index, so this mirrors the production identity shape.
    record = NodeRecord(
        entity_type="RADAR_SYSTEM",
        identity_fields={"system_name": name},
        name=name,
        extraction_confidence=0.9,
    )
    prov = ProvenanceMetadata(document_id=doc_id, page_numbers=[1])
    rids = store.upsert_nodes_batch_sync([record], prov)
    assert rids and rids[0].startswith("#"), f"upsert returned no RID: {rids!r}"

    yield store, record, rids[0]

    # Cleanup: the node is a global entity keyed by a unique uuid name, so
    # delete it directly (delete_document_graph_sync preserves global
    # entities by design). Best-effort — never fail teardown.
    try:
        store.delete_document_graph_sync(doc_id)
    except Exception:
        pass
    try:
        store._client.command_sync(
            store._database, "sql",
            "DELETE FROM RADAR_SYSTEM WHERE system_name = :name",
            {"name": name},
        )
    except Exception:
        pass


class TestNodeDurabilityAssertionLive:
    def test_real_rid_is_durable(self, upserted_radar):
        """A real, just-committed RID + its record must pass the assertion
        (no exception). This is the happy path against live ArcadeDB."""
        store, record, rid = upserted_radar

        # Must NOT raise.
        store._assert_nodes_durable_sync([rid], [record])

    def test_fake_position_rid_raises(self, upserted_radar):
        """A real-bucket / non-existent-position RID of the SAME vertex type
        must be detected as non-durable -> UpsertNotDurableError.

        This is the case the old ``count(*) FROM [list]`` query silently
        passed (it counts list entries, so a phantom position still counted).
        The fixed ``... FROM RADAR_SYSTEM WHERE @rid IN [...]`` returns 0 rows
        for the phantom, so the assertion correctly fires.
        """
        from app.services.arcadedb_graph import UpsertNotDurableError

        store, record, rid = upserted_radar
        fake_rid = _bump_position(rid)
        assert fake_rid != rid

        with pytest.raises(UpsertNotDurableError) as exc:
            store._assert_nodes_durable_sync([fake_rid], [record])
        # The corrected per-type query can name which RID was non-durable.
        assert fake_rid in str(exc.value)

    def test_mixed_real_and_fake_raises_naming_only_the_fake(self, upserted_radar):
        """A batch containing the real RID and a phantom (same type) must
        raise, and the error must name the phantom as missing but NOT the real
        one — proving the existence query distinguishes them per RID."""
        from app.services.arcadedb_graph import UpsertNotDurableError

        store, record, rid = upserted_radar
        fake_rid = _bump_position(rid)

        with pytest.raises(UpsertNotDurableError) as exc:
            store._assert_nodes_durable_sync([rid, fake_rid], [record, record])
        msg = str(exc.value)
        # The "Missing (not durable):" sample must contain the phantom and not
        # the real RID.
        missing_segment = msg.split("Missing (not durable):")[1].split(".")[0]
        assert fake_rid in missing_segment
        assert rid not in missing_segment

    def test_raw_count_list_query_counts_entries_not_rows(self, upserted_radar):
        """Regression guard documenting WHY the fix was needed: prove against
        the live DB that the OLD query shape (``count(*) FROM [<rid>...]``)
        counts list entries, so a phantom position is indistinguishable from a
        real row. If a future refactor reverts to that shape, this asserts the
        footgun is real."""
        store, record, rid = upserted_radar
        fake_rid = _bump_position(rid)

        rows = store._client.query_sync(
            store._database,
            "sql",
            f"SELECT count(*) AS count FROM [{rid}, {fake_rid}]",
        )
        # 2 — the phantom counted. This is the silent no-op the fix replaces.
        assert int(rows[0]["count"]) == 2

        # And the corrected per-type query returns only the real one.
        good = store._client.query_sync(
            store._database,
            "sql",
            f"SELECT count(*) AS count FROM RADAR_SYSTEM "
            f"WHERE @rid IN [{rid}, {fake_rid}]",
        )
        assert int(good[0]["count"]) == 1
