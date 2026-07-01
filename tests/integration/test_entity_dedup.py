"""VERIFICATION GATE — case-insensitive entity identity (dedup) end-to-end.

Task 4 of the case-insensitive-identity feature. Tasks 0-3 (case-insensitive
``LogicalIdentity`` in ``extraction_merge``, DB-layer normalized ``<field>_key``
WHERE/SET in ``arcadedb_graph``, ``<field>_key`` UNIQUE indexes, and the
one-time backfill/dedup migration ``scripts/migrate_entity_key_dedup.py``) are
DONE and the migration HAS ALREADY RUN against the live graph. This gate proves
the post-migration invariants hold and — most importantly — that a later
case/whitespace variant upsert converges on the existing vertex through the REAL
write path (``upsert_nodes_batch_sync``) instead of creating a duplicate.

Six areas, all against the LIVE ArcadeDB graph via the module-scoped
``arcadedb_store`` fixture (``tests/integration/conftest.py``), which skips
cleanly when ArcadeDB is unreachable:

  1. Dedup by normalized key      — 'fan song'/'spoon rest' → exactly 1 vertex
                                     each; uppercase losers gone.
  2. Survivor lineage intact      — Fan Song keeps merged fields + page + edges.
  3. Distinct variants preserved  — the 16-member 'Guideline' missile family
                                     stays 16 separate vertices.
  4. No null keys                 — every ``<field>_key`` is populated.
  5. Indexes exist                — the ``<field>_key`` UNIQUE indexes are live.
  6. Idempotent upsert (WRITES)   — upserting identity ``system_name='FAN SONG'``
                                     (uppercase) returns the EXISTING Fan Song
                                     RID, creates NO new vertex (count 92→92),
                                     and does NOT clobber the first-seen display
                                     casing ('Fan Song'). This is the definitive
                                     live proof of WHERE-on-``system_name_key``.

Read-only assertions are safe against production. Test 6 writes through the real
path but must MATCH the existing vertex and create nothing (asserted
count-before == count-after); it cleans up defensively if a bug ever caused a
new uppercase vertex to be inserted.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# Ground truth verified live (2026-07-01) with captured output; the tests
# re-assert these against the running graph.
_FAN_SONG_RID = "#37:4"
_RADAR_COUNT = 92
_MISSILE_COUNT = 145
_FAN_SONG_EDGES = 137
_GUIDELINE_VARIANTS = 16

# Document-scoped and global entity types with their identity ``<field>_key``
# column. Post-migration every one of these must be fully populated (0 nulls).
_KEY_COLUMNS = {
    "RADAR_SYSTEM": "system_name_key",
    "MISSILE_SYSTEM": "system_name_key",
    "SECTION": "section_number_key",
    "FIGURE": "figure_ref_key",
    "TABLE_REF": "table_ref_key",
    "IMAGE": "image_ref_key",
    "TEXT_BLOCK": "text_ref_key",
}


# ---------------------------------------------------------------------------
# Small read helpers over the live store's sync query path.
# ---------------------------------------------------------------------------


def _rows(store, sql: str, params: dict | None = None) -> list[dict]:
    return store.execute_query_sync(sql, params or {})


def _count(store, sql: str, params: dict | None = None) -> int:
    rows = _rows(store, sql, params)
    assert rows, f"count query returned no rows: {sql}"
    return int(rows[0]["c"])


# ---------------------------------------------------------------------------
# 1. Dedup by normalized key
# ---------------------------------------------------------------------------


def test_dedup_by_normalized_key(arcadedb_store):
    """'fan song'/'spoon rest' → exactly one RADAR_SYSTEM vertex each, and the
    uppercase display losers ('FAN SONG'/'SPOON REST') were merged + deleted."""
    store = arcadedb_store

    fan = _count(
        store,
        "SELECT count(*) AS c FROM RADAR_SYSTEM WHERE system_name_key = :k",
        {"k": "fan song"},
    )
    assert fan == 1, f"expected exactly 1 'fan song' vertex, got {fan}"

    spoon = _count(
        store,
        "SELECT count(*) AS c FROM RADAR_SYSTEM WHERE system_name_key = :k",
        {"k": "spoon rest"},
    )
    assert spoon == 1, f"expected exactly 1 'spoon rest' vertex, got {spoon}"

    # The uppercase display losers must be GONE (merged into the survivor).
    uppercase = _count(
        store,
        "SELECT count(*) AS c FROM RADAR_SYSTEM "
        "WHERE system_name = 'FAN SONG' OR system_name = 'SPOON REST'",
    )
    assert uppercase == 0, (
        f"uppercase losers still present ({uppercase} row(s)); dedup/migration "
        "did not delete them"
    )


# ---------------------------------------------------------------------------
# 2. Survivor lineage intact
# ---------------------------------------------------------------------------


def test_survivor_lineage_intact(arcadedb_store):
    """The Fan Song survivor keeps its merged fields, the page number folded in
    from the deleted loser, and its full edge complement (EXTRACTED_FROM present)."""
    store = arcadedb_store

    rows = _rows(
        store,
        "SELECT @rid AS rid, system_name, nominal_rf_mhz, tx_peak_power_kw, "
        "_page_numbers FROM RADAR_SYSTEM WHERE system_name_key = :k",
        {"k": "fan song"},
    )
    assert len(rows) == 1, f"expected 1 survivor, got {len(rows)}: {rows}"
    survivor = rows[0]

    assert survivor["rid"] == _FAN_SONG_RID, (
        f"survivor RID {survivor['rid']} != expected {_FAN_SONG_RID}"
    )
    assert survivor["system_name"] == "Fan Song", (
        f"survivor display name is {survivor['system_name']!r}, expected 'Fan Song'"
    )
    assert survivor["nominal_rf_mhz"] == 2450, (
        f"nominal_rf_mhz={survivor['nominal_rf_mhz']}, expected 2450"
    )
    assert survivor["tx_peak_power_kw"] == 300, (
        f"tx_peak_power_kw={survivor['tx_peak_power_kw']}, expected 300"
    )
    # Page 2 was contributed by the merged/deleted loser — proves lineage union.
    assert 2 in (survivor["_page_numbers"] or []), (
        f"_page_numbers {survivor['_page_numbers']} does not contain 2 "
        "(page folded in from the merged loser is missing)"
    )

    edges = _rows(
        store,
        f"SELECT both().size() AS edges, out('EXTRACTED_FROM').size() AS ef "
        f"FROM {_FAN_SONG_RID}",
    )
    assert edges, "edge-count query returned no rows"
    edge_count = int(edges[0]["edges"])
    ef = int(edges[0]["ef"])
    assert edge_count == _FAN_SONG_EDGES, (
        f"survivor edge count {edge_count} != expected {_FAN_SONG_EDGES}"
    )
    assert edge_count > 0
    assert ef > 0, "EXTRACTED_FROM edges absent on survivor (lineage lost)"


# ---------------------------------------------------------------------------
# 3. Distinct variants preserved
# ---------------------------------------------------------------------------


def test_distinct_variants_preserved(arcadedb_store):
    """The 'Guideline' missile family is 16 DISTINCT system_names — they must
    NOT have been collapsed by the case-insensitive dedup (they differ by more
    than case/whitespace, so each stays its own vertex)."""
    store = arcadedb_store

    # ArcadeDB SQL has no count(DISTINCT ...) / HAVING — select the rows and
    # assert distinctness in Python.
    rows = _rows(
        store,
        "SELECT system_name FROM MISSILE_SYSTEM WHERE name = :n",
        {"n": "Guideline"},
    )
    assert len(rows) == _GUIDELINE_VARIANTS, (
        f"expected {_GUIDELINE_VARIANTS} Guideline-family rows, got {len(rows)}: "
        f"{[r['system_name'] for r in rows]}"
    )
    names = [r["system_name"] for r in rows]
    assert len(set(names)) == _GUIDELINE_VARIANTS, (
        f"expected {_GUIDELINE_VARIANTS} DISTINCT system_names, got "
        f"{len(set(names))} distinct: {sorted(names)}"
    )


# ---------------------------------------------------------------------------
# 4. No null keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entity_type,key_col", sorted(_KEY_COLUMNS.items()))
def test_no_null_keys(arcadedb_store, entity_type, key_col):
    """Every ``<field>_key`` column is fully populated post-backfill (0 nulls)."""
    store = arcadedb_store
    nulls = _count(
        store,
        f"SELECT count(*) AS c FROM {entity_type} WHERE {key_col} IS NULL",
    )
    assert nulls == 0, (
        f"{entity_type}.{key_col} has {nulls} NULL row(s); backfill incomplete"
    )


# ---------------------------------------------------------------------------
# 5. Indexes exist
# ---------------------------------------------------------------------------


def test_key_indexes_exist(arcadedb_store):
    """The composite ``<field>_key`` UNIQUE indexes are live in the schema."""
    store = arcadedb_store
    rows = _rows(store, "SELECT name FROM schema:indexes")
    names = {r["name"] for r in rows}
    for expected in (
        "RADAR_SYSTEM[system_name_key,entity_type]",
        "MISSILE_SYSTEM[system_name_key,entity_type]",
    ):
        assert expected in names, (
            f"index {expected!r} not found. Present *_key indexes: "
            f"{sorted(n for n in names if '_key' in n)}"
        )


# ---------------------------------------------------------------------------
# 6. Idempotent upsert — the definitive end-to-end proof (WRITES, creates none)
# ---------------------------------------------------------------------------


def test_idempotent_upsert_uppercase_resolves_to_survivor(arcadedb_store):
    """Upsert identity ``system_name='FAN SONG'`` (uppercase) through the REAL
    write path and assert it converges on the existing 'Fan Song' vertex:

      (a) returns the SAME RID as the existing survivor (#37:4),
      (b) RADAR_SYSTEM count is UNCHANGED (no new vertex created), and
      (c) the display ``system_name`` stays first-seen 'Fan Song' (not clobbered
          to uppercase).

    This exercises WHERE-on-``system_name_key`` live: ``norm('FAN SONG')`` =
    'fan song' matches the survivor, so the UPSERT takes its UPDATE branch.
    """
    from app.services.graph_store import NodeRecord

    store = arcadedb_store

    # Resolve the survivor + preserve its confidence so the (unconditional)
    # mutable SET does not clobber real data. Empty properties dict keeps
    # nominal_rf_mhz / tx_peak_power_kw untouched (they are set only when passed).
    before = _rows(
        store,
        "SELECT @rid AS rid, extraction_confidence AS conf "
        "FROM RADAR_SYSTEM WHERE system_name_key = :k",
        {"k": "fan song"},
    )
    assert len(before) == 1, f"expected 1 pre-existing survivor, got {before}"
    survivor_rid = before[0]["rid"]
    survivor_conf = before[0]["conf"]
    assert survivor_rid == _FAN_SONG_RID

    count_before = _count(store, "SELECT count(*) AS c FROM RADAR_SYSTEM")
    assert count_before == _RADAR_COUNT, (
        f"pre-upsert RADAR_SYSTEM count {count_before} != baseline {_RADAR_COUNT}"
    )

    record = NodeRecord(
        entity_type="RADAR_SYSTEM",
        identity_fields={"system_name": "FAN SONG"},  # uppercase variant
        name="FAN SONG",
        properties={},  # do not touch survivor's mutable fields
        extraction_confidence=survivor_conf,  # preserve existing confidence
    )

    returned_rid = None
    try:
        # provenance=None: do NOT create a HAS_PROVENANCE edge (that would mutate
        # the survivor's edge set). Pure identity-resolution write.
        rids = store.upsert_nodes_batch_sync([record], provenance=None)
        assert rids, f"upsert returned no RID: {rids!r}"
        returned_rid = rids[0]

        # (a) same RID as the existing survivor
        assert returned_rid == survivor_rid, (
            f"uppercase 'FAN SONG' upsert returned {returned_rid}, expected the "
            f"existing survivor {survivor_rid} — WHERE-on-system_name_key dedup "
            "did NOT match the pre-existing vertex"
        )

        # (b) count unchanged — nothing created
        count_after = _count(store, "SELECT count(*) AS c FROM RADAR_SYSTEM")
        assert count_after == count_before, (
            f"RADAR_SYSTEM count changed {count_before} -> {count_after}; the "
            "uppercase upsert created a duplicate vertex instead of merging"
        )

        # (c) first-seen display casing preserved
        disp = _rows(
            store,
            "SELECT system_name FROM RADAR_SYSTEM WHERE system_name_key = :k",
            {"k": "fan song"},
        )
        assert len(disp) == 1, f"expected still exactly 1 vertex, got {disp}"
        assert disp[0]["system_name"] == "Fan Song", (
            f"display system_name was clobbered to {disp[0]['system_name']!r}; "
            "COALESCE first-seen write-once failed"
        )
    finally:
        # Defensive cleanup: if a regression ever inserts a NEW uppercase vertex
        # (returned RID differs from the survivor), remove it so a failed run
        # never leaves junk. The survivor's display name is 'Fan Song', so a
        # delete keyed on the uppercase display string can never hit it.
        if returned_rid is not None and returned_rid != survivor_rid:
            try:
                store.execute_command_sync(
                    "DELETE FROM RADAR_SYSTEM WHERE system_name = 'FAN SONG'"
                )
            except Exception:
                pass
