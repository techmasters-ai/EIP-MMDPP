"""Unit tests for normalized ``<field>_key`` resolution in the ArcadeDB
write layer (case-insensitive-entity-identity, Task 1).

Pure SQL/param-builder tests — no live DB. These pin the DB-layer twin of
the case-insensitive ``LogicalIdentity`` equality (Task 0): every node write
path and every relationship endpoint must resolve on ``norm(value)`` stored
in a ``<field>_key`` column so that a later case/whitespace variant
(``FAN SONG`` after ``Fan Song``) upserts onto — and edges attach to — the
existing vertex, while the raw display fields are preserved first-seen.
"""
from __future__ import annotations

from app.services.arcadedb_graph import (
    _build_node_upsert_clauses,
    _build_upsert_node_script,
    _build_upsert_relationship_script,
    _build_where,
    _key_fields,
)
from app.services.extraction_merge import norm
from app.services.graph_store import NodeRecord, RelationshipRecord


def _radar(system_name: str = "Fan Song", **kw) -> NodeRecord:
    return NodeRecord(
        entity_type="RADAR_SYSTEM",
        identity_fields={"system_name": system_name},
        name=system_name,
        **kw,
    )


# ---------------------------------------------------------------------------
# norm sanity — the value that must show up in the WHERE
# ---------------------------------------------------------------------------

def test_norm_collapses_case_and_whitespace():
    assert norm("Fan Song") == "fan song"
    assert norm("FAN SONG") == "fan song"
    assert norm("  Fan   Song ") == "fan song"


# ---------------------------------------------------------------------------
# _key_fields
# ---------------------------------------------------------------------------

def test_key_fields_maps_to_normalized_key():
    keys = _key_fields({"system_name": "FAN SONG"})
    assert keys == {"system_name_key": "fan song"}


def test_key_fields_skips_document_id():
    keys = _key_fields({"section_number": "3.A", "document_id": "doc-UUID-1"})
    assert keys == {"section_number_key": norm("3.A")}
    assert "document_id_key" not in keys
    # document_id itself is not remapped/normalized.
    assert "document_id" not in keys


# ---------------------------------------------------------------------------
# _build_where — normalized key matching (+ raw document_id)
# ---------------------------------------------------------------------------

def test_build_where_matches_on_field_key():
    where = _build_where({"system_name": "Fan Song"})
    assert where == "system_name_key = :system_name_key"
    # raw identity field is NOT matched directly.
    assert "system_name =" not in where


def test_build_where_keeps_document_id_raw():
    where = _build_where({"section_number": "3.A", "document_id": "doc-1"})
    assert "section_number_key = :section_number_key" in where
    assert "document_id = :document_id" in where
    assert "document_id_key" not in where


def test_build_where_suffix_disambiguates_batch_params():
    where = _build_where({"system_name": "Fan Song"}, suffix="_2")
    assert where == "system_name_key = :system_name_key_2"


# ---------------------------------------------------------------------------
# _build_node_upsert_clauses — WHERE on key, first-seen display preserved
# ---------------------------------------------------------------------------

def test_node_clauses_where_on_key_with_norm_value():
    set_clause, where_clause, params = _build_node_upsert_clauses(_radar("FAN SONG"))
    assert where_clause == "system_name_key = :system_name_key"
    # The bound key param carries the normalized value.
    assert params["system_name_key"] == "fan song"


def test_node_clauses_persist_key_column():
    set_clause, _where, _params = _build_node_upsert_clauses(_radar("Fan Song"))
    # The <field>_key column is SET so a later variant's WHERE can match it.
    assert "system_name_key = :system_name_key" in set_clause


def test_node_clauses_display_name_is_first_seen_coalesce():
    set_clause, _where, params = _build_node_upsert_clauses(_radar("Fan Song"))
    # name must NOT be force-overwritten on update.
    assert "name = COALESCE(name, :name)" in set_clause
    assert "name = :name," not in set_clause and not set_clause.endswith("name = :name")
    assert params["name"] == "Fan Song"


def test_node_clauses_raw_identity_field_is_first_seen_coalesce():
    set_clause, _where, params = _build_node_upsert_clauses(_radar("Fan Song"))
    # raw display casing persisted write-once (never clobbered by a later
    # variant), keyed on a distinct *_raw param.
    assert "system_name = COALESCE(system_name, :system_name_raw)" in set_clause
    assert params["system_name_raw"] == "Fan Song"


def test_node_clauses_mutable_fields_overwrite():
    rec = _radar("Fan Song", extraction_confidence=0.7, properties={"band": "S"})
    set_clause, _where, _params = _build_node_upsert_clauses(rec)
    # entity_type / confidence / non-identity properties are overwritten each merge.
    assert "extraction_confidence = :extraction_confidence" in set_clause
    assert "band = :band" in set_clause
    assert "COALESCE(band" not in set_clause


def test_node_clauses_document_scope_keeps_document_id_raw():
    rec = NodeRecord(
        entity_type="SECTION",
        identity_fields={"section_number": "3.A", "document_id": "doc-1"},
        name="3.A Overview",
    )
    set_clause, where_clause, params = _build_node_upsert_clauses(rec)
    assert "section_number_key = :section_number_key" in where_clause
    assert "document_id = :document_id" in where_clause
    assert params["section_number_key"] == norm("3.A")
    assert params["document_id"] == "doc-1"
    # document_id is neither normalized nor COALESCE'd as a display field.
    assert "document_id_key" not in set_clause
    assert "COALESCE(document_id" not in set_clause


def test_node_clauses_property_named_like_identity_not_duplicated():
    # A property colliding with an identity field name must not double-emit
    # (the identity display COALESCE wins).
    rec = _radar("Fan Song", properties={"system_name": "SHOULD_NOT_OVERWRITE"})
    set_clause, _where, params = _build_node_upsert_clauses(rec)
    assert set_clause.count("system_name = ") == 1
    assert "system_name = COALESCE(system_name, :system_name_raw)" in set_clause
    assert params["system_name_raw"] == "Fan Song"


# ---------------------------------------------------------------------------
# _build_upsert_node_script — batch builder
# ---------------------------------------------------------------------------

def test_batch_script_where_on_key_with_norm_value():
    script, params = _build_upsert_node_script([_radar("FAN SONG")])
    assert "WHERE system_name_key = :system_name_key_0" in script
    assert params["system_name_key_0"] == "fan song"
    # The raw identity value must not appear in the WHERE match.
    assert "WHERE system_name = :" not in script


def test_batch_script_name_first_seen_coalesce():
    script, params = _build_upsert_node_script([_radar("Fan Song")])
    assert "name = COALESCE(name, :name_0)" in script
    assert params["name_0"] == "Fan Song"


def test_batch_script_two_case_variants_share_norm_key():
    # Two case-variant records produce the SAME normalized key value, so both
    # WHERE clauses resolve to one logical vertex.
    script, params = _build_upsert_node_script([_radar("Fan Song"), _radar("FAN SONG")])
    assert params["system_name_key_0"] == "fan song"
    assert params["system_name_key_1"] == "fan song"
    assert script.count("UPSERT RETURN AFTER @rid WHERE") == 2


# ---------------------------------------------------------------------------
# _build_upsert_relationship_script — endpoints resolve on <field>_key
# ---------------------------------------------------------------------------

def _rel(from_sys: str = "FAN SONG", to_platform: str = "SA-2") -> RelationshipRecord:
    return RelationshipRecord(
        from_type="RADAR_SYSTEM",
        from_identity={"system_name": from_sys},
        to_type="MISSILE_SYSTEM",
        to_identity={"system_name": to_platform},
        rel_type="ASSOCIATED_WITH",
    )


def test_relationship_endpoints_resolve_on_key():
    script, params = _build_upsert_relationship_script([_rel()], provenance=None)
    # from/to SELECTs match the normalized key, not the raw value.
    assert "system_name_key = :f_system_name_key_0" in script
    assert "system_name_key = :t_system_name_key_0" in script
    assert params["f_system_name_key_0"] == "fan song"
    assert params["t_system_name_key_0"] == norm("SA-2")


def test_relationship_endpoint_document_id_stays_raw():
    rec = RelationshipRecord(
        from_type="SECTION",
        from_identity={"section_number": "3.A", "document_id": "doc-1"},
        to_type="RADAR_SYSTEM",
        to_identity={"system_name": "Fan Song"},
        rel_type="MENTIONS",
    )
    script, params = _build_upsert_relationship_script([rec], provenance=None)
    assert "section_number_key = :f_section_number_key_0" in script
    assert "document_id = :f_document_id_0" in script
    assert params["f_document_id_0"] == "doc-1"
    assert params["f_section_number_key_0"] == norm("3.A")
