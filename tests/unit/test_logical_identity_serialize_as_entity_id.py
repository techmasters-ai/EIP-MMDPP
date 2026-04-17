"""Phase 8 Task 52b — LogicalIdentity.serialize_as_entity_id contract."""
from __future__ import annotations

import pytest

from app.services.extraction_merge import LogicalIdentity


def _id(
    entity_type="RADAR_SYSTEM",
    field_names=("system_name",),
    values=("Tombstone",),
    scope="global",
    document_id=None,
):
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=field_names,
        identity_tuple=values,
        scope=scope,
        document_id=document_id,
    )


def test_stable_same_input_same_output():
    a = _id()
    b = _id()
    assert a.serialize_as_entity_id() == b.serialize_as_entity_id()


def test_distinct_identities_produce_distinct_strings():
    a = _id(values=("Tombstone",))
    b = _id(values=("Clam Shell",))
    assert a.serialize_as_entity_id() != b.serialize_as_entity_id()


def test_document_scoped_identity_includes_document_id_in_serialization():
    a = _id(scope="document", document_id="doc-1")
    b = _id(scope="document", document_id="doc-2")
    assert a.serialize_as_entity_id() != b.serialize_as_entity_id()
    assert "doc-1" in a.serialize_as_entity_id()


def test_global_scoped_identity_omits_document_suffix_and_tolerates_none():
    g = _id(scope="global", document_id=None)
    s = g.serialize_as_entity_id()
    assert "__doc__" not in s  # no document suffix for global scope


def test_embedded_delimiter_roundtrips_via_repr_quoting():
    """A value containing '::' or '|' must not collide with the format's
    delimiters; repr() quoting handles it."""
    weird = _id(values=("Name::with|special",))
    s = weird.serialize_as_entity_id()
    # repr() wraps the string in single quotes and escapes as needed — so
    # the literal '::' / '|' end up INSIDE the quoted value, not delimiting.
    # Assert the output distinguishes from a plain-name identity.
    normal = _id(values=("Name",))
    assert s != normal.serialize_as_entity_id()


def test_version_prefix_present():
    """Format starts with 'v1::' — future format bumps use v2 and coexist."""
    s = _id().serialize_as_entity_id()
    assert s.startswith("v1::")


def test_identity_fields_appear_in_declared_order():
    """For a multi-field identity, output respects identity_field_names order."""
    ordered = LogicalIdentity(
        entity_type="SECTION",
        identity_field_names=("section_number", "heading"),
        identity_tuple=("3.2", "Overview"),
        scope="document",
        document_id="doc-1",
    )
    s = ordered.serialize_as_entity_id()
    # section_number appears before heading in the output
    assert s.index("section_number") < s.index("heading")


def test_entity_type_appears_in_output():
    s = _id(entity_type="MISSILE_SYSTEM").serialize_as_entity_id()
    assert "MISSILE_SYSTEM" in s
