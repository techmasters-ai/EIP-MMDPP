"""Tests for case/whitespace-insensitive LogicalIdentity equality (Task 0).

Spec: two LogicalIdentity instances whose identity values differ only by
case/whitespace must be == and hash-equal (so they merge into one record
in the merge index), while the raw first-seen values are preserved for
display via identity_values_dict() / identity_tuple.
"""
from app.services.extraction_merge import LogicalIdentity, norm


def _make(system_name: str) -> LogicalIdentity:
    return LogicalIdentity(
        entity_type="RADAR_SYSTEM",
        identity_field_names=("system_name",),
        identity_tuple=(system_name,),
        scope="global",
        document_id=None,
    )


def _make_doc(system_name: str, document_id: str | None) -> LogicalIdentity:
    return LogicalIdentity(
        entity_type="RADAR_SYSTEM",
        identity_field_names=("system_name",),
        identity_tuple=(system_name,),
        scope="document",
        document_id=document_id,
    )


def test_norm():
    assert norm("  FAN   Song ") == "fan song"
    assert norm(None) == ""


def test_case_variants_equal_and_hash_equal():
    a = _make("Fan Song")
    b = _make("FAN SONG")
    assert a == b
    assert hash(a) == hash(b)


def test_distinct_names_not_equal():
    a = _make("Fan Song")
    b = _make("Low Blow")
    assert a != b


def test_display_values_stay_raw():
    a = _make("Fan Song")
    assert a.identity_values_dict()["system_name"] == "Fan Song"
    assert a.identity_tuple == ("Fan Song",)


def test_whitespace_only_variants_equal_and_hash_equal():
    # Same casing, differs only by leading/trailing/internal whitespace.
    a = _make("Fan Song")
    b = _make("  Fan   Song ")
    assert a == b
    assert hash(a) == hash(b)


def test_document_scope_different_document_id_not_equal():
    # Same normalized identity value, different documents → distinct entities.
    a = _make_doc("Fan Song", "DOC1")
    b = _make_doc("Fan Song", "DOC2")
    assert a != b


def test_document_scope_same_document_id_case_variant_merges():
    # Same document + case/whitespace variant of the value → merge.
    a = _make_doc("Fan Song", "DOC1")
    b = _make_doc("  FAN   song ", "DOC1")
    assert a == b
    assert hash(a) == hash(b)


def test_document_id_compared_raw_case_sensitive():
    # document_id is a UUID, compared RAW — case matters, so these differ.
    a = _make_doc("Fan Song", "DOC1")
    b = _make_doc("Fan Song", "doc1")
    assert a != b


def test_none_element_in_identity_tuple_does_not_crash():
    a = LogicalIdentity(
        entity_type="RADAR_SYSTEM",
        identity_field_names=("system_name", "designation"),
        identity_tuple=("Fan Song", None),
        scope="global",
        document_id=None,
    )
    # norm_key / hash / eq must all tolerate the None (norms to "").
    assert a.norm_key == ("RADAR_SYSTEM", ("fan song", ""), None)
    assert hash(a) == hash(a)
    b = LogicalIdentity(
        entity_type="RADAR_SYSTEM",
        identity_field_names=("system_name", "designation"),
        identity_tuple=("FAN SONG", None),
        scope="global",
        document_id=None,
    )
    assert a == b
