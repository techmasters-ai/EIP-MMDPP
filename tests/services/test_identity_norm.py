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
