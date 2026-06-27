"""Unit tests for the merged-chunk row accessors in extraction_chunk_index.py.

Phase 1 Task 1 of the merged-chunk routing plan
(`docs/superpowers/plans/2026-05-27-merged-chunk-routing.md`).

The three accessors provide None-coalescing reads for columns added to the
``ExtractionChunk`` vertex in this task:

* ``read_chunk_source_refs(row) -> list[str]``   — default ``[]``
* ``read_chunk_token_count(row) -> int``         — default ``0``
* ``read_chunk_index(row) -> int``               — default ``-1``

The accessors must NEVER return ``None``. They must handle:
    - native ArcadeDB LIST property values
    - JSON-encoded string fallback (``source_refs_json``)
    - legacy rows (column missing, or column explicitly ``None``)

Tests are unit (no ArcadeDB, no Ollama).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# read_chunk_source_refs
# ---------------------------------------------------------------------------


def test_read_chunk_source_refs_returns_native_list():
    """ArcadeDB LIST values come back as Python lists; pass through directly."""
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs": ["#/texts/35", "#/texts/36"], "chunk_index": 3}
    assert read_chunk_source_refs(row) == ["#/texts/35", "#/texts/36"]


def test_read_chunk_source_refs_returns_empty_list_for_legacy_row():
    """Legacy per-element row has no source_refs column at all → []."""
    from app.services.extraction_chunk_index import read_chunk_source_refs

    legacy = {"chunk_index": -1, "self_ref": "#/texts/0"}
    assert read_chunk_source_refs(legacy) == []


def test_read_chunk_source_refs_returns_empty_list_when_none():
    """Column explicitly None (legacy row before backfill) → []."""
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs": None, "chunk_index": -1}
    assert read_chunk_source_refs(row) == []


def test_read_chunk_source_refs_returns_empty_list_when_empty_list():
    """Already-empty list stays empty (no coercion to None or other weirdness)."""
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs": [], "chunk_index": 4}
    assert read_chunk_source_refs(row) == []


def test_read_chunk_source_refs_falls_back_to_json_string():
    """If LIST migration ever fails, JSON-string fallback under
    ``source_refs_json`` is the documented escape hatch. The accessor
    must transparently decode it.
    """
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs_json": '["#/texts/7", "#/texts/8"]'}
    assert read_chunk_source_refs(row) == ["#/texts/7", "#/texts/8"]


def test_read_chunk_source_refs_native_list_wins_over_json_string():
    """When both are present (transition state), the native LIST takes precedence."""
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {
        "source_refs": ["#/texts/1"],
        "source_refs_json": '["#/texts/999"]',
    }
    assert read_chunk_source_refs(row) == ["#/texts/1"]


def test_read_chunk_source_refs_handles_corrupt_json_string_as_empty():
    """A corrupt JSON-string fallback should not raise — coalesce to []."""
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs_json": "not valid json"}
    assert read_chunk_source_refs(row) == []


def test_read_chunk_source_refs_never_returns_none():
    """Documented contract: NEVER returns None across every input shape."""
    from app.services.extraction_chunk_index import read_chunk_source_refs

    for row in ({}, {"source_refs": None}, {"source_refs_json": None}):
        result = read_chunk_source_refs(row)
        assert result is not None
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# read_chunk_token_count
# ---------------------------------------------------------------------------


def test_read_chunk_token_count_returns_value():
    from app.services.extraction_chunk_index import read_chunk_token_count

    row = {"token_count": 312}
    assert read_chunk_token_count(row) == 312


def test_read_chunk_token_count_returns_zero_for_legacy_row():
    from app.services.extraction_chunk_index import read_chunk_token_count

    legacy = {"chunk_index": -1}  # token_count absent
    assert read_chunk_token_count(legacy) == 0


def test_read_chunk_token_count_returns_zero_when_none():
    from app.services.extraction_chunk_index import read_chunk_token_count

    null_row = {"chunk_index": 0, "token_count": None}
    assert read_chunk_token_count(null_row) == 0


def test_read_chunk_token_count_coerces_string_int():
    """ArcadeDB sometimes serialises INTEGER as a JSON string in HTTP replies;
    the accessor should coerce or fall back to 0 without raising.
    """
    from app.services.extraction_chunk_index import read_chunk_token_count

    row = {"token_count": "42"}
    assert read_chunk_token_count(row) == 42


def test_read_chunk_token_count_handles_garbage_as_zero():
    from app.services.extraction_chunk_index import read_chunk_token_count

    row = {"token_count": "not-a-number"}
    assert read_chunk_token_count(row) == 0


# ---------------------------------------------------------------------------
# read_chunk_index
# ---------------------------------------------------------------------------


def test_read_chunk_index_returns_value():
    from app.services.extraction_chunk_index import read_chunk_index

    row = {"chunk_index": 3}
    assert read_chunk_index(row) == 3


def test_read_chunk_index_returns_minus_one_for_legacy_row():
    from app.services.extraction_chunk_index import read_chunk_index

    legacy = {"self_ref": "#/texts/0"}  # chunk_index absent
    assert read_chunk_index(legacy) == -1


def test_read_chunk_index_returns_minus_one_when_none():
    from app.services.extraction_chunk_index import read_chunk_index

    null_row = {"chunk_index": None}
    assert read_chunk_index(null_row) == -1


def test_read_chunk_index_handles_zero_correctly():
    """chunk_index=0 is a valid merged-chunk position, NOT a falsy legacy marker."""
    from app.services.extraction_chunk_index import read_chunk_index

    row = {"chunk_index": 0}
    assert read_chunk_index(row) == 0


def test_read_chunk_index_handles_garbage_as_minus_one():
    from app.services.extraction_chunk_index import read_chunk_index

    row = {"chunk_index": "not-a-number"}
    assert read_chunk_index(row) == -1


# ---------------------------------------------------------------------------
# read_chunk_section_path — router-scoring Part 1 (section signal)
# ---------------------------------------------------------------------------


def test_read_chunk_section_path_returns_value():
    from app.services.extraction_chunk_index import read_chunk_section_path

    row = {"section_path": "Radar Systems > Fan Song"}
    assert read_chunk_section_path(row) == "Radar Systems > Fan Song"


def test_read_chunk_section_path_returns_none_for_legacy_row():
    """Legacy rows (indexed before Part 1) have no section_path column → None."""
    from app.services.extraction_chunk_index import read_chunk_section_path

    legacy = {"chunk_index": 3, "self_ref": "chunk_3"}
    assert read_chunk_section_path(legacy) is None


def test_read_chunk_section_path_returns_none_when_none():
    from app.services.extraction_chunk_index import read_chunk_section_path

    row = {"section_path": None, "chunk_index": 0}
    assert read_chunk_section_path(row) is None


def test_read_chunk_section_path_returns_none_for_blank_string():
    """A whitespace-only section_path is treated as 'no section' → None,
    so the section matcher never tries to match an empty crumb.
    """
    from app.services.extraction_chunk_index import read_chunk_section_path

    assert read_chunk_section_path({"section_path": ""}) is None
    assert read_chunk_section_path({"section_path": "   "}) is None


def test_read_chunk_section_path_coerces_non_string_to_string():
    """Defensive: a non-string, non-None value is coerced to str rather than
    raising (mirrors the never-crash contract of the other accessors).
    """
    from app.services.extraction_chunk_index import read_chunk_section_path

    assert read_chunk_section_path({"section_path": 123}) == "123"


# ---------------------------------------------------------------------------
# read_chunk_headings — router-scoring Part 1 (section signal)
# ---------------------------------------------------------------------------


def test_read_chunk_headings_returns_native_list():
    from app.services.extraction_chunk_index import read_chunk_headings

    row = {"headings": ["Radar Systems", "Fan Song"]}
    assert read_chunk_headings(row) == ["Radar Systems", "Fan Song"]


def test_read_chunk_headings_returns_empty_list_for_legacy_row():
    """Legacy rows have no headings column → [] (never None)."""
    from app.services.extraction_chunk_index import read_chunk_headings

    legacy = {"chunk_index": 3, "self_ref": "chunk_3"}
    assert read_chunk_headings(legacy) == []


def test_read_chunk_headings_returns_empty_list_when_none():
    from app.services.extraction_chunk_index import read_chunk_headings

    row = {"headings": None, "chunk_index": 0}
    assert read_chunk_headings(row) == []


def test_read_chunk_headings_returns_empty_list_when_empty_list():
    from app.services.extraction_chunk_index import read_chunk_headings

    row = {"headings": [], "chunk_index": 4}
    assert read_chunk_headings(row) == []


def test_read_chunk_headings_falls_back_to_json_string():
    """Mirror source_refs: a JSON-string fallback under ``headings_json`` is
    transparently decoded.
    """
    from app.services.extraction_chunk_index import read_chunk_headings

    row = {"headings_json": '["Radar Systems", "Fan Song"]'}
    assert read_chunk_headings(row) == ["Radar Systems", "Fan Song"]


def test_read_chunk_headings_native_list_wins_over_json_string():
    from app.services.extraction_chunk_index import read_chunk_headings

    row = {
        "headings": ["Radar Systems"],
        "headings_json": '["WRONG"]',
    }
    assert read_chunk_headings(row) == ["Radar Systems"]


def test_read_chunk_headings_handles_corrupt_json_string_as_empty():
    from app.services.extraction_chunk_index import read_chunk_headings

    row = {"headings_json": "not valid json"}
    assert read_chunk_headings(row) == []


def test_read_chunk_headings_filters_nones_from_list():
    from app.services.extraction_chunk_index import read_chunk_headings

    row = {"headings": [None, "Radar Systems", None]}
    assert read_chunk_headings(row) == ["Radar Systems"]


def test_read_chunk_headings_never_returns_none():
    from app.services.extraction_chunk_index import read_chunk_headings

    for row in ({}, {"headings": None}, {"headings_json": None}):
        result = read_chunk_headings(row)
        assert result is not None
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Object access (not just dicts) — accessors must accept object-style rows
# ---------------------------------------------------------------------------


def test_accessors_accept_object_like_rows():
    """ArcadeDB sometimes returns row objects with attribute access. The
    accessors take ``dict | object``; both must work.
    """
    from app.services.extraction_chunk_index import (
        read_chunk_headings,
        read_chunk_index,
        read_chunk_section_path,
        read_chunk_source_refs,
        read_chunk_token_count,
    )

    class _Row:
        chunk_index = 5
        source_refs = ["#/texts/7"]
        token_count = 128
        section_path = "Radar Systems > Fan Song"
        headings = ["Radar Systems", "Fan Song"]

    row = _Row()
    assert read_chunk_index(row) == 5
    assert read_chunk_source_refs(row) == ["#/texts/7"]
    assert read_chunk_token_count(row) == 128
    assert read_chunk_section_path(row) == "Radar Systems > Fan Song"
    assert read_chunk_headings(row) == ["Radar Systems", "Fan Song"]


# ---------------------------------------------------------------------------
# M2 — corrupt source_refs of unexpected type should still return [],
# but a warning must be emitted so the data-shape bug is observable.
# ---------------------------------------------------------------------------


def test_read_chunk_source_refs_warns_on_unexpected_type(caplog):
    """An int (or any non-list/non-string) in source_refs is a data-shape bug.

    The accessor must STILL return [] (safe), but emit a WARNING so the bug
    is visible — silently coercing to [] masks an upstream corruption.
    """
    import logging

    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs": 42, "chunk_index": 1}

    with caplog.at_level(logging.WARNING, logger="app.services.extraction_chunk_index"):
        result = read_chunk_source_refs(row)

    assert result == []
    # The warning must mention the corrupting type so operators can grep
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "source_refs" in r.getMessage()
    ]
    assert warning_records, (
        "read_chunk_source_refs must emit a WARNING when source_refs is an "
        "unexpected non-None type — silently returning [] hides upstream data bugs."
    )
    # Confirm the type name appears in the warning (helps grep for the bug source)
    assert any("int" in r.getMessage() for r in warning_records), (
        "Warning should mention the unexpected type name (e.g. 'int')."
    )


def test_read_chunk_source_refs_no_warning_on_legacy_none(caplog):
    """None is the documented legacy/backfill state and must NOT warn —
    a flood of warnings on legacy rows would drown the signal.
    """
    import logging

    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs": None}
    with caplog.at_level(logging.WARNING, logger="app.services.extraction_chunk_index"):
        result = read_chunk_source_refs(row)
    assert result == []
    warnings_for_us = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "source_refs" in r.getMessage()
    ]
    assert warnings_for_us == [], (
        "None is the documented legacy state — must not warn."
    )


# ---------------------------------------------------------------------------
# M3 — source_refs accessor must filter None elements out of the list,
# not stringify them to the literal "None".
# ---------------------------------------------------------------------------


def test_read_chunk_source_refs_filters_nones_from_list():
    """A None element in the native LIST path must be dropped, not coerced
    to the string ``"None"`` (which is almost certainly worse than dropping).
    """
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs": [None, "#/texts/35", None]}
    assert read_chunk_source_refs(row) == ["#/texts/35"]


def test_read_chunk_source_refs_filters_nones_from_json_fallback():
    """Same filter behaviour for the JSON-decode fallback path."""
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs_json": '[null, "#/texts/36"]'}
    assert read_chunk_source_refs(row) == ["#/texts/36"]


def test_read_chunk_source_refs_filters_nones_from_string_json_in_source_refs():
    """When source_refs holds a JSON-string (transition state), the None
    filter must apply to its decoded contents too.
    """
    from app.services.extraction_chunk_index import read_chunk_source_refs

    row = {"source_refs": '[null, "#/texts/37", null, "#/texts/38"]'}
    assert read_chunk_source_refs(row) == ["#/texts/37", "#/texts/38"]


# ---------------------------------------------------------------------------
# read_chunk_is_table — TABLE signal (is_table wiring)
# ---------------------------------------------------------------------------


def test_read_chunk_is_table_returns_false_for_legacy_row():
    """Legacy rows (indexed before the is_table wiring) lack the column
    entirely — the accessor must False-coalesce so historical runs keep
    scoring is_table=0.0 (no migration)."""
    from app.services.extraction_chunk_index import read_chunk_is_table

    row = {"chunk_index": 3, "chunk_text": "legacy"}
    assert read_chunk_is_table(row) is False


def test_read_chunk_is_table_returns_false_when_none():
    from app.services.extraction_chunk_index import read_chunk_is_table

    row = {"is_table": None}
    assert read_chunk_is_table(row) is False


def test_read_chunk_is_table_returns_native_bool():
    from app.services.extraction_chunk_index import read_chunk_is_table

    assert read_chunk_is_table({"is_table": True}) is True
    assert read_chunk_is_table({"is_table": False}) is False


def test_read_chunk_is_table_coerces_stringified_bool():
    """ArcadeDB HTTP serialisation occasionally stringifies scalar columns —
    only 'true'/'1'/'yes' (case-insensitive) read as True."""
    from app.services.extraction_chunk_index import read_chunk_is_table

    assert read_chunk_is_table({"is_table": "true"}) is True
    assert read_chunk_is_table({"is_table": "TRUE"}) is True
    assert read_chunk_is_table({"is_table": "1"}) is True
    assert read_chunk_is_table({"is_table": "yes"}) is True
    assert read_chunk_is_table({"is_table": "false"}) is False
    assert read_chunk_is_table({"is_table": "0"}) is False
    assert read_chunk_is_table({"is_table": ""}) is False
    assert read_chunk_is_table({"is_table": "garbage"}) is False


def test_read_chunk_is_table_coerces_numeric():
    from app.services.extraction_chunk_index import read_chunk_is_table

    assert read_chunk_is_table({"is_table": 1}) is True
    assert read_chunk_is_table({"is_table": 0}) is False


def test_read_chunk_is_table_handles_garbage_as_false():
    """Non-coercible types must NEVER raise — False is the safe default."""
    from app.services.extraction_chunk_index import read_chunk_is_table

    assert read_chunk_is_table({"is_table": ["unexpected"]}) is False
    assert read_chunk_is_table({"is_table": {"nested": True}}) is False


def test_read_chunk_is_table_accepts_object_like_rows():
    """Mirrors test_accessors_accept_object_like_rows for the new accessor."""
    from app.services.extraction_chunk_index import read_chunk_is_table

    class _Row:
        is_table = True

    class _LegacyRow:
        chunk_index = 7

    assert read_chunk_is_table(_Row()) is True
    assert read_chunk_is_table(_LegacyRow()) is False
