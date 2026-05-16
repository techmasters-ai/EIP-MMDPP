"""Tests for `_replace_raw_table_refs_in_body_children` — the in-place
substitution helper used by synth-only passes.

Regression guard for the architectural fix in docs/sa2_extraction_runs.md
(Run V9-POST-FIX-GATED): synth refs replace raw table refs at the original
document position rather than being appended at the end of body.children.
This preserves page-order proximity between table content and surrounding
prose, which the identity/prose passes were quietly relying on under v9.
"""
from __future__ import annotations

from app.services.table_normalization._pipeline_hooks import (
    _replace_raw_table_refs_in_body_children,
)


def test_simple_in_place_substitution_preserves_sibling_order():
    """Replace a `#/tables/0` ref with 3 synth `#/texts/N` refs in the middle
    of a list — surrounding siblings stay in their original positions."""
    doc = {
        "body": {
            "children": [
                {"$ref": "#/texts/10"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/11"},
            ],
        },
    }
    n = _replace_raw_table_refs_in_body_children(
        doc, {"#/tables/0": ["#/texts/316", "#/texts/317", "#/texts/318"]}
    )
    assert n == 1
    assert doc["body"]["children"] == [
        {"$ref": "#/texts/10"},
        {"$ref": "#/texts/316"},
        {"$ref": "#/texts/317"},
        {"$ref": "#/texts/318"},
        {"$ref": "#/texts/11"},
    ]


def test_multiple_tables_each_substituted():
    doc = {
        "body": {
            "children": [
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/5"},
                {"$ref": "#/tables/1"},
            ],
        },
    }
    n = _replace_raw_table_refs_in_body_children(
        doc, {
            "#/tables/0": ["#/texts/100"],
            "#/tables/1": ["#/texts/200", "#/texts/201"],
        }
    )
    assert n == 2
    assert doc["body"]["children"] == [
        {"$ref": "#/texts/100"},
        {"$ref": "#/texts/5"},
        {"$ref": "#/texts/200"},
        {"$ref": "#/texts/201"},
    ]


def test_table_ref_not_in_replacement_map_left_alone():
    """An OTHER-shape table (or any table whose normalization produced no
    synth refs) is NOT in the replacement map and should not be touched."""
    doc = {
        "body": {
            "children": [
                {"$ref": "#/tables/0"},  # normalized → has synth refs
                {"$ref": "#/tables/9"},  # not in map → stays
            ],
        },
    }
    n = _replace_raw_table_refs_in_body_children(
        doc, {"#/tables/0": ["#/texts/100"]}
    )
    assert n == 1
    assert doc["body"]["children"] == [
        {"$ref": "#/texts/100"},
        {"$ref": "#/tables/9"},
    ]


def test_nested_children_recursed():
    """A `#/tables/N` ref deep in a nested children tree still gets replaced."""
    doc = {
        "body": {
            "children": [
                {
                    "self_ref": "#/groups/0",
                    "children": [
                        {"$ref": "#/texts/1"},
                        {"$ref": "#/tables/0"},
                    ],
                },
            ],
        },
    }
    n = _replace_raw_table_refs_in_body_children(
        doc, {"#/tables/0": ["#/texts/100", "#/texts/101"]}
    )
    assert n == 1
    inner = doc["body"]["children"][0]["children"]
    assert inner == [
        {"$ref": "#/texts/1"},
        {"$ref": "#/texts/100"},
        {"$ref": "#/texts/101"},
    ]


def test_empty_replacement_map_is_noop():
    doc = {"body": {"children": [{"$ref": "#/tables/0"}]}}
    n = _replace_raw_table_refs_in_body_children(doc, {})
    assert n == 0
    assert doc["body"]["children"] == [{"$ref": "#/tables/0"}]


def test_tables_array_untouched():
    """The structured `tables[]` array must remain intact — overlay machinery
    reads it directly even after we remove the body.children $ref."""
    doc = {
        "tables": [{"self_ref": "#/tables/0", "data": "structured-table-data"}],
        "body": {"children": [{"$ref": "#/tables/0"}]},
    }
    _replace_raw_table_refs_in_body_children(
        doc, {"#/tables/0": ["#/texts/100"]}
    )
    assert doc["tables"] == [{"self_ref": "#/tables/0", "data": "structured-table-data"}]


def test_picture_and_other_refs_unaffected():
    """`#/pictures/N`, `#/texts/N`, and non-dict children pass through untouched."""
    doc = {
        "body": {
            "children": [
                {"$ref": "#/pictures/0"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/5"},
                "not-a-dict-sibling",
            ],
        },
    }
    n = _replace_raw_table_refs_in_body_children(
        doc, {"#/tables/0": ["#/texts/100"]}
    )
    assert n == 1
    assert doc["body"]["children"] == [
        {"$ref": "#/pictures/0"},
        {"$ref": "#/texts/100"},
        {"$ref": "#/texts/5"},
        "not-a-dict-sibling",
    ]
