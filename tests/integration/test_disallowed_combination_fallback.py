"""§9.1 flag-matrix row 4: disallowed-combination fallback.

When DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true AND
DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=true simultaneously, the
integration block in main.py:~574 must:
- Log an ERROR
- Fall back to today's behavior (neither path fires)

This is a unit-test against the integration logic — checking the flag-
matrix branches by reading the source.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAIN_PY = REPO_ROOT / "docker" / "docling-graph" / "app" / "main.py"


@pytest.fixture(scope="module")
def main_source() -> str:
    if not MAIN_PY.exists():
        pytest.skip(f"main.py missing at {MAIN_PY}")
    return MAIN_PY.read_text()


def test_flag_matrix_has_both_flags_branch(main_source):
    """Both flags-true branch exists and logs ERROR."""
    assert "if _norm_on and _exp_on:" in main_source, (
        "expected `if _norm_on and _exp_on:` branch in flag matrix"
    )
    # ERROR-level log mentioning both flag names + "true"
    pattern = re.compile(
        r"_norm_on and _exp_on.*?logger\.error\(.*?"
        r"DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED.*?"
        r"DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS.*?"
        r"true",
        re.DOTALL,
    )
    assert pattern.search(main_source), (
        "expected ERROR log naming both env vars + 'true' inside the "
        "_norm_on and _exp_on branch"
    )


def test_falls_back_to_today_behavior_when_both_true(main_source):
    """The branch must NOT call into normalize_tables or synthesize_table_facts."""
    # Find the both-flags-true branch
    m = re.search(
        r"if _norm_on and _exp_on:(.*?)elif _norm_on:",
        main_source, re.DOTALL,
    )
    assert m, "could not locate both-flags-true branch"
    branch_body = m.group(1)
    assert "normalize_tables" not in branch_body, (
        "disallowed-combo branch must not call normalize_tables"
    )
    assert "synthesize_table_facts" not in branch_body, (
        "disallowed-combo branch must not call synthesize_table_facts"
    )


def test_flag_matrix_has_elif_norm_on_branch(main_source):
    """The normalization path (elif _norm_on:) exists and calls normalize_tables."""
    m = re.search(
        r"elif _norm_on:(.*?)elif _exp_on:",
        main_source, re.DOTALL,
    )
    assert m, "could not locate elif _norm_on: branch"
    body = m.group(1)
    assert "normalize_tables" in body, "norm-on branch must call normalize_tables"
    assert "_text_item_from_chunk" in body, "norm-on branch must call _text_item_from_chunk"
    assert "_body_children.append" in body, (
        "norm-on branch must update body.children (per spec §10.1 bugfix)"
    )


def test_flag_matrix_has_elif_exp_on_branch(main_source):
    """The experimental path (elif _exp_on:) exists and calls synthesize_table_facts."""
    # Match all text after `elif _exp_on:` until the next elif/else/method boundary
    m = re.search(r"elif _exp_on:(.*?)(?:\n    elif|\n    else:|\n    if _is_empty)",
                  main_source, re.DOTALL)
    assert m, "could not locate elif _exp_on: branch"
    body = m.group(1)
    assert "synthesize_table_facts" in body, (
        "exp-on branch must call synthesize_table_facts"
    )


def test_per_pass_bridge_reset_present(main_source):
    """Per spec: _bridge_reset() must be called at the start of each pass to
    prevent cross-pass cell_refs leakage."""
    assert "_bridge_reset()" in main_source, (
        "expected per-pass bridge reset call (_bridge_reset()) before "
        "the flag-matrix branches"
    )
