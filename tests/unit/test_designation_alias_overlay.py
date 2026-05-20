"""Step 6: deterministic designation alias expansion from labeled
designation-row tables.

Given a NormalizedTable whose identity rows include Industry / Military /
NATO designations alongside a Missile Type (or equivalent canonical)
row, derive the per-column alias tuple. The result is consumed by the
missile_identity postprocess to populate `nomenclature` (Military) and
`name` (NATO) on existing missile round entities — NOT to create new
top-level entities.

Generic — no equipment names anywhere. Operates on identity-row labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.designation_alias_overlay import (
    DesignationAliases,
    expand_designation_aliases,
    DesignationRowKind,
)
from app.services.table_normalization.models import (
    CellRef, NormalizedCell, NormalizedColumn, NormalizedRow,
    NormalizedTable, Shape, TableSection,
)


def _column(col_idx: int, identity: dict[str, str]) -> NormalizedColumn:
    """Build a NormalizedColumn with arbitrary display_name (unused here)."""
    return NormalizedColumn(
        col_idx=col_idx,
        identity=identity,
        display_name=identity.get("Missile Type") or f"col-{col_idx}",
    )


def _table(columns: list[NormalizedColumn]) -> NormalizedTable:
    return NormalizedTable(
        table_index=0,
        self_ref="#/tables/0",
        caption=None,
        page_numbers=(),
        shape=Shape.COLUMN_MAJOR,
        rows=(),
        columns=tuple(columns),
        sections=(),
        cells=(),
        raw_markdown="",
    )


class TestSingleColumnExpansion:
    def test_canonical_with_military_and_nato(self):
        """Column 7 of SA-2 Table 0: 20D / S-75V / S-75M / SA-2C."""
        t = _table([_column(7, {
            "Industry Designation": "S-75V",
            "Military Designation": "S-75M",
            "NATO Designation": "SA-2C",
            "Missile Type": "20D",
        })])
        out = expand_designation_aliases([t])
        assert len(out) == 1
        a = out[0]
        assert a.canonical_entity == "20D"
        assert "S-75M" in a.nomenclature_aliases
        assert "SA-2C" in a.name_aliases

    def test_canonical_with_dup_industry_military(self):
        """Column 3: 13D / S-75 / S-75 / SA-2C — Industry duplicates Military."""
        t = _table([_column(3, {
            "Industry Designation": "S-75",
            "Military Designation": "S-75",
            "NATO Designation": "SA-2C",
            "Missile Type": "13D",
        })])
        out = expand_designation_aliases([t])
        a = out[0]
        assert a.canonical_entity == "13D"
        # Duplicates collapsed to single alias
        assert a.nomenclature_aliases == ("S-75",)
        assert a.name_aliases == ("SA-2C",)

    def test_canonical_with_empty_industry(self):
        """Column 5: 13DA / (empty) / S-75M1 / SA-2D."""
        t = _table([_column(5, {
            "Industry Designation": "",
            "Military Designation": "S-75M1",
            "NATO Designation": "SA-2D",
            "Missile Type": "13DA",
        })])
        out = expand_designation_aliases([t])
        a = out[0]
        assert a.canonical_entity == "13DA"
        assert a.nomenclature_aliases == ("S-75M1",)
        assert a.name_aliases == ("SA-2D",)


class TestNoCanonicalEntityRow:
    """Per user guidance: do NOT create new top-level entities when no
    canonical Missile-Type row exists; just skip the column."""

    def test_skip_when_missile_type_row_missing(self):
        t = _table([_column(2, {
            "Industry Designation": "SA-75",
            "Military Designation": "SA-75",
            "NATO Designation": "SA-2A",
            # No "Missile Type" entry
        })])
        out = expand_designation_aliases([t])
        assert out == []

    def test_skip_when_canonical_value_empty(self):
        t = _table([_column(1, {
            "Industry Designation": "X",
            "Military Designation": "Y",
            "NATO Designation": "Z",
            "Missile Type": "",  # empty canonical
        })])
        out = expand_designation_aliases([t])
        assert out == []


class TestMultiColumnSourceTable:
    """Representative SA-2 Table 0 columns.

    The source-correct mapping is:
      * SA-2C attaches to 13D / 20D / 20DP.
      * SA-2D attaches to 13DM.
    """

    def test_sa2c_and_sa2d_attach_to_source_columns(self):
        cols = [
            _column(3, {"Military Designation": "S-75",  "NATO Designation": "SA-2C", "Missile Type": "13D"}),
            _column(4, {"Military Designation": "S-75",  "NATO Designation": "SA-2D", "Missile Type": "13DM"}),
            _column(7, {"Military Designation": "S-75M", "NATO Designation": "SA-2C", "Missile Type": "20D"}),
            _column(8, {"Military Designation": "S-75M", "NATO Designation": "SA-2C", "Missile Type": "20DP"}),
        ]
        out = expand_designation_aliases([_table(cols)])
        by_entity = {a.canonical_entity: a for a in out}
        for round_name in ("13D", "20D", "20DP"):
            assert round_name in by_entity
            assert "SA-2C" in by_entity[round_name].name_aliases
        assert "13DM" in by_entity
        assert by_entity["13DM"].name_aliases == ("SA-2D",)

    def test_sa2e_attaches_to_15d_only(self):
        cols = [
            _column(11, {"Military Designation": "S-75", "NATO Designation": "SA-2E", "Missile Type": "15D"}),
        ]
        out = expand_designation_aliases([_table(cols)])
        by_entity = {a.canonical_entity: a for a in out}
        assert "15D" in by_entity
        assert by_entity["15D"].name_aliases == ("SA-2E",)


class TestRowKindRecognition:
    """Identity-row labels must match flexibly (case-insensitive,
    whitespace-collapsed) but only against the explicit allowed set."""

    def test_exact_label_match(self):
        from app.services.designation_alias_overlay import _classify_identity_label
        assert _classify_identity_label("Industry Designation") == DesignationRowKind.INDUSTRY
        assert _classify_identity_label("Military Designation") == DesignationRowKind.MILITARY
        assert _classify_identity_label("NATO Designation") == DesignationRowKind.NATO
        assert _classify_identity_label("Missile Type") == DesignationRowKind.CANONICAL_ENTITY

    def test_case_and_whitespace_insensitive(self):
        from app.services.designation_alias_overlay import _classify_identity_label
        assert _classify_identity_label("  industry  designation  ") == DesignationRowKind.INDUSTRY
        assert _classify_identity_label("NATO DESIGNATION") == DesignationRowKind.NATO

    def test_unrelated_label_returns_none(self):
        from app.services.designation_alias_overlay import _classify_identity_label
        assert _classify_identity_label("Fan Song Variant") is None
        assert _classify_identity_label("Max Range") is None
        assert _classify_identity_label("Section: Stage 1") is None


class TestGuardrails:
    """Don't attach aliases when the alias value equals the canonical
    name (self-loops) or when the alias cell is missing."""

    def test_skip_alias_equal_to_canonical_name(self):
        """If Industry='1D' and Missile Type='1D', skip the redundant alias."""
        t = _table([_column(2, {
            "Industry Designation": "1D",
            "Military Designation": "SA-75",
            "NATO Designation": "SA-2A",
            "Missile Type": "1D",
        })])
        out = expand_designation_aliases([t])
        a = out[0]
        # Industry "1D" == canonical "1D" → dropped from nomenclature
        # Military "SA-75" survives
        assert "1D" not in a.nomenclature_aliases
        assert "SA-75" in a.nomenclature_aliases

    def test_predecessor_mentions_not_in_table_not_attached(self):
        """SA-25 from the prose phrase "earlier SA-25/S-25/SA-1 Guild"
        is NOT in Table 0 — so it should NOT show up in any expansion."""
        t = _table([_column(2, {
            "Military Designation": "SA-75",
            "NATO Designation": "SA-2A",
            "Missile Type": "1D",
        })])
        out = expand_designation_aliases([t])
        a = out[0]
        all_aliases = list(a.nomenclature_aliases) + list(a.name_aliases)
        assert "SA-25" not in all_aliases
        assert "SA-1" not in all_aliases

    def test_empty_table_list_returns_empty(self):
        assert expand_designation_aliases([]) == []

    def test_value_equal_to_identity_label_dropped(self):
        """Merged label cells in source tables can leak the row label
        as a VALUE for adjacent columns (e.g. col 1 ends up with
        `{'Industry Designation': 'Industry Designation', ...,
        'Missile Type': '1D'}`). Such values must not become aliases."""
        t = _table([_column(2, {
            "Industry Designation": "Industry Designation",
            "Military Designation": "Military Designation",
            "NATO Designation": "NATO Designation",
            "Missile Type": "1D",
        })])
        out = expand_designation_aliases([t])
        # All non-canonical values are themselves identity-row labels →
        # this column contributes no real aliases.
        assert out == [], (
            f"label-string values must not be promoted to aliases: {out}"
        )

    def test_table_with_no_identity_rows_skipped(self):
        """A purely-numeric table with no Industry/Military/NATO rows
        contributes nothing."""
        t = _table([_column(2, {"Max Range": "29000", "Max Alt": "22000"})])
        out = expand_designation_aliases([t])
        assert out == []


class TestNonSA2SyntheticCorpus:
    """Document-generic contract: the same alias-expansion rules must
    work on a designation table from a completely different corpus
    (synthetic names, no SA-2 / S-75 hardcoding). The overlay operates
    on identity-row labels and column position, not on equipment names."""

    def test_synthetic_aam_corpus_three_rounds(self):
        """A synthetic air-to-air missile corpus with the same
        Industry/Military/NATO row layout as SA-2 Table 0."""
        t = _table([
            _column(2, {
                "Industry Designation": "FAKE-AAM-A",
                "Military Designation": "MIL-1",
                "NATO Designation": "AAM-CODE-A",
                "Missile Type": "ZX-01",
            }),
            _column(3, {
                "Industry Designation": "FAKE-AAM-B",
                "Military Designation": "MIL-2",
                "NATO Designation": "AAM-CODE-B",
                "Missile Type": "ZX-02",
            }),
        ])
        out = expand_designation_aliases([t])
        by_entity = {a.canonical_entity: a for a in out}
        # Each synthetic round gets its column's aliases attached
        assert by_entity["ZX-01"].name_aliases == ("AAM-CODE-A",)
        assert by_entity["ZX-01"].nomenclature_aliases == ("MIL-1", "FAKE-AAM-A")
        assert by_entity["ZX-02"].name_aliases == ("AAM-CODE-B",)
        assert by_entity["ZX-02"].nomenclature_aliases == ("MIL-2", "FAKE-AAM-B")

    def test_synthetic_corpus_with_round_designation_label(self):
        """`Round Designation` is an alternate canonical row label
        recognized alongside `Missile Type`."""
        t = _table([_column(2, {
            "Industry Designation": "ALPHA-99",
            "NATO Designation": "ALPHA-CODE",
            "Round Designation": "R-99",
        })])
        out = expand_designation_aliases([t])
        assert len(out) == 1
        assert out[0].canonical_entity == "R-99"
        assert out[0].nomenclature_aliases == ("ALPHA-99",)
        assert out[0].name_aliases == ("ALPHA-CODE",)

    def test_multi_column_alias_spread_synthetic(self):
        """Same NATO code spanning multiple synthetic columns must
        attach to every canonical round, mirroring the SA-2C spread."""
        cols = [
            _column(i + 2, {
                "Military Designation": f"M-{i}",
                "NATO Designation": "SHARED-NATO",
                "Missile Type": f"VARIANT-{i}",
            })
            for i in range(3)
        ]
        out = expand_designation_aliases([_table(cols)])
        by_entity = {a.canonical_entity: a for a in out}
        for i in range(3):
            assert f"VARIANT-{i}" in by_entity
            assert by_entity[f"VARIANT-{i}"].name_aliases == ("SHARED-NATO",)


class TestDesignationAliasBag:
    """Provenance: each alias bag must reference its source table_index."""

    def test_records_source_table_index(self):
        t = _table([_column(3, {
            "Military Designation": "S-75",
            "NATO Designation": "SA-2C",
            "Missile Type": "13D",
        })])
        out = expand_designation_aliases([t])
        assert out[0].source_table_index == 0


# ===== Bridge-level checks (preservation + diagnostic gating) =====
#
# The bridge `_apply_designation_alias_overlay_to_missile_systems` lives
# in docker/docling-graph/app/evidence_gate.py. These tests pin its
# behavior with respect to:
#   * preserving existing LLM-emitted aliases (no overwrite)
#   * deduping so a re-emission doesn't grow the joined string
#   * gating diagnostics to entities that actually received aliases

import importlib.util
from pathlib import Path as _Path

_SERVICE_APP_ROOT = _Path(__file__).resolve().parents[2] / "docker" / "docling-graph" / "app"

_NUM_EV_SPEC = importlib.util.spec_from_file_location(
    "app._numeric_evidence", _SERVICE_APP_ROOT / "_numeric_evidence.py"
)
_NUM_EV_MOD = importlib.util.module_from_spec(_NUM_EV_SPEC)
import sys as _sys
_sys.modules["app._numeric_evidence"] = _NUM_EV_MOD
assert _NUM_EV_SPEC.loader is not None
_NUM_EV_SPEC.loader.exec_module(_NUM_EV_MOD)

_EG_SPEC = importlib.util.spec_from_file_location(
    "docling_graph_evidence_gate", _SERVICE_APP_ROOT / "evidence_gate.py"
)
_EG = importlib.util.module_from_spec(_EG_SPEC)
assert _EG_SPEC.loader is not None
_EG_SPEC.loader.exec_module(_EG)
_apply_bridge = _EG._apply_designation_alias_overlay_to_missile_systems


def _table_with_4_rounds() -> NormalizedTable:
    return _table([
        _column(2, {"Military Designation": "S-75",  "NATO Designation": "SA-2C", "Missile Type": "13D"}),
        _column(3, {"Military Designation": "S-75M", "NATO Designation": "SA-2D", "Missile Type": "20DSU"}),
    ])


class TestBridgePreservesExistingAliases:
    def test_existing_nomenclature_kept_and_new_appended(self):
        rows = [{"system_name": "13D", "nomenclature": "S-75 Dvina", "name": None}]
        diag = _apply_bridge(rows, [_table_with_4_rounds()])
        # Existing kept; "S-75" appended (not a duplicate of "S-75 Dvina")
        assert "S-75 Dvina" in rows[0]["nomenclature"]
        assert "S-75" in rows[0]["nomenclature"].split(" / ")
        assert rows[0]["name"] == "SA-2C"
        assert "13D" in diag

    def test_duplicate_alias_not_added_twice(self):
        rows = [{"system_name": "13D", "nomenclature": "S-75", "name": "SA-2C"}]
        diag = _apply_bridge(rows, [_table_with_4_rounds()])
        # Nothing should change — both aliases already present
        assert rows[0]["nomenclature"] == "S-75"
        assert rows[0]["name"] == "SA-2C"
        assert "13D" not in diag, (
            "diagnostic must be empty when no aliases were actually added"
        )

    def test_diagnostic_only_lists_entities_with_additions(self):
        rows = [
            {"system_name": "13D", "nomenclature": "S-75", "name": "SA-2C"},   # no-op
            {"system_name": "20DSU", "nomenclature": None, "name": None},      # gets both
            {"system_name": "AIM-9", "nomenclature": None, "name": None},      # not in table
        ]
        diag = _apply_bridge(rows, [_table_with_4_rounds()])
        assert "13D" not in diag
        assert "20DSU" in diag
        assert diag["20DSU"] == {
            "nomenclature": ["S-75M"],
            "name": ["SA-2D"],
        }
        assert "AIM-9" not in diag

    def test_idempotent_when_called_twice(self):
        """Calling the overlay twice on the same row set must not grow
        the joined string further."""
        rows = [{"system_name": "13D", "nomenclature": None, "name": None}]
        _apply_bridge(rows, [_table_with_4_rounds()])
        nomenclature_after_first = rows[0]["nomenclature"]
        name_after_first = rows[0]["name"]
        _apply_bridge(rows, [_table_with_4_rounds()])
        assert rows[0]["nomenclature"] == nomenclature_after_first
        assert rows[0]["name"] == name_after_first

    def test_no_tables_is_noop(self):
        rows = [{"system_name": "13D", "nomenclature": None, "name": None}]
        diag = _apply_bridge(rows, None)
        assert rows[0]["nomenclature"] is None
        assert rows[0]["name"] is None
        assert diag == {}
