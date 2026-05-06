"""Unit tests for spec §5.2 helper functions (Mechanism A1).

These cover the four pure helpers that compose extract_table_overlay:
_classify_identity_row, _classify_cross_entity_ref,
_extract_alias_clusters, _pick_canonical.
"""
# _table_facts.py uses LAZY `from app._alias_map import …` inside
# function bodies. The conftest only APPENDS the docling-graph service
# root to sys.path, but the repo-root `app/` package wins on `from app
# import …`. So we PREPEND the docling-graph service root here (same
# pattern used by test_table_facts_resolve.py) so the lazy imports
# resolve to docker/docling-graph/app/_alias_map.py.
import sys
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _SERVICE_ROOT / "app"
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_SERVICE_ROOT))

import _table_facts as _tf  # noqa: E402  — after sys.path setup


def _load_table_facts():
    return _tf


# ---- _classify_identity_row -------------------------------------------------


def test_classify_identity_row_missile():
    tf = _load_table_facts()
    assert tf._classify_identity_row("Missile Type") == "MISSILE_SYSTEM"
    assert tf._classify_identity_row("Industry Designation") == "MISSILE_SYSTEM"


def test_classify_identity_row_radar():
    tf = _load_table_facts()
    assert tf._classify_identity_row("Radar Variant") == "RADAR_SYSTEM"


def test_classify_identity_row_cross_entity_ref_returns_none():
    """Fan Song Variant matches RADAR_SYSTEM in CROSS_ENTITY_REF_PATTERNS,
    not MISSILE_IDENTITY_LABELS — must be classified by
    _classify_cross_entity_ref instead, NOT by _classify_identity_row.
    Per spec §5.1 classification-order rule, cross-entity-ref check
    runs FIRST; identity-label check runs SECOND."""
    tf = _load_table_facts()
    assert tf._classify_identity_row("Fan Song Variant") is None


def test_classify_identity_row_spec_row_returns_none():
    tf = _load_table_facts()
    assert tf._classify_identity_row("Length mm") is None
    assert tf._classify_identity_row("") is None


# ---- _classify_cross_entity_ref --------------------------------------------


def test_classify_cross_entity_ref_fan_song():
    tf = _load_table_facts()
    assert tf._classify_cross_entity_ref("Fan Song Variant") == "RADAR_SYSTEM"


def test_classify_cross_entity_ref_unknown_returns_none():
    tf = _load_table_facts()
    assert tf._classify_cross_entity_ref("Missile Type") is None
    assert tf._classify_cross_entity_ref("Length mm") is None


# ---- _pick_canonical -------------------------------------------------------


def test_pick_canonical_picks_missile_type_first():
    tf = _load_table_facts()
    cluster = {
        "Missile Type": "1D",
        "Industry Designation": "SA-75",
        "NATO Designation": "SA-2A",
    }
    assert tf._pick_canonical(cluster, entity_type="MISSILE_SYSTEM") == "1D"


def test_pick_canonical_falls_back_when_missile_type_missing():
    tf = _load_table_facts()
    cluster = {
        "Industry Designation": "SA-75",
        "NATO Designation": "SA-2A",
    }
    assert tf._pick_canonical(cluster, entity_type="MISSILE_SYSTEM") == "SA-75"


def test_pick_canonical_alphabetic_fallback_for_no_priority_match():
    tf = _load_table_facts()
    cluster = {"Some Custom Label": "Z-1", "Another Label": "A-1"}
    # NFC + casefold sort → A-1 < Z-1
    assert tf._pick_canonical(cluster, entity_type="MISSILE_SYSTEM") == "A-1"


def test_pick_canonical_empty_cluster():
    tf = _load_table_facts()
    assert tf._pick_canonical({}, entity_type="MISSILE_SYSTEM") == ""


# ---- _extract_alias_clusters -----------------------------------------------


def _build_sa2_like_cells():
    """Synthetic 5×5 column-major table:
       row 0 col 0: Missile Type (row_header)
       row 1 col 0: Industry Designation (row_header)
       row 2 col 0: NATO Designation (row_header)
       row 3 col 0: Fan Song Variant (row_header) — cross-entity-ref
       row 4 col 0: Length mm (row_header) — spec row
       cols 1..4 hold the values for variants 1D / 13D / 13DM / 20D
    """
    cells = []
    labels = ("Missile Type", "Industry Designation", "NATO Designation",
              "Fan Song Variant", "Length mm")
    for r, label in enumerate(labels):
        cells.append({
            "start_row_offset_idx": r, "start_col_offset_idx": 0,
            "end_col_offset_idx": 1, "row_header": True, "text": label,
        })
    variants = (
        ("1D", "SA-75", "SA-2A", "RSNA-75", "10726"),
        ("13D", "S-75",  "SA-2C", "RSN-75",  "10726"),
        ("13DM", "S-75M", "SA-2D", "RSN-75M", "10841"),
        ("20D", "V-755", "SA-2F", "RSN-75V", "10841"),
    )
    for col_idx, col_vals in enumerate(variants, start=1):
        for r, val in enumerate(col_vals):
            cells.append({
                "start_row_offset_idx": r, "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx + 1, "row_header": False,
                "text": val,
            })
    return {"data": {"table_cells": cells, "num_rows": 5, "num_cols": 5}}


def test_extract_alias_clusters_builds_one_cluster_per_column():
    tf = _load_table_facts()
    table = _build_sa2_like_cells()
    clusters = tf._extract_alias_clusters(table, entity_type="MISSILE_SYSTEM")
    # One cluster per data column (4); each cluster has the three identity
    # rows but NOT the Fan Song row (cross-entity-ref) or the Length row
    # (spec row).
    assert len(clusters) == 4
    for cluster in clusters:
        assert "Fan Song Variant" not in cluster
        assert "Length mm" not in cluster
        assert "Missile Type" in cluster


def test_extract_alias_clusters_excludes_empty_cells():
    tf = _load_table_facts()
    table = _build_sa2_like_cells()
    # Blank out one identity cell.
    for cell in table["data"]["table_cells"]:
        if (cell.get("start_row_offset_idx") == 1
                and cell.get("start_col_offset_idx") == 1):
            cell["text"] = ""
    clusters = tf._extract_alias_clusters(table, entity_type="MISSILE_SYSTEM")
    # Column 1's cluster must NOT include "Industry Designation" anymore.
    assert clusters[0].get("Industry Designation") in (None, "")


def test_extract_alias_clusters_no_identity_rows_returns_empty():
    tf = _load_table_facts()
    cells = [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0,
         "end_col_offset_idx": 1, "row_header": True, "text": "Length mm"},
        {"start_row_offset_idx": 0, "start_col_offset_idx": 1,
         "end_col_offset_idx": 2, "row_header": False, "text": "10726"},
    ]
    table = {"data": {"table_cells": cells, "num_rows": 1, "num_cols": 2}}
    assert tf._extract_alias_clusters(table, entity_type="MISSILE_SYSTEM") == []
