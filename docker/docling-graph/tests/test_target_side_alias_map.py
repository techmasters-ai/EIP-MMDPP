"""Tests for Recommendation 2 — target-side alias maps for cross-entity rows.

The overlay's `alias_map_by_entity_type` now contains entries for BOTH the
winner entity type AND any target entity type referenced via cross-entity
rows. This lets the system_links resolver bridge OCR-spaced and family-
derived alias forms (e.g. `RSN- 75V` from a `Fan Song Variant` row maps
to `Fan Song` as candidate canonical).

All tests use generic identity tokens — no equipment-specific names in
the test logic. Fixture row labels like "Fan Song Variant" are used
because they're real Docling row labels that exercise the
CROSS_ENTITY_REF_PATTERNS classification path, but the generic mechanism
under test (family-name strip + OCR-space collapse) works for any
`<X> Variant` label.
"""
from __future__ import annotations
import importlib.util
import pathlib
import sys

_SR = pathlib.Path(__file__).resolve().parent.parent / "app"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load("app._numeric_evidence", _SR / "_numeric_evidence.py")
# Register the docling-graph app/schemas.py as `app.schemas` so
# `extract_table_overlay`'s lazy `from app.schemas import TableOverlay`
# resolves to the DG version (not the repo-root one which lacks TableOverlay).
_load("app.schemas", _SR / "schemas.py")
# Same trick for app._alias_map — needed by `_classify_identity_row` etc.
_load("app._alias_map", _SR / "_alias_map.py")
_tf = _load("_dgp_table_facts", _SR / "_table_facts.py")


# --- Unit-level: generic helpers ------------------------------------------

def test_extract_family_name_strips_variant_suffix():
    assert _tf._extract_cross_entity_family_name("Fan Song Variant") == "Fan Song"
    assert _tf._extract_cross_entity_family_name("Spoon Rest Variant") == "Spoon Rest"
    assert _tf._extract_cross_entity_family_name("Foo Variant") == "Foo"
    assert _tf._extract_cross_entity_family_name("Bar Variants") == "Bar"
    # Case-insensitive
    assert _tf._extract_cross_entity_family_name("xyz VARIANT") == "xyz"


def test_extract_family_name_returns_none_when_no_variant_suffix():
    assert _tf._extract_cross_entity_family_name("Plain Label") is None
    assert _tf._extract_cross_entity_family_name("") is None
    assert _tf._extract_cross_entity_family_name(None) is None
    # "Variant" alone is not a family + suffix — no name remains
    assert _tf._extract_cross_entity_family_name("Variant") is None


def test_ocr_space_collapse_handles_common_patterns():
    """`<token>- <token>` (dash followed by whitespace) → `<token>-<token>`."""
    assert _tf._normalize_ocr_spaces("RSN- 75V") == "RSN-75V"
    assert _tf._normalize_ocr_spaces("RSNA- 75M") == "RSNA-75M"
    # Multi-space too
    assert _tf._normalize_ocr_spaces("Some-  Token") == "Some-Token"
    # Strips outer whitespace
    assert _tf._normalize_ocr_spaces("  RSN- 75V  ") == "RSN-75V"


def test_ocr_space_collapse_leaves_clean_aliases_unchanged():
    assert _tf._normalize_ocr_spaces("RSN-75V") == "RSN-75V"
    assert _tf._normalize_ocr_spaces("Fan Song") == "Fan Song"  # no dash, no change
    assert _tf._normalize_ocr_spaces("") == ""


def test_normalize_target_alias_key_matches_resolver_shape():
    """The registration key must match what the resolver does at lookup
    time (`normalize_evidence_text`: uppercase + whitespace-collapse)."""
    assert _tf._normalize_target_alias_key("rsn-75v") == "RSN-75V"
    assert _tf._normalize_target_alias_key("Fan  Song") == "FAN SONG"
    assert _tf._normalize_target_alias_key("") == ""


# --- Integration: full overlay generation --------------------------------

def _table_with_cross_entity_row(missile_names, fan_song_variants, with_designations=False):
    """Build a Docling-shape table that qualifies for overlay extraction
    (≥4 data columns, identity rows with row_header flag). Includes:
      - Missile Type identity row
      - NATO Designation identity row
      - Fan Song Variant cross-entity-ref row
    Pads to 4 missile columns if fewer are provided.
    """
    # Pad to 4 columns minimum (_table_qualifies_for_overlay requires it)
    pad = max(0, 4 - len(missile_names))
    missile_names = list(missile_names) + [f"PAD{i}" for i in range(pad)]
    variants = list(fan_song_variants) + [f"PADV{i}" for i in range(pad)]

    cells = []
    num_cols = len(missile_names) + 1
    # Row 0: Missile Type identity row (row_header flag required for qualification)
    cells.append({
        "start_row_offset_idx": 0, "end_row_offset_idx": 0,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        "text": "Missile Type",
        "row_header": True,
    })
    for i, m in enumerate(missile_names):
        cells.append({
            "start_row_offset_idx": 0, "end_row_offset_idx": 0,
            "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
            "text": m,
        })
    # Row 1: NATO Designation (second identity row for qualification depth)
    cells.append({
        "start_row_offset_idx": 1, "end_row_offset_idx": 1,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        "text": "NATO Designation",
        "row_header": True,
    })
    for i, _ in enumerate(missile_names):
        cells.append({
            "start_row_offset_idx": 1, "end_row_offset_idx": 1,
            "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
            "text": f"NATO-{i}",
        })
    # Row 2: Fan Song Variant cross-entity row (target_alias source for Rec 2)
    cells.append({
        "start_row_offset_idx": 2, "end_row_offset_idx": 2,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        "text": "Fan Song Variant",
        "row_header": True,
    })
    for i, v in enumerate(variants):
        cells.append({
            "start_row_offset_idx": 2, "end_row_offset_idx": 2,
            "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
            "text": v,
        })
    # Row 3: Max Range spec row (required to reach ≥4 rows for shape detection)
    cells.append({
        "start_row_offset_idx": 3, "end_row_offset_idx": 3,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        "text": "Max Range",
        "row_header": True,
    })
    for i, _ in enumerate(missile_names):
        cells.append({
            "start_row_offset_idx": 3, "end_row_offset_idx": 3,
            "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
            "text": str(29000 + i * 1000),
        })
    return {
        "self_ref": "#/tables/0",
        "data": {"table_cells": cells, "num_rows": 4, "num_cols": num_cols},
    }


def test_overlay_emits_radar_aliases_alongside_missile_aliases():
    """Given a missile table with a `Fan Song Variant` cross-entity row,
    the overlay's `alias_map_by_entity_type` should contain BOTH
    MISSILE_SYSTEM (from the winner identity column) AND RADAR_SYSTEM
    (from the cross-entity target aliases bridged to `Fan Song` family).
    """
    table = _table_with_cross_entity_row(
        missile_names=["M1", "M2"],
        fan_song_variants=["RSN-75A", "RSN-75B"],
    )
    overlay, _ = _tf.extract_table_overlay({"tables": [table]})

    # MISSILE_SYSTEM map should exist (winner type), with the missile names
    assert "MISSILE_SYSTEM" in overlay.alias_map_by_entity_type
    # RADAR_SYSTEM map should now exist too (Rec 2)
    assert "RADAR_SYSTEM" in overlay.alias_map_by_entity_type
    radar_map = overlay.alias_map_by_entity_type["RADAR_SYSTEM"]
    # Each variant → "Fan Song" family canonical
    assert radar_map["RSN-75A"] == "Fan Song"
    assert radar_map["RSN-75B"] == "Fan Song"


def test_overlay_registers_ocr_normalized_alias_form_too():
    """When an alias has OCR-spacing artifacts (`RSN- 75V` with literal
    space after the dash), the overlay registers BOTH the raw and the
    OCR-normalized form so resolver lookups succeed either way."""
    table = _table_with_cross_entity_row(
        missile_names=["M1"],
        fan_song_variants=["RSN- 75V"],
    )
    overlay, _ = _tf.extract_table_overlay({"tables": [table]})

    radar_map = overlay.alias_map_by_entity_type.get("RADAR_SYSTEM", {})
    # Raw form (uppercased per resolver normalization)
    assert "RSN- 75V" in radar_map
    assert radar_map["RSN- 75V"] == "Fan Song"
    # OCR-normalized form
    assert "RSN-75V" in radar_map
    assert radar_map["RSN-75V"] == "Fan Song"


def test_overlay_no_radar_alias_map_when_no_cross_entity_row():
    """A missile table with NO cross-entity row → no RADAR_SYSTEM aliases.
    Validates that Rec 2 is purely additive — only fires when cross-entity
    rows exist."""
    cells = []
    # Identity row with row_header flag + 4 missile cols
    cells.append({
        "start_row_offset_idx": 0, "end_row_offset_idx": 0,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        "text": "Missile Type", "row_header": True,
    })
    for i in range(4):
        cells.append({
            "start_row_offset_idx": 0, "end_row_offset_idx": 0,
            "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
            "text": f"M{i+1}",
        })
    # Second identity row
    cells.append({
        "start_row_offset_idx": 1, "end_row_offset_idx": 1,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        "text": "NATO Designation", "row_header": True,
    })
    for i in range(4):
        cells.append({
            "start_row_offset_idx": 1, "end_row_offset_idx": 1,
            "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
            "text": f"NATO-{i}",
        })
    # Spec rows (no cross-entity)
    for r, label in [(2, "Max Range"), (3, "Min Range")]:
        cells.append({
            "start_row_offset_idx": r, "end_row_offset_idx": r,
            "start_col_offset_idx": 0, "end_col_offset_idx": 0,
            "text": label, "row_header": True,
        })
        for i in range(4):
            cells.append({
                "start_row_offset_idx": r, "end_row_offset_idx": r,
                "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
                "text": str(29000 + i * 1000),
            })
    table = {
        "self_ref": "#/tables/0",
        "data": {"table_cells": cells, "num_rows": 4, "num_cols": 5},
    }
    overlay, _ = _tf.extract_table_overlay({"tables": [table]})
    # No cross-entity row → no RADAR_SYSTEM entries should appear
    assert "RADAR_SYSTEM" not in overlay.alias_map_by_entity_type


def test_overlay_winner_alias_map_unchanged_by_target_alias_additions():
    """Verify Rec 2 doesn't accidentally affect the existing MISSILE_SYSTEM
    alias-map building."""
    table = _table_with_cross_entity_row(
        missile_names=["1D", "13D"],
        fan_song_variants=["RSN-75A", "RSN-75B"],
    )
    overlay, _ = _tf.extract_table_overlay({"tables": [table]})

    missile_map = overlay.alias_map_by_entity_type["MISSILE_SYSTEM"]
    # MISSILE aliases should still be there
    assert "1D" in missile_map
    assert "13D" in missile_map


def test_overlay_dedupes_same_alias_appearing_in_multiple_rows():
    """If the same variant alias appears in two cross-entity rows (rare
    but possible), setdefault means first-write wins — no overwrite,
    no exception."""
    cells = []
    # Identity row + 4 missile cols
    cells.append({
        "start_row_offset_idx": 0, "end_row_offset_idx": 0,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        "text": "Missile Type", "row_header": True,
    })
    for i in range(4):
        cells.append({
            "start_row_offset_idx": 0, "end_row_offset_idx": 0,
            "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
            "text": f"M{i+1}",
        })
    # Second identity row
    cells.append({
        "start_row_offset_idx": 1, "end_row_offset_idx": 1,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        "text": "NATO Designation", "row_header": True,
    })
    for i in range(4):
        cells.append({
            "start_row_offset_idx": 1, "end_row_offset_idx": 1,
            "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
            "text": f"NATO-{i}",
        })
    # Two Fan Song Variant rows with overlapping alias on col 1
    for r in (2, 3):
        cells.append({
            "start_row_offset_idx": r, "end_row_offset_idx": r,
            "start_col_offset_idx": 0, "end_col_offset_idx": 0,
            "text": "Fan Song Variant", "row_header": True,
        })
        for i in range(4):
            cells.append({
                "start_row_offset_idx": r, "end_row_offset_idx": r,
                "start_col_offset_idx": i + 1, "end_col_offset_idx": i + 1,
                "text": "RSN-75V" if i == 0 else f"RSN-Z{r}{i}",
            })
    table = {
        "self_ref": "#/tables/0",
        "data": {"table_cells": cells, "num_rows": 4, "num_cols": 5},
    }
    overlay, _ = _tf.extract_table_overlay({"tables": [table]})
    radar_map = overlay.alias_map_by_entity_type.get("RADAR_SYSTEM", {})
    assert radar_map.get("RSN-75V") == "Fan Song"  # first registration wins


# --- End-to-end: alias map enables resolution by the resolver ------------

def test_resolver_bridges_target_alias_via_overlay_alias_map():
    """End-to-end: overlay registers `RSN-75V → Fan Song` under
    RADAR_SYSTEM, and upstream has `Fan Song` as a RADAR_SYSTEM entity.
    Resolver should bridge the alias via the new alias map and return
    the upstream ref."""
    eg = _load("_dgp_evidence_gate", _SR / "evidence_gate.py")
    from types import SimpleNamespace

    table = _table_with_cross_entity_row(
        missile_names=["M1"],
        fan_song_variants=["RSN-75V"],
    )
    overlay, _ = _tf.extract_table_overlay({"tables": [table]})
    alias_map = overlay.alias_map_by_entity_type

    # Upstream catalog has `Fan Song` as RADAR_SYSTEM
    upstream = [
        SimpleNamespace(ref_id="MIS:M1", entity_type="MISSILE_SYSTEM",
                        identity_values={"system_name": "M1"}, display_label="M1"),
        SimpleNamespace(ref_id="RAD:Fan Song", entity_type="RADAR_SYSTEM",
                        identity_values={"system_name": "Fan Song"}, display_label="Fan Song"),
    ]
    name_to_ref_by_type = eg._build_upstream_name_map_by_type(upstream)

    # Direct lookup misses (Fan Song's upstream key is "FAN SONG", not "RSN-75V")
    # Alias fallback maps "RSN-75V" → "Fan Song" → upstream ref
    result = eg._resolve_ref("RSN-75V", "RADAR_SYSTEM", name_to_ref_by_type, alias_map)
    assert result == "RAD:Fan Song"


def test_resolver_returns_none_when_family_canonical_absent_from_upstream():
    """The handoff requires the bridge to be GATED by upstream presence.
    The overlay registers `RSN-75V → Fan Song` unconditionally; the
    resolver returns None if `Fan Song` isn't an upstream RADAR_SYSTEM
    entry. Verifies the gate is properly enforced AT RESOLUTION TIME."""
    eg = _load("_dgp_evidence_gate", _SR / "evidence_gate.py")
    from types import SimpleNamespace

    table = _table_with_cross_entity_row(
        missile_names=["M1"],
        fan_song_variants=["RSN-75V"],
    )
    overlay, _ = _tf.extract_table_overlay({"tables": [table]})
    alias_map = overlay.alias_map_by_entity_type

    # Upstream catalog does NOT contain `Fan Song`
    upstream = [
        SimpleNamespace(ref_id="MIS:M1", entity_type="MISSILE_SYSTEM",
                        identity_values={"system_name": "M1"}, display_label="M1"),
    ]
    name_to_ref_by_type = eg._build_upstream_name_map_by_type(upstream)

    # alias map exists, but Fan Song isn't in upstream — resolver returns None
    result = eg._resolve_ref("RSN-75V", "RADAR_SYSTEM", name_to_ref_by_type, alias_map)
    assert result is None
