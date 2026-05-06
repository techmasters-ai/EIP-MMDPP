"""Integration tests for synthesize_table_facts (spec §6 worked example).

Synthetic SA-2-shaped column-major table; verifies end-to-end emission for
each of the four missile passes, idempotence flag, max_synthesized cap, and
graceful error handling for malformed input.
"""
import sys
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_APP_PATH = _SERVICE_ROOT / "app"

# Mirror the path setup used by test_table_facts_coerce.py so that
# `from app._alias_map import ...` inside _table_facts resolves to the
# docling-graph app package, not the repo-root app package.
import importlib as _il
if "app" not in sys.modules or not hasattr(sys.modules["app"], "__path__") or \
        str(_APP_PATH) not in sys.modules["app"].__path__:
    for _k in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
        del sys.modules[_k]
    sys.path.insert(0, str(_APP_PATH))
    sys.path.insert(0, str(_SERVICE_ROOT))
    _il.import_module("app")

import _table_facts as _tf  # noqa: E402


def _load():
    return _tf


def _cell(text, row, col, *, row_header=False, col_span=1):
    return {
        "text": text,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + 1,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": 1, "col_span": col_span,
        "row_header": row_header, "column_header": False,
    }


def _sa2_shaped_doc():
    """SA-2-style variants table: 3 missile columns × 9 spec rows with
    embedded section keywords on rows 6-8."""
    cells = [
        _cell("Missile Type",        0, 0, row_header=True),
        _cell("1D",  0, 1), _cell("13D", 0, 2), _cell("13DM", 0, 3),

        _cell("Max Range km",        1, 0, row_header=True),
        _cell("29",  1, 1), _cell("34",  1, 2), _cell("43",   1, 3),

        _cell("Max Altitude km",     2, 0, row_header=True),
        _cell("22",  2, 1), _cell("27",  2, 2), _cell("30",   2, 3),

        _cell("Length mm",           3, 0, row_header=True),
        _cell("10726", 3, 1), _cell("10841", 3, 2), _cell("10841", 3, 3),

        _cell("Total Weight kg",     4, 0, row_header=True),
        _cell("2163",  4, 1), _cell("2283",  4, 2), _cell("2283",  4, 3),

        _cell("Max Speed m/s",       5, 0, row_header=True),
        _cell("",      5, 1), _cell("650",   5, 2), _cell("650",   5, 3),

        _cell("1st Stage Weight kg", 6, 0, row_header=True),
        _cell("1135",  6, 1), _cell("1032",  6, 2), _cell("1032",  6, 3),

        _cell("1st Stage Time sec",  7, 0, row_header=True),
        _cell("4.0",   7, 1), _cell("4.0",   7, 2), _cell("4.0",   7, 3),

        _cell("2nd Stage Weight kg", 8, 0, row_header=True),
        _cell("1028",  8, 1), _cell("1251",  8, 2), _cell("1251",  8, 3),
    ]
    return {
        "tables": [
            {
                "self_ref": "#/tables/0",
                "data": {"table_cells": cells, "num_rows": 9, "num_cols": 4},
                "prov": [{"page_no": 6}],
            }
        ],
        "texts": [],
        "body": {"children": []},
    }


def test_synthesizes_propulsion_facts():
    """missile_propulsion pass on SA-2 doc emits booster + sustain mass facts."""
    tf = _load()
    doc = _sa2_shaped_doc()
    out_doc, stats = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    assert stats.facts_emitted >= 6  # 3 entities × (booster_mass + sustain_mass) = 6
    text_set = {t["text"] for t in out_doc["texts"]}
    assert any("1D — booster_mass_kg = 1135" in t for t in text_set)
    assert any("13D — booster_mass_kg = 1032" in t for t in text_set)
    assert any("13DM — booster_mass_kg = 1032" in t for t in text_set)
    assert any("1D — sustain_mass_kg = 1028" in t for t in text_set)
    assert not any("max_intercept_km" in t for t in text_set)


def test_synthesizes_kinematics_facts():
    tf = _load()
    doc = _sa2_shaped_doc()
    out_doc, stats = tf.synthesize_table_facts(doc, active_pass="missile_kinematics")
    assert stats.facts_emitted >= 6  # 3 entities × (max_range + max_alt) = 6
    text_set = {t["text"] for t in out_doc["texts"]}
    assert any("1D — max_intercept_km = 29" in t for t in text_set)
    assert any("1D — max_altitude_km = 22" in t for t in text_set)
    assert not any("booster_mass_kg" in t for t in text_set)


def test_synthesizes_airframe_facts():
    tf = _load()
    doc = _sa2_shaped_doc()
    out_doc, stats = tf.synthesize_table_facts(doc, active_pass="missile_airframe")
    assert stats.facts_emitted >= 6  # 3 entities × (length + total_weight) = 6
    text_set = {t["text"] for t in out_doc["texts"]}
    # body_length_m is converted from mm: 10726 -> 10.726
    assert any("1D — body_length_m = 10.726" in t for t in text_set)
    assert any("1D — total_mass_kg = 2163" in t for t in text_set)


def test_idempotence_skips_second_call():
    tf = _load()
    doc = _sa2_shaped_doc()
    out1, stats1 = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    out2, stats2 = tf.synthesize_table_facts(out1, active_pass="missile_propulsion")
    assert stats1.idempotent_skip is False
    assert stats2.idempotent_skip is True
    assert stats2.facts_emitted == 0
    assert len(out2["texts"]) == len(out1["texts"])


def test_max_synthesized_cap():
    """max_synthesized=5 caps emission at 5 even when more would fire."""
    tf = _load()
    doc = _sa2_shaped_doc()
    out, stats = tf.synthesize_table_facts(
        doc, active_pass="missile_propulsion", max_synthesized=5,
    )
    assert stats.facts_emitted == 5
    assert stats.truncated_at_cap is True


def test_appends_to_body_children_so_chunker_walks_them():
    tf = _load()
    doc = _sa2_shaped_doc()
    out, _ = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    refs = {c.get("$ref") for c in out["body"]["children"]}
    for i in range(len(out["texts"])):
        assert f"#/texts/{i}" in refs


def test_handles_doc_with_no_tables():
    tf = _load()
    doc = {"tables": [], "texts": [], "body": {"children": []}}
    out, stats = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    assert stats.facts_emitted == 0
    assert stats.tables_seen == 0


def test_handles_malformed_doc():
    tf = _load()
    out, stats = tf.synthesize_table_facts({}, active_pass="missile_propulsion")
    assert stats.facts_emitted == 0


def test_unknown_pass_skips_with_warning():
    tf = _load()
    doc = _sa2_shaped_doc()
    out, stats = tf.synthesize_table_facts(doc, active_pass="nonexistent_pass")
    assert stats.facts_emitted == 0
    assert stats.tables_seen >= 1


def test_stats_counters_increment_correctly():
    tf = _load()
    doc = _sa2_shaped_doc()
    _, stats = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    assert stats.tables_seen == 1
    assert stats.tables_by_shape == {"column_major": 1}
    assert stats.sections_detected >= 2  # "1st Stage" + "2nd Stage"
    # Skipped rows (kinematics labels in propulsion pass): Max Range km,
    # Max Altitude km, Length mm, Total Weight kg, Max Speed m/s = 5 rows × 3 cols
    # (Max Speed 1D cell is empty and not counted, so actual minimum is 14).
    assert stats.rows_skipped_unresolvable >= 14
