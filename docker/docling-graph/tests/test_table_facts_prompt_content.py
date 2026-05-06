"""Prompt-content test (spec §8.4) — synthesizer's facts land in the LLM
user message in the exact emit_fact format. CI proxy for the full §20
end-to-end run."""
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


def _cell(text, row, col, *, row_header=False, col_span=1):
    return {
        "text": text,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + 1,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": 1, "col_span": col_span,
        "row_header": row_header, "column_header": False,
        "obj_type": "table_cell",
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


def test_prompt_contains_synthesized_facts_in_emit_format():
    """Run /extract-pass-equivalent against the SA-2 synthetic fixture;
    assert specific emit_fact lines appear in the captured user message."""
    doc = _sa2_shaped_doc()
    out, _ = _tf.synthesize_table_facts(doc, active_pass="missile_propulsion")

    # Concatenate all synthesized text — what the LLM ultimately sees in
    # whatever chunk the chunker assigns these to.
    rendered = "\n".join(t["text"] for t in out["texts"])

    # The exact-string assertions catch both presence AND format drift.
    # Note: the section keyword ("1st Stage" / "2nd Stage") is stripped from
    # label_text by detect_section_context before it reaches emit_fact, so the
    # source label is the post-strip form "Weight kg".
    assert (
        "1D — booster_mass_kg = 1135 [source: Weight kg row of variants table]"
        in rendered
    )
    assert (
        "13DM — sustain_mass_kg = 1251 [source: Weight kg row of variants table]"
        in rendered
    )
    assert (
        "13DM — booster_mass_kg = 1032 [source: Weight kg row of variants table]"
        in rendered
    )
