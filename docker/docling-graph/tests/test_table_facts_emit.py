"""Tests for emit_fact (spec §5.7)."""
import sys
from pathlib import Path

# Mirror the path setup used by test_table_facts_coerce.py so that
# `from app._alias_map import ...` inside _table_facts resolves to the
# docling-graph app package, not the repo-root app package.
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_APP_PATH = _SERVICE_ROOT / "app"
sys.path.insert(0, str(_APP_PATH))
sys.path.insert(0, str(_SERVICE_ROOT))

# Ensure the docling-graph app package is cached in sys.modules under 'app'
# BEFORE importing _table_facts (which does lazy `from app._alias_map import`
# inside function bodies). Without this, the repo-root app/ may win the race.
import importlib as _il
if "app" not in sys.modules or not hasattr(sys.modules["app"], "__path__") or \
        str(_APP_PATH) not in sys.modules["app"].__path__:
    # Evict any stale repo-root 'app' entries and load the DG one.
    for _k in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
        del sys.modules[_k]
    _il.import_module("app")

import _table_facts as _tf  # noqa: E402


def _load():
    return _tf


def test_emit_fact_textitem_schema_completeness():
    """Returned dict must satisfy DoclingDocument's TextItem union variant."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="booster_mass_kg",
        value=1135.0,
        source_label="1st Stage Weight kg",
        text_idx=42,
    )
    assert item["self_ref"] == "#/texts/42"
    assert item["parent"] == {"$ref": "#/body"}
    assert item["children"] == []
    assert item["content_layer"] == "body"
    assert item["label"] == "text"
    assert item["prov"] == []
    assert item["orig"] == item["text"]


def test_emit_fact_text_format_integer_valued_float():
    """1135.0 is an integer-valued float; _format_value trims trailing .0."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="booster_mass_kg",
        value=1135.0,
        source_label="1st Stage Weight kg",
        text_idx=42,
    )
    assert item["text"] == "1D — booster_mass_kg = 1135 [source: 1st Stage Weight kg row of variants table]"


def test_emit_fact_text_format_decimal_float():
    """10.726 has a non-zero fractional part; _format_value preserves it."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="body_length_m",
        value=10.726,
        source_label="Length mm",
        text_idx=42,
    )
    assert item["text"] == "1D — body_length_m = 10.726 [source: Length mm row of variants table]"


def test_emit_fact_string_value():
    """String values render verbatim without trailing .0."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="booster_thrust",
        value="dual-pulse Mark 104",
        source_label="1st Stage Thrust",
        text_idx=43,
    )
    assert "1D — booster_thrust = dual-pulse Mark 104 [source: 1st Stage Thrust row of variants table]" == item["text"]


def test_emit_fact_int_value_formatted_without_decimal():
    """Integer values should not show a trailing .0 — '1135' not '1135.0'."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="booster_mass_kg",
        value=1135,  # int
        source_label="1st Stage Weight kg",
        text_idx=44,
    )
    assert " = 1135 " in item["text"]
