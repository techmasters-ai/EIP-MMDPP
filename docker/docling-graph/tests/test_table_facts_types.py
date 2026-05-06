"""Type-shape tests for _table_facts.py.

Verifies the core data types declared in spec §5.0 — Shape enum, LabelRow
TypedDict, ParsedValue and FactStats dataclasses. These are the contract every
other component in the synthesizer depends on; if they drift the rest of the
pipeline silently misbehaves.
"""
import importlib.util
import sys
from dataclasses import is_dataclass, fields as dataclass_fields
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load_table_facts():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_table_facts", _FACTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["docling_graph_service_table_facts"] = module
    spec.loader.exec_module(module)
    return module


def test_shape_enum_has_required_members():
    tf = _load_table_facts()
    assert tf.Shape.COLUMN_MAJOR.value == "column_major"
    assert tf.Shape.ROW_MAJOR.value == "row_major"
    assert tf.Shape.HYBRID.value == "hybrid"
    assert tf.Shape.OTHER.value == "other"


def test_label_row_typed_dict_keys():
    tf = _load_table_facts()
    expected = {"row_idx", "label_text", "label_col_span", "data_cells"}
    assert set(tf.LabelRow.__annotations__.keys()) == expected


def test_parsed_value_is_frozen_dataclass():
    tf = _load_table_facts()
    assert is_dataclass(tf.ParsedValue)
    field_names = {f.name for f in dataclass_fields(tf.ParsedValue)}
    assert field_names == {"value", "unit_inferred", "conversion_factor", "raw_text"}
    pv = tf.ParsedValue(value=1135.0, unit_inferred="kg", conversion_factor=1.0, raw_text="1135")
    try:
        pv.value = 0
    except Exception:
        return
    raise AssertionError("ParsedValue must be frozen — assignment should raise")


def test_parsed_value_supports_positional_construction():
    """§6 worked example uses ParsedValue(1135, 'kg', 1.0, '1135') positionally."""
    tf = _load_table_facts()
    pv = tf.ParsedValue(1135.0, "kg", 1.0, "1135")
    assert pv.value == 1135.0
    assert pv.unit_inferred == "kg"


def test_fact_stats_is_mutable_dataclass_with_defaults():
    tf = _load_table_facts()
    assert is_dataclass(tf.FactStats)
    fs = tf.FactStats()
    assert fs.tables_seen == 0
    assert fs.facts_emitted == 0
    assert fs.tables_by_shape == {}
    assert fs.hybrid_collisions == 0
    assert fs.truncated_at_cap is False
    assert fs.idempotent_skip is False
    fs.facts_emitted += 1
    assert fs.facts_emitted == 1


def test_fact_stats_empty_classmethod():
    tf = _load_table_facts()
    fs = tf.FactStats.empty()
    assert isinstance(fs, tf.FactStats)
    assert fs.facts_emitted == 0


def test_fact_stats_as_dict():
    tf = _load_table_facts()
    fs = tf.FactStats(tables_seen=3, facts_emitted=33)
    fs.tables_by_shape["column_major"] = 1
    d = fs.as_dict()
    assert isinstance(d, dict)
    assert d["tables_seen"] == 3
    assert d["facts_emitted"] == 33
    assert d["tables_by_shape"] == {"column_major": 1}


def test_fact_stats_default_factory_isolates_instances():
    """Each FactStats instance must get its own tables_by_shape dict."""
    tf = _load_table_facts()
    a = tf.FactStats()
    b = tf.FactStats()
    a.tables_by_shape["column_major"] = 1
    assert "column_major" not in b.tables_by_shape


def test_alias_key_typealias_exists():
    tf = _load_table_facts()
    assert hasattr(tf, "AliasKey")
    assert hasattr(tf, "SectionContext")
