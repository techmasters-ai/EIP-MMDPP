"""Drift guard for _alias_map.py (spec §8.3).

The structured ALIAS_MAP in _alias_map.py and the §12b prose in
prompt_rules.DELTA_SYSTEM_PROMPT are paired SSoTs. These tests catch
drift in either direction: a new alias added to the map without a prose
mention, or a renamed schema field that the alias map still points at.
"""
import importlib.util
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _facts():
    return _load("docling_graph_service_table_facts", _APP_DIR / "_table_facts.py")


def _aliases():
    return _load("docling_graph_service_alias_map", _APP_DIR / "_alias_map.py")


def _delta_prompt() -> str:
    from ontology_bundles._shared.prompt_rules import DELTA_SYSTEM_PROMPT
    return DELTA_SYSTEM_PROMPT


def test_alias_map_labels_appear_in_prompt_rule():
    """Every ALIAS_MAP key's label_normalized appears as a token in §12b prose."""
    tf = _facts()
    am = _aliases()
    prose_normalized = tf.normalize_label(_delta_prompt())
    prose_tokens = set(prose_normalized.split())

    for (label_norm, _section, _pass), _field in am.ALIAS_MAP.items():
        for token in label_norm.split():
            assert token in prose_tokens, (
                f"Token {token!r} from ALIAS_MAP label {label_norm!r} "
                f"missing from §12b prose tokens. Either add it to §12b "
                f"or drop it from ALIAS_MAP."
            )


def test_section_keywords_appear_in_prompt_rule():
    """Every SECTION_KEYWORDS entry appears as a contiguous phrase in §12b prose."""
    tf = _facts()
    am = _aliases()
    prose_normalized = tf.normalize_label(_delta_prompt())
    for keyword in am.SECTION_KEYWORDS:
        keyword_norm = tf.normalize_label(keyword)
        assert keyword_norm in prose_normalized, (
            f"Section keyword {keyword!r} (normalized {keyword_norm!r}) "
            f"missing from §12b prose. Add to §12b before adding to "
            f"SECTION_KEYWORDS."
        )


def test_unit_table_keys_match_field_suffix_classes():
    """FIELD_SUFFIX_TO_UNIT_CLASS values must all be UNIT_TABLE keys."""
    am = _aliases()
    unit_classes = set(am.UNIT_TABLE.keys())
    referenced = set(am.FIELD_SUFFIX_TO_UNIT_CLASS.values())
    missing = referenced - unit_classes
    assert not missing, (
        f"FIELD_SUFFIX_TO_UNIT_CLASS references unit classes not in "
        f"UNIT_TABLE: {missing}"
    )


def test_unit_table_includes_canonical_unit():
    """Each unit class's table must include the canonical unit at factor 1.0
    (e.g., length_m table must include 'm' -> 1.0)."""
    am = _aliases()
    canonical_units = {
        "length_m": "m",
        "length_km": "km",
        "mass_kg": "kg",
        "time_sec": "sec",
        "velocity_mps": "mps",
        "frequency_mhz": "mhz",
        "gain_dbi": "dbi",
        "power_kw": "kw",
        "power_dbw": "dbw",
        "angle_deg": "deg",
    }
    for cls, unit in canonical_units.items():
        if cls not in am.UNIT_TABLE:
            continue
        assert unit in am.UNIT_TABLE[cls], f"{cls} missing canonical unit {unit!r}"
        assert am.UNIT_TABLE[cls][unit] == 1.0, f"{cls}[{unit!r}] should be 1.0"


def _collect_all_field_names(template_cls) -> set[str]:
    """Two-level walk: template_cls -> each entity item class -> model_fields.
    Mirrors the pattern in app/_field_provenance_helpers.py and
    docling-graph's provenance walker. Catches schema-side drift if a field
    is renamed or removed."""
    all_fields = set(template_cls.model_fields.keys())
    for fname, finfo in template_cls.model_fields.items():
        # If this field's annotation is a list[ItemClass] or ItemClass,
        # introspect the item class's fields too.
        annotation = finfo.annotation
        item_cls = None
        # list[X] case
        if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
            args = getattr(annotation, "__args__", ())
            if args and hasattr(args[0], "model_fields"):
                item_cls = args[0]
        # Direct BaseModel case
        elif hasattr(annotation, "model_fields"):
            item_cls = annotation
        if item_cls is not None:
            all_fields.update(item_cls.model_fields.keys())
    return all_fields


def test_alias_map_target_fields_exist_on_schemas():
    """Every ALIAS_MAP value (target schema field) must exist as a field on
    the schema for the corresponding pass. Catches drift where a schema is
    refactored and the alias map still points at a renamed/removed field."""
    am = _aliases()
    # Group ALIAS_MAP entries by pass.
    by_pass: dict[str, set[str]] = {}
    for (_label, _section, pass_name), schema_field in am.ALIAS_MAP.items():
        by_pass.setdefault(pass_name, set()).add(schema_field)

    # Resolve template classes via the bundle loader. Mirrors how main.py
    # loads them at extract-pass time. Use _load helper to dynamically import.
    bundles = _load("docling_graph_service_bundles", _APP_DIR / "bundles.py")
    for pass_name, fields in by_pass.items():
        template_cls = bundles.load_pass_template("air_defense_v3", pass_name)
        actual_fields = _collect_all_field_names(template_cls)
        missing = fields - actual_fields
        assert not missing, (
            f"ALIAS_MAP entries for pass {pass_name!r} reference fields "
            f"{missing!r} that do not exist on the schema. The schema may "
            f"have been refactored; reconcile the alias map."
        )
