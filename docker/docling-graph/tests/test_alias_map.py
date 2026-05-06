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
