"""Tests for normalize_label (spec §5.5).

This function is the single source of truth for label normalization across
the resolver and the §8.3 drift guard — both must use it so they assert the
same equality."""
import importlib.util
import sys
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_table_facts_norm", _FACTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["docling_graph_service_table_facts_norm"] = module
    spec.loader.exec_module(module)
    return module


def test_lowercase_and_whitespace_collapse():
    tf = _load()
    assert tf.normalize_label("  Length  mm  ") == "length mm"
    assert tf.normalize_label("Total Weight kg") == "total weight kg"


def test_punctuation_stripped_but_hyphens_preserved():
    """Hyphens distinguish 'SA-2' from 'SA 2'. Other punctuation goes."""
    tf = _load()
    assert tf.normalize_label("SA-2") == "sa-2"
    assert tf.normalize_label("SA-2 Guideline") == "sa-2 guideline"
    assert tf.normalize_label("Weight, kg") == "weight kg"
    assert tf.normalize_label("Weight (kg)") == "weight kg"
    assert tf.normalize_label("Weight/kg") == "weight kg"
    assert tf.normalize_label("Mass.") == "mass"


def test_dash_class_collapsed_to_ascii_hyphen():
    """En-dash, em-dash, figure-dash all map to ASCII hyphen so '13D-A',
    '13D–A', '13D—A' compare equal."""
    tf = _load()
    assert tf.normalize_label("13D–A") == tf.normalize_label("13D-A")
    assert tf.normalize_label("13D—A") == tf.normalize_label("13D-A")
    assert tf.normalize_label("13D‒A") == tf.normalize_label("13D-A")


def test_nfkc_fold_collapses_full_width_and_compatibility_chars():
    """Full-width digits (e.g., '１') and compatibility characters fold to
    ASCII so OCR-extracted CJK-context tables still match."""
    tf = _load()
    assert tf.normalize_label("１D") == tf.normalize_label("1D")
    assert tf.normalize_label("ª") == "a"


def test_idempotent():
    tf = _load()
    once = tf.normalize_label("1st Stage Weight kg")
    twice = tf.normalize_label(once)
    assert once == twice


def test_empty_string():
    tf = _load()
    assert tf.normalize_label("") == ""
    assert tf.normalize_label("   ") == ""
