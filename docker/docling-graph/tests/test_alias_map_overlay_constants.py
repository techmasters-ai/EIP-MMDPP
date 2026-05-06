"""Drift guards for _alias_map.py overlay constants (spec §8.2).

These tests pin the structure and content invariants of the four new
constants (MISSILE_IDENTITY_LABELS, RADAR_IDENTITY_LABELS,
CROSS_ENTITY_REF_PATTERNS, CANONICAL_PRIORITY) so that future edits
cannot silently break the overlay's classification rules.
"""
# Existing test convention in this directory loads service modules via
# importlib.spec_from_file_location (see test_table_facts_*.py). The
# conftest appends the service root to sys.path AT THE END, which is
# not enough on its own — the repo-root `app/` package wins on
# `from app import …`. Direct importlib loading sidesteps that.
import importlib.util
from pathlib import Path

_ALIAS_PATH = Path(__file__).resolve().parent.parent / "app" / "_alias_map.py"


def _load_alias_map():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_alias_map_overlay", _ALIAS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missile_identity_labels_excludes_bare_variant_and_designation():
    """Spec §5.1: bare 'variant' and 'designation' MUST NOT appear; they
    create false positives via cross-entity-ref rows like
    'Fan Song Variant'."""
    am = _load_alias_map()
    labels = tuple(s.lower() for s in am.MISSILE_IDENTITY_LABELS)
    assert "variant" not in labels, (
        "bare 'variant' would misclassify Fan Song Variant rows as missile aliases"
    )
    assert "designation" not in labels


def test_identity_labels_have_canonical_priority_coverage():
    """Every label in MISSILE_IDENTITY_LABELS appears (case-insensitive
    substring) in CANONICAL_PRIORITY['MISSILE_SYSTEM'] OR is documented
    as intentional fallback. Catches new label without priority entry."""
    am = _load_alias_map()
    priority = tuple(s.lower() for s in am.CANONICAL_PRIORITY["MISSILE_SYSTEM"])
    for label in am.MISSILE_IDENTITY_LABELS:
        norm = label.lower()
        # Match any priority entry that's a substring of the label, or vice-versa.
        assert any(p in norm or norm in p for p in priority), (
            f"identity label {label!r} has no CANONICAL_PRIORITY entry"
        )


def test_cross_entity_ref_patterns_dont_overlap_identity_labels():
    """A label can't be both a missile-identity row AND a cross-entity-ref
    row. CROSS_ENTITY_REF_PATTERNS keys must not match MISSILE_IDENTITY_LABELS
    or RADAR_IDENTITY_LABELS after normalization."""
    am = _load_alias_map()
    refs = set(am.CROSS_ENTITY_REF_PATTERNS.keys())
    missile = {s.lower() for s in am.MISSILE_IDENTITY_LABELS}
    radar = {s.lower() for s in am.RADAR_IDENTITY_LABELS}
    assert refs.isdisjoint(missile)
    assert refs.isdisjoint(radar)


def test_canonical_priority_uses_display_labels():
    """CANONICAL_PRIORITY entries are user-facing label patterns
    (Title Case with spaces, e.g., 'Missile Type'), not schema field
    names (snake_case)."""
    am = _load_alias_map()
    for entity_type, entries in am.CANONICAL_PRIORITY.items():
        for entry in entries:
            assert " " in entry, (
                f"{entity_type}: {entry!r} is missing a space — looks like a "
                f"schema field name, not a display label"
            )
            assert "_" not in entry, (
                f"{entity_type}: {entry!r} contains underscore — display "
                f"labels should be Title Case with spaces"
            )
            assert entry[0].isupper(), (
                f"{entity_type}: {entry!r} should start uppercase"
            )
