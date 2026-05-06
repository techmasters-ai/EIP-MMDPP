"""Tests for resolve_alias (spec §5.5).

The actual data-driven resolution is exhaustively tested in
test_alias_map_missile.py and test_alias_map_radar.py via direct dict
lookups; these tests verify the resolve_alias wrapper handles
normalization and the None-fallback path.
"""
import sys
from pathlib import Path

# Add app and parent (for 'app' module) to path.
app_path = Path(__file__).resolve().parent.parent / "app"
parent_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(app_path))
sys.path.insert(0, str(parent_path))

import _table_facts as tf


def test_resolve_uses_normalize_label():
    """Caller passes raw label; resolve_alias normalizes internally."""
    assert tf.resolve_alias("Length mm", None, "missile_airframe") == "body_length_m"
    assert tf.resolve_alias("LENGTH MM", None, "missile_airframe") == "body_length_m"
    assert tf.resolve_alias("  length, mm  ", None, "missile_airframe") == "body_length_m"


def test_resolve_returns_none_for_unknown_label():
    assert tf.resolve_alias("Mystery Field", None, "missile_airframe") is None


def test_resolve_section_conditional():
    assert tf.resolve_alias("Weight kg", "1st Stage", "missile_propulsion") == "booster_mass_kg"
    assert tf.resolve_alias("Weight kg", None, "missile_propulsion") is None
    assert tf.resolve_alias("Weight kg", None, "missile_airframe") == "total_mass_kg"


def test_resolve_pass_conditional():
    assert tf.resolve_alias("Length mm", None, "missile_airframe") == "body_length_m"
    assert tf.resolve_alias("Length mm", None, "missile_propulsion") is None
