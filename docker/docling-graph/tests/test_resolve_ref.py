"""Tests for `_resolve_ref` — the type-segregated upstream-ref resolver
used by `_postprocess_air_defense_system_links` for cross_entity_hints.

Regression guard for the cross-type leak found 2026-05-16 during review:
the prior implementation used a type-agnostic name_to_ref map, so a
missile name that happened to collide with a radar name in the upstream
catalog would silently leak the radar ref where a missile was requested.
These tests pin the type-segregated behavior.

All fixtures use generic identity tokens (M-X, R-Y, etc.) — never
document-specific equipment names.
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
_eg = _load("_dgp_evidence_gate", _SR / "evidence_gate.py")
_resolve_ref = _eg._resolve_ref
_build_upstream_name_map_by_type = _eg._build_upstream_name_map_by_type


def _norm(s: str) -> str:
    return _eg.normalize_evidence_text(s)


# --- direct-hit, same-type --------------------------------------------------

def test_direct_hit_within_same_type():
    """name → ref via direct upstream lookup within the same entity_type."""
    upstream_by_type = {"MISSILE_SYSTEM": {_norm("M-X"): "ref-mis-001"}}
    result = _resolve_ref("M-X", "MISSILE_SYSTEM", upstream_by_type, None)
    assert result == "ref-mis-001"


def test_direct_hit_case_insensitive():
    """name lookup normalizes case before consulting the per-type map."""
    upstream_by_type = {"MISSILE_SYSTEM": {_norm("M-X"): "ref-mis-001"}}
    result = _resolve_ref("m-x", "MISSILE_SYSTEM", upstream_by_type, None)
    assert result == "ref-mis-001"


# --- cross-type leak prevention --------------------------------------------

def test_cross_type_leak_is_blocked_at_direct_hit():
    """name collides across types: upstream has the SAME token registered
    under RADAR_SYSTEM, but caller asks for MISSILE_SYSTEM. Must return
    None — never leak the radar ref."""
    upstream_by_type = {"RADAR_SYSTEM": {_norm("M-X"): "ref-radar-001"}}
    result = _resolve_ref("M-X", "MISSILE_SYSTEM", upstream_by_type, None)
    assert result is None, (
        f"cross-type leak detected: asked MISSILE_SYSTEM, got radar ref {result!r}"
    )


def test_cross_type_leak_is_blocked_at_alias_fallback():
    """Alias map has the name → canonical mapping, but only under the
    wrong entity_type. Must not cross-resolve."""
    upstream_by_type = {"RADAR_SYSTEM": {_norm("canonical-r"): "ref-radar"}}
    alias_map = {"RADAR_SYSTEM": {_norm("alias-x"): "canonical-r"}}
    # Caller asks for MISSILE — must NOT find the radar via alias path
    result = _resolve_ref("alias-x", "MISSILE_SYSTEM", upstream_by_type, alias_map)
    assert result is None


# --- alias-fallback within the same type -----------------------------------

def test_alias_fallback_within_same_type():
    """When name doesn't directly hit upstream, overlay alias map can
    bridge alias → canonical, then canonical → upstream ref — all within
    the same entity_type."""
    upstream_by_type = {"MISSILE_SYSTEM": {_norm("canonical-m"): "ref-mis"}}
    alias_map = {"MISSILE_SYSTEM": {_norm("alias-x"): "canonical-m"}}
    result = _resolve_ref("alias-x", "MISSILE_SYSTEM", upstream_by_type, alias_map)
    assert result == "ref-mis"


def test_alias_fallback_canonical_not_in_upstream_returns_none():
    """If the canonical that the alias maps to isn't in upstream, return
    None (don't fabricate a ref)."""
    upstream_by_type = {"MISSILE_SYSTEM": {}}
    alias_map = {"MISSILE_SYSTEM": {_norm("alias-x"): "missing-canonical"}}
    result = _resolve_ref("alias-x", "MISSILE_SYSTEM", upstream_by_type, alias_map)
    assert result is None


# --- defensive behavior ----------------------------------------------------

def test_none_alias_map_does_not_crash():
    upstream_by_type = {"MISSILE_SYSTEM": {_norm("M-X"): "ref-mis"}}
    result = _resolve_ref("M-X", "MISSILE_SYSTEM", upstream_by_type, None)
    assert result == "ref-mis"


def test_empty_inputs_return_none():
    assert _resolve_ref("", "MISSILE_SYSTEM", {}, None) is None
    assert _resolve_ref("X", "MISSILE_SYSTEM", {}, None) is None
    assert _resolve_ref(None, "MISSILE_SYSTEM", {}, None) is None  # type: ignore[arg-type]


def test_missing_entity_type_returns_none():
    """Without entity_type, type-segregated resolution can't happen — return
    None (don't fall back to global lookup, which would re-introduce the
    cross-type leak)."""
    upstream_by_type = {"MISSILE_SYSTEM": {_norm("M-X"): "ref-mis"}}
    assert _resolve_ref("M-X", None, upstream_by_type, None) is None
    assert _resolve_ref("M-X", "", upstream_by_type, None) is None


# --- _build_upstream_name_map_by_type --------------------------------------

def test_build_per_type_segregates_by_entity_type():
    """Two entities with the SAME system_name but different entity_types
    must end up in different type maps — never overwriting each other."""
    from types import SimpleNamespace
    upstream = [
        SimpleNamespace(
            ref_id="MISSILE_SYSTEM:M-X",
            entity_type="MISSILE_SYSTEM",
            identity_values={"system_name": "M-X"},
            display_label="M-X",
        ),
        SimpleNamespace(
            ref_id="RADAR_SYSTEM:M-X",
            entity_type="RADAR_SYSTEM",
            identity_values={"system_name": "M-X"},
            display_label="M-X",
        ),
    ]
    result = _build_upstream_name_map_by_type(upstream)
    assert result["MISSILE_SYSTEM"][_norm("M-X")] == "MISSILE_SYSTEM:M-X"
    assert result["RADAR_SYSTEM"][_norm("M-X")] == "RADAR_SYSTEM:M-X"


def test_build_per_type_skips_entities_without_entity_type():
    """Defensive: an upstream entity missing entity_type is skipped (not
    silently coerced to some default type which could leak)."""
    from types import SimpleNamespace
    upstream = [
        SimpleNamespace(
            ref_id="MISSILE_SYSTEM:M-X",
            identity_values={"system_name": "M-X"},
            display_label="M-X",
            # entity_type intentionally absent
        ),
    ]
    result = _build_upstream_name_map_by_type(upstream)
    assert result == {}


def test_build_per_type_registers_aliases_under_same_type():
    """Aliases get the entity's type, not a separate type."""
    from types import SimpleNamespace
    upstream = [
        SimpleNamespace(
            ref_id="MISSILE_SYSTEM:M-X",
            entity_type="MISSILE_SYSTEM",
            identity_values={"system_name": "M-X"},
            display_label="M-X",
            aliases=["alias-1", "alias-2"],
        ),
    ]
    result = _build_upstream_name_map_by_type(upstream)
    assert result["MISSILE_SYSTEM"][_norm("M-X")] == "MISSILE_SYSTEM:M-X"
    assert result["MISSILE_SYSTEM"][_norm("alias-1")] == "MISSILE_SYSTEM:M-X"
    assert result["MISSILE_SYSTEM"][_norm("alias-2")] == "MISSILE_SYSTEM:M-X"
