"""Tests for Recommendation 1 — unresolved-cross-entity-hint diagnostics
emitted by `_postprocess_air_defense_system_links`.

The diagnostics make hint-resolution failures visible in the postprocess
output so downstream callers can distinguish "no hints were generated"
from "hints generated, all resolved" from "hints generated, N failed".
"""
from __future__ import annotations
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

_SR = pathlib.Path(__file__).resolve().parent.parent / "app"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load("app._numeric_evidence", _SR / "_numeric_evidence.py")
_eg = _load("_dgp_evidence_gate", _SR / "evidence_gate.py")


def _upstream(ref_id, entity_type, system_name):
    return SimpleNamespace(
        ref_id=ref_id,
        entity_type=entity_type,
        identity_values={"system_name": system_name},
        display_label=system_name,
    )


def _hint(source_canonical, source_entity_type, target_alias, target_entity_type):
    return SimpleNamespace(
        source_canonical=source_canonical,
        source_entity_type=source_entity_type,
        target_alias=target_alias,
        target_entity_type=target_entity_type,
        relationship_kind="associated_with",
    )


def _call(upstream, hints, alias_map=None):
    return _eg._postprocess_air_defense_system_links(
        {"relationships": []},
        "",
        upstream,
        hints,
        alias_map_by_entity_type=alias_map,
    )


# --- diagnostics structure -------------------------------------------------

def test_diagnostics_always_present_when_hints_passed():
    """Even when zero hints fail, the unresolved_cross_entity_hints key
    appears (with count=0). Lets downstream code distinguish 'no hints'
    from 'hints all resolved'."""
    upstream = [
        _upstream("MIS:M1", "MISSILE_SYSTEM", "M1"),
        _upstream("RAD:R1", "RADAR_SYSTEM", "R1"),
    ]
    hints = [_hint("M1", "MISSILE_SYSTEM", "R1", "RADAR_SYSTEM")]
    _, stats = _call(upstream, hints)
    assert "unresolved_cross_entity_hints" in stats
    assert stats["unresolved_cross_entity_hints"]["count"] == 0
    assert stats["unresolved_cross_entity_hints"]["samples"] == []


def test_diagnostics_absent_when_no_hints_passed_at_all():
    """When cross_entity_hints is None (not just empty list), the
    diagnostic key is absent — distinguishes 'overlay didn't run' from
    'overlay ran but produced empty hint list'."""
    _, stats = _eg._postprocess_air_defense_system_links(
        {"relationships": []},
        "",
        [_upstream("MIS:M1", "MISSILE_SYSTEM", "M1")],
        None,  # explicitly None
    )
    assert "unresolved_cross_entity_hints" not in stats


def test_diagnostics_present_when_empty_hint_list():
    """Empty hint list ≠ None — diagnostic should appear with count=0."""
    _, stats = _call([_upstream("MIS:M1", "MISSILE_SYSTEM", "M1")], [])
    assert "unresolved_cross_entity_hints" in stats
    assert stats["unresolved_cross_entity_hints"]["count"] == 0


# --- source unresolved -----------------------------------------------------

def test_unresolved_source():
    """Hint with source not in upstream → diagnostic records reason=source_unresolved."""
    upstream = [_upstream("RAD:R1", "RADAR_SYSTEM", "R1")]
    hints = [_hint("UNKNOWN_MISSILE", "MISSILE_SYSTEM", "R1", "RADAR_SYSTEM")]
    _, stats = _call(upstream, hints)
    diag = stats["unresolved_cross_entity_hints"]
    assert diag["count"] == 1
    sample = diag["samples"][0]
    assert sample["source_alias"] == "UNKNOWN_MISSILE"
    assert sample["target_alias"] == "R1"
    assert sample["source_type"] == "MISSILE_SYSTEM"
    assert sample["target_type"] == "RADAR_SYSTEM"
    assert sample["source_resolved"] is False
    assert sample["target_resolved"] is True
    assert sample["reason"] == "source_unresolved"


# --- target unresolved -----------------------------------------------------

def test_unresolved_target():
    """Hint with target not in upstream → reason=target_unresolved.
    Models the SA-2 case where `RSN- 75V` doesn't match upstream
    `Fan Song`."""
    upstream = [_upstream("MIS:M1", "MISSILE_SYSTEM", "M1")]
    hints = [_hint("M1", "MISSILE_SYSTEM", "RSN- 75V", "RADAR_SYSTEM")]
    _, stats = _call(upstream, hints)
    diag = stats["unresolved_cross_entity_hints"]
    assert diag["count"] == 1
    sample = diag["samples"][0]
    assert sample["source_resolved"] is True
    assert sample["target_resolved"] is False
    assert sample["reason"] == "target_unresolved"
    assert sample["target_alias"] == "RSN- 75V"


# --- both unresolved -------------------------------------------------------

def test_both_unresolved():
    """Neither source nor target in upstream → reason=both_unresolved."""
    upstream = []
    hints = [_hint("X", "MISSILE_SYSTEM", "Y", "RADAR_SYSTEM")]
    _, stats = _call(upstream, hints)
    diag = stats["unresolved_cross_entity_hints"]
    assert diag["count"] == 1
    sample = diag["samples"][0]
    assert sample["source_resolved"] is False
    assert sample["target_resolved"] is False
    assert sample["reason"] == "both_unresolved"


# --- all resolved → samples empty, count zero ------------------------------

def test_all_resolved_yields_zero_count_and_empty_samples():
    upstream = [
        _upstream("MIS:M1", "MISSILE_SYSTEM", "M1"),
        _upstream("RAD:R1", "RADAR_SYSTEM", "R1"),
        _upstream("MIS:M2", "MISSILE_SYSTEM", "M2"),
        _upstream("RAD:R2", "RADAR_SYSTEM", "R2"),
    ]
    hints = [
        _hint("M1", "MISSILE_SYSTEM", "R1", "RADAR_SYSTEM"),
        _hint("M2", "MISSILE_SYSTEM", "R2", "RADAR_SYSTEM"),
    ]
    pass_output, stats = _call(upstream, hints)
    diag = stats["unresolved_cross_entity_hints"]
    assert diag["count"] == 0
    assert diag["samples"] == []
    # promoted_from_cross_entity_hints should have both edges
    assert "promoted_from_cross_entity_hints" in stats
    assert len(stats["promoted_from_cross_entity_hints"]) == 2


# --- bounded sample collection ---------------------------------------------

def test_unresolved_sample_count_is_bounded():
    """When many hints fail, only the first N samples are kept; count
    reflects the true total."""
    upstream = []  # nothing resolves
    hints = [_hint(f"M{i}", "MISSILE_SYSTEM", f"R{i}", "RADAR_SYSTEM") for i in range(50)]
    _, stats = _call(upstream, hints)
    diag = stats["unresolved_cross_entity_hints"]
    assert diag["count"] == 50
    # cap is 20 per the implementation
    assert len(diag["samples"]) == 20


# --- existing behavior preserved -------------------------------------------

def test_promoted_behavior_unchanged():
    """Adding the diagnostic must NOT change the promoted_from_cross_entity_hints
    output for cases where some hints resolve."""
    upstream = [
        _upstream("MIS:M1", "MISSILE_SYSTEM", "M1"),
        _upstream("RAD:R1", "RADAR_SYSTEM", "R1"),
    ]
    hints = [
        _hint("M1", "MISSILE_SYSTEM", "R1", "RADAR_SYSTEM"),
        _hint("UNKNOWN", "MISSILE_SYSTEM", "R1", "RADAR_SYSTEM"),
    ]
    pass_output, stats = _call(upstream, hints)
    # one promoted, one unresolved
    assert len(stats["promoted_from_cross_entity_hints"]) == 1
    assert stats["promoted_from_cross_entity_hints"][0]["from_ref_id"] == "MIS:M1"
    assert stats["promoted_from_cross_entity_hints"][0]["to_ref_id"] == "RAD:R1"
    assert stats["unresolved_cross_entity_hints"]["count"] == 1


def test_diagnostic_uses_dict_hints_too():
    """Hints can arrive as dicts (not just dataclass-like) — the diagnostic
    should record from those too."""
    upstream = [_upstream("RAD:R1", "RADAR_SYSTEM", "R1")]
    hints = [{
        "source_canonical": "UNKNOWN_MISSILE",
        "source_entity_type": "MISSILE_SYSTEM",
        "target_alias": "R1",
        "target_entity_type": "RADAR_SYSTEM",
        "relationship_kind": "associated_with",
    }]
    _, stats = _call(upstream, hints)
    diag = stats["unresolved_cross_entity_hints"]
    assert diag["count"] == 1
    assert diag["samples"][0]["source_alias"] == "UNKNOWN_MISSILE"
