"""Item 3: role-aware CUES validation in system_links postprocess.

For each LLM-emitted RADAR_SYSTEM → RADAR_SYSTEM relationship:
  * If source role ∈ {SEARCH, HEIGHT_FINDER} AND target role ∈ {FIRE_CONTROL},
    retype to CUES (same direction).
  * If LLM-emitted direction is reversed (source=FIRE_CONTROL, target=SEARCH/HEIGHT_FINDER)
    AND both roles strongly support inversion, FLIP direction and set CUES.
  * If either role is missing, ambiguous, or unsupported, leave unchanged
    and emit a diagnostic.

Constraints:
  * NEVER touches missile↔radar or any non-radar edges.
  * NEVER creates new edges.
  * NEVER uses equipment-specific names — operates purely on entity_type
    + role string.
  * Promoted-from-hint rels (deterministic) are not eligible for retype —
    only LLM-emitted rels are processed.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_SERVICE_APP_ROOT = Path(__file__).resolve().parent.parent / "app"

_NUM_EV_SPEC = importlib.util.spec_from_file_location(
    "app._numeric_evidence", _SERVICE_APP_ROOT / "_numeric_evidence.py"
)
_NUM_EV_MOD = importlib.util.module_from_spec(_NUM_EV_SPEC)
sys.modules["app._numeric_evidence"] = _NUM_EV_MOD
assert _NUM_EV_SPEC.loader is not None
_NUM_EV_SPEC.loader.exec_module(_NUM_EV_MOD)

_MODULE_PATH = _SERVICE_APP_ROOT / "evidence_gate.py"
_SPEC = importlib.util.spec_from_file_location("docling_graph_evidence_gate", _MODULE_PATH)
_EVIDENCE_GATE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_EVIDENCE_GATE)


def _make_radar(name: str, role: str | None) -> SimpleNamespace:
    """Build an upstream EntityRef-shaped object with a role in properties."""
    return SimpleNamespace(
        ref_id=f"RADAR_SYSTEM:{name}",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": name},
        display_label=name,
        aliases=None,
        properties={"emitter_function": role} if role else None,
    )


def _make_missile(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        ref_id=f"MISSILE_SYSTEM:{name}",
        entity_type="MISSILE_SYSTEM",
        identity_values={"system_name": name},
        display_label=name,
        aliases=None,
        properties=None,
    )


# ===== Unit tests on the helper =====

def test_build_role_map_extracts_emitter_function():
    """`_build_role_map_by_ref` returns {ref_id → role} for RADAR_SYSTEM
    entities only."""
    upstream = [
        _make_radar("SearchOne", "SEARCH"),
        _make_radar("FCOne", "FIRE_CONTROL"),
        _make_radar("NoRole", None),
        _make_missile("Missile1"),
    ]
    role_map = _EVIDENCE_GATE._build_role_map_by_ref(upstream)
    assert role_map == {
        "RADAR_SYSTEM:SearchOne": "SEARCH",
        "RADAR_SYSTEM:FCOne": "FIRE_CONTROL",
    }


def test_retype_search_to_fire_control_becomes_cues():
    """Same direction (search→FC) gets retyped from ASSOCIATED_WITH to CUES."""
    rels = [
        {"rel_type": "ASSOCIATED_WITH",
         "from_ref_id": "RADAR_SYSTEM:Searcher",
         "to_ref_id": "RADAR_SYSTEM:Engager",
         "confidence": 0.9},
    ]
    role_map = {
        "RADAR_SYSTEM:Searcher": "SEARCH",
        "RADAR_SYSTEM:Engager": "FIRE_CONTROL",
    }
    diag = _EVIDENCE_GATE._retype_radar_radar_to_cues(rels, role_map)
    assert rels[0]["rel_type"] == "CUES"
    assert rels[0]["from_ref_id"] == "RADAR_SYSTEM:Searcher"  # unchanged direction
    assert rels[0]["to_ref_id"] == "RADAR_SYSTEM:Engager"
    assert len(diag["retyped"]) == 1
    assert diag["retyped"][0]["reason"] == "search_to_fire_control"


def test_retype_height_finder_to_fire_control_becomes_cues():
    """HEIGHT_FINDER counts as source-side role per user spec."""
    rels = [
        {"rel_type": "ASSOCIATED_WITH",
         "from_ref_id": "RADAR_SYSTEM:HFinder",
         "to_ref_id": "RADAR_SYSTEM:Engager",
         "confidence": 0.9},
    ]
    role_map = {
        "RADAR_SYSTEM:HFinder": "HEIGHT_FINDER",
        "RADAR_SYSTEM:Engager": "FIRE_CONTROL",
    }
    diag = _EVIDENCE_GATE._retype_radar_radar_to_cues(rels, role_map)
    assert rels[0]["rel_type"] == "CUES"
    assert diag["retyped"][0]["reason"] == "height_finder_to_fire_control"


def test_flip_reversed_direction_when_roles_support_it():
    """LLM emitted FC→SEARCH; roles strongly support reversal → flip + CUES."""
    rels = [
        {"rel_type": "ASSOCIATED_WITH",
         "from_ref_id": "RADAR_SYSTEM:Engager",
         "to_ref_id": "RADAR_SYSTEM:Searcher",
         "confidence": 0.9},
    ]
    role_map = {
        "RADAR_SYSTEM:Engager": "FIRE_CONTROL",
        "RADAR_SYSTEM:Searcher": "SEARCH",
    }
    diag = _EVIDENCE_GATE._retype_radar_radar_to_cues(rels, role_map)
    assert rels[0]["rel_type"] == "CUES"
    # Flipped direction
    assert rels[0]["from_ref_id"] == "RADAR_SYSTEM:Searcher"
    assert rels[0]["to_ref_id"] == "RADAR_SYSTEM:Engager"
    assert len(diag["flipped"]) == 1


def test_missing_role_leaves_unchanged_with_diagnostic():
    """When source role is missing, edge is left unchanged but flagged."""
    rels = [
        {"rel_type": "ASSOCIATED_WITH",
         "from_ref_id": "RADAR_SYSTEM:Unknown",
         "to_ref_id": "RADAR_SYSTEM:Engager",
         "confidence": 0.9},
    ]
    role_map = {
        "RADAR_SYSTEM:Engager": "FIRE_CONTROL",
        # Unknown has no role mapping
    }
    diag = _EVIDENCE_GATE._retype_radar_radar_to_cues(rels, role_map)
    assert rels[0]["rel_type"] == "ASSOCIATED_WITH"  # unchanged
    assert len(diag["skipped"]) == 1
    assert "missing_role" in diag["skipped"][0]["reason"]


def test_ambiguous_role_pair_leaves_unchanged():
    """Both endpoints same role (e.g., FC→FC) — not a valid CUES direction;
    leave unchanged with diagnostic."""
    rels = [
        {"rel_type": "ASSOCIATED_WITH",
         "from_ref_id": "RADAR_SYSTEM:FC1",
         "to_ref_id": "RADAR_SYSTEM:FC2",
         "confidence": 0.9},
    ]
    role_map = {
        "RADAR_SYSTEM:FC1": "FIRE_CONTROL",
        "RADAR_SYSTEM:FC2": "FIRE_CONTROL",
    }
    diag = _EVIDENCE_GATE._retype_radar_radar_to_cues(rels, role_map)
    assert rels[0]["rel_type"] == "ASSOCIATED_WITH"
    assert len(diag["skipped"]) == 1


def test_never_touches_missile_radar_edges():
    """Cross-type edges (MISSILE → RADAR or vice versa) are never eligible."""
    rels = [
        {"rel_type": "ASSOCIATED_WITH",
         "from_ref_id": "MISSILE_SYSTEM:M1",
         "to_ref_id": "RADAR_SYSTEM:R1",
         "confidence": 1.0},
        {"rel_type": "ASSOCIATED_WITH",
         "from_ref_id": "RADAR_SYSTEM:R1",
         "to_ref_id": "MISSILE_SYSTEM:M1",
         "confidence": 1.0},
    ]
    role_map = {"RADAR_SYSTEM:R1": "FIRE_CONTROL"}
    diag = _EVIDENCE_GATE._retype_radar_radar_to_cues(rels, role_map)
    for r in rels:
        assert r["rel_type"] == "ASSOCIATED_WITH"  # unchanged
    # Cross-type rels not even counted as skipped — they're out-of-scope
    assert len(diag.get("retyped", [])) == 0
    assert len(diag.get("flipped", [])) == 0


def test_does_not_retype_cues_edges_already_typed():
    """If the LLM already emitted CUES correctly, no diagnostic noise."""
    rels = [
        {"rel_type": "CUES",
         "from_ref_id": "RADAR_SYSTEM:Searcher",
         "to_ref_id": "RADAR_SYSTEM:Engager",
         "confidence": 0.9},
    ]
    role_map = {
        "RADAR_SYSTEM:Searcher": "SEARCH",
        "RADAR_SYSTEM:Engager": "FIRE_CONTROL",
    }
    diag = _EVIDENCE_GATE._retype_radar_radar_to_cues(rels, role_map)
    assert rels[0]["rel_type"] == "CUES"
    # Should not be counted as "retyped" since it was already CUES
    assert len(diag.get("retyped", [])) == 0


def test_does_not_create_new_edges():
    """Operate in-place on the rels list. Never add new entries."""
    rels = [
        {"rel_type": "ASSOCIATED_WITH",
         "from_ref_id": "RADAR_SYSTEM:Searcher",
         "to_ref_id": "RADAR_SYSTEM:Engager",
         "confidence": 0.9},
    ]
    role_map = {
        "RADAR_SYSTEM:Searcher": "SEARCH",
        "RADAR_SYSTEM:Engager": "FIRE_CONTROL",
    }
    before_count = len(rels)
    _EVIDENCE_GATE._retype_radar_radar_to_cues(rels, role_map)
    assert len(rels) == before_count


# ===== Integration test against postprocess =====

def test_postprocess_integration_retypes_search_to_fire_control():
    """End-to-end: feed pass_output through _postprocess_air_defense_system_links
    with role info on upstream_entities. Result has CUES for the search→FC edge."""
    upstream = [
        _make_radar("Searcher", "SEARCH"),
        _make_radar("Engager", "FIRE_CONTROL"),
    ]
    pass_output = {
        "relationships": [
            {"rel_type": "ASSOCIATED_WITH",
             "from_ref_id": "RADAR_SYSTEM:Searcher",
             "to_ref_id": "RADAR_SYSTEM:Engager",
             "confidence": 0.9},
        ]
    }
    out, stats = _EVIDENCE_GATE._postprocess_air_defense_system_links(
        pass_output, evidence_text="", upstream_entities=upstream,
        cross_entity_hints=None, alias_map_by_entity_type=None,
    )
    rels = out["relationships"]
    assert len(rels) == 1
    assert rels[0]["rel_type"] == "CUES"
    # Stats should expose the retype diagnostic
    assert "role_aware_cues" in stats
    assert len(stats["role_aware_cues"].get("retyped", [])) == 1
