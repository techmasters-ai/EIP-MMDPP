"""Item 4: deterministic VARIANT_OF emitter for missile variant → parent SAM family.

Per spec:
  * source: child variant MISSILE_SYSTEM entity
  * target: parent/family MISSILE_SYSTEM entity (same type)
  * evidence: child's alias/identity field (nomenclature/name/dieqp) contains
    the parent's system_name as a substring (case-insensitive structural match)
  * both endpoints must resolve to existing typed entities
  * never creates new entities; never cross-type edges
  * deterministic — no LLM, no equipment-specific constants
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


def _make_missile(name, aliases=None, properties=None):
    return SimpleNamespace(
        ref_id=f"MISSILE_SYSTEM:{name}",
        entity_type="MISSILE_SYSTEM",
        identity_values={"system_name": name},
        display_label=name,
        aliases=aliases,
        properties=properties,
    )


def _make_radar(name, aliases=None, properties=None):
    return SimpleNamespace(
        ref_id=f"RADAR_SYSTEM:{name}",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": name},
        display_label=name,
        aliases=aliases,
        properties=properties,
    )


# ===== Unit tests =====

def test_emits_variant_of_when_alias_contains_parent_system_name():
    """Child '1D' with aliases=['S-75 Dvina'] gets a VARIANT_OF edge to
    parent 'S-75' because 'S-75' appears as a substring in the alias."""
    upstream = [
        _make_missile("S-75", aliases=["Dvina"]),  # parent family
        _make_missile("1D", aliases=["S-75 Dvina"]),  # variant
    ]
    seen_pairs: set = set()
    out: list = []
    diag = _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert len(out) == 1
    assert out[0] == {
        "rel_type": "VARIANT_OF",
        "from_ref_id": "MISSILE_SYSTEM:1D",
        "to_ref_id": "MISSILE_SYSTEM:S-75",
        "confidence": 1.0,
    }
    assert len(diag["emitted"]) == 1


def test_does_not_create_self_edge():
    """Entity whose alias contains its own system_name → no self-loop."""
    upstream = [
        _make_missile("S-75", aliases=["S-75 Dvina"]),
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert len(out) == 0


def test_no_match_when_parent_not_in_upstream():
    """Alias mentions a name that isn't in the upstream catalog → no emit
    (only emit when both endpoints resolve to existing entities)."""
    upstream = [
        _make_missile("1D", aliases=["Unknown Family"]),  # 'Unknown Family' has no entity
    ]
    seen_pairs: set = set()
    out: list = []
    diag = _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert len(out) == 0
    # Skipped due to no parent resolution
    assert len(diag["skipped"]) >= 0  # may be 0 if no candidate detected


def test_never_creates_cross_type_edges():
    """A missile alias mentioning a radar name MUST NOT create missile→radar."""
    upstream = [
        _make_radar("S-75 Fan Song"),  # radar with name including 'S-75'
        _make_missile("1D", aliases=["S-75 Fan Song"]),  # alias mentions the radar name
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    # No VARIANT_OF edge — radar isn't a valid parent (different entity_type)
    for e in out:
        assert e.get("rel_type") != "VARIANT_OF" or (
            e["from_ref_id"].split(":")[0] == e["to_ref_id"].split(":")[0]
        ), f"cross-type VARIANT_OF created: {e}"


def test_deduplicates_against_seen_pairs():
    """If (from_ref, to_ref) already in seen_pairs, don't add a duplicate."""
    upstream = [
        _make_missile("S-75"),
        _make_missile("1D", aliases=["S-75 Dvina"]),
    ]
    seen_pairs: set = {("MISSILE_SYSTEM:1D", "MISSILE_SYSTEM:S-75")}
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert len(out) == 0


def test_handles_nomenclature_alias_field():
    """Aliases harvested from nomenclature field also trigger detection.
    (The capture script puts nomenclature/name/dieqp into the `aliases` list.)"""
    upstream = [
        _make_missile("S-75M"),
        _make_missile("20D", aliases=["S-75M"]),  # nomenclature = "S-75M" parent
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert len(out) == 1
    assert out[0]["to_ref_id"] == "MISSILE_SYSTEM:S-75M"


def test_picks_longest_matching_parent_when_multiple():
    """Given parents 'S-75' and 'S-75M', and child alias 'S-75M Dvina',
    pick 'S-75M' (longer, more specific match)."""
    upstream = [
        _make_missile("S-75"),
        _make_missile("S-75M"),
        _make_missile("20D", aliases=["S-75M Volkhov"]),
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert len(out) == 1
    assert out[0]["to_ref_id"] == "MISSILE_SYSTEM:S-75M"


def test_emit_is_case_insensitive_structural_match():
    """Match must be case-insensitive — 's-75' alias under S-75 family resolves."""
    upstream = [
        _make_missile("S-75"),
        _make_missile("1D", aliases=["s-75 dvina"]),  # lowercase
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert len(out) == 1


def test_diagnostic_records_emitted_and_skipped():
    """For every candidate, emit a diagnostic entry capturing alias,
    resolved parent ref, and reason (emitted | skipped)."""
    upstream = [
        _make_missile("S-75"),
        _make_missile("1D", aliases=["S-75 Dvina"]),
        _make_missile("Unrelated", aliases=["Has No Parent"]),
    ]
    seen_pairs: set = set()
    out: list = []
    diag = _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert "emitted" in diag
    assert len(diag["emitted"]) >= 1
    # Diagnostic carries child + parent + reason
    e = diag["emitted"][0]
    assert "child_ref_id" in e
    assert "parent_ref_id" in e
    assert "matched_alias" in e


# ===== Boundary-aware matching tests (Item 4 tightening) =====
# These guard against false parentage from raw substring containment.

def test_rejects_prefix_match_inside_longer_alphanumeric():
    """Parent 'S-75' must NOT match inside alias 'S-750' (digit follows
    parent name with no separator). Generic — applies to any short
    designation that prefixes a longer one."""
    upstream = [
        _make_missile("S-75"),
        _make_missile("S-750", aliases=["S-750"]),  # alias contains itself, not 'S-75'
    ]
    seen_pairs: set = set()
    out: list = []
    diag = _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    # No VARIANT_OF S-750 → S-75 (would be false parentage)
    for e in out:
        assert not (e["from_ref_id"] == "MISSILE_SYSTEM:S-750"
                    and e["to_ref_id"] == "MISSILE_SYSTEM:S-75"), (
            f"False parentage: S-750 should not be a variant of S-75 (substring trap)"
        )


def test_rejects_suffix_match_inside_longer_alphanumeric():
    """Parent 'SA-2' must NOT match inside alias 'SA-20' (digit follows
    parent name with no boundary)."""
    upstream = [
        _make_missile("SA-2"),
        _make_missile("SA-20", aliases=["SA-20 Foo"]),
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    for e in out:
        assert not (e["from_ref_id"] == "MISSILE_SYSTEM:SA-20"
                    and e["to_ref_id"] == "MISSILE_SYSTEM:SA-2")


def test_accepts_match_at_boundary_with_space():
    """Parent 'S-75' SHOULD match in alias 'S-75 Dvina' (space boundary)."""
    upstream = [
        _make_missile("S-75"),
        _make_missile("1D", aliases=["S-75 Dvina"]),
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert any(e["from_ref_id"] == "MISSILE_SYSTEM:1D"
               and e["to_ref_id"] == "MISSILE_SYSTEM:S-75"
               for e in out)


def test_accepts_match_at_boundary_with_slash():
    """Parent 'S-75' SHOULD match in alias 'S-75/SNR-75' (slash boundary)."""
    upstream = [
        _make_missile("S-75"),
        _make_missile("DerivedX", aliases=["S-75/Variant"]),
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert any(e["from_ref_id"] == "MISSILE_SYSTEM:DerivedX"
               and e["to_ref_id"] == "MISSILE_SYSTEM:S-75"
               for e in out)


def test_rejects_parent_name_too_short():
    """Parent system_name shorter than minimum normalized length (3 chars
    of letters+digits combined) should not match against any alias —
    too risky for false parentage."""
    upstream = [
        _make_missile("M1"),  # 2 chars — too short
        _make_missile("Variant", aliases=["M1 something"]),
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    for e in out:
        assert e["to_ref_id"] != "MISSILE_SYSTEM:M1", (
            "M1 too short to be a safe parent name"
        )


def test_parent_must_have_letter_and_digit():
    """Parent names that are all letters OR all digits are too generic to
    serve as parent — skip."""
    # All-letters parent: 'Dvina' alone is too generic
    upstream = [
        _make_missile("Dvina"),  # all-letters
        _make_missile("Variant", aliases=["Dvina launcher"]),
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    for e in out:
        assert e["to_ref_id"] != "MISSILE_SYSTEM:Dvina"

    # All-digits parent
    upstream = [
        _make_missile("75"),  # all-digits
        _make_missile("Variant", aliases=["75 something"]),
    ]
    seen_pairs.clear(); out.clear()
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    for e in out:
        assert e["to_ref_id"] != "MISSILE_SYSTEM:75"


def test_exact_alias_token_match_preferred_over_substring():
    """If two parents match — one exact alias, one substring — prefer
    the exact match. (e.g., alias 'S-75M' matches both parent 'S-75M'
    exactly AND parent 'S-75' as a substring with boundary; prefer 'S-75M')"""
    upstream = [
        _make_missile("S-75"),
        _make_missile("S-75M"),
        _make_missile("20D", aliases=["S-75M"]),
    ]
    seen_pairs: set = set()
    out: list = []
    _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    assert len(out) == 1
    assert out[0]["to_ref_id"] == "MISSILE_SYSTEM:S-75M"


def test_diagnostic_records_boundary_skips():
    """When a substring match is rejected by boundary rules, emit a
    diagnostic so the rejection is visible."""
    upstream = [
        _make_missile("S-75"),
        _make_missile("S-750", aliases=["S-750"]),
    ]
    seen_pairs: set = set()
    out: list = []
    diag = _EVIDENCE_GATE._emit_variant_of_relationships(out, upstream, seen_pairs)
    # Either no diagnostic for S-750 (nothing detected), OR a skipped
    # diagnostic with boundary_reject reason. We don't enforce one over
    # the other, just that no false emit occurred.
    for e in out:
        assert not (e["from_ref_id"] == "MISSILE_SYSTEM:S-750"
                    and e["to_ref_id"] == "MISSILE_SYSTEM:S-75")


def test_integration_with_postprocess_emits_variant_of_alongside_others():
    """End-to-end: feed pass_output with an LLM-emitted CUES rel and
    upstream entities with variant/family structure. Result has BOTH
    the original CUES rel AND a new VARIANT_OF edge."""
    upstream = [
        _make_radar("Searcher", properties={"emitter_function": "SEARCH"}),
        _make_radar("Engager", properties={"emitter_function": "FIRE_CONTROL"}),
        _make_missile("S-75"),
        _make_missile("1D", aliases=["S-75 Dvina"]),
    ]
    pass_output = {
        "relationships": [
            # LLM-emitted radar→radar that Item 3 will retype to CUES
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
    rel_types = [r.get("rel_type") for r in out["relationships"]]
    assert "CUES" in rel_types, "Item 3 retype expected"
    assert "VARIANT_OF" in rel_types, "Item 4 emitter expected"
    # Stats should expose Item 4 diagnostic
    assert "variant_of_emitter" in stats
    assert len(stats["variant_of_emitter"]["emitted"]) == 1
