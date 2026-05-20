"""Tests for `_infer_radar_emitter_function` — deterministic radar role inference.

Three goals:
  1. HEIGHT_FINDER must be inferred when "HEIGHTFINDING" / "HEIGHT FINDER"
     text is near the entity (currently never returns this enum).
  2. The nearest role marker wins, not the first-listed marker. A chunk
     with both "ACQUISITION RADAR" (close to entity) and "ENGAGEMENT
     RADAR" (farther) must return SEARCH, not FIRE_CONTROL.
  3. The broad "MISSILE GUIDANCE" / "GUIDED UP TO" context override
     no longer overrides explicit role labels. It only fires when no
     explicit role marker is in the window.
"""
import importlib.util
import sys
from pathlib import Path

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

_infer = _EVIDENCE_GATE._infer_radar_emitter_function


# ===== HEIGHT_FINDER inference =====

def test_heightfinding_radar_returns_height_finder():
    """`<entity> HEIGHTFINDING RADAR` → HEIGHT_FINDER."""
    evidence = "PRV-10 KONUS HEIGHTFINDING RADAR"
    assert _infer("PRV-10 Konus", evidence) == "HEIGHT_FINDER"
    assert _infer("Konus", evidence) == "HEIGHT_FINDER"


def test_heightfinding_radars_plural_returns_height_finder():
    """`<entity> HEIGHTFINDING RADARS` (plural) → HEIGHT_FINDER."""
    evidence = "PRV-10 KONUS / PRV-11 VERSHINA / SIDE NET HEIGHTFINDING RADARS"
    for name in ("Konus", "Vershina", "Side Net"):
        assert _infer(name, evidence) == "HEIGHT_FINDER", f"{name} should be HEIGHT_FINDER"


def test_height_finder_hyphenated_returns_height_finder():
    """`<entity> HEIGHT-FINDING RADAR` (hyphenated) → HEIGHT_FINDER."""
    evidence = "P-15 HEIGHT-FINDING RADAR"
    assert _infer("P-15", evidence) == "HEIGHT_FINDER"


def test_height_finder_short_form_returns_height_finder():
    """`<entity> HEIGHT FINDER` (no -ing) → HEIGHT_FINDER."""
    evidence = "EARLY MODEL PRV-10 HEIGHT FINDER"
    assert _infer("PRV-10", evidence) == "HEIGHT_FINDER"


# ===== Nearest-marker wins (the Spoon Rest bug) =====

def test_explicit_acquisition_radar_overrides_distant_engagement_text():
    """The Spoon Rest scenario: an Acquisition Radar entity in a chunk that
    also discusses Fan Song's engagement role. Acquisition (which is
    closer to the entity name) must win."""
    # Construct evidence where:
    #   - "SPOON REST D/E ACQUISITION RADAR" is right next to the entity
    #   - "ENGAGEMENT RADAR" prose about Fan Song is 100+ chars away
    evidence = (
        "P-18-2/P-18M SPOON REST D/E ACQUISITION RADAR. "
        "Some intervening text describing the system. "
        "Some more intervening text to push the engagement-radar mention farther away. "
        "The FAN SONG is the ENGAGEMENT RADAR for the S-75 / SA-2 family."
    )
    # Spoon Rest's nearest role marker is ACQUISITION RADAR → must return SEARCH
    assert _infer("Spoon Rest", evidence) == "SEARCH"


def test_engagement_radar_close_returns_fire_control():
    """When ENGAGEMENT RADAR is the nearest marker, return FIRE_CONTROL."""
    evidence = "RSNA-75/SNR-75 FAN SONG ENGAGEMENT RADAR is the primary tracking radar."
    assert _infer("Fan Song", evidence) == "FIRE_CONTROL"


def test_heightfinding_close_beats_engagement_far():
    """If HEIGHTFINDING is close to the entity and ENGAGEMENT RADAR is far,
    HEIGHTFINDING wins → HEIGHT_FINDER."""
    evidence = (
        "PRV-10 KONUS HEIGHTFINDING RADAR is used for altitude measurement. "
        "More than enough text to push the engagement reference far away. "
        "More text. More text. The FAN SONG is the ENGAGEMENT RADAR elsewhere."
    )
    assert _infer("Konus", evidence) == "HEIGHT_FINDER"


# ===== MISSILE GUIDANCE override is narrowed =====

def test_missile_guidance_does_not_override_explicit_acquisition():
    """Spoon Rest mentioned in a chunk that also contains 'MISSILE GUIDANCE'
    prose elsewhere must NOT be force-labeled FIRE_CONTROL when its own
    explicit role label says ACQUISITION."""
    evidence = (
        "NITEL P-18-2/P-18M SPOON REST D/E ACQUISITION RADAR. "
        "Fan Song handles MISSILE GUIDANCE for the SA-2 family."
    )
    # Spoon Rest's own role is ACQUISITION → SEARCH must win over the
    # generic MISSILE GUIDANCE proximity to "Spoon Rest"
    assert _infer("Spoon Rest", evidence) == "SEARCH"


def test_missile_guidance_only_fires_when_no_explicit_role_marker():
    """The MISSILE GUIDANCE override IS still valid as a fallback —
    when no explicit role label is in the window for the entity.
    (Production evidence_text is uppercase via `normalize_evidence_text`.)"""
    evidence = "THE FAN SONG HANDLES MISSILE GUIDANCE FOR THE SAM."
    # Fan Song has no nearby ACQUISITION/HEIGHTFINDING/etc; MISSILE GUIDANCE
    # fallback fires → FIRE_CONTROL
    assert _infer("Fan Song", evidence) == "FIRE_CONTROL"


# ===== Existing behavior preserved =====

def test_acquisition_radar_direct_concat():
    """Direct concatenation form `<entity> ACQUISITION RADAR` still works."""
    evidence = "FLAT FACE ACQUISITION RADAR"
    assert _infer("Flat Face", evidence) == "SEARCH"


def test_engagement_radar_direct_concat():
    evidence = "FAN SONG ENGAGEMENT RADAR"
    assert _infer("Fan Song", evidence) == "FIRE_CONTROL"


def test_guidance_radar_returns_fire_control():
    evidence = "SNR-75 GUIDANCE RADAR controls the missile."
    assert _infer("SNR-75", evidence) == "FIRE_CONTROL"


def test_no_evidence_returns_none():
    """No role-marker text near the entity → return None (let LLM value stand)."""
    evidence = "SOMETHING UNRELATED ABOUT WEATHER AND FOOTBALL"
    assert _infer("Fan Song", evidence) is None


def test_non_string_input_returns_none():
    """Defensive: non-string system_name returns None."""
    assert _infer(None, "some evidence") is None
    assert _infer(42, "some evidence") is None


# ===== Post-entity marker beats pre-entity marker =====
#
# Generic principle: a role label that appears AFTER an entity binds
# to that entity (or its slash-group), while a role label that appears
# BEFORE it belongs to whichever entity preceded the marker. Nearest-
# absolute-distance can wrongly bind a preceding marker to the next
# entity (the Konus production case below). Post markers must win over
# pre markers in the same window.


def test_konus_in_full_radar_paragraph_inherits_from_slash_group():
    """Production SA-2 case: the chunk text concatenates multiple radar
    entries with their role labels. ACQUISITION RADAR appears BEFORE
    Konus (binding to Spoon Rest), and HEIGHTFINDING RADARS appears
    AFTER Konus (binding to the Konus/Vershina/Side Net slash-group).
    Konus must take the post marker, not the pre one."""
    evidence = (
        "P-18-2/P-18M SPOON REST D/E ACQUISITION RADAR PRV-10 KONUS / "
        "PRV-11 VERSHINA / SIDE NET HEIGHTFINDING RADARS RD-75 AMAZONKA"
    )
    assert _infer("Spoon Rest", evidence) == "SEARCH"
    assert _infer("Konus", evidence) == "HEIGHT_FINDER"
    assert _infer("PRV-10 Konus", evidence) == "HEIGHT_FINDER"
    assert _infer("Vershina", evidence) == "HEIGHT_FINDER"
    assert _infer("Side Net", evidence) == "HEIGHT_FINDER"


def test_post_marker_beats_closer_pre_marker():
    """Generic: even when the pre-entity marker is closer in absolute
    distance, the post marker wins because pre markers belong to the
    prior entity in the sequence."""
    # ACQUISITION RADAR is 5 chars before Konus; HEIGHTFINDING is 30+
    # chars after. Pre is closer, but post must win.
    evidence = "ALPHA ACQUISITION RADAR KONUS / BRAVO / CHARLIE HEIGHTFINDING RADARS"
    assert _infer("Konus", evidence) == "HEIGHT_FINDER"


def test_pre_marker_used_as_fallback_when_no_post_marker():
    """When NO marker appears after the entity but a pre marker is in
    the window, the pre marker is used as a fallback (some doc layouts
    do put the role label first)."""
    evidence = "SEARCH RADAR: P-15 FLAT FACE"
    assert _infer("P-15 Flat Face", evidence) == "SEARCH"


def test_spoon_rest_not_misled_by_distant_heightfinder_in_window():
    """Regression guard for the production paragraph: SPOON REST has
    ACQUISITION RADAR as the IMMEDIATELY following post marker, and
    HEIGHTFINDING RADARS appears further along (still within 80-char
    window). The closer post marker must win → SEARCH."""
    evidence = (
        "P-18-2/P-18M SPOON REST D/E ACQUISITION RADAR PRV-10 KONUS / "
        "PRV-11 VERSHINA / SIDE NET HEIGHTFINDING RADARS"
    )
    assert _infer("Spoon Rest", evidence) == "SEARCH"
    assert _infer("Spoon Rest D/E", evidence) == "SEARCH"


def test_fan_song_still_fire_control_with_other_radars_listed_after():
    """Regression guard: Fan Song's role label is ENGAGEMENT RADAR (Phase
    1 direct-concat). Listing other radars with their role labels after
    Fan Song must not affect Fan Song's role."""
    evidence = (
        "FAN SONG ENGAGEMENT RADAR P-18-2/P-18M SPOON REST D/E "
        "ACQUISITION RADAR PRV-10 KONUS / SIDE NET HEIGHTFINDING RADARS"
    )
    assert _infer("Fan Song", evidence) == "FIRE_CONTROL"
