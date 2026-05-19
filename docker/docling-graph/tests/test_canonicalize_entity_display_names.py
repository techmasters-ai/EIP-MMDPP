"""Item 5: post-extraction display-name canonicalization.

Conservative OCR-artifact cleanup applied to entity identity fields
(system_name, nomenclature) AFTER the LLM emits them. Generic — no
equipment-specific names anywhere.

Constraints (per Item 5 spec):
  * collapse hyphen-space: `RSN- 75M` → `RSN-75M`
  * collapse slash spacing: `RSNA-75 / SNR-75` → `RSNA-75/SNR-75`
  * normalize repeated whitespace
  * NO semantic canonicalization: `Fan Song` stays `Fan Song`; `S-75` ≠ `SA-75`
  * type-scoped: missile postprocess never touches radar fields
  * preserve original via diagnostics

Test scope:
  - The pure helper `_canonicalize_display_name(value: str) -> str`
  - Integration with `_postprocess_air_defense_radars` and
    `_postprocess_air_defense_missiles` — verify the canonical name is
    rewritten and the original surface form is captured in diagnostics.
"""
import importlib.util
import sys
from pathlib import Path

_SERVICE_APP_ROOT = Path(__file__).resolve().parent.parent / "app"

# Pre-register `app._numeric_evidence` so evidence_gate.py can resolve its import.
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


# ===== Unit tests on the pure helper =====

def test_canonicalize_collapses_hyphen_space():
    """`RSN- 75M` → `RSN-75M` (hyphen followed by whitespace is OCR artifact)."""
    fn = _EVIDENCE_GATE._canonicalize_display_name
    assert fn("RSN- 75M") == "RSN-75M"
    assert fn("RSNA- 75M") == "RSNA-75M"
    assert fn("Some- Token") == "Some-Token"


def test_canonicalize_collapses_slash_spacing():
    """`RSNA-75 / SNR-75` → `RSNA-75/SNR-75` (slash with spaces is OCR artifact)."""
    fn = _EVIDENCE_GATE._canonicalize_display_name
    assert fn("RSNA-75 / SNR-75") == "RSNA-75/SNR-75"
    assert fn("A /B") == "A/B"
    assert fn("A/ B") == "A/B"


def test_canonicalize_normalizes_repeated_whitespace():
    """Multiple spaces collapse to one; leading/trailing trimmed."""
    fn = _EVIDENCE_GATE._canonicalize_display_name
    assert fn("Fan  Song") == "Fan Song"
    assert fn("  Fan Song  ") == "Fan Song"
    assert fn("A\t\tB") == "A B"


def test_canonicalize_no_semantic_equivalence():
    """No domain-aware merging — these stay distinct."""
    fn = _EVIDENCE_GATE._canonicalize_display_name
    # Different model numbers — not equivalent
    assert fn("S-75") != fn("SA-75")
    # Different designations — not equivalent
    assert fn("Fan Song") != fn("SNR-75")
    # Different designations even with similar prefix
    assert fn("RSN-75") != fn("SNR-75")


def test_canonicalize_idempotent():
    """Already-clean values pass through unchanged."""
    fn = _EVIDENCE_GATE._canonicalize_display_name
    assert fn("RSN-75M") == "RSN-75M"
    assert fn("Fan Song") == "Fan Song"
    assert fn("S-75") == "S-75"
    # Applying twice = once
    assert fn(fn("RSN- 75M")) == fn("RSN- 75M")


def test_canonicalize_handles_none_and_non_string():
    """Defensive: non-string inputs return unchanged (function is on identity fields)."""
    fn = _EVIDENCE_GATE._canonicalize_display_name
    assert fn(None) is None
    assert fn("") == ""
    assert fn(42) == 42


# ===== Integration tests: radar postprocess =====

def test_radar_postprocess_rewrites_canonical_and_emits_diagnostic():
    """A radar row with OCR-spaced system_name gets canonicalized; original
    captured in diagnostics."""
    pass_output = {
        "radar_systems": [
            {"system_name": "RSN- 75M", "emitter_function": "missile guidance radar"},
        ]
    }
    out, stats = _EVIDENCE_GATE._postprocess_air_defense_radars(pass_output, evidence_text="")
    assert out["radar_systems"][0]["system_name"] == "RSN-75M"
    # Diagnostic preserves original
    assert "display_name_canonicalized" in stats
    rewrites = stats["display_name_canonicalized"]
    assert isinstance(rewrites, list) and len(rewrites) == 1
    assert rewrites[0]["original"] == "RSN- 75M"
    assert rewrites[0]["canonical"] == "RSN-75M"
    assert rewrites[0]["field"] == "system_name"


def test_radar_postprocess_no_diagnostic_when_already_clean():
    """No-op when nothing to canonicalize."""
    pass_output = {
        "radar_systems": [
            {"system_name": "Fan Song", "emitter_function": "missile guidance radar"},
        ]
    }
    out, stats = _EVIDENCE_GATE._postprocess_air_defense_radars(pass_output, evidence_text="")
    assert out["radar_systems"][0]["system_name"] == "Fan Song"
    assert "display_name_canonicalized" not in stats


# ===== Integration tests: missile postprocess (type-scoped) =====

def test_missile_postprocess_rewrites_and_radar_untouched():
    """Missile postprocess canonicalizes missile system_name; never touches radar payloads."""
    pass_output = {
        "missile_systems": [
            {"system_name": "5Ya- 23"},
        ]
    }
    out, stats = _EVIDENCE_GATE._postprocess_air_defense_missiles(pass_output, evidence_text="")
    assert out["missile_systems"][0]["system_name"] == "5Ya-23"
    rewrites = stats.get("display_name_canonicalized", [])
    assert len(rewrites) == 1
    assert rewrites[0]["original"] == "5Ya- 23"
    assert rewrites[0]["canonical"] == "5Ya-23"


# ===== Cross-type isolation =====

def test_radar_and_missile_postprocess_independent():
    """A missile-only pass_output never gets radar canonicalization applied (and vice versa).
    Verifies type-scoping per Item 5 guardrail #2 (operates per pass)."""
    # Radar postprocess on a doc with NO radar_systems — should noop
    pass_output_missile_only = {
        "missile_systems": [{"system_name": "5Ya- 23"}],
    }
    out, stats = _EVIDENCE_GATE._postprocess_air_defense_radars(
        pass_output_missile_only, evidence_text=""
    )
    # Radar postprocess shouldn't synthesize a radar from missile data, nor canonicalize it
    assert "missile_systems" not in stats.get("display_name_canonicalized", {})
    assert out["missile_systems"][0]["system_name"] == "5Ya- 23"  # untouched


# ===== Multiple identity fields per entity =====

def test_canonicalize_applies_to_nomenclature_too():
    """Nomenclature is also an identity-style field on radar/missile schemas.
    OCR-spaced nomenclature should be canonicalized (and diagnostic recorded)
    alongside system_name.

    Evidence-text must contain the original (raw) nomenclature surface form,
    because `_clear_unsupported_radar_properties` runs BEFORE canonicalization
    and nulls anything not quoted verbatim in the batch text. The test
    mirrors how the LLM would have extracted this value from a real chunk.
    """
    raw_nomenclature = "RSN- 75M Almaz"
    pass_output = {
        "radar_systems": [
            {"system_name": "RSN-75M", "nomenclature": raw_nomenclature},
        ]
    }
    # evidence_text must be pre-normalized (uppercased) — that's the shape
    # `collect_batch_evidence_text` produces in production. The quoting
    # check normalizes the candidate value but compares against the
    # already-normalized evidence_text directly.
    evidence_text = f"THE RADAR KNOWN AS {raw_nomenclature.upper()} WAS USED IN..."
    out, stats = _EVIDENCE_GATE._postprocess_air_defense_radars(
        pass_output, evidence_text=evidence_text
    )
    assert out["radar_systems"][0]["nomenclature"] == "RSN-75M Almaz"
    rewrites = stats.get("display_name_canonicalized", [])
    fields = {r["field"] for r in rewrites}
    assert "nomenclature" in fields
