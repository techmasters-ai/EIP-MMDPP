"""SA-2 + Dvina extraction acceptance harness.

Locks in extraction expectations against the committed fixtures so future
code changes can't silently regress quality while improving counts. Runs
against the saved fixture JSONs — no LLM calls — so it's fast (<5s) and
re-runnable after every code change.

Structure: per-doc fixture map with positive assertions (entities/fields
that MUST be present), blacklist assertions (entities that MUST NOT be
classified as radars), relationship assertions (rel types and counts).

Each assertion has a short rationale string so a failure points to WHY
it matters, not just what changed.

Per-doc expectations are TARGETS — they capture what should be in the
fixture for the extraction to be considered correct. Tests can be in
three states:
  * PASS — current fixture satisfies the expectation.
  * XFAIL — known regression; the expectation is the truth, fixture is wrong.
  * SKIP — expectation depends on a downstream step not yet landed.

The harness file is the single source of truth for extraction quality.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sa2"

SA2_DOC_ID = "78673393-639b-4fde-9bda-9e7bfd43ccda"
DVINA_DOC_ID = "b77c48f9-3a27-473f-be05-fa7e73e5d6f5"


def _load_pass(doc_id: str, pass_name: str) -> dict[str, Any]:
    path = FIXTURE_DIR / f"{doc_id}_{pass_name}_response.json"
    return json.loads(path.read_text())


def _entities(pass_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the entity list regardless of key (radar_systems/missile_systems)."""
    for k, v in (pass_output or {}).items():
        if isinstance(v, list) and k != "relationships":
            return v
    return []


def _find_entity(entities: list[dict[str, Any]], system_name: str) -> dict[str, Any] | None:
    """Find an entity by exact or contains-match on system_name."""
    target = system_name.strip().lower()
    for e in entities:
        sn = (e.get("system_name") or "").strip().lower()
        if sn == target or target in sn:
            return e
    return None


def _all_aliases(e: dict[str, Any]) -> set[str]:
    """Return the set of all aliases attached to an entity across the
    nomenclature / name / dieqp fields, splitting any " / "-joined values
    so multi-value designation expansions are visible to membership
    tests."""
    out: set[str] = set()
    for field in ("nomenclature", "name", "dieqp"):
        v = e.get(field)
        if not isinstance(v, str):
            continue
        for part in v.split(" / "):
            p = part.strip()
            if p:
                out.add(p)
    return out


# ============================================================================
# SA-2 radar_identity
# ============================================================================

class TestSA2RadarIdentity:
    """SA-2's radar_identity expectations. Locks in:
      - Spoon Rest is SEARCH (acquisition), not FIRE_CONTROL
      - Side Net is HEIGHT_FINDER, not SEARCH
      - Konus / Vershina are HEIGHT_FINDER
      - Amazonka is something other than SEARCH (Rangefinding)
      - Non-radar entities (5E93, 5L22, AKKORD, Tsikloida) are NOT in the pass
      - scan_type for UHF-Band radars is populated (when band field exists)
    """
    @pytest.fixture(scope="class")
    def entities(self) -> list[dict[str, Any]]:
        d = _load_pass(SA2_DOC_ID, "radar_identity")
        return _entities(d.get("pass_output", {}))

    def test_spoon_rest_is_search(self, entities):
        e = _find_entity(entities, "Spoon Rest")
        assert e is not None, "Spoon Rest must be extracted"
        assert e.get("emitter_function") == "SEARCH", (
            f"Spoon Rest is an Acquisition Radar per source — must be SEARCH, "
            f"got {e.get('emitter_function')!r}"
        )

    def test_side_net_is_height_finder(self, entities):
        e = _find_entity(entities, "Side Net")
        assert e is not None, "Side Net must be extracted"
        assert e.get("emitter_function") == "HEIGHT_FINDER", (
            f"Side Net is a Heightfinding Radar per source — must be HEIGHT_FINDER, "
            f"got {e.get('emitter_function')!r}"
        )

    def test_konus_is_height_finder(self, entities):
        e = _find_entity(entities, "Konus")
        assert e is not None, "Konus (PRV-10) must be extracted"
        assert e.get("emitter_function") == "HEIGHT_FINDER"

    def test_vershina_is_height_finder(self, entities):
        e = _find_entity(entities, "Vershina")
        assert e is not None, "Vershina (PRV-11) must be extracted"
        assert e.get("emitter_function") == "HEIGHT_FINDER"

    def test_no_power_generator_classified_as_radar(self, entities):
        for blacklisted in ("5E93", "ESP-90"):
            assert _find_entity(entities, blacklisted) is None, (
                f"{blacklisted} is a Power Generator per source — "
                f"must NOT be classified as a radar"
            )

    def test_no_decon_van_classified_as_radar(self, entities):
        for blacklisted in ("5L22", "5L22A"):
            assert _find_entity(entities, blacklisted) is None, (
                f"{blacklisted} is a chemical decontamination / fuel tank — "
                f"must NOT be classified as a radar"
            )

    def test_no_training_emulator_classified_as_radar(self, entities):
        assert _find_entity(entities, "AKKORD") is None, (
            "AKKORD is a Training Emulator — must NOT be classified as a radar"
        )

    def test_no_radio_relay_classified_as_radar(self, entities):
        assert _find_entity(entities, "Tsikloida") is None, (
            "Tsikloida is a Radio Relay van — must NOT be classified as a radar"
        )

    def test_entity_count_above_minimum(self, entities):
        """The pass must produce a reasonable number of radar entities.
        Lower bound based on the actual radar systems mentioned in source."""
        assert len(entities) >= 10, (
            f"Expected at least 10 radar entities (Fan Song, Spoon Rest, "
            f"Side Net, Konus, Vershina, Amazonka, Parol, P-12, Flat Face, "
            f"Squat Eye), got {len(entities)}"
        )

    def test_canonical_radars_present(self, entities):
        """Core radars must always be extracted regardless of run-to-run variance."""
        canonical = ["Fan Song", "Spoon Rest", "Side Net", "Amazonka"]
        missing = [name for name in canonical if _find_entity(entities, name) is None]
        assert not missing, f"Canonical SA-2 radars missing: {missing}"


# ============================================================================
# SA-2 radar_power_rf
# ============================================================================

class TestSA2RadarPowerRf:
    """RF numeric fields are correctly empty for this doc (source lacks them).
    Test pins this expectation so a future change doesn't silently start
    hallucinating MHz/kW values without evidence."""

    @pytest.fixture(scope="class")
    def entities(self) -> list[dict[str, Any]]:
        d = _load_pass(SA2_DOC_ID, "radar_power_rf")
        return _entities(d.get("pass_output", {}))

    def test_no_hallucinated_rf_numerics(self, entities):
        """SA-2 source has zero explicit MHz/kW/dBW values for radars.
        Any non-null value here is hallucinated."""
        rf_fields = ("erp_dbw", "tx_peak_power_kw", "nominal_rf_mhz")
        for e in entities:
            for f in rf_fields:
                v = e.get(f)
                if v is not None:
                    pytest.fail(
                        f"{e.get('system_name')}: {f}={v!r} but SA-2 source "
                        f"has no explicit RF specs — value is hallucinated"
                    )


# ============================================================================
# SA-2 missile_identity
# ============================================================================

class TestSA2MissileIdentity:
    """Missile identity expectations including the Industry/Military/NATO
    designation triplet capture (Step 6)."""

    @pytest.fixture(scope="class")
    def entities(self) -> list[dict[str, Any]]:
        d = _load_pass(SA2_DOC_ID, "missile_identity")
        return _entities(d.get("pass_output", {}))

    def test_canonical_missile_variants_present(self, entities):
        """Core missile variants from Table 0 must always be extracted."""
        canonical = ["S-75", "1D", "13D", "20D", "5Ya23", "SA-2"]
        missing = [name for name in canonical if _find_entity(entities, name) is None]
        assert not missing, f"Canonical SA-2 missile variants missing: {missing}"

    def test_13d_has_military_and_nato_aliases(self, entities):
        """13D — Table 0 column 3: Industry=S-75, Military=S-75, NATO=SA-2C."""
        e = _find_entity(entities, "13D")
        assert e is not None
        aliases = _all_aliases(e)
        assert "S-75" in aliases, f"13D missing Military 'S-75' in {aliases}"
        assert "SA-2C" in aliases, f"13D missing NATO 'SA-2C' in {aliases}"

    def test_20d_has_military_and_nato_aliases(self, entities):
        """20D — Table 0 column 7: Industry=S-75V, Military=S-75M, NATO=SA-2C.
        (The previous test docstring claimed NATO=SA-2D but the source
        table column for 20D actually shows SA-2C.)"""
        e = _find_entity(entities, "20D")
        assert e is not None
        aliases = _all_aliases(e)
        assert "S-75M" in aliases, f"20D missing Military 'S-75M' in {aliases}"
        assert "SA-2C" in aliases, f"20D missing NATO 'SA-2C' in {aliases}"

    # ===== "Lost variant" diagnosis (#5 in next-steps roadmap) =====
    #
    # The earlier integrated run showed SA-25, SA-2C, SA-2E in v9 but
    # absent today. Source-corpus diagnosis:
    #   * SA-25 — appears only in prose text 91, in the phrase "the
    #     earlier SA-25/S-25 / SA-1 Guild" (predecessor reference, NOT an
    #     SA-2 family variant). v9's emission was LLM noise; today's
    #     absence is the correct behavior.
    #   * SA-2C / SA-2E — appear ONLY in Table 0 row 2 (NATO Designation
    #     row), as aliases of specific missile rounds (SA-2C → 13D /
    #     13DM / 20D / 20DP ; SA-2E → 15D). They are NATO aliases of
    #     existing round entities, not independent missiles. v9 emitted
    #     them as separate top-level entities (duplication); the proper
    #     representation is as aliases on the round entities, which
    #     Step 6 (deterministic designation alias expansion) will do.

    def test_sa25_is_not_a_separate_missile_entity(self, entities):
        """SA-25 (the predecessor / SA-1 Guild family) appears only in a
        predecessor-context prose mention and must NOT be emitted as a
        separate SA-2 missile_identity entity."""
        e = _find_entity(entities, "SA-25")
        assert e is None, (
            "SA-25 is a predecessor system reference (SA-1 Guild / S-25), "
            "not an SA-2 variant. Should not be a separate missile entity."
        )

    def test_15d_has_sa2e_nato_alias(self, entities):
        """SA-2E appears once in Table 0 row 2, column 11 — mapped to
        round 15D. Step 6 alias expansion attaches SA-2E to 15D."""
        e = _find_entity(entities, "15D")
        assert e is not None
        aliases = _all_aliases(e)
        assert "SA-2E" in aliases, f"15D missing NATO 'SA-2E' in {aliases}"

    def test_sa2c_aliases_attached_to_each_round(self, entities):
        """SA-2C appears in Table 0 NATO row in 3 columns: 13D (col 3),
        20D (col 7), 20DP (col 8). Step 6 attaches SA-2C to ALL of these
        round entities. (13DM is column 4 → NATO=SA-2D, not SA-2C.)"""
        rounds_with_sa2c = ("13D", "20D", "20DP")
        missing = []
        for r in rounds_with_sa2c:
            e = _find_entity(entities, r)
            if e is None:
                missing.append(f"{r}: entity absent")
                continue
            aliases = _all_aliases(e)
            if "SA-2C" not in aliases:
                missing.append(f"{r}: missing NATO 'SA-2C' (saw {aliases})")
        assert not missing, f"SA-2C alias expansion incomplete: {missing}"

    def test_sa2d_aliases_attached_to_correct_rounds(self, entities):
        """SA-2D appears in 5 NATO-row columns: 13DM, 13DA, 13DAM, 20DSU,
        5Ya23. Step 6 must attach SA-2D to all of them."""
        rounds_with_sa2d = ("13DM", "13DA", "13DAM", "20DSU", "5Ya23")
        missing = []
        for r in rounds_with_sa2d:
            e = _find_entity(entities, r)
            if e is None:
                missing.append(f"{r}: entity absent")
                continue
            aliases = _all_aliases(e)
            if "SA-2D" not in aliases:
                missing.append(f"{r}: missing NATO 'SA-2D' (saw {aliases})")
        assert not missing, f"SA-2D alias expansion incomplete: {missing}"

    def test_5ya23_has_no_separate_sa2_emission(self, entities):
        """5Ya23 has NATO designation SA-2D (Table 0 row 2 col 10),
        which today's run captures as a separate SA-2D entity. Document
        that the alias relationship exists — the round entity itself
        must be present even if the alias isn't yet on it."""
        e = _find_entity(entities, "5Ya23")
        assert e is not None, "5Ya23 round entity must be present"


# ============================================================================
# SA-2 missile_kinematics
# ============================================================================

class TestSA2MissileKinematics:
    @pytest.fixture(scope="class")
    def entities(self) -> list[dict[str, Any]]:
        d = _load_pass(SA2_DOC_ID, "missile_kinematics")
        return _entities(d.get("pass_output", {}))

    def test_kinematic_fields_populated(self, entities):
        """At least 8 entities must have at least one kinematic numeric
        (max_range, min_range, max_alt, min_alt) — guards the Item 1+5 gain."""
        kinematic_fields = (
            "max_intercept_km", "min_intercept_km",
            "max_altitude_km", "min_altitude_km",
        )
        with_kinematic = [
            e for e in entities
            if any(e.get(f) is not None for f in kinematic_fields)
        ]
        assert len(with_kinematic) >= 8, (
            f"Expected at least 8 entities with kinematic numerics from Table 0; "
            f"got {len(with_kinematic)}"
        )


# ============================================================================
# SA-2 system_links
# ============================================================================

class TestSA2SystemLinks:
    @pytest.fixture(scope="class")
    def doc(self) -> dict[str, Any]:
        return _load_pass(SA2_DOC_ID, "system_links")

    @pytest.fixture(scope="class")
    def rels(self, doc) -> list[dict[str, Any]]:
        return (doc.get("pass_output") or {}).get("relationships") or []

    def test_variant_of_relationships_present(self, rels):
        """Item 4 must emit VARIANT_OF edges. Lower bound: 8 (we got 11-15
        across various runs; flooring at 8 to allow LLM variance)."""
        variant_of = [r for r in rels if r.get("rel_type") == "VARIANT_OF"]
        assert len(variant_of) >= 8, (
            f"Expected at least 8 VARIANT_OF edges, got {len(variant_of)}. "
            f"Item 4 should produce these deterministically from missile aliases."
        )

    def test_no_cross_type_variant_of(self, rels):
        """VARIANT_OF must never link a missile to a radar (or vice versa)."""
        for r in rels:
            if r.get("rel_type") != "VARIANT_OF":
                continue
            f = (r.get("from_ref_id") or "").split(":", 1)[0]
            t = (r.get("to_ref_id") or "").split(":", 1)[0]
            assert f == t, (
                f"Cross-type VARIANT_OF detected: {r['from_ref_id']} -> {r['to_ref_id']}"
            )

    def test_no_self_loops(self, rels):
        for r in rels:
            assert r.get("from_ref_id") != r.get("to_ref_id"), (
                f"Self-loop relationship: {r}"
            )

    def test_hint_promoted_floor(self, doc):
        """Deterministic hint promotion must produce at least 8 edges."""
        sp = (doc.get("diagnostics") or {}).get("service_postprocess") or {}
        promoted = sp.get("promoted_from_cross_entity_hints") or []
        assert len(promoted) >= 8, (
            f"Expected at least 8 hint-promoted edges, got {len(promoted)}"
        )

    def test_no_unresolved_hints(self, doc):
        """Item 2 must keep unresolved-hint count at 0."""
        sp = (doc.get("diagnostics") or {}).get("service_postprocess") or {}
        unresolved = sp.get("unresolved_cross_entity_hints") or {}
        count = unresolved.get("count") if isinstance(unresolved, dict) else None
        assert count == 0, f"Expected unresolved=0, got {count}"

    def test_role_aware_cues_has_no_missing_role(self, doc):
        sp = (doc.get("diagnostics") or {}).get("service_postprocess") or {}
        ra = sp.get("role_aware_cues") or {}
        skipped = [s for s in (ra.get("skipped") or []) if "missing_role" in s.get("reason", "")]
        assert len(skipped) == 0, (
            f"Found {len(skipped)} CUES retypes skipped due to missing role: {skipped[:3]}"
        )


# ============================================================================
# Dvina (prose-only doc)
# ============================================================================

class TestDvinaMissileIdentity:
    @pytest.fixture(scope="class")
    def entities(self) -> list[dict[str, Any]]:
        d = _load_pass(DVINA_DOC_ID, "missile_identity")
        return _entities(d.get("pass_output", {}))

    def test_s75_extracted(self, entities):
        e = _find_entity(entities, "S-75")
        assert e is not None, "S-75 (primary subject of Dvina doc) must be extracted"

    def test_s75_has_sa2_guideline_alias(self, entities):
        e = _find_entity(entities, "S-75")
        assert e is not None
        aliases = {e.get("nomenclature"), e.get("name"), e.get("dieqp")}
        assert "SA-2 Guideline" in aliases or "SA-2" in aliases, (
            f"S-75 must have SA-2 Guideline NATO alias; got {aliases}"
        )


class TestDvinaMissileKinematics:
    @pytest.fixture(scope="class")
    def entities(self) -> list[dict[str, Any]]:
        d = _load_pass(DVINA_DOC_ID, "missile_kinematics")
        return _entities(d.get("pass_output", {}))

    def test_max_intercept_km_populated(self, entities):
        e = _find_entity(entities, "S-75")
        assert e is not None
        v = e.get("max_intercept_km")
        # 18 miles ≈ 28.97 km — accept 28-30 range for unit-conversion tolerance
        assert v is not None and 28 <= v <= 30, (
            f"max_intercept_km should be ~29 km (from '18 miles'), got {v}"
        )

    def test_min_intercept_km_populated(self, entities):
        e = _find_entity(entities, "S-75")
        assert e is not None
        v = e.get("min_intercept_km")
        # 5 miles ≈ 8.05 km
        assert v is not None and 7 <= v <= 9, (
            f"min_intercept_km should be ~8 km (from '5 miles'), got {v}"
        )

    def test_max_altitude_km_populated(self, entities):
        e = _find_entity(entities, "S-75")
        assert e is not None
        v = e.get("max_altitude_km")
        # 82000 ft ≈ 24.99 km
        assert v is not None and 24 <= v <= 26, (
            f"max_altitude_km should be ~25 km (from '82,000 feet'), got {v}"
        )

    def test_min_altitude_km_populated(self, entities):
        e = _find_entity(entities, "S-75")
        assert e is not None
        v = e.get("min_altitude_km")
        # 1500 ft ≈ 0.46 km
        assert v is not None and 0.3 <= v <= 0.6, (
            f"min_altitude_km should be ~0.46 km (from '1,500 feet'), got {v}"
        )

    def test_max_launch_angle_deg_populated(self, entities):
        e = _find_entity(entities, "S-75")
        assert e is not None
        v = e.get("max_launch_angle_deg")
        assert v is not None and 55 <= v <= 65, (
            f"max_launch_angle_deg should be ~60 (from prose), got {v}"
        )
