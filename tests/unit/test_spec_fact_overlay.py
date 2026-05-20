"""Step 4: spec fact overlay for prose key-value blocks.

The overlay scans docling document texts[] for labeled spec fragments
that the chunker would otherwise split into fragments too small for the
LLM to extract reliably. It normalizes them into structured facts that
can be (a) re-emitted as a SPEC_FACTS preamble for the chunk containing
the relevant entity, and (b) consumed by the evidence_gate postprocess
as a deterministic numeric source.

Generic patterns covered (no equipment names anywhere):
  * `LABEL: VALUE UNIT`                — single-line label-value pair
  * `LABEL: VALUE/VALUE UNIT/UNIT`     — paired max/min syntax
  * `LABEL:` on its own line followed by `VALUE UNIT` on the next       — bullet-style data box (Dvina)
  * `LABEL VALUE UNIT`                 — space-only delimiter

Unit normalization:
  * Imperial → metric where explicit (miles → km, feet → km, lbs → kg)
  * "Mach N" → preserved as label (no numeric speed conversion attempt)
  * "N degrees" / "N°" → degrees (no conversion needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.spec_overlay import (
    parse_spec_facts_from_evidence_text,
    parse_spec_facts_from_texts,
    SpecFact,
)


def _texts(*items: str) -> list[dict]:
    return [{"text": t, "label": None} for t in items]


# ===== Label : value (single-line) =====

class TestSingleLineLabelValue:
    def test_length_with_feet_unit(self):
        facts = parse_spec_facts_from_texts(_texts("Length: 35 feet"))
        f = next((x for x in facts if x.label_canonical == "length"), None)
        assert f is not None
        assert f.unit_raw == "feet"
        # 35 ft = 10.668 m
        assert abs(f.value_metric_m - 35 * 0.3048) < 1e-9
        # also rendered in km
        assert abs(f.value_metric_km - (35 * 0.3048) / 1000.0) < 1e-9

    def test_length_with_inches_unit(self):
        facts = parse_spec_facts_from_texts(_texts("Booster diameter: 26 inches"))
        f = next((x for x in facts if x.label_canonical == "diameter"), None)
        assert f is not None
        assert abs(f.value_metric_m - 26 * 0.0254) < 1e-6
        assert f.unit_raw == "inches"

    def test_weight_with_pounds_unit(self):
        facts = parse_spec_facts_from_texts(_texts("Weight: 5,040 pounds"))
        f = next((x for x in facts if x.label_canonical == "weight"), None)
        assert f is not None
        # 5040 lb → 2286 kg
        assert abs(f.value_metric_kg - 5040 * 0.453592) < 0.1

    def test_speed_mach_preserved_as_label(self):
        facts = parse_spec_facts_from_texts(_texts("Maximum speed: Mach 3"))
        f = next((x for x in facts if x.label_canonical == "max_speed"), None)
        assert f is not None
        assert f.value_raw == "Mach 3"


# ===== Paired max/min syntax =====

class TestPairedMaxMin:
    def test_range_paired_miles(self):
        """`Max/min effective range: 18 miles/5 miles` → two facts (max + min)."""
        facts = parse_spec_facts_from_texts(_texts("Max/min effective range: 18 miles/5 miles"))
        max_f = next((x for x in facts if x.label_canonical == "max_range"), None)
        min_f = next((x for x in facts if x.label_canonical == "min_range"), None)
        assert max_f is not None
        assert min_f is not None
        # 18 miles ≈ 28.97 km; 5 miles ≈ 8.05 km
        assert abs(max_f.value_metric_km - 28.97) < 0.5
        assert abs(min_f.value_metric_km - 8.05) < 0.5

    def test_altitude_paired_feet(self):
        """`Max/min effective altitude: 82,000 feet/1,500 feet` → two facts."""
        facts = parse_spec_facts_from_texts(_texts("Max/min effective altitude: 82,000 feet/1,500 feet"))
        max_f = next((x for x in facts if x.label_canonical == "max_altitude"), None)
        min_f = next((x for x in facts if x.label_canonical == "min_altitude"), None)
        assert max_f is not None
        assert min_f is not None
        # 82000 ft ≈ 24.99 km; 1500 ft ≈ 0.4572 km
        assert abs(max_f.value_metric_km - 24.99) < 0.1
        assert abs(min_f.value_metric_km - 0.4572) < 0.05


# ===== Bullet-style data box (label : on line 1, value on line 2) =====

class TestBulletDataBox:
    def test_label_value_on_separate_text_items(self):
        """Dvina structure: `Length:` on one text item, `35 feet` on the next.
        Overlay must reconstruct."""
        facts = parse_spec_facts_from_texts(_texts("Length:", "35 feet"))
        f = next((x for x in facts if x.label_canonical == "length"), None)
        assert f is not None
        assert abs(f.value_metric_m - 35 * 0.3048) < 1e-6

    def test_multi_line_paired_data_box(self):
        """Real Dvina-style: each field is `Label:` then `Value Unit`."""
        items = _texts(
            "Length:",                "35 feet",
            "Booster diameter:",      "26 inches",
            "Weight:",                "5,040 pounds",
            "Maximum speed:",         "Mach 3",
            "Max/min effective range:", "18 miles/5 miles",
            "Max/min effective altitude:", "82,000 feet/1,500 feet",
        )
        facts = parse_spec_facts_from_texts(items)
        labels = {f.label_canonical for f in facts}
        assert {"length", "diameter", "weight", "max_speed",
                "max_range", "min_range", "max_altitude", "min_altitude"} <= labels


# ===== Negative cases — no anchor, no fact =====

class TestNoFalsePositives:
    def test_prose_with_value_but_no_label_does_not_match(self):
        facts = parse_spec_facts_from_texts(_texts(
            "The missile travelled 18 miles before impact, reaching 82,000 feet at apex.",
        ))
        # No "Range:" / "Altitude:" / etc. anchor → no facts
        assert facts == []

    def test_rotated_360_degrees_does_not_emit_launch_angle(self):
        """Azimuth rotation phrases must NOT produce launch_angle facts."""
        facts = parse_spec_facts_from_texts(_texts(
            "The launcher rotated 360 degrees."
        ))
        assert not any(f.label_canonical == "launch_angle" for f in facts)


# ===== Launch angle anchor =====

class TestLaunchAngleFact:
    def test_launched_at_n_degrees(self):
        facts = parse_spec_facts_from_texts(_texts(
            "Usually launched the missile at 60 degrees."
        ))
        f = next((x for x in facts if x.label_canonical == "launch_angle"), None)
        assert f is not None
        assert f.value_raw == "60"
        assert f.unit_raw in ("degrees", "°")

    def test_max_launch_angle_label(self):
        facts = parse_spec_facts_from_texts(_texts("Max launch angle: 75°"))
        f = next((x for x in facts if x.label_canonical == "launch_angle"), None)
        assert f is not None
        assert f.value_raw == "75"


# ===== SpecFact integrity =====

class TestSpecFactIntegrity:
    def test_fact_has_source_text_idx(self):
        """Each fact must carry the text index it came from for provenance."""
        items = _texts("ignored", "Length: 35 feet", "also ignored")
        facts = parse_spec_facts_from_texts(items)
        f = next((x for x in facts if x.label_canonical == "length"), None)
        assert f is not None
        assert f.source_text_idx == 1

    def test_fact_preserves_raw_phrase(self):
        items = _texts("Max/min effective range: 18 miles/5 miles")
        facts = parse_spec_facts_from_texts(items)
        for f in facts:
            if f.label_canonical in ("max_range", "min_range"):
                assert "miles" in f.raw_phrase


# ===== Long-form "Maximum/minimum" label (Dvina) =====
#
# The Dvina source uses the spelled-out form `Maximum/minimum effective
# range:` / `Maximum/minimum effective altitude:` rather than the
# abbreviated `Max/min ...`. Both must canonicalize to the same paired
# emitters, with the FIRST value attached to the max field and the
# SECOND to the min field.

class TestLongFormPairedLabel:
    def test_long_form_range_bullet_pair(self):
        facts = parse_spec_facts_from_texts(_texts(
            "Maximum/minimum effective range:",
            "18 miles/5 miles",
        ))
        max_f = next((x for x in facts if x.label_canonical == "max_range"), None)
        min_f = next((x for x in facts if x.label_canonical == "min_range"), None)
        assert max_f is not None
        assert min_f is not None
        assert abs(max_f.value_metric_km - 28.97) < 0.5
        assert abs(min_f.value_metric_km - 8.05) < 0.5

    def test_long_form_altitude_bullet_pair(self):
        facts = parse_spec_facts_from_texts(_texts(
            "Maximum/minimum effective altitude:",
            "82,000 feet/1,500 feet",
        ))
        max_f = next((x for x in facts if x.label_canonical == "max_altitude"), None)
        min_f = next((x for x in facts if x.label_canonical == "min_altitude"), None)
        assert max_f is not None
        assert min_f is not None
        assert abs(max_f.value_metric_km - 24.99) < 0.1
        assert abs(min_f.value_metric_km - 0.4572) < 0.05

    def test_long_form_range_single_line(self):
        facts = parse_spec_facts_from_texts(_texts(
            "Maximum/minimum effective range: 18 miles/5 miles",
        ))
        labels = {f.label_canonical for f in facts}
        assert "max_range" in labels
        assert "min_range" in labels


# ===== Production inline-evidence path =====
#
# In production, `normalize_evidence_text` joins multiple chunks into one
# uppercased blob. The inline scanner must still recognize both short and
# long label forms and route values to the correct max/min fields.

class TestInlineEvidencePairedLabels:
    def test_inline_max_min_long_form_range(self):
        ev = (
            "MAXIMUM/MINIMUM EFFECTIVE RANGE: 18 MILES/5 MILES "
            "MAXIMUM/MINIMUM EFFECTIVE ALTITUDE: 82,000 FEET/1,500 FEET"
        )
        facts = parse_spec_facts_from_evidence_text(ev)
        labels = [(f.label_canonical, f.value_metric_km) for f in facts]
        max_r = next((km for lbl, km in labels if lbl == "max_range"), None)
        min_r = next((km for lbl, km in labels if lbl == "min_range"), None)
        max_a = next((km for lbl, km in labels if lbl == "max_altitude"), None)
        min_a = next((km for lbl, km in labels if lbl == "min_altitude"), None)
        assert max_r is not None and abs(max_r - 28.97) < 0.5
        assert min_r is not None and abs(min_r - 8.05) < 0.5
        assert max_a is not None and abs(max_a - 24.99) < 0.1
        assert min_a is not None and abs(min_a - 0.4572) < 0.05

    def test_inline_short_form_range_still_works(self):
        """Regression guard — the existing abbreviated form must keep working."""
        ev = "MAX/MIN EFFECTIVE RANGE: 18 MILES/5 MILES"
        facts = parse_spec_facts_from_evidence_text(ev)
        labels = {f.label_canonical for f in facts}
        assert "max_range" in labels
        assert "min_range" in labels

    def test_inline_first_value_is_max(self):
        """The first value in the pair MUST become max_*, the second min_*."""
        ev = "MAXIMUM/MINIMUM EFFECTIVE RANGE: 18 MILES/5 MILES"
        facts = parse_spec_facts_from_evidence_text(ev)
        max_f = next((f for f in facts if f.label_canonical == "max_range"), None)
        min_f = next((f for f in facts if f.label_canonical == "min_range"), None)
        assert max_f is not None and max_f.value_raw == "18"
        assert min_f is not None and min_f.value_raw == "5"
