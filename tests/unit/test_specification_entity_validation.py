"""Regression tests for SpecificationEntity coercion across the three
schema modules that define it (radar_domain, missile_domain, other_systems).

Context: the docling-graph logs were showing repeated validation salvage
events of the form:

    specifications -> N -> value: Input should be a valid string

because the LLM was emitting numeric values for SpecificationEntity.value.
A pre-validator using coerce_optional_text() now normalizes these to
strings in all three modules. These tests lock in that behavior so the
fix cannot land in only one domain pass.
"""
import pytest

from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import (
    SpecificationEntity as RadarSpec,
)
from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import (
    SpecificationEntity as MissileSpec,
)
from ontology_bundles.air_defense_v3.extraction_schemas.other_systems import (
    SpecificationEntity as OtherSpec,
)

SPEC_CLASSES = [RadarSpec, MissileSpec, OtherSpec]


@pytest.mark.parametrize("spec_cls", SPEC_CLASSES)
class TestSpecificationEntityCoercion:
    def test_value_accepts_string(self, spec_cls):
        s = spec_cls(parameter="frequency", value="2.4", unit="GHz")
        assert s.value == "2.4"

    def test_value_accepts_int_and_normalizes(self, spec_cls):
        s = spec_cls(parameter="range", value=150, unit="km")
        assert s.value == "150"

    def test_value_accepts_float_and_normalizes(self, spec_cls):
        s = spec_cls(parameter="frequency", value=2.4, unit="GHz")
        assert s.value == "2.4"

    def test_value_empty_string_coerces_to_none(self, spec_cls):
        """B-18: SPECIFICATION is now a value-object component (is_entity=False).
        All fields are Optional; coerce_optional_text normalises empty
        strings to None. Content-based identity (A0-1) dedupes the
        resulting (None, None)-ish records."""
        s = spec_cls(parameter="range", value="", unit="km")
        assert s.value is None

    def test_value_whitespace_coerces_to_none(self, spec_cls):
        s = spec_cls(parameter="range", value="   ", unit="km")
        assert s.value is None

    def test_value_none_accepted(self, spec_cls):
        s = spec_cls(parameter="range", value=None, unit="km")
        assert s.value is None

    def test_parameter_accepts_int_and_normalizes(self, spec_cls):
        # LLM has been seen to emit numeric "parameter" values when the
        # source document uses numeric table headers.
        s = spec_cls(parameter=5, value="high", unit=None)
        assert s.parameter == "5"

    def test_unit_accepts_int_and_normalizes(self, spec_cls):
        s = spec_cls(parameter="range", value="150", unit=100)
        assert s.unit == "100"

    def test_value_strips_whitespace(self, spec_cls):
        s = spec_cls(parameter="range", value="  150 km  ", unit=None)
        assert s.value == "150 km"

    def test_nested_dict_value_coerces_to_none(self, spec_cls):
        """Nested dicts coerce to None (via coerce_optional_text). B-18:
        now Optional; content-based identity dedupes — no ValidationError."""
        s = spec_cls(parameter="range", value={"amount": 150}, unit="km")
        assert s.value is None

    def test_list_value_coerces_to_none(self, spec_cls):
        s = spec_cls(parameter="range", value=[150, 200], unit="km")
        assert s.value is None


@pytest.mark.parametrize("spec_cls", SPEC_CLASSES)
def test_numeric_and_string_values_produce_identical_spec(spec_cls):
    """Identity stability: a spec with value=150 and value='150' must
    serialize identically, otherwise extraction_merge would treat them
    as distinct SPECIFICATION records and fragment the graph."""
    a = spec_cls(parameter="range", value=150, unit="km")
    b = spec_cls(parameter="range", value="150", unit="km")
    assert a.model_dump() == b.model_dump()
