"""Tests for ontology_bundles.air_defense_v3.validators."""
from enum import Enum

import pytest

from ontology_bundles.air_defense_v3.validators import (
    coerce_optional_int,
    coerce_optional_float,
    coerce_optional_text,
    coerce_optional_confidence,
    normalize_enum,
    _normalize_enum,
)


class _SampleEnum(str, Enum):
    RADAR = "RADAR"
    SONAR = "SONAR"
    FAN_SONG = "FAN_SONG"


class TestCoerceOptionalInt:
    def test_none_returns_none(self):
        assert coerce_optional_int(None) is None

    def test_int_returns_int(self):
        assert coerce_optional_int(5) == 5

    def test_numeric_string(self):
        assert coerce_optional_int("5") == 5

    def test_negative_string(self):
        assert coerce_optional_int("-12") == -12

    def test_empty_string_returns_none(self):
        assert coerce_optional_int("") is None

    def test_whitespace_string_returns_none(self):
        assert coerce_optional_int("   ") is None

    def test_embedded_number(self):
        assert coerce_optional_int("page 5 of 10") == 5

    def test_unparseable_returns_none(self):
        assert coerce_optional_int("unknown") is None

    def test_bool_returns_none(self):
        # bool is a subclass of int; passing True/False almost always means
        # the LLM conflated a boolean field with a numeric one. Return None
        # to drop rather than coerce to 1/0 and pollute the graph.
        assert coerce_optional_int(True) is None
        assert coerce_optional_int(False) is None


class TestCoerceOptionalFloat:
    def test_none_returns_none(self):
        assert coerce_optional_float(None) is None

    def test_float_returns_float(self):
        assert coerce_optional_float(3.14) == 3.14

    def test_int_coerces_to_float(self):
        assert coerce_optional_float(5) == 5.0

    def test_decimal_string(self):
        assert coerce_optional_float("3.14") == 3.14

    def test_unparseable_returns_none(self):
        assert coerce_optional_float("abc") is None

    def test_bool_returns_none(self):
        # Same reasoning as int: True/False in a numeric field is an LLM
        # conflation bug; drop rather than coerce to 1.0/0.0.
        assert coerce_optional_float(True) is None
        assert coerce_optional_float(False) is None


class TestCoerceOptionalConfidence:
    def test_none_returns_none(self):
        assert coerce_optional_confidence(None) is None

    def test_valid_float(self):
        assert coerce_optional_confidence(0.75) == 0.75

    def test_percentage_over_one(self):
        assert coerce_optional_confidence(85) == 0.85

    def test_text_high(self):
        assert coerce_optional_confidence("high") == 0.9

    def test_text_medium(self):
        assert coerce_optional_confidence("medium") == 0.6

    def test_text_low(self):
        assert coerce_optional_confidence("low") == 0.3

    def test_unparseable_returns_none(self):
        assert coerce_optional_confidence("vague") is None

    def test_explicit_zero_preserved(self):
        # Regression: the 'or 0.8' bug would have defaulted this.
        assert coerce_optional_confidence(0.0) == 0.0

    def test_bool_returns_none(self):
        # A boolean value for confidence is meaningless — return None rather
        # than coerce True to 1.0 (which would spuriously mark an extraction
        # as maximally confident).
        assert coerce_optional_confidence(True) is None
        assert coerce_optional_confidence(False) is None


class TestCoerceOptionalText:
    def test_none_returns_none(self):
        assert coerce_optional_text(None) is None

    def test_empty_string_returns_none(self):
        assert coerce_optional_text("") is None

    def test_whitespace_string_returns_none(self):
        assert coerce_optional_text("   ") is None

    def test_string_passes_through_stripped(self):
        assert coerce_optional_text("  hello  ") == "hello"

    def test_int_becomes_string(self):
        # Primary fix: the LLM emits integers for SpecificationEntity.value
        # and Pydantic rejects them with "Input should be a valid string".
        assert coerce_optional_text(150) == "150"

    def test_negative_int_becomes_string(self):
        assert coerce_optional_text(-42) == "-42"

    def test_float_becomes_string(self):
        assert coerce_optional_text(150.5) == "150.5"

    def test_whole_float_becomes_string(self):
        # 150.0 should render as '150.0' — don't silently lose the decimal.
        assert coerce_optional_text(150.0) == "150.0"

    def test_bool_returns_none(self):
        # bools are not intentionally preserved — SPECIFICATION.value of True
        # is almost always an LLM mistake, so drop it rather than emit 'True'.
        assert coerce_optional_text(True) is None
        assert coerce_optional_text(False) is None

    def test_dict_returns_none(self):
        # Never silently stringify nested dicts — that would produce
        # unstable identity strings like "{'a': 1}".
        assert coerce_optional_text({"a": 1}) is None

    def test_list_returns_none(self):
        # Same reasoning as dicts: no stable coercion rule.
        assert coerce_optional_text([1, 2, 3]) is None


class TestNormalizeEnum:
    def test_exact_match(self):
        validator = normalize_enum({"RADAR", "SONAR"})
        assert validator("RADAR") == "RADAR"

    def test_case_insensitive(self):
        validator = normalize_enum({"RADAR"})
        assert validator("radar") == "RADAR"

    def test_space_to_underscore(self):
        validator = normalize_enum({"FOO_BAR"})
        assert validator("foo bar") == "FOO_BAR"

    def test_unknown_returns_none(self):
        validator = normalize_enum({"RADAR"})
        assert validator("UNKNOWN") is None

    def test_none_returns_none(self):
        validator = normalize_enum({"RADAR"})
        assert validator(None) is None


class TestNormalizeEnumClassForm:
    """_normalize_enum(enum_cls, v) — the docs-signature form."""

    def test_exact_match(self):
        assert _normalize_enum(_SampleEnum, "RADAR") == "RADAR"

    def test_case_insensitive(self):
        assert _normalize_enum(_SampleEnum, "radar") == "RADAR"

    def test_space_to_underscore(self):
        # 'fan song' -> 'FAN_SONG' (enum value).
        assert _normalize_enum(_SampleEnum, "fan song") == "FAN_SONG"

    def test_unknown_returns_none(self):
        assert _normalize_enum(_SampleEnum, "UNKNOWN") is None

    def test_none_returns_none(self):
        assert _normalize_enum(_SampleEnum, None) is None

    def test_non_string_returns_none(self):
        assert _normalize_enum(_SampleEnum, 123) is None

    def test_empty_string_returns_none(self):
        assert _normalize_enum(_SampleEnum, "") is None

    def test_whitespace_string_returns_none(self):
        assert _normalize_enum(_SampleEnum, "   ") is None

    def test_enum_instance_passes_through_as_value(self):
        # Passing an actual enum member returns its string value.
        assert _normalize_enum(_SampleEnum, _SampleEnum.RADAR) == "RADAR"
