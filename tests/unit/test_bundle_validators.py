"""Tests for ontology_bundles.air_defense_v3.validators."""
import pytest

from ontology_bundles.air_defense_v3.validators import (
    coerce_optional_int,
    coerce_optional_float,
    coerce_optional_confidence,
    normalize_enum,
)


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
