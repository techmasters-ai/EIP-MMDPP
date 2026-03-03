"""Unit tests for app.services.graph helper functions.

Pure unit tests — no database required. Only _sanitize_label and _parse_agtype
are exercised here; the Cypher-level functions are covered by integration tests.
"""

import pytest

pytestmark = pytest.mark.unit


class TestSanitizeLabel:
    def test_simple_word(self):
        from app.services.graph import _sanitize_label

        assert _sanitize_label("COMPONENT") == "COMPONENT"

    def test_spaces_replaced_with_underscore(self):
        from app.services.graph import _sanitize_label

        assert _sanitize_label("EQUIPMENT SYSTEM") == "EQUIPMENT_SYSTEM"

    def test_hyphens_replaced(self):
        from app.services.graph import _sanitize_label

        assert _sanitize_label("MIL-STD") == "MIL_STD"

    def test_leading_digit_prefixed(self):
        from app.services.graph import _sanitize_label

        result = _sanitize_label("1553B")
        assert result.startswith("_"), f"Expected underscore prefix, got: {result}"

    def test_empty_string_returns_unknown(self):
        from app.services.graph import _sanitize_label

        assert _sanitize_label("") == "UNKNOWN"

    def test_only_special_chars_returns_unknown(self):
        from app.services.graph import _sanitize_label

        # All non-alphanumeric chars become underscores, all-underscore is fine
        result = _sanitize_label("!@#")
        assert result  # should not be empty
        assert all(c in "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" for c in result)

    def test_already_valid_label(self):
        from app.services.graph import _sanitize_label

        assert _sanitize_label("EQUIPMENT_SYSTEM") == "EQUIPMENT_SYSTEM"

    def test_mixed_case_preserved(self):
        from app.services.graph import _sanitize_label

        assert _sanitize_label("EntityType") == "EntityType"


class TestParseAgtype:
    def test_none_returns_none(self):
        from app.services.graph import _parse_agtype

        assert _parse_agtype(None) is None

    def test_null_string_returns_none(self):
        from app.services.graph import _parse_agtype

        assert _parse_agtype("null") is None

    def test_json_object_parsed(self):
        from app.services.graph import _parse_agtype

        result = _parse_agtype('{"name": "Patriot", "type": "EQUIPMENT_SYSTEM"}')
        assert isinstance(result, dict)
        assert result["name"] == "Patriot"

    def test_json_number_parsed(self):
        from app.services.graph import _parse_agtype

        assert _parse_agtype("42") == 42
        assert _parse_agtype("3.14") == pytest.approx(3.14)

    def test_json_boolean_parsed(self):
        from app.services.graph import _parse_agtype

        assert _parse_agtype("true") is True
        assert _parse_agtype("false") is False

    def test_json_array_parsed(self):
        from app.services.graph import _parse_agtype

        result = _parse_agtype('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_quoted_string_stripped(self):
        from app.services.graph import _parse_agtype

        assert _parse_agtype('"CONTAINS"') == "CONTAINS"

    def test_plain_string_returned_as_is(self):
        from app.services.graph import _parse_agtype

        assert _parse_agtype("CONTAINS") == "CONTAINS"

    def test_whitespace_stripped(self):
        from app.services.graph import _parse_agtype

        assert _parse_agtype("  null  ") is None
        assert _parse_agtype('  "hello"  ') == "hello"
