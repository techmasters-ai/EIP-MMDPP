"""A-2: lenient coercers must log on unrecoverable input, silent on None.

None is the "field absent" signal — logging it would drown useful
diagnostics. Unrecoverable non-None input (e.g. "not a number",
bool, dict) is a signal worth surfacing: it means the LLM returned
something the schema couldn't accept.
"""
import logging

from ontology_bundles.air_defense_v3.validators import (
    coerce_optional_confidence,
    coerce_optional_float,
    coerce_optional_int,
)

_LOGGER = "ontology_bundles.air_defense_v3.validators"


def test_coerce_optional_int_logs_on_unrecoverable(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = coerce_optional_int("not a number")
    assert result is None
    assert any("unrecoverable" in rec.message.lower() for rec in caplog.records)


def test_coerce_optional_int_no_log_on_none(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert coerce_optional_int(None) is None
    assert not caplog.records


def test_coerce_optional_int_no_log_on_empty_string(caplog):
    """Empty string is the 'absent field' variant — not worth warning."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert coerce_optional_int("") is None
        assert coerce_optional_int("   ") is None
    assert not caplog.records


def test_coerce_optional_float_logs_on_unrecoverable(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert coerce_optional_float("abc") is None
    assert any("unrecoverable" in rec.message.lower() for rec in caplog.records)


def test_coerce_optional_float_no_log_on_none(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert coerce_optional_float(None) is None
    assert not caplog.records


def test_coerce_optional_confidence_logs_on_unrecoverable(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert coerce_optional_confidence("definitely sure") is None
    assert any("unrecoverable" in rec.message.lower() for rec in caplog.records)


def test_coerce_optional_confidence_no_log_on_none(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert coerce_optional_confidence(None) is None
    assert not caplog.records


def test_coerce_optional_confidence_no_log_on_valid_bucket(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert coerce_optional_confidence("high") == 0.9
    assert not caplog.records
