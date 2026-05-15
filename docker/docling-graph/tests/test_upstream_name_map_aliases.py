"""Tests for _build_upstream_name_map alias handling.

The name map is the bridge between table-cell names in synthesized chunks
and the upstream ref_id catalog. Without aliases support, the relationship
pass could only match entities by their primary system_name."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SERVICE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def evidence_gate(dg_app_module):
    """Load docling-graph's evidence_gate from the same package the dg
    main module imports it from. The dg_app_module fixture does the
    sys.path swap; we just locate the loaded module afterwards.

    Falls back to direct importlib loading when running this test file in
    isolation (no combined-test sys.path collision)."""
    import sys as _sys
    if "app.evidence_gate" in _sys.modules:
        return _sys.modules["app.evidence_gate"]
    # Direct load — works only when this test file is run standalone from
    # docker/docling-graph/ as the cwd. We add the service root path so
    # `from app._numeric_evidence import ...` succeeds.
    service_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(service_root))
    try:
        from app import evidence_gate as eg  # type: ignore  # noqa: WPS433
    finally:
        if str(service_root) in sys.path:
            sys.path.remove(str(service_root))
    return eg


def _ref(ref_id: str, system_name: str | None = None, *, display_label=None, aliases=None):
    return SimpleNamespace(
        ref_id=ref_id,
        identity_values={"system_name": system_name} if system_name else {},
        display_label=display_label,
        aliases=aliases,
    )


def test_empty_input_returns_empty_dict(evidence_gate):
    assert evidence_gate._build_upstream_name_map(None) == {}
    assert evidence_gate._build_upstream_name_map([]) == {}


def test_primary_system_name_registers(evidence_gate):
    norm = evidence_gate.normalize_evidence_text
    m = evidence_gate._build_upstream_name_map(
        [_ref("E001", system_name="Fan Song")]
    )
    assert m[norm("Fan Song")] == "E001"


def test_aliases_register_under_their_own_keys(evidence_gate):
    norm = evidence_gate.normalize_evidence_text
    ref = _ref(
        "E020", system_name="1D",
        display_label="1D",
        aliases=["SA-75", "SA-2A", "Guideline"],
    )
    m = evidence_gate._build_upstream_name_map([ref])
    assert m[norm("1D")] == "E020"
    assert m[norm("SA-75")] == "E020"
    assert m[norm("SA-2A")] == "E020"
    assert m[norm("Guideline")] == "E020"


def test_first_registration_wins_on_conflict(evidence_gate):
    norm = evidence_gate.normalize_evidence_text
    refs = [
        _ref("E001", system_name="Guideline"),
        _ref("E020", system_name="1D", aliases=["Guideline"]),
    ]
    m = evidence_gate._build_upstream_name_map(refs)
    assert m[norm("Guideline")] == "E001"
    assert m[norm("1D")] == "E020"


def test_no_aliases_does_not_crash(evidence_gate):
    norm = evidence_gate.normalize_evidence_text
    m = evidence_gate._build_upstream_name_map([
        _ref("E001", system_name="Fan Song", display_label="Fan Song", aliases=None),
        _ref("E002", system_name="SNR-75", display_label="SNR-75", aliases=[]),
    ])
    assert m[norm("Fan Song")] == "E001"
    assert m[norm("SNR-75")] == "E002"


def test_aliases_with_whitespace_and_blank_ignored(evidence_gate):
    norm = evidence_gate.normalize_evidence_text
    ref = _ref(
        "E001", system_name="Fan Song",
        aliases=["", "  ", "RSN-75"],
    )
    m = evidence_gate._build_upstream_name_map([ref])
    keys_for_e001 = {k for k, v in m.items() if v == "E001"}
    assert norm("RSN-75") in keys_for_e001
    assert norm("Fan Song") in keys_for_e001


def test_alias_not_overwriting_primary_within_same_ref(evidence_gate):
    norm = evidence_gate.normalize_evidence_text
    ref = _ref(
        "E020", system_name="1D",
        display_label="1D",
        aliases=["1D", "SA-75"],
    )
    m = evidence_gate._build_upstream_name_map([ref])
    assert m[norm("1D")] == "E020"
    assert m[norm("SA-75")] == "E020"
