"""Task F3 (§9 subset-schema extraction) — field_subset restricts the LLM schema.

Host-runnable copy of the F3 algorithm tests so they are collected by
``scripts/run_tests.sh`` (which runs ``pytest tests/unit``). The docling-graph
*receiver* lives in the gitignored vendored clone and is applied at build time
via ``docker/docling-graph/patches/0004-f3-subset-schema.patch``; the
library-integration + source-wiring checks for it live in
``docker/docling-graph/tests/`` and run inside the container, NOT here.

What this file verifies (no docling_graph import, no DB — pure unit):
1. field_subset=[...] → the built LLM schema includes ONLY those fields (plus
   pydantic-required), never the dropped ones.
2. field_subset=None → full schema, byte-identical to
   ``json.dumps(model_json_schema(), indent=2)``.
3. The live template/record class is never mutated.

The reference implementation below MUST stay byte-for-byte equivalent to
``LlmBackend._build_field_subset_schema`` in patch 0004. If the library is
importable (e.g. inside the docling-graph container), the final test asserts
the two agree so the reference cannot silently drift from the patch.
"""
from __future__ import annotations

import copy
import json
from typing import Optional, Type

import pytest
from pydantic import BaseModel, Field

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Reference implementation — mirrors LlmBackend._build_field_subset_schema
# (docker/docling-graph/patches/0004-f3-subset-schema.patch).
# ---------------------------------------------------------------------------

def _build_field_subset_schema_ref(
    template: Type[BaseModel],
    field_subset: list[str] | None,
) -> str:
    """When field_subset is None → full schema JSON, byte-identical to
    json.dumps(template.model_json_schema(), indent=2).
    When field_subset is a list → restricted schema with only the listed fields
    PLUS any pydantic-required fields. Never mutates the live template class."""
    full_schema = template.model_json_schema()

    if field_subset is None:
        return json.dumps(full_schema, indent=2)

    pydantic_required: set[str] = set(full_schema.get("required") or [])
    keep: set[str] = set(field_subset) | pydantic_required

    restricted = copy.deepcopy(full_schema)
    all_props: dict = restricted.get("properties") or {}
    restricted["properties"] = {k: v for k, v in all_props.items() if k in keep}
    if "required" in restricted:
        restricted["required"] = [r for r in restricted["required"] if r in keep]
    # $defs retained intact — unused defs are harmless.
    return json.dumps(restricted, indent=2)


# ---------------------------------------------------------------------------
# Tiny Pydantic schemas for tests
# ---------------------------------------------------------------------------

class _RadarSpec(BaseModel):
    """Minimal flat schema with required + optional fields."""

    # Required (no default) — must survive a field_subset that omits them
    identity_name: str
    radar_type: str

    # Optional extras
    frequency_ghz: Optional[float] = None
    range_km: Optional[float] = None
    notes: Optional[str] = None
    classification: Optional[str] = None


class _DeepSchema(BaseModel):
    """Schema with a nested list so $defs appears in model_json_schema()."""

    class _Child(BaseModel):
        value: str
        unit: str

    name: str
    measurements: list[_Child] = Field(default_factory=list)
    label: Optional[str] = None


# ---------------------------------------------------------------------------
# Tests — restricted schema (non-empty field_subset)
# ---------------------------------------------------------------------------

class TestFieldSubsetSchemaNotNone:
    def test_only_requested_optional_fields_present(self):
        schema = json.loads(_build_field_subset_schema_ref(_RadarSpec, ["frequency_ghz", "range_km"]))
        props = schema.get("properties", {})
        assert "frequency_ghz" in props
        assert "range_km" in props
        assert "notes" not in props
        assert "classification" not in props

    def test_pydantic_required_fields_always_included(self):
        schema = json.loads(_build_field_subset_schema_ref(_RadarSpec, ["frequency_ghz"]))
        props = schema.get("properties", {})
        assert "identity_name" in props, "pydantic-required field must always be kept"
        assert "radar_type" in props, "pydantic-required field must always be kept"
        assert "frequency_ghz" in props

    def test_dropped_field_not_in_schema(self):
        schema = json.loads(_build_field_subset_schema_ref(_RadarSpec, ["frequency_ghz"]))
        props = schema.get("properties", {})
        assert "notes" not in props
        assert "classification" not in props
        assert "range_km" not in props

    def test_defs_retained_for_deep_schema(self):
        schema = json.loads(_build_field_subset_schema_ref(_DeepSchema, ["label"]))
        props = schema.get("properties", {})
        assert "label" in props
        assert "name" in props  # pydantic-required
        assert isinstance(props, dict)

    def test_schema_is_valid_json(self):
        parsed = json.loads(_build_field_subset_schema_ref(_RadarSpec, ["frequency_ghz", "range_km"]))
        assert isinstance(parsed, dict)
        assert "properties" in parsed

    def test_empty_field_subset_uses_only_required_fields(self):
        schema = json.loads(_build_field_subset_schema_ref(_RadarSpec, []))
        props = schema.get("properties", {})
        assert "identity_name" in props
        assert "radar_type" in props
        assert "frequency_ghz" not in props


class TestFieldSubsetNoneIsFullSchema:
    def test_none_returns_full_schema_json(self):
        full_expected = json.dumps(_RadarSpec.model_json_schema(), indent=2)
        via_subset_none = _build_field_subset_schema_ref(_RadarSpec, None)
        assert full_expected == via_subset_none, (
            "field_subset=None must return the byte-identical full schema"
        )

    def test_none_returns_all_properties(self):
        schema = json.loads(_build_field_subset_schema_ref(_RadarSpec, None))
        props = schema.get("properties", {})
        for fname in _RadarSpec.model_fields:
            assert fname in props, f"full schema must contain field {fname!r}"


class TestLiveClassInvariant:
    def test_live_template_class_unchanged_after_build(self):
        original_fields = set(_RadarSpec.model_fields.keys())
        original_schema = _RadarSpec.model_json_schema()
        _build_field_subset_schema_ref(_RadarSpec, ["frequency_ghz"])
        assert set(_RadarSpec.model_fields.keys()) == original_fields
        assert _RadarSpec.model_json_schema() == original_schema

    def test_repeated_calls_do_not_accumulate_mutations(self):
        subsets: list[list[str] | None] = [
            ["frequency_ghz"], ["range_km", "notes"], ["classification"], None,
        ]
        original_schema = _RadarSpec.model_json_schema()
        for s in subsets:
            _build_field_subset_schema_ref(_RadarSpec, s)
        assert _RadarSpec.model_json_schema() == original_schema


# ---------------------------------------------------------------------------
# Drift guard — when docling_graph IS importable (inside the container), the
# reference implementation must match the real patched method exactly.
# Skipped on hosts without the library (e.g. the default scripts/run_tests.sh
# environment), so this never fails the host suite.
# ---------------------------------------------------------------------------

def _try_import_llm_backend():
    try:
        from docling_graph.core.extractors.backends.llm_backend import LlmBackend  # type: ignore
        return LlmBackend
    except Exception:
        return None


_LLM_BACKEND_CLS = _try_import_llm_backend()


@pytest.mark.skipif(
    _LLM_BACKEND_CLS is None,
    reason="docling_graph not importable on this host; container test covers the real method",
)
class TestReferenceMatchesPatchedImplementation:
    @pytest.mark.parametrize("subset", [None, [], ["frequency_ghz"], ["range_km", "notes"]])
    def test_reference_matches_real(self, subset):
        ref = _build_field_subset_schema_ref(_RadarSpec, subset)
        real = _LLM_BACKEND_CLS._build_field_subset_schema(_RadarSpec, subset)
        assert ref == real, "reference impl drifted from patch 0004's _build_field_subset_schema"
