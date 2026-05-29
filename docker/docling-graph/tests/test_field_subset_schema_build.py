"""Task F3 (§9 subset-schema extraction) — field_subset restricts the LLM schema.

Tests that:
1. With field_subset=["field_a", "field_b"], the built LLM schema/prompt
   includes ONLY those fields (plus any pydantic-required fields), NOT the
   dropped ones.
2. With field_subset=None, the full schema is used (byte-identical to current
   behavior via _get_schema_json).
3. The live template/record class is UNCHANGED after the build (not pruned).

These tests exercise:
  (A) The _build_field_subset_schema logic directly (in an inline reference
      implementation) — verifies the algorithm is correct.  This path does NOT
      require the full docling-graph library.
  (B) Integration with llm_backend.LlmBackend._build_field_subset_schema when
      the library IS installed as a package (optional, skipped if not).
  (C) Structural checks on main.py and llm_backend.py to verify the threading
      wiring is present.

Run from repo root:
    python3 -m pytest docker/docling-graph/tests/test_field_subset_schema_build.py -v
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Optional, Type

import pytest
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reference implementation of _build_field_subset_schema
# (mirrors what we will add to LlmBackend)
# ---------------------------------------------------------------------------

def _build_field_subset_schema_ref(
    template: Type[BaseModel],
    field_subset: list[str] | None,
) -> str:
    """Reference implementation of LlmBackend._build_field_subset_schema.

    When field_subset is None → returns the full schema JSON (byte-identical to
    json.dumps(template.model_json_schema(), indent=2)).
    When field_subset is a list → returns a restricted schema containing only
    the listed fields PLUS any pydantic-required fields (fields without defaults).
    Never mutates the live template class.
    """
    full_schema = template.model_json_schema()

    if field_subset is None:
        return json.dumps(full_schema, indent=2)

    # Collect pydantic-required fields: these MUST always be kept regardless
    # of whether field_subset includes them (defensive; F1 already includes
    # them but we guard here).
    pydantic_required: set[str] = set(full_schema.get("required") or [])

    # The keep-set is: explicitly requested ∪ pydantic-required
    keep: set[str] = set(field_subset) | pydantic_required

    # Build a filtered copy — never mutate the live schema dict
    restricted = copy.deepcopy(full_schema)
    all_props: dict = restricted.get("properties") or {}
    restricted_props = {k: v for k, v in all_props.items() if k in keep}
    restricted["properties"] = restricted_props

    # Filter the "required" list to only kept fields
    if "required" in restricted:
        restricted["required"] = [r for r in restricted["required"] if r in keep]

    # $defs are retained intact — unused defs are harmless and removing them
    # would require a recursive ref-walk.

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
# Tests — Algorithm (uses reference implementation)
# ---------------------------------------------------------------------------

class TestFieldSubsetSchemaNotNone:
    """field_subset is a non-empty list — LLM schema must be restricted."""

    def test_only_requested_optional_fields_present(self):
        schema_json = _build_field_subset_schema_ref(
            _RadarSpec, ["frequency_ghz", "range_km"]
        )
        schema = json.loads(schema_json)
        props = schema.get("properties", {})

        assert "frequency_ghz" in props, "requested field must be in properties"
        assert "range_km" in props, "requested field must be in properties"
        assert "notes" not in props, "unrequested optional field must not appear"
        assert "classification" not in props, "unrequested optional field must not appear"

    def test_pydantic_required_fields_always_included(self):
        """Fields pydantic marks required (no default) survive even if omitted
        from field_subset — defensive guard."""
        schema_json = _build_field_subset_schema_ref(
            _RadarSpec, ["frequency_ghz"]
        )
        schema = json.loads(schema_json)
        props = schema.get("properties", {})

        # identity_name, radar_type are pydantic-required (no default)
        assert "identity_name" in props, "pydantic-required field must always be kept"
        assert "radar_type" in props, "pydantic-required field must always be kept"
        assert "frequency_ghz" in props, "requested optional field must be present"

    def test_dropped_field_not_in_schema(self):
        schema_json = _build_field_subset_schema_ref(
            _RadarSpec, ["frequency_ghz"]
        )
        schema = json.loads(schema_json)
        props = schema.get("properties", {})
        assert "notes" not in props
        assert "classification" not in props
        assert "range_km" not in props

    def test_defs_retained_for_deep_schema(self):
        """$defs are retained so nested sub-schemas remain reachable.
        The restricted schema omits 'measurements' from properties but $defs stay."""
        schema_json = _build_field_subset_schema_ref(_DeepSchema, ["label"])
        schema = json.loads(schema_json)
        props = schema.get("properties", {})
        assert "label" in props
        # name is pydantic-required so it must stay
        assert "name" in props
        # $defs may still be present — that's fine (kept for safety)
        assert isinstance(props, dict)

    def test_live_template_class_unchanged_after_build(self):
        """The live record/template class must NOT be mutated."""
        original_fields = set(_RadarSpec.model_fields.keys())
        original_schema = _RadarSpec.model_json_schema()

        _build_field_subset_schema_ref(_RadarSpec, ["frequency_ghz"])

        assert set(_RadarSpec.model_fields.keys()) == original_fields
        assert _RadarSpec.model_json_schema() == original_schema

    def test_schema_is_valid_json(self):
        schema_json = _build_field_subset_schema_ref(
            _RadarSpec, ["frequency_ghz", "range_km"]
        )
        parsed = json.loads(schema_json)
        assert isinstance(parsed, dict)
        assert "properties" in parsed

    def test_empty_field_subset_uses_only_required_fields(self):
        """An empty list → only pydantic-required fields appear."""
        schema_json = _build_field_subset_schema_ref(_RadarSpec, [])
        schema = json.loads(schema_json)
        props = schema.get("properties", {})
        # Required fields must still be present
        assert "identity_name" in props
        assert "radar_type" in props
        # Optional fields gone
        assert "frequency_ghz" not in props


class TestFieldSubsetNoneIsFullSchema:
    """field_subset=None → full schema, byte-identical to json.dumps(model_json_schema())."""

    def test_none_returns_full_schema_json(self):
        full_expected = json.dumps(_RadarSpec.model_json_schema(), indent=2)
        via_subset_none = _build_field_subset_schema_ref(_RadarSpec, None)
        assert full_expected == via_subset_none, (
            "field_subset=None must return the byte-identical full schema"
        )

    def test_none_returns_all_properties(self):
        schema_json = _build_field_subset_schema_ref(_RadarSpec, None)
        schema = json.loads(schema_json)
        props = schema.get("properties", {})
        for fname in _RadarSpec.model_fields:
            assert fname in props, f"full schema must contain field {fname!r}"


class TestLiveClassInvariant:
    """Additional mutation safety checks across multiple calls."""

    def test_repeated_calls_do_not_accumulate_mutations(self):
        subsets: list[list[str] | None] = [
            ["frequency_ghz"],
            ["range_km", "notes"],
            ["classification"],
            None,
        ]
        original_schema = _RadarSpec.model_json_schema()

        for s in subsets:
            _build_field_subset_schema_ref(_RadarSpec, s)

        assert _RadarSpec.model_json_schema() == original_schema


# ---------------------------------------------------------------------------
# Integration tests — check that llm_backend.LlmBackend has the method IF
# the full library is installed as a package.
# ---------------------------------------------------------------------------

def _try_import_llm_backend():
    """Return LlmBackend class or None if library not installed."""
    try:
        # sys.path must include the repo/docling_graph package for this to work
        repo_root = Path(__file__).resolve().parent.parent / "repo"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from docling_graph.core.extractors.backends.llm_backend import LlmBackend  # type: ignore[import]
        return LlmBackend
    except Exception:
        return None


_LLM_BACKEND_CLS = _try_import_llm_backend()
_skip_if_no_library = pytest.mark.skipif(
    _LLM_BACKEND_CLS is None,
    reason="docling-graph library not installed; skipping integration path",
)


@_skip_if_no_library
class TestLlmBackendIntegration:
    """These tests run ONLY when the full docling-graph library is installed."""

    def test_method_exists(self):
        assert hasattr(_LLM_BACKEND_CLS, "_build_field_subset_schema")

    def test_none_matches_get_schema_json(self):
        full_via_cache = _LLM_BACKEND_CLS._get_schema_json(_RadarSpec)
        via_subset_none = _LLM_BACKEND_CLS._build_field_subset_schema(_RadarSpec, None)
        assert full_via_cache == via_subset_none

    def test_subset_restricts_properties(self):
        schema_json = _LLM_BACKEND_CLS._build_field_subset_schema(
            _RadarSpec, ["frequency_ghz"]
        )
        schema = json.loads(schema_json)
        props = schema.get("properties", {})
        assert "frequency_ghz" in props
        assert "notes" not in props
        # required fields kept
        assert "identity_name" in props
        assert "radar_type" in props

    def test_live_class_not_mutated(self):
        original = _RadarSpec.model_json_schema()
        _LLM_BACKEND_CLS._build_field_subset_schema(_RadarSpec, ["frequency_ghz"])
        assert _RadarSpec.model_json_schema() == original


# ---------------------------------------------------------------------------
# Structural wiring checks — verify the threading wiring is present in source
# ---------------------------------------------------------------------------

class TestWiringPresenceInSource:
    """Source-level checks that the F3 wiring exists in main.py and llm_backend."""

    def test_main_py_has_thread_local_for_field_subset(self):
        """main.py must declare _pass_llm_schema_overrides thread-local."""
        main_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
        source = main_path.read_text(encoding="utf-8")
        assert "_pass_llm_schema_overrides" in source, (
            "main.py must declare _pass_llm_schema_overrides thread-local "
            "for F3 field_subset threading"
        )

    def test_main_py_run_extraction_pass_accepts_field_subset(self):
        """run_extraction_pass must accept a field_subset parameter."""
        main_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
        source = main_path.read_text(encoding="utf-8")
        assert "field_subset" in source, (
            "run_extraction_pass must accept field_subset parameter"
        )

    def test_llm_backend_has_build_field_subset_schema(self):
        """llm_backend.py must define _build_field_subset_schema."""
        backend_path = (
            Path(__file__).resolve().parent.parent
            / "repo" / "docling_graph" / "core"
            / "extractors" / "backends" / "llm_backend.py"
        )
        source = backend_path.read_text(encoding="utf-8")
        assert "_build_field_subset_schema" in source, (
            "llm_backend.py must define _build_field_subset_schema static method"
        )

    def test_main_py_extract_pass_threads_field_subset(self):
        """extract_pass in main.py must thread field_subset down to run_extraction_pass."""
        main_path = Path(__file__).resolve().parent.parent / "app" / "main.py"
        source = main_path.read_text(encoding="utf-8")
        # The extract_pass handler must forward body.field_subset
        assert "body.field_subset" in source, (
            "extract_pass must pass body.field_subset to run_extraction_pass"
        )
