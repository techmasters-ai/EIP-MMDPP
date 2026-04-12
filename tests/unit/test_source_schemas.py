"""Tests for the bundle-threaded Source schemas and new ReingestRequest.

Task 3.7 + spec §7.4 bundle-threading subsection. Strictly additive:
existing callers that don't pass the new bundle fields keep working.
"""
import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.sources import (
    ReingestRequest,
    SourceCreate,
    SourceResponse,
)


class TestSourceCreateBundleFields:
    def test_without_bundle_fields_works(self):
        """Existing callers that only set name should keep working."""
        s = SourceCreate(name="test-source")
        assert s.name == "test-source"
        assert s.default_ontology_bundle_key is None
        assert s.default_use_case_key is None

    def test_with_default_ontology_bundle_key(self):
        s = SourceCreate(name="test", default_ontology_bundle_key="air_defense_v3")
        assert s.default_ontology_bundle_key == "air_defense_v3"

    def test_with_default_use_case_key(self):
        s = SourceCreate(
            name="test",
            default_use_case_key="air_defense_v3_use_case",
        )
        assert s.default_use_case_key == "air_defense_v3_use_case"


class TestSourceResponseBundleFields:
    def test_response_exposes_default_ontology_bundle_key(self):
        """The field appears on the JSON schema so UIs can display it."""
        schema = TypeAdapter(SourceResponse).json_schema()
        assert "default_ontology_bundle_key" in schema.get("properties", {})
        assert "default_use_case_key" in schema.get("properties", {})


class TestReingestRequest:
    def test_defaults_mode_to_full(self):
        """No body or {} body must default to mode='full' (previous behavior)."""
        r = ReingestRequest()
        assert r.mode == "full"
        assert r.ontology_bundle_key is None
        assert r.use_case_key is None

    def test_accepts_all_valid_modes(self):
        for m in ("full", "embeddings_only", "graph_only"):
            r = ReingestRequest(mode=m)
            assert r.mode == m

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValidationError):
            ReingestRequest(mode="experimental")

    def test_accepts_optional_bundle_override(self):
        r = ReingestRequest(ontology_bundle_key="air_defense_v3")
        assert r.ontology_bundle_key == "air_defense_v3"
        # Still defaults mode to 'full'
        assert r.mode == "full"

    def test_accepts_all_fields_together(self):
        r = ReingestRequest(
            mode="graph_only",
            ontology_bundle_key="air_defense_v3",
            use_case_key="demo_use_case",
        )
        assert r.mode == "graph_only"
        assert r.ontology_bundle_key == "air_defense_v3"
        assert r.use_case_key == "demo_use_case"
