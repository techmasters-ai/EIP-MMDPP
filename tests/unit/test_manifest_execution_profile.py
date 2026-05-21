"""C2 — ExecutionProfile schema validation tests.

Verifies the optional ``execution:`` block on PassManifest:
- accepted in various shapes (partial, full, null, omitted)
- rejected for bad types
- back-compat with manifests that have no execution block
- real air_defense_v3 manifests load after Iter 1 annotations

Run standalone:
    python3 -m pytest tests/unit/test_manifest_execution_profile.py -v
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError


_MISSING = object()


def _raw_pass(
    name: str = "radar_identity",
    input_mode: str = "document_only",
    *,
    phase: str = "identity",
    execution=_MISSING,
) -> dict:
    d = {
        "name": name,
        "required": False,
        "kind": "entities",
        "input_mode": input_mode,
        "module": "extraction_schemas.placeholder",
        "template_class": "PlaceholderPass",
        "phase": phase,
    }
    if execution is not _MISSING:
        d["execution"] = execution
    return d


# ---------------------------------------------------------------------------
# 1. execution block omitted → PassManifest.execution is None
# ---------------------------------------------------------------------------

class TestExecutionOmitted:

    def test_omitted_execution_is_none(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass())
        assert p.execution is None

    def test_omitted_execution_back_compat_with_existing_passes(self):
        """All existing passes (no execution key) parse cleanly."""
        from app.services.ontology_bundles import PassManifest
        for name in ("radar_power_rf", "missile_kinematics", "system_links"):
            p = PassManifest.model_validate(
                _raw_pass(name, phase="field_group")
            )
            assert p.execution is None


# ---------------------------------------------------------------------------
# 2. execution: null → None
# ---------------------------------------------------------------------------

class TestExecutionNull:

    def test_explicit_null_is_none(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass(execution=None))
        assert p.execution is None


# ---------------------------------------------------------------------------
# 3. Partial execution block — only llm_batch_token_size
# ---------------------------------------------------------------------------

class TestPartialExecutionBlock:

    def test_only_llm_batch_token_size(self):
        from app.services.ontology_bundles import PassManifest, ExecutionProfile
        p = PassManifest.model_validate(_raw_pass(execution={"llm_batch_token_size": 2048}))
        assert p.execution is not None
        assert p.execution.llm_batch_token_size == 2048
        assert p.execution.temperature is None
        assert p.execution.max_tokens is None
        assert p.execution.chunk_max_tokens is None

    def test_only_temperature(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass(execution={"temperature": 0.3}))
        assert p.execution is not None
        assert p.execution.temperature == pytest.approx(0.3)
        assert p.execution.llm_batch_token_size is None

    def test_only_max_tokens(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass(execution={"max_tokens": 8192}))
        assert p.execution is not None
        assert p.execution.max_tokens == 8192

    def test_only_chunk_max_tokens(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass(execution={"chunk_max_tokens": 256}))
        assert p.execution is not None
        assert p.execution.chunk_max_tokens == 256


# ---------------------------------------------------------------------------
# 4. Full execution block — all four fields
# ---------------------------------------------------------------------------

class TestFullExecutionBlock:

    def test_all_four_fields(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass(execution={
            "chunk_max_tokens": 256,
            "llm_batch_token_size": 2048,
            "temperature": 0.2,
            "max_tokens": 8192,
        }))
        assert p.execution is not None
        assert p.execution.chunk_max_tokens == 256
        assert p.execution.llm_batch_token_size == 2048
        assert p.execution.temperature == pytest.approx(0.2)
        assert p.execution.max_tokens == 8192


# ---------------------------------------------------------------------------
# 5. Bad types are rejected
# ---------------------------------------------------------------------------

class TestExecutionBadTypes:

    def test_string_for_llm_batch_token_size_raises(self):
        from app.services.ontology_bundles import PassManifest
        with pytest.raises(ValidationError):
            PassManifest.model_validate(_raw_pass(execution={"llm_batch_token_size": "two-thousand"}))

    def test_string_for_temperature_raises(self):
        from app.services.ontology_bundles import PassManifest
        with pytest.raises(ValidationError):
            PassManifest.model_validate(_raw_pass(execution={"temperature": "warm"}))

    def test_list_for_execution_raises(self):
        from app.services.ontology_bundles import PassManifest
        with pytest.raises(ValidationError):
            PassManifest.model_validate(_raw_pass(execution=["llm_batch_token_size", 2048]))


# ---------------------------------------------------------------------------
# 6. Real manifests load after Iter 1 annotations
# ---------------------------------------------------------------------------

class TestRealManifestsWithExecutionAnnotations:

    def test_air_defense_v3_subset_loads_cleanly(self):
        """air_defense_v3_baseline_subset has identity passes annotated with execution."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3_baseline_subset")
        for p in m.passes:
            # Every pass loads; execution is None or valid ExecutionProfile
            if p.execution is not None:
                from app.services.ontology_bundles import ExecutionProfile
                assert isinstance(p.execution, ExecutionProfile)

    def test_air_defense_v3_loads_cleanly(self):
        """Full bundle with identity-pass execution annotations loads cleanly."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3")
        for p in m.passes:
            if p.execution is not None:
                from app.services.ontology_bundles import ExecutionProfile
                assert isinstance(p.execution, ExecutionProfile)

    def test_identity_passes_have_llm_batch_token_size_2048(self):
        """Iter 1: identity passes carry llm_batch_token_size=2048."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3_baseline_subset")
        identity_passes = [p for p in m.passes if p.phase == "identity"]
        assert identity_passes, "Expected at least one identity pass"
        for p in identity_passes:
            assert p.execution is not None, (
                f"Identity pass {p.name!r} should have execution block after Iter 1"
            )
            assert p.execution.llm_batch_token_size == 2048, (
                f"Identity pass {p.name!r} should have llm_batch_token_size=2048"
            )

    def test_non_identity_passes_have_no_execution_block(self):
        """Iter 1: only identity passes are annotated; other passes have no block."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3_baseline_subset")
        for p in m.passes:
            if p.phase != "identity":
                assert p.execution is None, (
                    f"Non-identity pass {p.name!r} (phase={p.phase!r}) should not have "
                    f"execution block in Iter 1; got: {p.execution!r}"
                )
