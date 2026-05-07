"""PR 3 post-deletion smoke test. Task 5.7.

Verifies every legacy file is deleted, every CI lint passes,
derive_ontology_graph has no feature-flag branch, and Settings has
no legacy flag fields. This is the final verification gate before
PR 3 is approved.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestDeletedFilesGone:
    """Every file marked for deletion in spec §7.5 must not exist."""

    @pytest.mark.parametrize("rel_path", [
        "docker/docling-graph/app/template_builder.py",
        "app/services/layered_extraction.py",
        "app/services/ontology_layers.py",
        "ontology/layer_map.yaml",
        "ontology/ontology.yaml",
    ])
    def test_file_deleted(self, rel_path):
        p = REPO_ROOT / rel_path
        assert not p.exists(), f"{rel_path} should have been deleted in Task 5.2"

    @pytest.mark.parametrize("rel_path", [
        "tests/unit/test_layered_extraction.py",
        "tests/unit/test_ontology_layers.py",
        "docker/docling-graph/tests/test_template_builder.py",
    ])
    def test_test_file_deleted(self, rel_path):
        p = REPO_ROOT / rel_path
        assert not p.exists(), f"{rel_path} should have been deleted in Task 5.2"


class TestCILintsPass:
    def test_ci_lints_all_pass(self):
        """All 8 legacy-reference lints must pass."""
        result = subprocess.run(
            ["./tools/ci_lints.sh"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"CI lints failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "All CI lints passed." in result.stdout


class TestOnlyNewPathExists:
    def test_no_legacy_branch_in_derive_ontology_graph(self):
        """derive_ontology_graph is now a thin dispatcher (Task 8). It must
        reference _claim_and_dispatch_pass (per-pass fan-out) and must NOT
        reference the deleted legacy helper or any feature-flag branch."""
        from app.workers import pipeline
        src = inspect.getsource(pipeline.derive_ontology_graph.run)
        assert "_derive_ontology_graph_legacy" not in src, (
            "derive_ontology_graph still references the legacy branch"
        )
        assert "_derive_ontology_graph_bundle_passes" not in src, (
            "derive_ontology_graph still references the deleted monolithic helper"
        )
        assert "_claim_and_dispatch_pass" in src, (
            "derive_ontology_graph dispatcher must use _claim_and_dispatch_pass"
        )

    def test_no_legacy_function_exists(self):
        """_derive_ontology_graph_legacy must not exist as a module attribute."""
        from app.workers import pipeline
        assert not hasattr(pipeline, "_derive_ontology_graph_legacy"), (
            "_derive_ontology_graph_legacy still exists in pipeline.py"
        )


class TestSettingsCleanup:
    def test_no_graph_extraction_engine(self):
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert not hasattr(s, "graph_extraction_engine")

    def test_no_graph_layered_fields(self):
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        for field_name in [
            "graph_layered_shadow_mode",
            "graph_layered_fail_open_to_single_pass",
            "graph_layered_extraction_enabled",
            "graph_layered_cross_layer_enabled",
            "graph_layered_max_passes",
        ]:
            assert not hasattr(s, field_name), (
                f"Settings still has legacy field: {field_name}"
            )

    def test_default_ontology_bundle_key_still_present(self):
        """The bundle default setting must survive the cleanup."""
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.default_ontology_bundle_key == "air_defense_v3"


class TestNewPathFunctional:
    def test_coverage_checker_passes(self):
        """The coverage checker must still pass on air_defense_v3."""
        python = REPO_ROOT / ".venv" / "bin" / "python"
        result = subprocess.run(
            [str(python), "tools/check_extraction_coverage.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASS air_defense_v3" in result.stdout

    def test_extraction_status_endpoint_registered(self):
        """The new /extraction-status endpoint exists on the router."""
        from app.api.v1.sources import router
        routes = [r.path for r in router.routes]
        assert "/documents/{document_id}/extraction-status" in routes

    def test_schema_count_in_health_response(self):
        """HealthResponse uses schema_count (not template_count).

        The docling-graph schemas live in a separate package root
        (docker/docling-graph/) that is not on the worker venv's import
        path. We verify via grep instead of direct import.
        """
        schema_file = REPO_ROOT / "docker" / "docling-graph" / "app" / "schemas.py"
        content = schema_file.read_text()
        assert "schema_count" in content, "HealthResponse missing schema_count field"
        assert "template_count" not in content, "HealthResponse still has template_count (should be renamed)"
