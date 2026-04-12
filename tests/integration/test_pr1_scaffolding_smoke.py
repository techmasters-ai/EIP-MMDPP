"""PR 1 scaffolding smoke test.

Verifies every Chunk-1/Chunk-2 exit criterion from spec §7.3:
- The air_defense_v3 bundle directory is complete.
- ontology/ontology.yaml is a symlink into the bundle.
- tools/check_extraction_coverage.py exits 0 with "PASS air_defense_v3".
- load_ontology() returns the bundle ontology.
- load_bundle_manifest() parses all five passes.
- Migration 0015 has been applied (new StageRun columns exist).

The end-to-end canary run (legacy path still produces a graph) is in
tests/e2e/test_full_pipeline.py and is documented in the PR description
rather than asserted here, because it requires the full compose stack.

Task 2.8 of the extraction-refactor plan.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def test_bundle_directory_complete():
    p = REPO_ROOT / "ontology_bundles" / "air_defense_v3"
    assert p.is_dir(), f"Bundle directory missing: {p}"
    for required in [
        "ontology.yaml", "manifest.yaml", "coverage.yaml",
        "validators.py", "derive_rules.py",
    ]:
        assert (p / required).exists(), f"Missing bundle file: {required}"
    for pass_file in [
        "reference", "radar_domain", "missile_domain",
        "other_systems", "system_links",
    ]:
        f = p / "extraction_schemas" / f"{pass_file}.py"
        assert f.exists(), f"Missing extraction schema: {pass_file}.py"


def test_ontology_symlink_resolves():
    p = REPO_ROOT / "ontology" / "ontology.yaml"
    assert p.is_symlink(), f"{p} is not a symlink"
    resolved = p.resolve()
    assert resolved.name == "ontology.yaml"
    assert "ontology_bundles/air_defense_v3" in str(resolved)


def test_coverage_checker_passes():
    """tools/check_extraction_coverage.py exits 0 and reports PASS on the bundle."""
    # Use the venv python so the subprocess has pydantic, yaml, etc.
    # System python does not have those deps.
    python = VENV_PYTHON if VENV_PYTHON.exists() else sys.executable
    result = subprocess.run(
        [str(python), "tools/check_extraction_coverage.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Checker failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "PASS air_defense_v3" in result.stdout


def test_load_ontology_returns_bundle_ontology():
    from app.services.ontology_templates import load_ontology
    ont = load_ontology()
    assert isinstance(ont, dict)
    assert "entity_types" in ont
    assert any(e.get("name") == "RADAR_SYSTEM" for e in ont["entity_types"])


def test_load_bundle_manifest_has_five_passes():
    from app.services.ontology_bundles import load_bundle_manifest
    m = load_bundle_manifest("air_defense_v3")
    assert len(m.passes) == 5
    assert {p.name for p in m.passes} == {
        "reference", "radar_domain", "missile_domain",
        "other_systems", "system_links",
    }


def _resolve_postgres_env() -> dict[str, str] | None:
    """Return an env dict that lets tests reach postgres, or None if unreachable.

    Mirrors the pattern in tests/unit/test_migration_0015.py: the default
    hostname `postgres:5435` only resolves inside the compose network, so
    we fall back to localhost:5435 (the host-exposed port) when running
    from the host.
    """
    env = {**os.environ}
    candidates = [
        (env.get("POSTGRES_HOST", "postgres"), int(env.get("POSTGRES_PORT", "5435"))),
        ("localhost", 5435),
        ("127.0.0.1", 5435),
    ]
    for host, port in candidates:
        try:
            with socket.create_connection((host, port), timeout=1):
                pass
        except (OSError, socket.gaierror):
            continue
        env["POSTGRES_HOST"] = host
        env["POSTGRES_PORT"] = str(port)
        return env
    return None


def test_migration_0015_applied():
    """The four StageRun columns added by 0015 must exist in the live DB."""
    env = _resolve_postgres_env()
    if env is None:
        pytest.skip("postgres not reachable; compose stack must be up for this test")

    # Reconfigure settings with the host-resolved env so create_engine works
    os.environ["POSTGRES_HOST"] = env["POSTGRES_HOST"]
    os.environ["POSTGRES_PORT"] = env["POSTGRES_PORT"]
    from app.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]
    settings = get_settings()

    from sqlalchemy import create_engine, text
    engine = create_engine(settings.sync_database_url)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'ingest'
                  AND table_name = 'stage_runs'
                  AND column_name IN (
                      'pass_name', 'execution_status',
                      'yield_status', 'rollback_executed'
                  )
                """
            ))
            cols = {row[0] for row in result}
    finally:
        engine.dispose()

    assert cols == {
        "pass_name", "execution_status", "yield_status", "rollback_executed"
    }, f"Missing columns; found only: {cols}"
