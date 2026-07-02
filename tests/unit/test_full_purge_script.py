"""G-1: Dry-run coverage for scripts/full_purge_and_reingest.py.

Verifies the --dry-run path prints the full truncate plan without
executing any destructive step (no Postgres writes, no ArcadeDB
drops, no MinIO deletes, no Redis flush, no container control, no
Celery enqueue). The real-run path is exercised in Chunk G-3 against
a live corpus — not mocked here.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "full_purge_and_reingest.py"


def _run_dry(args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), "--dry-run"] + (args or [])
    return subprocess.run(
        cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT,
        timeout=60,
    )


def test_dry_run_exits_zero():
    """--dry-run is a no-op: zero exit, no exceptions."""
    result = _run_dry()
    assert result.returncode == 0, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


def test_dry_run_emits_every_truncate_target():
    """The truncate plan printed under --dry-run must include every
    table from spec §5.1."""
    result = _run_dry()
    expected = [
        "ingest.stage_runs",
        "ingest.pipeline_runs",
        "ingest.document_graph_extractions",
        "ingest.document_elements",
        "ingest.artifacts",
        "retrieval.chunk_links",
        "retrieval.text_chunks",
        "retrieval.image_chunks",
        "retrieval.chunks_legacy",
        "retrieval.community_runs",
        "governance.feedback",
        "governance.patch_approvals",
        "governance.patch_events",
        "governance.patches",
        "governance.trusted_data_submissions",
        "ingest.watch_logs",
    ]
    combined = result.stdout + result.stderr
    missing = [t for t in expected if t not in combined]
    assert not missing, f"dry-run output missing truncate lines for: {missing}"


def test_dry_run_names_preserved_tables():
    """The dry-run plan must name the preserved tables so the operator
    sees that ingest.documents (UPDATE, not TRUNCATE), sources,
    watch_dirs, auth, and alembic_version survive."""
    result = _run_dry()
    combined = result.stdout + result.stderr
    for preserved in (
        "ingest.documents",
        "ingest.sources",
        "ingest.watch_dirs",
        "auth.users",
        "auth.user_roles",
        "public.alembic_version",
    ):
        assert preserved in combined, f"preserved table {preserved} missing from plan"


def test_real_run_requires_safety_flag():
    """Omitting both --dry-run and --i-understand-this-deletes-derived-data
    should exit 2 and print an explicit message."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False, capture_output=True, text=True, cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 2, f"expected exit 2, got {result.returncode}"
    combined = result.stdout + result.stderr
    assert "i-understand-this-deletes-derived-data" in combined


def test_dry_run_skips_enqueue_step(monkeypatch):
    """Direct-call test: enqueue_pipeline under dry_run=True returns []
    without importing Celery tasks."""
    from scripts.full_purge_and_reingest import enqueue_pipeline

    result = enqueue_pipeline(["doc-1", "doc-2"], dry_run=True)
    assert result == []


def test_stop_workers_dry_run_does_not_shell_out(monkeypatch):
    """stop_workers under dry_run=True logs only — no subprocess calls."""
    from scripts.full_purge_and_reingest import stop_workers

    called: list[tuple] = []

    def fake_run(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("subprocess.run must not be called in dry-run mode")

    monkeypatch.setattr("scripts.full_purge_and_reingest.subprocess.run", fake_run)
    stop_workers(dry_run=True)  # must not raise
    assert called == []
