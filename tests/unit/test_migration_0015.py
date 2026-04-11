"""Smoke test for Alembic migration 0015.

Asserts upgrade + downgrade both apply cleanly against the configured
database. Run via .venv/bin/pytest; the test uses subprocess to invoke
the .venv alembic binary explicitly so it works regardless of PATH.

Requires postgres to be running (docker compose up -d postgres) and
DATABASE_URL_SYNC env var pointing at the host-accessible URL, e.g.:
  DATABASE_URL_SYNC=postgresql+psycopg2://eip:eip_secret@localhost:5435/eip
"""
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC = REPO_ROOT / ".venv" / "bin" / "alembic"


def _alembic(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(ALEMBIC), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )


def test_migration_0015_applies_cleanly():
    """Upgrade head → 0015. Verify it exits 0."""
    result = _alembic("upgrade", "head")
    assert result.returncode == 0, (
        f"Migration failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_migration_0015_downgrade_then_reapply():
    """Downgrade to 0014, then re-apply head."""
    down = _alembic("downgrade", "0014")
    assert down.returncode == 0, f"Downgrade failed:\n{down.stderr}"

    # Re-apply so later tests see the head schema.
    up = _alembic("upgrade", "head")
    assert up.returncode == 0, f"Re-upgrade failed:\n{up.stderr}"
