"""Smoke test for Alembic migration 0015.

Asserts upgrade + downgrade both apply cleanly against the configured
database. Uses subprocess to invoke the .venv alembic binary explicitly
so it works regardless of PATH.

When run from the host (not inside the compose network), the default
settings point at the unreachable hostname `postgres:5435`. In that
case we auto-substitute `POSTGRES_HOST=localhost` and `POSTGRES_PORT=5435`
so the subprocess can reach the host-exposed postgres port. The test
is skipped if postgres is not reachable at either address.
"""
import os
import socket
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC = REPO_ROOT / ".venv" / "bin" / "alembic"


def _resolve_postgres_env() -> dict[str, str] | None:
    """Return an env dict that lets alembic reach postgres, or None if unreachable.

    Tries settings' default hostname first, then falls back to
    localhost:5435 (the host-exposed port from docker-compose.yml).
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


def _alembic(*args: str) -> subprocess.CompletedProcess:
    env = _resolve_postgres_env()
    if env is None:
        pytest.skip("postgres is not reachable at postgres:5435, localhost:5435, or 127.0.0.1:5435")
    return subprocess.run(
        [str(ALEMBIC), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
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
