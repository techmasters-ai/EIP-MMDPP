"""Integration tests for health and readiness endpoints."""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_liveness(async_client):
    response = await async_client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readiness_returns_checks(async_client):
    response = await async_client.get("/v1/health/ready")
    # May be degraded if test infra not running, but must return 200 with checks
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "postgres" in data["checks"]
    assert "redis" in data["checks"]
    assert "minio" in data["checks"]
