# Auto-Update Docling & Docling-Graph on Start

## Problem

`manage.sh --start` rebuilds Docker images, but Docker layer caching means pip packages are only installed once — when the requirements file changes. Even with `>=` pinning, cached layers prevent picking up new PyPI releases.

## Solution

Add a `CACHE_BUST` build arg to the Docling and Docling-Graph Dockerfiles, placed immediately before the `pip install` step. `manage.sh` passes a fresh timestamp on every `--start`, invalidating the pip layer while preserving cached base image and apt layers.

## Scope

**Affected packages:**
- `docling` (+ `docling-core`) — `docker/docling/requirements.txt`
- `docling-graph` — `docker/docling-graph/requirements.txt`

**Not in scope:**
- Main API/worker Dockerfile (`graphrag` dependency addressed in separate Microsoft GraphRAG project)
- Version locking or rollback mechanisms

## Changes

### 1. Dockerfile changes

Both `docker/docling/Dockerfile` and `docker/docling-graph/Dockerfile`:

```dockerfile
# Place immediately before pip install
ARG CACHE_BUST
RUN pip install --no-cache-dir -r requirements.txt
```

### 2. Requirements pinning

`docker/docling/requirements.txt` — relax exact pins to minimums:

| Before | After |
|--------|-------|
| `docling==2.76.0` | `docling>=2.76.0` |
| `docling-core==2.67.1` | `docling-core>=2.67.1` |

`docker/docling-graph/requirements.txt` — no change needed (already uses `>=`).

### 3. manage.sh changes

In the `--start` and `--start-split` code paths, replace the single `docker compose up -d --build` with:

```bash
docker compose build --build-arg CACHE_BUST=$(date +%s) docling docling-graph
docker compose up -d
```

This targets only the two services that need fresh packages. Infrastructure services and the main API image are unaffected.

### 4. Non-changes

- `--restart` remains fast (no rebuild)
- `--stop`, `--status`, `--logs`, and all other commands are unaffected
- docker-compose.yml is unchanged
