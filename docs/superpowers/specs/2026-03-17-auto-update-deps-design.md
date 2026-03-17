# Auto-Update Docling & Docling-Graph on Start

## Problem

`manage.sh --start` rebuilds Docker images, but Docker layer caching means pip packages are only installed once — when the requirements file changes. Even with `>=` pinning, cached layers prevent picking up new PyPI releases.

## Solution

Add a `CACHE_BUST` build arg to the Docling and Docling-Graph Dockerfiles, placed so it invalidates only the pip install layers that contain the updatable packages. `manage.sh` passes a fresh timestamp on every `--start`, invalidating those layers while preserving cached base image, apt, flash-attn build, and model download layers.

## Scope

**Affected packages:**
- `docling` (+ `docling-core`) — `docker/docling/requirements.txt`
- `docling-graph` — `docker/docling-graph/requirements.txt`

**Not in scope:**
- Main API/worker Dockerfile (`graphrag` dependency addressed in separate Microsoft GraphRAG project)
- Version locking or rollback mechanisms

## Changes

### 1. Docling Dockerfile restructure

The current Docling Dockerfile installs everything in one pip layer, then builds flash-attn, then downloads models. Placing `CACHE_BUST` before pip install would invalidate **all** subsequent layers including the expensive flash-attn build (~10-60 min) and model downloads (~3-4 GB).

**Fix**: Split requirements into stable and updatable. Install stable deps + flash-attn + models first (cached normally), then cache-bust only the updatable deps.

```dockerfile
# --- Stable deps (cached normally) ---
COPY requirements-stable.txt .
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir -r requirements-stable.txt

# flash-attn build (expensive, cached)
ENV MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Model downloads (cached)
ENV HF_HOME=/opt/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-docling-258M')"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-vision-3.3-2b')"

# --- Updatable deps (cache-busted on every start) ---
ARG CACHE_BUST
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
```

**`requirements-stable.txt`** — heavy/slow-to-build deps that rarely change:
```
torch>=2.5.0
transformers>=4.40.0
huggingface-hub>=0.20.0
Pillow>=11.0.0
python-multipart>=0.0.20
```

**`requirements.txt`** — the packages we want to auto-update:
```
docling>=2.76.0
docling-core>=2.67.1
```

The `--upgrade` flag ensures pip checks PyPI for newer versions even if an older version from the stable layer already satisfies the constraint.

### 2. Docling-Graph Dockerfile

Simpler — no expensive build steps. Just add `CACHE_BUST` before the existing pip install and add `--upgrade`:

```dockerfile
ARG CACHE_BUST
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
```

No changes to `requirements.txt` needed (already uses `>=` pinning).

### 3. manage.sh changes

In the `--start` and `--start-split` code paths, add a targeted build step before the existing `up` command. Uses the existing `dc` wrapper function for consistency:

```bash
info "Updating docling & docling-graph packages..."
dc build --build-arg CACHE_BUST=$(date +%s) docling docling-graph
dc "${profile_args[@]}" up -d --build
```

The `dc build` targets only the two services. The subsequent `dc up -d --build` still rebuilds any other services whose context changed (e.g., API image when `./app/` code changes), preserving existing behavior.

### 4. Non-changes

- `--restart` remains fast (no rebuild)
- `--stop`, `--status`, `--logs`, `--blow-away`, and all other commands are unaffected
- `docker-compose.yml` is unchanged
- Profile args (`--profile split`) are not needed for the `dc build` command since `docling` and `docling-graph` are top-level services not gated by profiles
