# Auto-Update Docling & Docling-Graph Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every `manage.sh --start` pulls the latest PyPI versions of Docling and Docling-Graph.

**Architecture:** Split Docling requirements into stable (torch, flash-attn, models) and updatable (docling, docling-core) files. Add a `CACHE_BUST` Docker build arg before the updatable pip install. manage.sh passes a fresh timestamp on every start.

**Tech Stack:** Docker, bash, pip

**Spec:** `docs/superpowers/specs/2026-03-17-auto-update-deps-design.md`

---

## Chunk 1: Implementation

### Task 1: Split Docling requirements into stable and updatable files

**Files:**
- Create: `docker/docling/requirements-stable.txt`
- Modify: `docker/docling/requirements.txt`

- [ ] **Step 1: Create `requirements-stable.txt`**

Move the heavy/slow-to-build dependencies that rarely change:

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
transformers>=4.40.0
torch>=2.5.0
huggingface-hub>=0.20.0
Pillow>=11.0.0
python-multipart>=0.0.20
```

- [ ] **Step 2: Update `requirements.txt` to only contain updatable packages**

Change from exact pins to minimum pins:

```
docling>=2.76.0
docling-core>=2.67.1
```

- [ ] **Step 3: Commit**

```bash
git add docker/docling/requirements-stable.txt docker/docling/requirements.txt
git commit -m "refactor: split Docling requirements into stable and updatable"
```

---

### Task 2: Restructure Docling Dockerfile for cache-busted updates

**Files:**
- Modify: `docker/docling/Dockerfile:86-102`

- [ ] **Step 1: Restructure the pip install and CACHE_BUST layers**

Replace the current block (lines 86-102):

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir -r requirements.txt
```

With:

```dockerfile
# --- Stable deps (cached normally) ---
COPY requirements-stable.txt .
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir -r requirements-stable.txt
```

Then, after the existing flash-attn build and model download layers (which remain unchanged), add the updatable deps block:

```dockerfile
# --- Updatable deps (cache-busted on every start) ---
ARG CACHE_BUST
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
```

This goes immediately after line 102 (`RUN python3 -c "... granite-vision-3.3-2b ..."`) and before the `ENV HF_HUB_OFFLINE=1` line.

- [ ] **Step 2: Verify Dockerfile builds**

```bash
docker compose build docling
```

Expected: Build succeeds. Stable deps and flash-attn are cached from previous build. Updatable layer runs pip install for docling/docling-core.

- [ ] **Step 3: Commit**

```bash
git add docker/docling/Dockerfile
git commit -m "feat: add CACHE_BUST arg to Docling Dockerfile for auto-updates"
```

---

### Task 3: Add CACHE_BUST to Docling-Graph Dockerfile

**Files:**
- Modify: `docker/docling-graph/Dockerfile:83-84`

- [ ] **Step 1: Add CACHE_BUST arg before pip install**

Replace:

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

With:

```dockerfile
ARG CACHE_BUST
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
```

- [ ] **Step 2: Verify Dockerfile builds**

```bash
docker compose build docling-graph
```

Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add docker/docling-graph/Dockerfile
git commit -m "feat: add CACHE_BUST arg to Docling-Graph Dockerfile for auto-updates"
```

---

### Task 4: Update manage.sh to pass CACHE_BUST on every start

**Files:**
- Modify: `manage.sh:140-143`

- [ ] **Step 1: Add targeted build before the existing up command**

Replace line 143:

```bash
  dc "${profile_args[@]}" up -d --build
```

With:

```bash
  info "Updating docling & docling-graph packages..."
  dc build --build-arg CACHE_BUST="$(date +%s)" docling docling-graph
  dc "${profile_args[@]}" up -d --build
```

- [ ] **Step 2: Verify manage.sh runs correctly**

```bash
bash manage.sh --start
```

Expected: Logs show "Updating docling & docling-graph packages...", then the two services rebuild with fresh pip installs. All other services start normally. Health checks pass.

- [ ] **Step 3: Commit**

```bash
git add manage.sh
git commit -m "feat: auto-update Docling and Docling-Graph on every start"
```
