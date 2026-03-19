---
name: open-issues-2026-03-18
description: Comprehensive list of open issues reported during 2026-03-18 session, being worked through systematically
type: project
---

## Open Issues (2026-03-18)

### 1. API container missing Ollama DNS — text search and embedding broken
- **Symptom:** `Query strategy QueryStrategy.basic failed: [Errno -3] Temporary failure in name resolution`
- **Root cause:** `api` service in docker-compose.yml is missing `extra_hosts: - "ollama:host-gateway"`. Workers and docling-graph have it, API does not. The `embed_texts()` function (rewritten to use Ollama API) can't resolve the `ollama` hostname.
- **Fix:** Add `extra_hosts` to the `api` service in docker-compose.yml
- **Status:** IDENTIFIED, not yet fixed

### 2. Reranker crashes — no NVIDIA driver in API container
- **Symptom:** `Query strategy QueryStrategy.hybrid failed: Found no NVIDIA driver on your system`
- **Root cause:** `.env` has `RERANKER_DEVICE=cuda` but the API container has no GPU access
- **Fix:** Change `RERANKER_DEVICE=cpu` in `.env`
- **Status:** IDENTIFIED, not yet fixed

### 3. Multi-modal search returns nothing
- **Symptom:** No results from hybrid/multi-modal queries
- **Root cause:** Cascading failure from issues #1 (embedding fails) and #2 (reranker crashes)
- **Fix:** Resolves once #1 and #2 are fixed
- **Status:** BLOCKED by #1 and #2

### 4. GraphRAG queries (Local, Global, Drift, Basic) return nothing
- **Symptom:** All four GraphRAG query modes return empty results
- **Root cause:** GraphRAG indexing failed previously (connection error + PipelineRunResult.errors bug). Code fixes applied but containers need rebuild. Also API container needs Ollama DNS (#1) for GraphRAG search to work.
- **Fix:** Rebuild containers, trigger reindex after #1 is fixed
- **Status:** BLOCKED by #1

### 5. Ontology graph view shows single node despite multiple entities
- **Symptom:** "Fan Song" search returns neighbors with different entity_types (PLATFORM, MODULATION, RF_EMISSION, RF_SIGNATURE) but graph shows only one node
- **Root cause:** Backend `get_neighborhood_graph_async` and frontend `toGraphElements` were updated to key by UUID, but the code changes haven't been deployed yet (containers need rebuild)
- **Fix:** Rebuild containers
- **Status:** CODE FIXED, needs deploy

### 6. Bounding boxes severely misaligned in DoclingDocument viewer
- **Symptom:** Bounding boxes land on unrelated content
- **Root cause:** `images_scale=2.0` caused mismatch between page image resolution and bounding box coordinate space. Changed to `1.0` but documents need re-ingesting to regenerate the Docling JSON with correct coordinates.
- **Fix:** Rebuild docling container + re-ingest documents
- **Status:** CODE FIXED, needs deploy + reingest

### 7. DoclingDocument viewer doesn't fill the window
- **Symptom:** Viewer modal doesn't take up the full available space
- **Root cause:** CSS sizing issue in DoclingViewer component or the iframe
- **Fix:** Adjust CSS/iframe sizing
- **Status:** NOT YET INVESTIGATED

### 8. `.txt` files have no viewer
- **Symptom:** Text files uploaded to the system don't show in the DoclingDocument viewer
- **Root cause:** DoclingDocument JSON isn't generated for .txt files, and no fallback viewer exists
- **Fix:** Add a plain text viewer fallback when DoclingDocument JSON isn't available
- **Status:** NOT YET INVESTIGATED
