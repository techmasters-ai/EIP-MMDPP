# GraphRAG Fixes + Drift/Basic Query Modes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix GraphRAG indexing/query failures caused by misconfigured env vars, fix the PipelineRunResult API mismatch, add Drift and Basic query modes to the frontend, and fix orphan node graph rendering on the Ontology tab.

**Architecture:** Four independent fixes: (1) env var name corrections in `.env`, (2) error field name fix in `graphrag_service.py`, (3) frontend additions to `client.ts` and `QueryPage.tsx`, (4) Neo4j query restructuring in `neo4j_graph.py` for isolated nodes.

**Tech Stack:** Python/FastAPI, TypeScript/React, Neo4j Cypher, Microsoft GraphRAG v3, Pydantic Settings

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `.env` | Modify | Fix env var names so Pydantic Settings reads them |
| `app/services/graphrag_service.py` | Modify | Fix `PipelineRunResult.error` (singular) |
| `frontend/src/api/client.ts` | Modify | Add `graphrag_drift` and `graphrag_basic` to `QueryStrategy` type |
| `frontend/src/components/QueryPage.tsx` | Modify | Add Drift/Basic to MODES array and provenance labels |
| `app/services/neo4j_graph.py` | Modify | Restructure `get_neighborhood_graph_async` so fallback always runs |

---

## Chunk 1: Backend Fixes

### Task 1: Fix .env GraphRAG config env var names

**Files:**
- Modify: `.env:75-79`

**Root cause:** Pydantic Settings reads env vars matching field names (case-insensitive). The config field is `graphrag_llm_model` but `.env` has `GRAPHRAG_MODEL` — the `_LLM_` segment is missing. Also, `graphrag_llm_api_base` defaults to `http://localhost:11434/v1` which doesn't resolve inside Docker (Ollama is at `http://ollama:11434/v1`).

- [ ] **Step 1: Update `.env` GraphRAG section**

Replace lines 75-79:

```env
# GraphRAG
GRAPHRAG_LLM_MODEL=gpt-oss:120b
GRAPHRAG_LLM_API_BASE=http://ollama:11434/v1
GRAPHRAG_EMBEDDING_MODEL=nomic-embed-text
GRAPHRAG_INDEXING_ENABLED=true
GRAPHRAG_INDEXING_INTERVAL_MINUTES=60
GRAPHRAG_MAX_CLUSTER_SIZE=10
```

- [ ] **Step 2: Verify config picks up the values**

```bash
docker compose exec api python -c "
from app.config import get_settings
s = get_settings()
print(f'model={s.graphrag_llm_model}')
print(f'api_base={s.graphrag_llm_api_base}')
print(f'embed={s.graphrag_embedding_model}')
"
```

Expected:
```
model=gpt-oss:120b
api_base=http://ollama:11434/v1
embed=nomic-embed-text
```

- [ ] **Step 3: Commit**

```bash
git add .env
git commit -m "fix: correct GraphRAG env var names to match Pydantic Settings fields"
```

---

### Task 2: Fix PipelineRunResult.errors AttributeError

**Files:**
- Modify: `app/services/graphrag_service.py:92-99`

**Root cause:** GraphRAG v3's `PipelineRunResult` dataclass has `error: BaseException | None` (singular), not `errors` (plural). The current code crashes with `AttributeError: 'PipelineRunResult' object has no attribute 'errors'`.

- [ ] **Step 1: Fix the error field access**

Replace lines 92-99 in `app/services/graphrag_service.py`:

```python
    for result in results:
        if result.error:
            logger.warning(
                "GraphRAG workflow %s error: %s",
                result.workflow, result.error,
            )
        else:
            logger.info("GraphRAG workflow %s completed", result.workflow)
```

- [ ] **Step 2: Commit**

```bash
git add app/services/graphrag_service.py
git commit -m "fix: use PipelineRunResult.error (singular) for GraphRAG v3 API"
```

---

### Task 3: Fix orphan node graph rendering in Ontology tab

**Files:**
- Modify: `app/services/neo4j_graph.py:491-586`

**Root cause:** When an entity has no relationships, the main Cypher query's `UNWIND` produces zero rows. If any error occurs during the main query, the fallback query never executes because both are inside the same `try` block. The function returns `{nodes: [], edges: []}` and the frontend shows "No graph data found."

- [ ] **Step 1: Restructure `get_neighborhood_graph_async`**

Replace the function body (lines 527-586) with two independent try/except blocks:

```python
    center: dict[str, Any] | None = None
    nodes_map: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    # Main query: fetch neighborhood (may return zero rows for orphan nodes)
    try:
        async with driver.session() as session:
            result = await session.run(query, name=entity_name, limit=limit)
            records = await result.data()

            for r in records:
                if center is None and r.get("center_props"):
                    center = dict(r["center_props"])
                    center["entity_type"] = r["center_type"]
                    nodes_map[entity_name] = center

                source_name = r.get("source")
                target_name = r.get("target")
                if not source_name or not target_name:
                    continue

                if source_name not in nodes_map and r.get("source_props"):
                    node = dict(r["source_props"])
                    node["entity_type"] = r["source_type"]
                    nodes_map[source_name] = node
                if target_name not in nodes_map and r.get("target_props"):
                    node = dict(r["target_props"])
                    node["entity_type"] = r["target_type"]
                    nodes_map[target_name] = node

                edge: dict[str, Any] = {
                    "source": source_name,
                    "target": target_name,
                    "rel_type": r.get("rel_type", "UNKNOWN"),
                }
                if r.get("rel_props"):
                    edge.update(r["rel_props"])
                edges.append(edge)
    except Exception as e:
        logger.warning("get_neighborhood_graph_async main query failed for '%s': %s", entity_name, e)

    # Fallback: always fetch center node if not found by main query
    if center is None:
        try:
            q2 = """
                MATCH (n:Entity {name: $name})
                RETURN properties(n) AS props, n.entity_type AS entity_type
                LIMIT 1
            """
            async with driver.session() as session:
                result = await session.run(q2, name=entity_name)
                records = await result.data()
                if records:
                    center = dict(records[0]["props"])
                    center["entity_type"] = records[0]["entity_type"]
                    nodes_map[entity_name] = center
        except Exception as e:
            logger.warning("get_neighborhood_graph_async fallback failed for '%s': %s", entity_name, e)

    return {
        "center": center,
        "nodes": list(nodes_map.values()),
        "edges": edges,
    }
```

- [ ] **Step 2: Commit**

```bash
git add app/services/neo4j_graph.py
git commit -m "fix: ensure orphan node fallback always runs in get_neighborhood_graph_async"
```

---

## Chunk 2: Frontend Changes

### Task 4: Add GraphRAG Drift and Basic modes to frontend

**Files:**
- Modify: `frontend/src/api/client.ts:53`
- Modify: `frontend/src/components/QueryPage.tsx:10-31, 246-270, 335-339`

- [ ] **Step 1: Update TypeScript QueryStrategy type**

In `frontend/src/api/client.ts` line 53, replace:

```typescript
export type QueryStrategy = "basic" | "hybrid" | "graphrag_local" | "graphrag_global";
```

with:

```typescript
export type QueryStrategy = "basic" | "hybrid" | "graphrag_local" | "graphrag_global" | "graphrag_drift" | "graphrag_basic";
```

- [ ] **Step 2: Add MODES entries in QueryPage.tsx**

In `frontend/src/components/QueryPage.tsx`, add two entries to the `MODES` array after the GraphRAG Global entry (after line 30):

```typescript
  {
    strategy: "graphrag_drift",
    label: "GraphRAG Drift",
    description: "Community-informed expansion search (DRIFT)",
  },
  {
    strategy: "graphrag_basic",
    label: "GraphRAG Basic",
    description: "Vector search over GraphRAG text units",
  },
```

- [ ] **Step 3: Add provenance labels for Drift and Basic**

In `frontend/src/components/QueryPage.tsx`, in the `ResultCard` component, add detection variables alongside the existing `isGraphRAGLocal`/`isGraphRAGGlobal` (around line 246):

```typescript
  const isGraphRAGDrift = ctx?.source === "graphrag_drift";
  const isGraphRAGBasic = ctx?.source === "graphrag_basic";
```

Then add label branches in the provenance section (after the `isGraphRAGGlobal` block around line 269):

```typescript
  } else if (isGraphRAGDrift) {
    provenanceLabel = "GraphRAG Drift: community-informed expansion";
  } else if (isGraphRAGBasic) {
    provenanceLabel = "GraphRAG Basic: text unit vector search";
  }
```

- [ ] **Step 4: TypeScript build check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/client.ts frontend/src/components/QueryPage.tsx
git commit -m "feat: add GraphRAG Drift and Basic query modes to search UI"
```

---

## Post-Implementation: Trigger GraphRAG Reindex

After deploying the env var fixes, manually trigger a reindex:

```bash
docker compose restart worker-graph beat
# Then trigger indexing:
curl -X POST http://localhost:8005/v1/graphrag/index
```

Monitor logs:
```bash
docker compose logs worker-graph --tail=50 -f | grep -i graphrag
```

Expected: indexing completes without `Connection error` or `AttributeError`, creates `communities.parquet` and `community_reports.parquet`.
