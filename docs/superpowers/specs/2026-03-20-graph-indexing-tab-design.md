# Graph Indexing Tab — Design Spec

**Date:** 2026-03-20
**Status:** Approved

## Overview

Add a "Graph Indexing" tab to the Graph Explorer page that lets users manually trigger GraphRAG indexing and shows when the next automatic run will occur.

## Backend

### New endpoint: `GET /v1/settings/graphrag`

Returns current GraphRAG configuration (read-only, from env vars) plus last indexing timestamp (from Redis).

```json
{
  "indexing_enabled": true,
  "indexing_interval_minutes": 60,
  "last_indexing_at": "2026-03-20T17:09:31Z"
}
```

- `indexing_enabled` and `indexing_interval_minutes` read from `get_settings()`
- `last_indexing_at` read from Redis key `graphrag:last_indexed_at` (null if never run)

### Task update: `run_graphrag_indexing_task`

After successful indexing, write `r.set("graphrag:last_indexed_at", utc_iso_string)` to Redis.

### Existing endpoint: `POST /v1/graphrag/index`

No changes. Already dispatches the indexing Celery task with Redis lock to prevent overlapping runs.

## Frontend

### GraphExplorer.tsx changes

Add `"indexing"` as 4th tab value. Tab label: "Graph Indexing".

```typescript
type Tab = "search" | "entity" | "relationship" | "indexing";
```

### New component: `GraphIndexingPanel`

Inline in GraphExplorer.tsx (follows existing pattern — EntityForm and RelationshipForm are in the same file).

**On mount:** calls `GET /v1/settings/graphrag` to load config.

**Display:**
- "Automatic indexing runs every **{N} minutes**" (from `indexing_interval_minutes`)
- "Next auto indexing will run in **{X} minutes**" (computed: `interval - minutes_since_last_run`, clamped to 0; shows "Pending first run" if `last_indexing_at` is null)
- "Run Indexing Now" button (primary style, disabled while request in flight)
- Status text: "Indexing started" / "Already running" / error message

**Countdown:** Recomputed every 60 seconds via `setInterval`.

### API client additions

```typescript
interface GraphRAGSettings {
  indexing_enabled: boolean;
  indexing_interval_minutes: number;
  last_indexing_at: string | null;
}

function getGraphRAGSettings(): Promise<GraphRAGSettings>
function triggerGraphRAGIndexing(): Promise<{ status: string; task_id: string }>
```

## Testing

- Unit test: `GET /v1/settings/graphrag` returns correct shape
- Unit test: `last_indexing_at` written to Redis after successful indexing
- Existing `POST /v1/graphrag/index` endpoint already covered
