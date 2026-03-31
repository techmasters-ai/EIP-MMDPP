# Async GraphRAG Queries Design

**Date:** 2026-03-31
**Status:** Approved

## Problem

GraphRAG queries (local, global, drift, basic) invoke LLM calls that take 1-3+ minutes. The browser drops the idle HTTP connection before the server responds, producing "NetworkError when attempting to fetch resource." The server continues processing after the browser gives up, wasting GPU/LLM time with no one to receive the result.

## Solution

Decouple query submission from result retrieval using Celery tasks and polling. The frontend submits a query, receives a job ID immediately, polls for completion, then fetches the result.

## API Endpoints

Three new endpoints under `/v1/retrieval/graphrag/`:

### POST `/v1/retrieval/graphrag/submit`

Submits a GraphRAG query as an async job.

**Request:** Same `UnifiedQueryRequest` schema. Strategy must be one of `graphrag_local`, `graphrag_global`, `graphrag_drift`, `graphrag_basic` or the endpoint returns 400.

**Response (202):**
```json
{
  "job_id": "celery-task-uuid",
  "status": "pending"
}
```

### GET `/v1/retrieval/graphrag/status/{job_id}`

Polls job status.

**Response (200):**
```json
{
  "job_id": "celery-task-uuid",
  "status": "pending | running | completed | failed",
  "error": null
}
```

Status mapping from Celery states:
- `PENDING` -> `pending`
- `STARTED` -> `running`
- `SUCCESS` -> `completed`
- `FAILURE` / `REVOKED` -> `failed`

### GET `/v1/retrieval/graphrag/result/{job_id}`

Fetches the full result once completed.

**Response (200):** Same `UnifiedQueryResponse` schema as the existing sync endpoint.

**Error responses:**
- 404: Job not found or expired (Redis TTL is 24h)
- 409: Job still in progress (status is not `completed`)

## Schemas

Two new schemas in `app/schemas/retrieval.py`:

```python
class GraphRAGJobSubmitResponse(APIModel):
    job_id: str
    status: str  # always "pending"

class GraphRAGJobStatusResponse(APIModel):
    job_id: str
    status: str  # pending | running | completed | failed
    error: Optional[str] = None
```

## Celery Task

Add the new task to the existing `app/workers/graphrag_tasks.py` (co-locate with indexing/tuning tasks).

One task: `run_graphrag_query_task`

- Receives serialized `UnifiedQueryRequest` dict (with UUIDs as strings for JSON serialization)
- Routes to the `graph` queue (same workers with access to GraphRAG data volume)
- `soft_time_limit=300` (5 min), `time_limit=360` (6 min)
- No Redis lock needed (concurrent queries are fine)

The task:
1. Deserializes the request
2. Calls the appropriate search function (`local_search`, `global_search`, `drift_search`, `basic_search`) from `graphrag_service` — these are sync functions that internally use `_run_async()` with `asyncio.new_event_loop()`, which works fine in Celery's threaded workers (same pattern as `run_graphrag_indexing_task`)
3. Constructs the `QueryResultItem` list and `UnifiedQueryResponse` envelope — replicating the response-building logic from the async `_graphrag_*_query` wrappers in `retrieval.py` (modality, context source label, score). This is straightforward since each wrapper is ~15 lines.
4. Applies min_confidence filtering on the result list
5. Returns the response dict via `UnifiedQueryResponse.model_dump(mode="json")` to ensure UUIDs serialize as strings for the Redis JSON result backend

Note: `_backfill_content_text` and `_populate_image_urls` are not needed — GraphRAG results return modality `"graphrag_response"` with `content_text` already populated from the LLM response, and no image URLs apply.

Precondition errors (indexing not complete, no community reports) are caught and returned as a result dict with an `error` field, so the status endpoint surfaces them as `failed` with the message.

### Job tracking

On submit, store a tracking key in Redis: `SET graphrag:job:{job_id} 1 EX 86400`. The status endpoint checks for this key first — if absent, return 404 ("Job not found or expired"). This distinguishes a real pending task from a garbage/typo job ID (Celery returns `PENDING` for both).

## Task Routing

Add to `celery_app.py` task_routes:
```python
"app.workers.graphrag_tasks.run_graphrag_query_task": {"queue": "graph"},
```

No change to `include` needed — `app.workers.graphrag_tasks` is already in the include list.

## Frontend Changes

### `frontend/src/api/client.ts`

Three new functions:
- `submitGraphRAGQuery(params)` -> POST `/v1/retrieval/graphrag/submit`
- `getGraphRAGQueryStatus(jobId)` -> GET `/v1/retrieval/graphrag/status/{job_id}`
- `getGraphRAGQueryResult(jobId)` -> GET `/v1/retrieval/graphrag/result/{job_id}`

No AbortController timeout needed on any of these (all return quickly).

### `frontend/src/components/QueryPage.tsx`

Modify `handleQuery`:
- If strategy is `graphrag_*`: submit -> poll status with backoff (1s, 2s, 4s, 8s, capped at 10s) -> on `completed` fetch result -> set results
- If strategy is `basic` or `hybrid`: unchanged (sync `unifiedQuery` call)
- On `failed`: display error from status response
- Polling uses chained `setTimeout` (not `setInterval`) for backoff; pending timeout cleared on unmount
- Loading spinner stays active during polling (same UX)

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid strategy on submit | 400 error |
| Indexing not complete | Task returns error, status shows `failed` with message |
| Task timeout (>5 min) | Celery `SoftTimeLimitExceeded`, status shows `failed` |
| Worker crash mid-query | `task_reject_on_worker_lost=True` rejects immediately back to queue; polling continues |
| Result fetched before completion | 409 "Job still in progress" |
| Job ID not found / expired | 404 "Job not found or expired" |
| Frontend navigates away during poll | `clearInterval` on unmount, task completes in background |
| Bogus/typo job ID | 404 via Redis tracking key check (not Celery's phantom PENDING) |
| Rapid-fire submissions | No explicit rate limit — `graph` queue worker count (1-2) naturally limits parallelism; excess tasks queue |

## What Doesn't Change

- `POST /v1/retrieval/query` remains functional for all strategies (backward compatible)
- Non-GraphRAG query paths (`basic`, `hybrid`) stay synchronous
- GraphRAG service layer (`graphrag_service.py`) is called identically
- Indexing, auto-tuning, document ingest pipelines untouched
- No new infrastructure (reuses existing Celery workers, Redis, `graph` queue)

## Files Changed

| File | Change |
|------|--------|
| `app/workers/graphrag_tasks.py` | Add `run_graphrag_query_task` |
| `app/api/v1/retrieval.py` | 3 new endpoints (submit, status, result) |
| `app/schemas/retrieval.py` | 2 new schemas |
| `app/workers/celery_app.py` | Task route for new task |
| `frontend/src/api/client.ts` | 3 new API functions |
| `frontend/src/components/QueryPage.tsx` | Async polling for graphrag strategies |
