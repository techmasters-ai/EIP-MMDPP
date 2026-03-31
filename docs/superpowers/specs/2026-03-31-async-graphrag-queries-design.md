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

New file: `app/workers/graphrag_query_tasks.py`

One task: `run_graphrag_query_task`

- Receives serialized `UnifiedQueryRequest` dict
- Routes to the `graph` queue (same workers with access to GraphRAG data volume)
- `soft_time_limit=300` (5 min), `time_limit=360` (6 min)
- No Redis lock needed (concurrent queries are fine)

The task:
1. Deserializes the request
2. Calls the appropriate search function (`local_search`, `global_search`, `drift_search`, `basic_search`) from `graphrag_service`
3. Applies min_confidence filtering
4. Backfills content_text from Postgres
5. Populates presigned image URLs
6. Returns the serialized `UnifiedQueryResponse` dict

Precondition errors (indexing not complete, no community reports) are caught and returned as a result dict with an `error` field, so the status endpoint surfaces them as `failed` with the message.

## Task Routing

Add to `celery_app.py` task_routes:
```python
"app.workers.graphrag_query_tasks.run_graphrag_query_task": {"queue": "graph"},
```

## Frontend Changes

### `frontend/src/api/client.ts`

Three new functions:
- `submitGraphRAGQuery(params)` -> POST `/v1/retrieval/graphrag/submit`
- `getGraphRAGQueryStatus(jobId)` -> GET `/v1/retrieval/graphrag/status/{job_id}`
- `getGraphRAGQueryResult(jobId)` -> GET `/v1/retrieval/graphrag/result/{job_id}`

No AbortController timeout needed on any of these (all return quickly).

### `frontend/src/components/QueryPage.tsx`

Modify `handleQuery`:
- If strategy is `graphrag_*`: submit -> poll status every 3s -> on `completed` fetch result -> set results
- If strategy is `basic` or `hybrid`: unchanged (sync `unifiedQuery` call)
- On `failed`: display error from status response
- Polling interval cleared on component unmount to prevent leaks
- Loading spinner stays active during polling (same UX)

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid strategy on submit | 400 error |
| Indexing not complete | Task returns error, status shows `failed` with message |
| Task timeout (>5 min) | Celery `SoftTimeLimitExceeded`, status shows `failed` |
| Worker crash mid-query | `acks_late` redelivers task, polling continues |
| Result fetched before completion | 409 "Job still in progress" |
| Job ID not found / expired | 404 "Job not found or expired" |
| Frontend navigates away during poll | `clearInterval` on unmount, task completes in background |

## What Doesn't Change

- `POST /v1/retrieval/query` remains functional for all strategies (backward compatible)
- Non-GraphRAG query paths (`basic`, `hybrid`) stay synchronous
- GraphRAG service layer (`graphrag_service.py`) is called identically
- Indexing, auto-tuning, document ingest pipelines untouched
- No new infrastructure (reuses existing Celery workers, Redis, `graph` queue)

## Files Changed

| File | Change |
|------|--------|
| `app/workers/graphrag_query_tasks.py` | **New** - Celery task |
| `app/api/v1/retrieval.py` | 3 new endpoints (submit, status, result) |
| `app/schemas/retrieval.py` | 2 new schemas |
| `app/workers/celery_app.py` | Task route for new task |
| `frontend/src/api/client.ts` | 3 new API functions |
| `frontend/src/components/QueryPage.tsx` | Async polling for graphrag strategies |
