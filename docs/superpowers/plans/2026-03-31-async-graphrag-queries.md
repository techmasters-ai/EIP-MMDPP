# Async GraphRAG Queries Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 4 GraphRAG query strategies from synchronous HTTP to async Celery jobs with submit/poll/fetch endpoints, eliminating browser timeout errors.

**Architecture:** Frontend submits query -> receives job_id immediately -> polls status with exponential backoff -> fetches result when complete. Backend dispatches query as a Celery task on the `graph` queue, stores result in Redis via Celery result backend (24h TTL). Redis tracking key distinguishes real jobs from bogus IDs.

**Tech Stack:** FastAPI, Celery + Redis, React/TypeScript

**Spec:** `docs/superpowers/specs/2026-03-31-async-graphrag-queries-design.md`

---

## Chunk 1: Backend — Schemas, Celery Task, API Endpoints

### Task 1: Add response schemas

**Files:**
- Modify: `app/schemas/retrieval.py`

- [ ] **Step 1: Add the two new schemas at the end of `app/schemas/retrieval.py`**

After the existing `DoclingDocumentResponse` class, add:

```python
class GraphRAGJobSubmitResponse(APIModel):
    job_id: str
    status: str  # always "pending"


class GraphRAGJobStatusResponse(APIModel):
    job_id: str
    status: str  # pending | running | completed | failed
    error: Optional[str] = None
```

- [ ] **Step 2: Run schema validation tests**

Run: `pytest tests/unit/test_retrieval_schemas.py -v`
Expected: All existing tests PASS, no regressions.

- [ ] **Step 3: Commit**

```bash
git add app/schemas/retrieval.py
git commit -m "feat: add GraphRAG async job response schemas"
```

---

### Task 2: Add Celery task for GraphRAG queries

**Files:**
- Modify: `app/workers/graphrag_tasks.py`
- Modify: `app/workers/celery_app.py`

- [ ] **Step 1: Add `run_graphrag_query_task` to `app/workers/graphrag_tasks.py`**

Add at the end of the file, after `run_graphrag_auto_tune_task`:

```python
_GRAPHRAG_SEARCH_FN = {
    "graphrag_local": "local_search",
    "graphrag_global": "global_search",
    "graphrag_drift": "drift_search",
    "graphrag_basic": "basic_search",
}

@celery_app.task(soft_time_limit=300, time_limit=360)
def run_graphrag_query_task(request_dict: dict) -> dict:
    """Run a GraphRAG query as an async Celery task.

    Returns a serialized UnifiedQueryResponse dict on success,
    or a dict with an 'error' key on failure.
    """
    from app.services import graphrag_service

    strategy = request_dict.get("strategy", "")
    query_text = request_dict.get("query_text", "")
    min_confidence = request_dict.get("min_confidence")

    fn_name = _GRAPHRAG_SEARCH_FN.get(strategy)
    if not fn_name:
        return {"error": f"Invalid GraphRAG strategy: {strategy}"}

    if not query_text:
        return {"error": "query_text is required for GraphRAG queries"}

    search_fn = getattr(graphrag_service, fn_name)
    try:
        graphrag_result = search_fn(query_text)
    except Exception as exc:
        logger.exception("GraphRAG %s query failed", strategy)
        return {"error": str(exc)}

    response = graphrag_result.get("response", "")
    if not response:
        error_key = graphrag_result.get("error", "")
        if error_key == "communities_not_indexed":
            return {
                "error": "GraphRAG indexing has not completed yet. "
                "Run indexing and wait for community detection to finish.",
            }
        # drift and basic return empty on no results; local/global are errors
        if strategy in ("graphrag_local", "graphrag_global"):
            source_label = strategy
            return {"error": f"GraphRAG {source_label}: no results found."}
        # drift/basic: return empty results
        return _build_response(strategy, query_text, [])

    result_item = {
        "score": 1.0,
        "modality": "graphrag_response",
        "content_text": response,
        "classification": "UNCLASSIFIED",
        "context": {
            "source": strategy,
            "graphrag_context": graphrag_result.get("context", {}),
        },
    }

    results = [result_item]

    # Apply min_confidence filter
    if min_confidence is not None:
        results = [r for r in results if r["score"] >= min_confidence]

    return _build_response(strategy, query_text, results)


def _build_response(strategy: str, query_text: str, results: list[dict]) -> dict:
    return {
        "query_text": query_text,
        "query_image": None,
        "strategy": strategy,
        "modality_filter": "all",
        "results": results,
        "total": len(results),
    }
```

- [ ] **Step 2: Add task route in `app/workers/celery_app.py`**

In the `task_routes` dict, add after the existing `run_graphrag_auto_tune_task` entry:

```python
"app.workers.graphrag_tasks.run_graphrag_query_task": {"queue": "graph"},
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `pytest tests/unit/ -v -k "graphrag" --timeout=30`
Expected: All existing graphrag tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app/workers/graphrag_tasks.py app/workers/celery_app.py
git commit -m "feat: add Celery task for async GraphRAG queries"
```

---

### Task 3: Add the 3 API endpoints

**Files:**
- Modify: `app/api/v1/retrieval.py`

- [ ] **Step 1: Add imports at the top of `app/api/v1/retrieval.py`**

Add to the existing imports from `app.schemas.retrieval`:

```python
from app.schemas.retrieval import (
    # ... existing imports ...
    GraphRAGJobSubmitResponse,
    GraphRAGJobStatusResponse,
)
```

- [ ] **Step 2: Add the 3 endpoints before the existing `trigger_graphrag_indexing` endpoint**

Add these endpoints in `app/api/v1/retrieval.py`. Find the section with `@router.post("/graphrag/index")` and add these before it:

```python
# ---------------------------------------------------------------------------
# Async GraphRAG query — submit / status / result
# ---------------------------------------------------------------------------

_GRAPHRAG_STRATEGIES = {"graphrag_local", "graphrag_global", "graphrag_drift", "graphrag_basic"}


@router.post("/retrieval/graphrag/submit", response_model=GraphRAGJobSubmitResponse,
             status_code=202)
async def submit_graphrag_query(body: UnifiedQueryRequest):
    """Submit a GraphRAG query as an async job. Returns a job_id for polling."""
    if body.strategy.value not in _GRAPHRAG_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy must be one of {sorted(_GRAPHRAG_STRATEGIES)}, "
            f"got '{body.strategy.value}'",
        )

    from app.workers.graphrag_tasks import run_graphrag_query_task

    request_dict = body.model_dump(mode="json")
    task = run_graphrag_query_task.delay(request_dict)

    # Track job ID in Redis so we can distinguish real jobs from bogus IDs
    import redis
    from app.config import get_settings
    settings = get_settings()
    try:
        r = redis.from_url(settings.celery_broker_url)
        r.set(f"graphrag:job:{task.id}", "1", ex=86400)
    except Exception:
        pass  # Non-fatal: status endpoint falls back to Celery state

    return GraphRAGJobSubmitResponse(job_id=str(task.id), status="pending")


@router.get("/retrieval/graphrag/status/{job_id}", response_model=GraphRAGJobStatusResponse)
async def get_graphrag_query_status(job_id: str):
    """Poll the status of an async GraphRAG query job."""
    import redis
    from app.config import get_settings
    from app.workers.celery_app import celery_app

    settings = get_settings()

    # Check tracking key first — reject bogus IDs
    try:
        r = redis.from_url(settings.celery_broker_url)
        if not r.exists(f"graphrag:job:{job_id}"):
            raise HTTPException(status_code=404, detail="Job not found or expired")
    except HTTPException:
        raise
    except Exception:
        pass  # Redis unavailable — fall through to Celery check

    result = celery_app.AsyncResult(job_id)
    state = result.state  # PENDING, STARTED, SUCCESS, FAILURE, REVOKED

    status_map = {
        "PENDING": "pending",
        "STARTED": "running",
        "SUCCESS": "completed",
        "FAILURE": "failed",
        "REVOKED": "failed",
    }
    status = status_map.get(state, "pending")

    error = None
    if state == "FAILURE":
        error = str(result.result) if result.result else "Task failed"
    elif state == "SUCCESS" and isinstance(result.result, dict):
        error = result.result.get("error")
        if error:
            status = "failed"

    return GraphRAGJobStatusResponse(job_id=job_id, status=status, error=error)


@router.get("/retrieval/graphrag/result/{job_id}")
async def get_graphrag_query_result(job_id: str):
    """Fetch the result of a completed async GraphRAG query job."""
    import redis
    from app.config import get_settings
    from app.workers.celery_app import celery_app

    settings = get_settings()

    # Check tracking key
    try:
        r = redis.from_url(settings.celery_broker_url)
        if not r.exists(f"graphrag:job:{job_id}"):
            raise HTTPException(status_code=404, detail="Job not found or expired")
    except HTTPException:
        raise
    except Exception:
        pass

    result = celery_app.AsyncResult(job_id)

    if result.state != "SUCCESS":
        raise HTTPException(status_code=409, detail="Job still in progress")

    data = result.result
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Unexpected result format")

    if "error" in data:
        raise HTTPException(status_code=422, detail=data["error"])

    return data
```

- [ ] **Step 3: Run existing retrieval tests**

Run: `pytest tests/unit/test_query_coverage.py tests/unit/test_retrieval_helpers.py -v --timeout=30`
Expected: All existing tests PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "feat: add submit/status/result endpoints for async GraphRAG queries"
```

---

## Chunk 2: Frontend — API Client, QueryPage Polling

### Task 4: Add frontend API functions

**Files:**
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Add TypeScript interfaces and API functions**

Add after the existing `unifiedQuery` function (around line 282), before the `// Graph Store` section:

```typescript
// ---------------------------------------------------------------------------
// Async GraphRAG Query (submit / poll / fetch)
// ---------------------------------------------------------------------------

export interface GraphRAGJobSubmitResponse {
  job_id: string;
  status: string;
}

export interface GraphRAGJobStatusResponse {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  error?: string | null;
}

export async function submitGraphRAGQuery(params: {
  query_text?: string;
  strategy: QueryStrategy;
  top_k?: number;
  min_confidence?: number;
  include_context?: boolean;
}): Promise<GraphRAGJobSubmitResponse> {
  const res = await fetch("/v1/retrieval/graphrag/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 10, include_context: true, ...params }),
  });
  return handleResponse<GraphRAGJobSubmitResponse>(res);
}

export async function getGraphRAGQueryStatus(
  jobId: string,
): Promise<GraphRAGJobStatusResponse> {
  const res = await fetch(`/v1/retrieval/graphrag/status/${jobId}`);
  return handleResponse<GraphRAGJobStatusResponse>(res);
}

export async function getGraphRAGQueryResult(
  jobId: string,
): Promise<UnifiedQueryResponse> {
  const res = await fetch(`/v1/retrieval/graphrag/result/${jobId}`);
  return handleResponse<UnifiedQueryResponse>(res);
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat: add async GraphRAG query API functions"
```

---

### Task 5: Update QueryPage to use async polling for GraphRAG strategies

**Files:**
- Modify: `frontend/src/components/QueryPage.tsx`

- [ ] **Step 1: Add imports**

Update the import line at the top of `QueryPage.tsx`:

```typescript
import {
  unifiedQuery,
  submitGraphRAGQuery,
  getGraphRAGQueryStatus,
  getGraphRAGQueryResult,
  getGraphNeighborhood,
  getRetrievalSettings,
  type QueryStrategy,
  type ModalityFilter,
  type QueryResultItem,
} from "../api/client";
```

- [ ] **Step 2: Add a ref for timeout cleanup and a helper to check if a strategy is GraphRAG**

Add after the existing state declarations inside the component (near the top of the component function body, after the `useState` calls):

```typescript
const pollTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

// Cleanup polling on unmount
useEffect(() => {
  return () => {
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
    }
  };
}, []);

const isGraphRAG = (strategy: QueryStrategy) =>
  strategy.startsWith("graphrag_");
```

- [ ] **Step 3: Replace the `handleQuery` function body**

Replace the existing `handleQuery` async function with:

```typescript
const handleQuery = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!hasQuery) return;

  setLoading(true);
  setError(null);
  setResults(null);
  setTotalResults(0);
  setElapsed(null);
  const t0 = performance.now();

  // Cancel any in-flight polling from a previous query
  if (pollTimeoutRef.current) {
    clearTimeout(pollTimeoutRef.current);
    pollTimeoutRef.current = null;
  }

  try {
    if (isGraphRAG(selected.strategy)) {
      // Async path: submit -> poll -> fetch
      const { job_id } = await submitGraphRAGQuery({
        query_text: queryText.trim() || undefined,
        strategy: selected.strategy,
        top_k: topK,
        min_confidence: minConfidence,
        include_context: true,
      });

      // Poll with exponential backoff: 1s, 2s, 4s, 8s, capped at 10s
      let delay = 1000;
      const poll = async () => {
        try {
          const status = await getGraphRAGQueryStatus(job_id);
          if (status.status === "completed") {
            const res = await getGraphRAGQueryResult(job_id);
            setResults(res.results);
            setTotalResults(res.total);
            setElapsed(Math.round(performance.now() - t0));
            setLoading(false);
          } else if (status.status === "failed") {
            setError(status.error || "GraphRAG query failed");
            setLoading(false);
          } else {
            // Still pending/running — schedule next poll
            delay = Math.min(delay * 2, 10000);
            pollTimeoutRef.current = setTimeout(() => void poll(), delay);
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : "Polling failed");
          setLoading(false);
        }
      };

      pollTimeoutRef.current = setTimeout(() => void poll(), delay);
    } else {
      // Sync path: basic & hybrid
      const res = await unifiedQuery({
        query_text: queryText.trim() || undefined,
        query_image: queryImage || undefined,
        strategy: selected.strategy,
        modality_filter: selected.strategy === "hybrid" ? modalityFilter : "all",
        top_k: topK,
        reranker_top_n: rerankerTopN,
        min_confidence: minConfidence,
        include_context: true,
      });
      setResults(res.results);
      setTotalResults(res.total);
      setElapsed(Math.round(performance.now() - t0));
    }
  } catch (err) {
    setError(err instanceof Error ? err.message : "Query failed");
  } finally {
    // Only clear loading for sync path; async path clears in poll callback
    if (!isGraphRAG(selected.strategy)) {
      setLoading(false);
    }
  }
};
```

- [ ] **Step 4: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 5: Build frontend**

Run: `cd frontend && npm run build`
Expected: Build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/QueryPage.tsx
git commit -m "feat: async polling for GraphRAG queries in QueryPage"
```

---

## Chunk 3: Testing & Verification

### Task 6: Add unit tests for the new Celery task and endpoints

**Files:**
- Create: `tests/unit/test_graphrag_query_task.py`

- [ ] **Step 1: Create the test file**

```python
"""Unit tests for async GraphRAG query task and endpoints."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# Stub graphrag/pandas if not available (same pattern as test_query_coverage.py)
class _AutoStubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock

for mod_name in [
    "pandas", "graphrag", "graphrag.api", "graphrag.config",
    "graphrag.config.enums", "graphrag.query", "litellm",
    "nest_asyncio2", "graphrag.index.update",
    "graphrag.index.update.incremental_index",
    "graphrag_llm", "graphrag_llm.embedding",
    "graphrag_llm.embedding.lite_llm_embedding",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _AutoStubModule(mod_name)


class TestRunGraphRAGQueryTask:
    """Tests for run_graphrag_query_task Celery task."""

    @patch("app.services.graphrag_service.local_search")
    def test_local_search_success(self, mock_search):
        mock_search.return_value = {
            "response": "Fan Song is a radar system.",
            "context": {"entities": []},
        }
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": "Fan Song"}
        )
        assert result["strategy"] == "graphrag_local"
        assert result["total"] == 1
        assert result["results"][0]["content_text"] == "Fan Song is a radar system."
        assert result["results"][0]["modality"] == "graphrag_response"
        assert "error" not in result

    @patch("app.services.graphrag_service.global_search")
    def test_global_search_success(self, mock_search):
        mock_search.return_value = {
            "response": "Community summary.",
            "context": {},
        }
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_global", "query_text": "overview"}
        )
        assert result["strategy"] == "graphrag_global"
        assert result["total"] == 1

    @patch("app.services.graphrag_service.local_search")
    def test_communities_not_indexed_error(self, mock_search):
        mock_search.return_value = {"response": "", "error": "communities_not_indexed"}
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": "test"}
        )
        assert "error" in result
        assert "indexing has not completed" in result["error"]

    @patch("app.services.graphrag_service.drift_search")
    def test_drift_empty_returns_no_error(self, mock_search):
        mock_search.return_value = {"response": ""}
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_drift", "query_text": "test"}
        )
        assert "error" not in result
        assert result["total"] == 0

    def test_invalid_strategy(self):
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "basic", "query_text": "test"}
        )
        assert "error" in result
        assert "Invalid" in result["error"]

    def test_missing_query_text(self):
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": ""}
        )
        assert "error" in result

    @patch("app.services.graphrag_service.local_search")
    def test_min_confidence_filter(self, mock_search):
        mock_search.return_value = {
            "response": "Answer.",
            "context": {},
        }
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": "test",
             "min_confidence": 1.5}
        )
        assert result["total"] == 0  # score=1.0 < 1.5

    @patch("app.services.graphrag_service.local_search")
    def test_search_exception(self, mock_search):
        mock_search.side_effect = RuntimeError("LLM timeout")
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": "test"}
        )
        assert "error" in result
        assert "LLM timeout" in result["error"]
```

- [ ] **Step 2: Run the new tests**

Run: `pytest tests/unit/test_graphrag_query_task.py -v --timeout=30`
Expected: All tests PASS.

- [ ] **Step 3: Run the full unit test suite**

Run: `pytest tests/unit/ -v --timeout=60`
Expected: All tests PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_graphrag_query_task.py
git commit -m "test: add unit tests for async GraphRAG query task"
```

---

### Task 7: End-to-end verification

- [ ] **Step 1: Rebuild the containers**

```bash
docker compose build api worker worker-graph
docker compose up -d
```

- [ ] **Step 2: Test the submit endpoint**

```bash
curl -s -X POST http://localhost:8005/v1/retrieval/graphrag/submit \
  -H "Content-Type: application/json" \
  -d '{"query_text": "Fan Song", "strategy": "graphrag_local"}' | python3 -m json.tool
```

Expected: `{"job_id": "<uuid>", "status": "pending"}`

- [ ] **Step 3: Test the status endpoint**

Using the job_id from step 2:

```bash
curl -s http://localhost:8005/v1/retrieval/graphrag/status/<job_id> | python3 -m json.tool
```

Expected: `{"job_id": "...", "status": "running"|"completed"|"pending", "error": null}`

- [ ] **Step 4: Test the result endpoint**

Once status is `completed`:

```bash
curl -s http://localhost:8005/v1/retrieval/graphrag/result/<job_id> | python3 -m json.tool
```

Expected: Full `UnifiedQueryResponse` JSON with GraphRAG results.

- [ ] **Step 5: Test bogus job ID returns 404**

```bash
curl -s http://localhost:8005/v1/retrieval/graphrag/status/bogus-id
```

Expected: `{"detail": "Job not found or expired"}` with HTTP 404.

- [ ] **Step 6: Test invalid strategy returns 400**

```bash
curl -s -X POST http://localhost:8005/v1/retrieval/graphrag/submit \
  -H "Content-Type: application/json" \
  -d '{"query_text": "test", "strategy": "basic"}'
```

Expected: HTTP 400.

- [ ] **Step 7: Test from the browser UI**

Open the app, select "GraphRAG Local", enter "Fan Song", submit. Verify:
- No "NetworkError" — query submits instantly
- Loading spinner stays active during polling
- Results appear when done

- [ ] **Step 8: Final commit**

```bash
git commit --allow-empty -m "chore: verified async GraphRAG queries end-to-end"
```
