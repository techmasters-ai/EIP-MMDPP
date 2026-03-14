# Codebase Simplification Plan (Approach B — Moderate Refactor)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Remove dead code, eliminate defensive over-engineering, consolidate duplicated patterns, and move configuration out of code — across the entire codebase.

**Architecture:** Surgical changes to existing files. No new services, no API changes, no schema migrations. All changes are internal refactors that preserve existing behavior.

**Tech Stack:** Python 3.11, Pydantic-settings, SQLAlchemy, Celery, Qdrant, Neo4j, YAML

---

### Task 1: Remove dead code from reranker and extraction

**Files:**
- Modify: `app/services/reranker.py:20-23` (delete `_get_settings_safe`)
- Modify: `app/services/reranker.py:31-32,55-56` (replace `_get_settings_safe()` → `get_settings()`, replace `getattr` → direct access)
- Modify: `app/services/extraction.py:327-358` (delete dead `chunk_text` function)
- Test: `tests/unit/test_reranker.py` (verify tests still pass)

**Step 1: Delete `_get_settings_safe` and fix getattr in reranker.py**

In `app/services/reranker.py`:
- Delete lines 20-23 (`_get_settings_safe` function)
- Line 31: `_get_settings_safe()` → `get_settings()`; `getattr(settings, "reranker_model", ...)` → `settings.reranker_model`
- Line 32: `getattr(settings, "reranker_device", ...)` → `settings.reranker_device`
- Line 55: `_get_settings_safe()` → `get_settings()`
- Line 56: `getattr(settings, "reranker_enabled", True)` → `settings.reranker_enabled`

```python
@lru_cache(maxsize=1)
def _get_reranker_model():
    """Load and cache the cross-encoder reranker model."""
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    logger.info("Loading reranker model: %s (device=%s)", settings.reranker_model, settings.reranker_device)
    model = CrossEncoder(settings.reranker_model, device=settings.reranker_device)
    logger.info("Reranker model loaded")
    return model


def rerank(query, candidates, top_k=10):
    settings = get_settings()
    if not settings.reranker_enabled:
        return candidates
    # ... rest unchanged
```

**Step 2: Delete dead `chunk_text` function from extraction.py**

Delete `app/services/extraction.py` lines 327-358 (the `chunk_text` function). It was replaced by `structure_aware_chunk` in `app/services/chunking.py` and is no longer imported anywhere.

**Step 3: Run tests**

Run: `python -m pytest tests/unit/test_reranker.py -v`
Expected: All 4 tests pass

**Step 4: Verify no remaining references to deleted code**

Run: `grep -rn "chunk_text\|_get_settings_safe" app/ --include="*.py" | grep -v "chunk_text=" | grep -v "\.chunk_text" | grep -v "content_text"`
Expected: No hits for `_get_settings_safe`; no imports of `chunk_text` from `extraction`

**Step 5: Commit**

```bash
git add app/services/reranker.py app/services/extraction.py
git commit -m "chore: remove dead _get_settings_safe and chunk_text functions"
```

---

### Task 2: Remove dead config settings

**Files:**
- Modify: `app/config.py:21,87-88,149-155,192-210` (delete dead fields)
- Modify: `app/api/v1/_retrieval_helpers.py:70-102` (update after config changes)
- Test: Run full unit suite

**Step 1: Delete dead settings from config.py**

Remove these fields from `app/config.py`:
- Line 21: `secret_key: str = "change-me"` — unused (JWT uses its own key)
- Line 87: `ollama_vlm_model: str = "llava"` — deprecated comment says so
- Line 88: `ollama_embedding_model: str = "nomic-embed-text"` — unused (BGE is the embedding model)
- Lines 149-151: `ocr_tesseract_confidence_threshold` and `ocr_easyocr_confidence_threshold` — hardcoded in `extraction.py` function signatures, never read from config
- Line 155: `abac_policy_path: str = "/app/policy/abac.yaml"` — Phase 3 future, no code references it

**Step 2: Remove duplicate ontology relation weights (lines 192-210)**

Lines 192-210 define 18 `retrieval_onto_weight_*` fields (e.g. `associated_with`, `installed_on`, etc.) that duplicate the purpose of lines 230-239. The lines 192-210 weights use lowercase snake_case keys while lines 230-239 use UPPER_CASE keys.

In `_retrieval_helpers.py:70-102`, `get_ontology_relation_weights()` maps both sets. After removing lines 192-210 from config, update the helper to load these weights from the ontology YAML instead (Task 6 handles this — for now just delete the config fields and their references in the helper).

For this task: delete lines 192-210 from config.py, and remove the "New ontology relation weights" block (lines 83-101) from `_retrieval_helpers.py:get_ontology_relation_weights()`.

**Step 3: Run tests**

Run: `python -m pytest tests/unit/ -v`
Expected: All pass (no code references deleted fields)

**Step 4: Verify no remaining references**

Run: `grep -rn "secret_key\|ollama_vlm_model\|ollama_embedding_model\|ocr_tesseract_confidence\|ocr_easyocr_confidence\|abac_policy_path" app/ --include="*.py"`
Expected: No hits

**Step 5: Commit**

```bash
git add app/config.py app/api/v1/_retrieval_helpers.py
git commit -m "chore: remove unused config fields (secret_key, ollama_vlm/embed, ocr thresholds, abac_policy)"
```

---

### Task 3: Clean up defensive getattr in retrieval.py

**Files:**
- Modify: `app/api/v1/retrieval.py:152-162` (replace getattr with direct access)

**Step 1: Fix `_apply_reranker` getattr usage**

In `app/api/v1/retrieval.py` lines 152-162, `QueryResultItem` is a Pydantic model — its fields are guaranteed. Replace:

```python
# Before (lines 152-162)
rerank_input = [
    {
        "chunk_id": str(getattr(r, "chunk_id", "")),
        "content_text": getattr(r, "content_text", "") or "",
        "score": getattr(r, "score", 0.0),
        "artifact_id": getattr(r, "artifact_id", None),
        "document_id": getattr(r, "document_id", None),
        "modality": getattr(r, "modality", "text"),
        "page_number": getattr(r, "page_number", None),
        "classification": getattr(r, "classification", "UNCLASSIFIED"),
    }
    for r in results[:_s.reranker_top_n]
]
```

With:

```python
# After
rerank_input = [
    {
        "chunk_id": str(r.chunk_id or ""),
        "content_text": r.content_text or "",
        "score": r.score,
        "artifact_id": r.artifact_id,
        "document_id": r.document_id,
        "modality": r.modality,
        "page_number": r.page_number,
        "classification": r.classification,
    }
    for r in results[:_s.reranker_top_n]
]
```

**Step 2: Fix getattr in `_rescore_expanded_chunks`**

Line 326: `getattr(c, "context", None)` → `c.context` (it's a field on `QueryResultItem`)

**Step 3: Run tests**

Run: `python -m pytest tests/unit/test_retrieval_pipeline.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "chore: replace defensive getattr with direct attribute access in retrieval"
```

---

### Task 4: Consolidate Qdrant store functions

**Files:**
- Modify: `app/services/qdrant_store.py:118-319` (consolidate 8 functions → generic)
- Modify: callers in `app/workers/pipeline.py`, `app/api/v1/retrieval.py`, `app/workers/trusted_data_tasks.py`
- Test: `tests/unit/test_qdrant_store.py` (if exists), full unit suite

**Step 1: Write generic upsert and search functions**

Replace the 4 upsert functions (`upsert_text_vector`, `upsert_image_vector`, `upsert_text_vectors_batch`, `upsert_image_vectors_batch`) with 2 generic ones:

```python
def upsert_vectors(client, collection: str, points: list[PointStruct]) -> None:
    """Upsert points to any collection."""
    client.upsert(collection_name=collection, points=points)


def upsert_vector(
    client,
    collection: str,
    point_id: uuid.UUID | str,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    """Upsert a single point to any collection."""
    upsert_vectors(client, collection, [
        PointStruct(id=str(point_id), vector=vector, payload=payload)
    ])
```

Replace the 4 search functions (`search_text_vectors`, `search_image_vectors`, `search_text_vectors_async`, `search_image_vectors_async`) with 2 generic ones:

```python
def search_vectors(
    client, collection: str, query_vector: list[float],
    limit: int = 20, filters: dict | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search any collection (sync)."""
    qdrant_filter = _build_filter(filters) if filters else None
    kwargs = dict(collection_name=collection, query=query_vector,
                  limit=limit, query_filter=qdrant_filter, with_payload=True)
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    results = client.query_points(**kwargs)
    return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in results.points]


async def search_vectors_async(
    client, collection: str, query_vector: list[float],
    limit: int = 20, filters: dict | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search any collection (async)."""
    qdrant_filter = _build_filter(filters) if filters else None
    kwargs = dict(collection_name=collection, query=query_vector,
                  limit=limit, query_filter=qdrant_filter, with_payload=True)
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    results = await client.query_points(**kwargs)
    return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in results.points]
```

**Step 2: Add backward-compat aliases (thin wrappers)**

Keep the old function names as thin wrappers that pass the collection name, so callers can migrate incrementally:

```python
def upsert_text_vectors_batch(client, points):
    upsert_vectors(client, get_settings().qdrant_text_collection, points)

def search_text_vectors_async(client, query_vector, limit=20, filters=None, score_threshold=None):
    return search_vectors_async(client, get_settings().qdrant_text_collection, query_vector, limit, filters, score_threshold)
# ... etc for each old name
```

**Step 3: Run tests**

Run: `python -m pytest tests/unit/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add app/services/qdrant_store.py
git commit -m "refactor: consolidate 8 Qdrant functions into generic upsert/search with aliases"
```

---

### Task 5: Extract shared `_get_db` context manager for workers

**Files:**
- Create: `app/workers/_db.py`
- Modify: `app/workers/pipeline.py:109-112` (import from `_db.py`)
- Modify: `app/workers/trusted_data_tasks.py:14-17` (import from `_db.py`)
- Test: Run full unit suite

**Step 1: Create shared helper**

Create `app/workers/_db.py`:

```python
"""Shared DB session helper for Celery workers."""

from contextlib import contextmanager


def get_worker_db():
    """Get a synchronous DB session for Celery worker use."""
    from app.db.session import get_sync_session
    return get_sync_session()


@contextmanager
def worker_db_session():
    """Context manager that yields a sync DB session and closes it."""
    db = get_worker_db()
    try:
        yield db
    finally:
        db.close()
```

**Step 2: Update pipeline.py**

Replace `_get_db()` definition (lines 109-112) with import:

```python
from app.workers._db import get_worker_db as _get_db
```

**Step 3: Update trusted_data_tasks.py**

Replace `_get_db()` definition (lines 14-17) with import:

```python
from app.workers._db import get_worker_db as _get_db
```

**Step 4: Run tests**

Run: `python -m pytest tests/unit/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git add app/workers/_db.py app/workers/pipeline.py app/workers/trusted_data_tasks.py
git commit -m "refactor: extract shared _get_db helper for Celery workers"
```

---

### Task 6: Move ontology relation weights from config to YAML

**Files:**
- Modify: `ontology/base.yaml` (add `scoring_weights` section)
- Modify: `app/api/v1/_retrieval_helpers.py:70-102` (load from YAML instead of config)
- Modify: `app/config.py:229-239` (delete `retrieval_onto_weight_*` fields)
- Remove from `.env`: all `RETRIEVAL_ONTO_WEIGHT_*` lines
- Test: `tests/unit/test_retrieval_helpers.py` or relevant tests

**Step 1: Add scoring weights to ontology YAML**

Add to `ontology/base.yaml` at the top level:

```yaml
scoring_weights:
  # Ontology relation weights for retrieval fusion scoring
  IS_VARIANT_OF: 0.95
  USES_COMPONENT: 0.92
  IS_SUBSYSTEM_OF: 0.90
  CONTAINS: 0.90
  PART_OF: 0.90
  INTERFACES_WITH: 0.85
  OPERATES_ON: 0.85
  MEETS_STANDARD: 0.80
  RELATED_TO: 0.75
  default: 0.70
```

**Step 2: Update `get_ontology_relation_weights()` to load from YAML**

```python
import yaml
from functools import lru_cache
from pathlib import Path

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "ontology" / "base.yaml"

@lru_cache(maxsize=1)
def _load_scoring_weights() -> dict[str, float]:
    """Load ontology relation scoring weights from base.yaml."""
    with open(_ONTOLOGY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("scoring_weights", {})


def get_ontology_relation_weights() -> dict[str, float]:
    return _load_scoring_weights()
```

**Step 3: Delete the 10 `retrieval_onto_weight_*` fields from config.py (lines 229-239)**

Also delete `retrieval_onto_weight_default` (line 239).

**Step 4: Remove `RETRIEVAL_ONTO_WEIGHT_*` lines from `.env`**

Remove lines 126-135 from `.env`.

**Step 5: Run tests**

Run: `python -m pytest tests/unit/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add ontology/base.yaml app/api/v1/_retrieval_helpers.py app/config.py .env
git commit -m "refactor: move ontology relation weights from env config to ontology YAML"
```

---

### Task 7: Consolidate retrieval.py lookup patterns and _retrieval_helpers cleanup

**Files:**
- Modify: `app/api/v1/retrieval.py:910-947` (merge `_lookup_text_chunk` and `_lookup_image_chunk`)
- Modify: `app/api/v1/_retrieval_helpers.py:36-57` (remove `_LazyFloat` class)
- Modify: `app/api/v1/_retrieval_helpers.py:150-193` (merge `build_text_filters` and `build_image_filters`)

**Step 1: Merge the two lookup functions**

`_lookup_text_chunk` (lines 910-927) and `_lookup_image_chunk` (lines 930-947) are identical except for the table name. Merge into one:

```python
async def _lookup_chunk(
    db: AsyncSession, chunk_id: str, table: str = "text_chunks",
    include_context: bool = True,
) -> QueryResultItem | None:
    sql = text(f"""
        SELECT id, artifact_id, document_id, chunk_text, modality,
               page_number, classification
        FROM retrieval.{table} WHERE id = :cid
    """)
    result = await db.execute(sql, {"cid": chunk_id})
    row = result.fetchone()
    if not row:
        return None
    return QueryResultItem(
        chunk_id=row[0], artifact_id=row[1], document_id=row[2],
        score=0.0, modality=row[4],
        content_text=row[3] if include_context else None,
        page_number=row[5], classification=row[6],
    )
```

Update `_lookup_chunk_by_type` to use it:

```python
async def _lookup_chunk_by_type(db, chunk_id, chunk_type, include_context=True):
    table = "image_chunks" if chunk_type == "image_chunk" else "text_chunks"
    return await _lookup_chunk(db, chunk_id, table, include_context)
```

Note: the table name is internal (not user-supplied), so the f-string is safe.

**Step 2: Remove `_LazyFloat` class**

The `_LazyFloat` class (lines 36-57) and module-level `CROSS_MODAL_DECAY`/`ONTOLOGY_DECAY` are unused — `get_cross_modal_decay()` and `get_ontology_decay()` are called directly. Delete lines 36-57.

**Step 3: Merge `build_text_filters` and `build_image_filters`**

These two functions (lines 150-193) are identical except for the table alias prefix (`tc.` vs `ic.`). Merge:

```python
def build_filters(body: UnifiedQueryRequest, alias: str = "tc") -> tuple[str, dict]:
    """Return (WHERE clause suffix, bind params dict)."""
    clauses = ""
    params: dict = {}
    if body.filters:
        if body.filters.classification:
            clauses += f" AND {alias}.classification = :filter_classification"
            params["filter_classification"] = body.filters.classification
        if body.filters.document_ids:
            clauses += f" AND {alias}.document_id = ANY(:filter_doc_ids)"
            params["filter_doc_ids"] = [str(d) for d in body.filters.document_ids]
        if body.filters.modalities:
            clauses += f" AND {alias}.modality = ANY(:filter_modalities)"
            params["filter_modalities"] = body.filters.modalities
        if body.filters.source_ids:
            clauses += f" AND {alias}.document_id IN (SELECT id FROM ingest.documents WHERE source_id = ANY(:filter_source_ids))"
            params["filter_source_ids"] = [str(s) for s in body.filters.source_ids]
    return clauses, params
```

Keep old names as aliases for backward compat:

```python
def build_text_filters(body): return build_filters(body, "tc")
def build_image_filters(body): return build_filters(body, "ic")
```

**Step 4: Run tests**

Run: `python -m pytest tests/unit/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git add app/api/v1/retrieval.py app/api/v1/_retrieval_helpers.py
git commit -m "refactor: consolidate duplicate lookup/filter patterns in retrieval"
```

---

### Task 8: Consolidate Neo4j mock helpers in tests

**Files:**
- Modify: `tests/conftest.py` (already has `mock_neo4j_driver` and `mock_neo4j_async_driver`)
- Modify: `tests/unit/test_neo4j_graph_operations.py:18-28` (delete local `_mock_driver`, use conftest fixture)
- Modify: `tests/unit/test_graphrag_service.py:18-28` (delete local `_mock_driver`, use conftest fixture)

**Step 1: Update test_neo4j_graph_operations.py**

Delete the local `_mock_driver` function (lines 18-28). Update all test methods to accept `mock_neo4j_driver` fixture and unpack `driver, session = mock_neo4j_driver`.

**Step 2: Update test_graphrag_service.py**

Same approach — delete local `_mock_driver`, use the `mock_neo4j_driver` fixture from conftest.

**Step 3: Run affected tests**

Run: `python -m pytest tests/unit/test_neo4j_graph_operations.py tests/unit/test_graphrag_service.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/conftest.py tests/unit/test_neo4j_graph_operations.py tests/unit/test_graphrag_service.py
git commit -m "test: consolidate Neo4j mock helpers to conftest"
```

---

### Task 9: Use env vars for hardcoded model names in manage.sh

**Files:**
- Modify: `manage.sh:154-195` (replace hardcoded model names with env vars)

**Step 1: Replace hardcoded model names**

In `manage.sh` `cmd_start()`, replace:
- `'models--BAAI--bge-large-en-v1.5'` → use `TEXT_EMBEDDING_MODEL` env var (already loaded from `.env`)
- `'models--BAAI--bge-reranker-v2-m3'` → use `RERANKER_MODEL` env var
- `"llama3.1:8b"` → use `DOCLING_GRAPH_MODEL` env var

The HuggingFace cache directory uses `models--ORG--NAME` format (dashes replace slashes). Convert at runtime:

```bash
local text_model_dir
text_model_dir="models--$(echo "${TEXT_EMBEDDING_MODEL}" | tr '/' '--')"

local reranker_model_dir
reranker_model_dir="models--$(echo "${RERANKER_MODEL}" | tr '/' '--')"
```

For Ollama, replace `"llama3.1:8b"` with `"${DOCLING_GRAPH_MODEL}"`.

**Step 2: Test manually**

Run: `./manage.sh --help` (verify no syntax errors)

**Step 3: Commit**

```bash
git add manage.sh
git commit -m "chore: use env vars for model names in manage.sh instead of hardcoding"
```

---

### Task 10: Run full test suite and update README

**Files:**
- Test: `./scripts/run_tests.sh` (full suite)
- Modify: `README.md` (update if needed)

**Step 1: Run full test suite**

Run: `python -m pytest tests/unit/ -v`
Expected: All 367+ tests pass

**Step 2: Grep for any remaining dead references**

Run: `grep -rn "_get_settings_safe\|chunk_text.*import\|ollama_vlm_model\|ollama_embedding_model\|ocr_tesseract_confidence\|abac_policy_path\|secret_key.*change-me" app/ tests/ --include="*.py"`
Expected: No hits

**Step 3: Update README if needed**

If any user-facing changes were made (none expected), update README.md.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: verify simplification pass — all tests passing"
```
