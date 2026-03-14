# Docling-Graph Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Replace hand-rolled LLM extraction with a standalone Docling-Graph Docker service, merge the two ontologies, and make GraphRAG community reports ontology-aware.

**Architecture:** Docling-Graph runs as an independent FastAPI service (port 8002) that receives document text and returns ontology-typed entities/relationships. The EIP-MMDPP worker calls it via HTTP, gates on confidence, and imports into Neo4j. GraphRAG uses ontology weights for community detection and ontology descriptions in report prompts.

**Tech Stack:** docling-graph, LiteLLM, FastAPI, Pydantic v2, NetworkX, httpx, Redis (concurrency gate), Alembic

**Design doc:** `docs/plans/2026-03-14-docling-graph-integration-design.md`

---

### Task 0: Merge Ontologies

**Files:**
- Create: `ontology/ontology.yaml`
- Delete: `ontology/base.yaml`, `ontology/base_v1.yaml`
- Modify: `app/services/ontology_templates.py:17` (change `_ONTOLOGY_PATH`)

**Step 1: Create merged ontology**

Copy `ontology/base.yaml` to `ontology/ontology.yaml`. Update the header:
```yaml
version: "3.0.0"
```

Remove the 5 redundant legacy relationship types from `relationship_types`:
- `IS_SUBSYSTEM_OF` (use `PART_OF`)
- `IMPLEMENTS` (use `PROVIDES`)
- `MEETS_STANDARD` (use `SPECIFIED_BY`)
- `DESCRIBED_IN` (use `MENTIONED_IN`)
- `PERFORMED_BY` (use `OPERATED_BY`)

Remove all `validation_matrix` entries referencing deleted relationship types (lines 1233-1243 in base.yaml).

Add replacement entries where needed:
```yaml
- {source: SUBSYSTEM, relationship: PART_OF, target: RADAR_SYSTEM}
- {source: SUBSYSTEM, relationship: PART_OF, target: MISSILE_SYSTEM}
- {source: EQUIPMENT_SYSTEM, relationship: PROVIDES, target: CAPABILITY}
- {source: SUBSYSTEM, relationship: PROVIDES, target: CAPABILITY}
- {source: EQUIPMENT_SYSTEM, relationship: SPECIFIED_BY, target: STANDARD}
- {source: COMPONENT, relationship: SPECIFIED_BY, target: STANDARD}
- {source: PROCEDURE, relationship: OPERATED_BY, target: ORGANIZATION}
```

Remove the `# Legacy types retained for backward compatibility` comments and section markers.

**Step 2: Update ontology path**

In `app/services/ontology_templates.py:17`, change:
```python
_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "ontology" / "base.yaml"
```
to:
```python
_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "ontology" / "ontology.yaml"
```

**Step 3: Update all references to old ontology files**

Search for `base_v1.yaml` and `base.yaml` in:
- `app/services/ontology_templates.py` — `_ONTOLOGY_PATH` (done above)
- `app/workers/pipeline.py` — any hardcoded paths
- `tests/` — any test fixtures referencing ontology files
- `docs/` — update references in design docs

**Step 4: Delete old files**

```bash
git rm ontology/base.yaml ontology/base_v1.yaml
```

**Step 5: Run existing tests**

```bash
./scripts/run_tests.sh
```

Expected: All tests pass (ontology loading now points to `ontology.yaml`).

**Step 6: Commit**

```bash
git add ontology/ontology.yaml app/services/ontology_templates.py
git commit -m "refactor: merge ontologies into ontology.yaml v3.0.0

Remove 5 redundant legacy relationship types (IS_SUBSYSTEM_OF,
IMPLEMENTS, MEETS_STANDARD, DESCRIBED_IN, PERFORMED_BY) and update
validation matrix with replacement entries."
```

---

### Task 1: Create Docling-Graph Docker Service

**Files:**
- Create: `docker/docling-graph/Dockerfile`
- Create: `docker/docling-graph/requirements.txt`
- Create: `docker/docling-graph/app/__init__.py`
- Create: `docker/docling-graph/app/schemas.py`
- Create: `docker/docling-graph/app/templates.py`
- Create: `docker/docling-graph/app/main.py`

**Step 1: Create requirements.txt**

```
docker/docling-graph/requirements.txt
```

```text
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
docling-graph>=0.1.0
litellm>=1.0.0
networkx>=3.0
pyyaml>=6.0
httpx>=0.27.0
```

**Step 2: Create Pydantic request/response schemas**

```
docker/docling-graph/app/schemas.py
```

```python
"""Request and response models for Docling-Graph extraction service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    document_id: str = Field(description="UUID of the document being extracted")
    text: str = Field(description="Full document text to extract from")
    ontology_version: str | None = Field(
        default=None,
        description="Expected ontology version (logged if mismatched)",
    )


class ExtractedEntityResponse(BaseModel):
    name: str
    entity_type: str
    confidence: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelationshipResponse(BaseModel):
    from_name: str
    from_type: str
    rel_type: str
    to_name: str
    to_type: str
    confidence: float = 1.0


class ExtractionResponse(BaseModel):
    entities: list[ExtractedEntityResponse] = Field(default_factory=list)
    relationships: list[ExtractedRelationshipResponse] = Field(default_factory=list)
    ontology_version: str
    model: str
    provider: str
```

**Step 3: Create YAML-to-template generator**

```
docker/docling-graph/app/templates.py
```

This module reads `ontology.yaml` at startup and generates Pydantic template classes for `docling-graph`. Each entity type becomes a class with `model_config = {'is_entity': True, 'graph_id_fields': [...]}` and `edge()` declarations for its relationships.

```python
"""Generate docling-graph Pydantic templates from ontology YAML."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_ontology(path: str | Path) -> dict:
    """Load and return the ontology YAML as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_templates(ontology: dict) -> dict[str, Any]:
    """Build docling-graph Pydantic template classes from ontology definition.

    Returns a dict mapping entity type name to the generated Pydantic class.
    """
    from pydantic import BaseModel, Field

    try:
        from docling_graph.utils import edge
    except ImportError:
        edge = None
        logger.warning("docling_graph.utils.edge not available; edge declarations skipped")

    entity_defs = {et["name"]: et for et in ontology.get("entity_types", [])}
    rel_defs = ontology.get("relationship_types", [])
    validation_matrix = ontology.get("validation_matrix", [])

    # Build lookup: source_type -> [(rel_type, target_type)]
    source_rels: dict[str, list[tuple[str, str]]] = {}
    for entry in validation_matrix:
        src = entry.get("source", "")
        rel = entry.get("relationship", "")
        tgt = entry.get("target", "")
        if src and rel and tgt:
            source_rels.setdefault(src, []).append((rel, tgt))

    templates: dict[str, Any] = {}

    for etype_name, etype_def in entity_defs.items():
        # Build property fields
        props = etype_def.get("properties", {}).get("properties", {})
        fields: dict[str, Any] = {}
        annotations: dict[str, Any] = {}

        for prop_name, prop_def in props.items():
            ptype = prop_def.get("type", "string")
            py_type = {"string": str, "integer": int, "number": float}.get(ptype, str)
            description = prop_def.get("description", "")
            example = prop_def.get("example")
            pattern = prop_def.get("pattern")

            field_kwargs: dict[str, Any] = {"default": None, "description": description}
            if example is not None:
                field_kwargs["examples"] = [example]
            if pattern:
                field_kwargs["pattern"] = pattern

            annotations[prop_name] = py_type | None
            fields[prop_name] = Field(**field_kwargs)

        # Determine graph_id_fields (use 'name' or first string property)
        id_fields = []
        for candidate in ["name", "system_name", "designation", "title"]:
            if candidate in props:
                id_fields = [candidate]
                break
        if not id_fields and props:
            id_fields = [next(iter(props))]

        # Build model_config
        model_config = {
            "is_entity": True,
            "graph_id_fields": id_fields,
        }

        # Create the class dynamically
        namespace = {"__annotations__": annotations, "model_config": model_config}
        namespace.update(fields)

        cls = type(etype_name, (BaseModel,), namespace)
        cls.__doc__ = etype_def.get("description", "")
        templates[etype_name] = cls

    logger.info("Generated %d Pydantic templates from ontology", len(templates))
    return templates


def get_ontology_version(ontology: dict) -> str:
    """Return the ontology version string."""
    return ontology.get("version", "unknown")
```

**Step 4: Create FastAPI application**

```
docker/docling-graph/app/main.py
```

```python
"""Docling-Graph extraction service — FastAPI application."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractedEntityResponse,
    ExtractedRelationshipResponse,
)
from .templates import build_templates, get_ontology_version, load_ontology

logger = logging.getLogger(__name__)

# Module-level state populated at startup
_ontology: dict = {}
_templates: dict = {}
_ontology_version: str = "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ontology, _templates, _ontology_version

    ontology_path = os.environ.get("ONTOLOGY_PATH", "/app/ontology/ontology.yaml")
    logger.info("Loading ontology from %s", ontology_path)

    _ontology = load_ontology(ontology_path)
    _ontology_version = get_ontology_version(_ontology)
    _templates = build_templates(_ontology)

    logger.info("Ontology v%s loaded: %d templates", _ontology_version, len(_templates))
    yield


app = FastAPI(title="Docling-Graph Extraction Service", lifespan=lifespan)


@app.get("/health")
async def health():
    if not _templates:
        raise HTTPException(status_code=503, detail="Templates not loaded")
    return {"status": "ok", "ontology_version": _ontology_version, "template_count": len(_templates)}


@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    if request.ontology_version and request.ontology_version != _ontology_version:
        logger.warning(
            "Ontology version mismatch: client=%s, service=%s",
            request.ontology_version,
            _ontology_version,
        )

    provider = os.environ.get("LLM_PROVIDER", "ollama")
    model = os.environ.get("LLM_MODEL", "llama3.2")

    try:
        from docling_graph import run_pipeline

        config = {
            "source": request.text,
            "template": list(_templates.values()),
            "backend": "llm",
            "provider_override": provider,
            "model_override": model,
            "structured_output": True,
            "processing_mode": "many-to-one",
        }

        start = time.time()
        context = run_pipeline(config)
        elapsed = time.time() - start
        logger.info("Extraction completed in %.1fs for doc %s", elapsed, request.document_id)

        # Convert docling-graph output to our response format
        graph = context.knowledge_graph
        entities = []
        for node_id, data in graph.nodes(data=True):
            entities.append(ExtractedEntityResponse(
                name=data.get("name", str(node_id)),
                entity_type=data.get("entity_type", "UNKNOWN"),
                confidence=data.get("confidence", 1.0),
                properties={k: v for k, v in data.items() if k not in ("name", "entity_type", "confidence")},
            ))

        relationships = []
        for src, tgt, data in graph.edges(data=True):
            src_data = graph.nodes[src]
            tgt_data = graph.nodes[tgt]
            relationships.append(ExtractedRelationshipResponse(
                from_name=src_data.get("name", str(src)),
                from_type=src_data.get("entity_type", "UNKNOWN"),
                rel_type=data.get("relationship_type", data.get("label", "RELATED_TO")),
                to_name=tgt_data.get("name", str(tgt)),
                to_type=tgt_data.get("entity_type", "UNKNOWN"),
                confidence=data.get("confidence", 1.0),
            ))

        return ExtractionResponse(
            entities=entities,
            relationships=relationships,
            ontology_version=_ontology_version,
            model=model,
            provider=provider,
        )

    except Exception:
        logger.exception("Extraction failed for document %s", request.document_id)
        raise HTTPException(status_code=503, detail="Extraction failed")
```

**Step 5: Create empty __init__.py**

```
docker/docling-graph/app/__init__.py
```
(empty file)

**Step 6: Create Dockerfile**

```
docker/docling-graph/Dockerfile
```

Use the same FIPS bypass shim pattern as `docker/docling/Dockerfile` (stage 0). Stage 1 installs Python 3.11, pip installs requirements, copies app code.

```dockerfile
# syntax=docker/dockerfile:1

# Stage 0 — FIPS bypass shim
FROM gcc:12-bookworm AS fips-bypass

RUN echo '0' > /tmp/fips_disabled
COPY --from=docker/docling/Dockerfile /tmp/fips_bypass.c /tmp/fips_bypass.c
# (inline the same fips_bypass.c from docker/docling/Dockerfile)
RUN cat > /tmp/fips_bypass.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <fcntl.h>

static const char *FIPS_PATH = "/proc/sys/crypto/fips_enabled";
static const char *FAKE_PATH = "/usr/local/lib/fips_disabled";

FILE *fopen(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "fopen");
    if (path && strcmp(path, FIPS_PATH) == 0)
        return real(FAKE_PATH, mode);
    return real(path, mode);
}
FILE *fopen64(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "fopen64");
    if (path && strcmp(path, FIPS_PATH) == 0)
        return real(FAKE_PATH, mode);
    return real(path, mode);
}
int open(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "open");
    if (path && strcmp(path, FIPS_PATH) == 0)
        return real(FAKE_PATH, flags);
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags);
        int mode = va_arg(ap, int); va_end(ap);
        return real(path, flags, mode);
    }
    return real(path, flags);
}
int open64(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "open64");
    if (path && strcmp(path, FIPS_PATH) == 0)
        return real(FAKE_PATH, flags);
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags);
        int mode = va_arg(ap, int); va_end(ap);
        return real(path, flags, mode);
    }
    return real(path, flags);
}
CEOF
RUN gcc -shared -fPIC -o /tmp/libfips_bypass.so /tmp/fips_bypass.c -ldl

# Stage 1 — Docling-Graph service
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=fips-bypass /tmp/libfips_bypass.so /usr/local/lib/
COPY --from=fips-bypass /tmp/fips_disabled /usr/local/lib/fips_disabled

RUN LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get update \
    && LD_PRELOAD=/usr/local/lib/libfips_bypass.so apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/local/lib/libfips_bypass.so

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
```

**Step 7: Run a quick build test**

```bash
docker build -t docling-graph-test docker/docling-graph/
```

Expected: Image builds successfully.

**Step 8: Commit**

```bash
git add docker/docling-graph/
git commit -m "feat: add Docling-Graph extraction service

Standalone FastAPI service wrapping docling-graph package.
Generates Pydantic templates from volume-mounted ontology YAML.
Routes LLM calls via LiteLLM to Ollama or OpenAI."
```

---

### Task 2: Add Docling-Graph to Docker Compose

**Files:**
- Modify: `docker-compose.yml` (insert after docling block, ~line 145)
- Modify: `docker-compose.test.yml` (add test override)
- Modify: `env.example` (add new env vars)

**Step 1: Add service to docker-compose.yml**

Insert after the docling service block (after line 145, before the API service):

```yaml
  # ---------------------------------------------------------------------------
  # Docling-Graph — ontology-driven entity/relationship extraction
  # ---------------------------------------------------------------------------
  docling-graph:
    build:
      context: ./docker/docling-graph
      dockerfile: Dockerfile
    restart: unless-stopped
    volumes:
      - ./ontology:/app/ontology:ro
    environment:
      ONTOLOGY_PATH: /app/ontology/ontology.yaml
      LLM_PROVIDER: ${DOCLING_GRAPH_LLM_PROVIDER:-ollama}
      LLM_MODEL: ${DOCLING_GRAPH_LLM_MODEL:-llama3.2}
      OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://ollama:11434}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      OPENAI_BASE_URL: ${OPENAI_BASE_URL:-}
    ports:
      - "${DOCLING_GRAPH_PORT:-8002}:8002"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 15s
      retries: 10
      start_period: 60s
    networks:
      - default
```

Add `docling-graph: condition: service_healthy` to `depends_on` for `worker` and `worker-ingest` services (alongside existing `docling` dependency).

**Step 2: Add test override to docker-compose.test.yml**

After the existing docling override:

```yaml
  docling-graph:
    environment:
      LLM_PROVIDER: mock
      ONTOLOGY_PATH: /app/ontology/ontology.yaml
    deploy: {}
```

**Step 3: Add env vars to env.example**

Add under a new section:

```bash
# --- Docling-Graph Service ---
DOCLING_GRAPH_LLM_PROVIDER=ollama
DOCLING_GRAPH_LLM_MODEL=llama3.2
DOCLING_GRAPH_PORT=8002
DOCLING_GRAPH_BASE_URL=http://docling-graph:8002
DOCLING_GRAPH_CONCURRENCY=2
DOCLING_GRAPH_TIMEOUT=300
```

**Step 4: Commit**

```bash
git add docker-compose.yml docker-compose.test.yml env.example
git commit -m "chore: add Docling-Graph service to Docker Compose

Standalone extraction service with ontology volume mount.
LLM routed via LiteLLM (Ollama or OpenAI). Mock mode for tests."
```

---

### Task 3: Add Config Settings for Docling-Graph Client

**Files:**
- Modify: `app/config.py:78-104` (replace extraction settings)

**Step 1: Add new settings, remove old ones**

In `app/config.py`, replace the LLM/extraction settings block (lines ~78-104) with:

```python
# --- Docling-Graph service ---
docling_graph_base_url: str = "http://docling-graph:8002"
docling_graph_concurrency: int = 2
docling_graph_timeout: int = 300
```

Keep these existing settings (they're still used):
- `graph_node_min_confidence` (line ~106)
- `graph_rel_min_confidence` (line ~108)
- `neo4j_*` settings
- `graphrag_*` settings

Remove these settings (now handled by the Docling-Graph service):
- `llm_provider`
- `openai_api_key`
- `ollama_base_url`
- `ollama_llm_concurrency`
- `ollama_num_ctx`
- `docling_graph_model`
- `docling_graph_timeout` (old one, for direct LLM calls)
- `graph_extraction_chunk_size`
- `graph_extraction_chunk_overlap`
- `docling_graph_require_llm`
- `docling_graph_max_tokens`
- `docling_graph_retry_attempts`
- `docling_graph_retry_backoff_seconds`
- `docling_review_confidence_threshold`

Note: `graphrag_model` stays — GraphRAG still calls Ollama directly for report generation.

**Step 2: Run tests**

```bash
./scripts/run_tests.sh
```

Fix any test failures due to removed config fields.

**Step 3: Commit**

```bash
git add app/config.py
git commit -m "refactor: replace LLM extraction config with Docling-Graph client settings"
```

---

### Task 4: Rewrite docling_graph_service.py as HTTP Client

**Files:**
- Modify: `app/services/docling_graph_service.py` (full rewrite)
- Test: `tests/test_docling_graph_client.py`

**Step 1: Write the failing test**

```python
# tests/test_docling_graph_client.py
"""Tests for Docling-Graph HTTP client."""

import json
from unittest.mock import patch, MagicMock

import httpx
import pytest

from app.services.docling_graph_service import extract_graph, DeterministicExtractionError


@pytest.fixture
def mock_extraction_response():
    return {
        "entities": [
            {"name": "Tombstone", "entity_type": "RADAR_SYSTEM", "confidence": 0.9, "properties": {}},
            {"name": "X-band", "entity_type": "FREQUENCY_BAND", "confidence": 0.85, "properties": {}},
        ],
        "relationships": [
            {
                "from_name": "Tombstone",
                "from_type": "RADAR_SYSTEM",
                "rel_type": "OPERATES_IN_BAND",
                "to_name": "X-band",
                "to_type": "FREQUENCY_BAND",
                "confidence": 0.8,
            },
        ],
        "ontology_version": "3.0.0",
        "model": "llama3.2",
        "provider": "ollama",
    }


def test_extract_graph_success(mock_extraction_response):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_extraction_response
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = extract_graph("Some radar text about Tombstone", "doc-123")

    assert len(result["entities"]) == 2
    assert result["entities"][0]["name"] == "Tombstone"
    assert len(result["relationships"]) == 1
    assert result["ontology_version"] == "3.0.0"


def test_extract_graph_service_error():
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Service Unavailable", request=MagicMock(), response=mock_response,
    )

    with patch("httpx.post", return_value=mock_response):
        with pytest.raises(httpx.HTTPStatusError):
            extract_graph("Some text", "doc-456")


def test_extract_graph_empty_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "entities": [],
        "relationships": [],
        "ontology_version": "3.0.0",
        "model": "llama3.2",
        "provider": "ollama",
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = extract_graph("No entities here", "doc-789")

    assert result["entities"] == []
    assert result["relationships"] == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_docling_graph_client.py -v
```

Expected: FAIL (module has old API)

**Step 3: Rewrite docling_graph_service.py**

Replace the entire file contents:

```python
"""Docling-Graph HTTP client.

Calls the standalone Docling-Graph service for ontology-driven
entity/relationship extraction. Replaces the previous in-process
LLM extraction pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class DeterministicExtractionError(ValueError):
    """Extraction failure that will not resolve on retry."""


def extract_graph(
    text: str,
    document_id: str,
    *,
    ontology_version: str | None = None,
) -> dict[str, Any]:
    """Extract entities and relationships via the Docling-Graph service.

    Returns a dict with keys: entities, relationships, ontology_version, model, provider.
    Raises httpx.HTTPStatusError on service errors (caller should retry).
    """
    settings = get_settings()
    url = f"{settings.docling_graph_base_url}/extract"
    timeout = settings.docling_graph_timeout

    payload = {
        "document_id": document_id,
        "text": text,
    }
    if ontology_version:
        payload["ontology_version"] = ontology_version

    logger.info(
        "Calling Docling-Graph service for document %s (%d chars)",
        document_id,
        len(text),
    )

    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    result = response.json()

    entity_count = len(result.get("entities", []))
    rel_count = len(result.get("relationships", []))
    logger.info(
        "Docling-Graph returned %d entities, %d relationships for document %s (model=%s)",
        entity_count,
        rel_count,
        document_id,
        result.get("model", "unknown"),
    )

    return result
```

**Step 4: Run tests**

```bash
pytest tests/test_docling_graph_client.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add app/services/docling_graph_service.py tests/test_docling_graph_client.py
git commit -m "refactor: rewrite docling_graph_service.py as HTTP client

Replace ~750 lines of in-process LLM extraction with a thin httpx
client that calls the standalone Docling-Graph service."
```

---

### Task 5: Simplify ontology_templates.py

**Files:**
- Modify: `app/services/ontology_templates.py`

**Step 1: Identify what to delete**

Delete these (now in Docling-Graph service):
- `build_extraction_prompt()` function
- `DocumentExtractionResult` Pydantic model
- `ExtractedEntity` Pydantic model
- `ExtractedRelationship` Pydantic model
- `generate_entity_templates()` function
- `generate_relationship_templates()` function
- Any few-shot prompt text

Keep these (used by GraphRAG + retrieval):
- `load_ontology()`
- `load_validation_matrix()`
- `build_entity_type_names()`
- `build_relationship_type_names()`
- `_ONTOLOGY_PATH`

**Step 2: Remove unused imports**

Remove any imports only used by deleted functions.

**Step 3: Run tests**

```bash
./scripts/run_tests.sh
```

Fix any tests that imported deleted symbols.

**Step 4: Commit**

```bash
git add app/services/ontology_templates.py
git commit -m "refactor: remove extraction prompt/model code from ontology_templates

Extraction is now handled by the Docling-Graph service. Keep
load_ontology, validation matrix, and type name helpers for
GraphRAG and retrieval."
```

---

### Task 6: Update Pipeline Task derive_ontology_graph

**Files:**
- Modify: `app/workers/pipeline.py:1401-1672`

**Step 1: Rewrite derive_ontology_graph body**

Keep the task decorator and signature. Replace the body:

1. Keep: Reading `DocumentElement` rows and joining into `full_text`
2. Replace: LLM extraction call → `docling_graph_service.extract_graph(full_text, document_id)`
3. Remove: NER fallback path (lines ~1503-1519)
4. Remove: Chunk-splitting logic (Docling-Graph handles internally)
5. Keep: Confidence gating logic (filter by `graph_node_min_confidence`, `graph_rel_min_confidence`)
6. Keep: `upsert_nodes_batch()` / `upsert_relationships_batch()` calls
7. Keep: Entity mention detection (`_build_entity_mentions()`)
8. Keep: `document_graph_extractions` upsert
9. Keep: Exception handling (retry on `httpx.HTTPStatusError`, no retry on `DeterministicExtractionError`)

Remove import of `ner` module. Add import of `httpx` for error handling.

**Step 2: Run tests**

```bash
./scripts/run_tests.sh
```

Expected: Pipeline tests pass with mocked Docling-Graph responses.

**Step 3: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "refactor: derive_ontology_graph calls Docling-Graph service

Replace in-process LLM extraction and NER fallback with HTTP
call to standalone Docling-Graph service. Keep confidence gating,
Neo4j import, and entity-mention detection."
```

---

### Task 7: GraphRAG Ontology-Weighted Community Detection

**Files:**
- Modify: `app/services/graphrag_service.py:225-274` (`_detect_communities`)

**Step 1: Write failing test**

```python
# tests/test_graphrag_weighted.py
"""Tests for ontology-weighted GraphRAG community detection."""

from unittest.mock import patch

from app.services.graphrag_service import _detect_communities


def test_weighted_communities_group_equipment_hierarchy():
    """Equipment connected by PART_OF (weight 0.90) should cluster together,
    not with distantly-connected entities via MENTIONED_IN (weight 0.70)."""
    entities = [
        {"name": "Tombstone", "entity_type": "RADAR_SYSTEM", "id": "1"},
        {"name": "Antenna", "entity_type": "ANTENNA", "id": "2"},
        {"name": "TM-9-1425", "entity_type": "DOCUMENT", "id": "3"},
        {"name": "Patriot", "entity_type": "EQUIPMENT_SYSTEM", "id": "4"},
    ]
    relationships = [
        {"source": "Tombstone", "target": "Antenna", "type": "HAS_ANTENNA"},
        {"source": "Tombstone", "target": "TM-9-1425", "type": "MENTIONED_IN"},
        {"source": "Patriot", "target": "TM-9-1425", "type": "MENTIONED_IN"},
    ]

    communities = _detect_communities(entities, relationships, None)
    assert len(communities) >= 1

    # Find which community Tombstone and Antenna are in
    tombstone_community = None
    antenna_community = None
    for c in communities:
        if "Tombstone" in c["entity_names"]:
            tombstone_community = c["community_id"]
        if "Antenna" in c["entity_names"]:
            antenna_community = c["community_id"]

    # Tombstone and Antenna should be in the same community (strong PART_OF edge)
    assert tombstone_community == antenna_community
```

**Step 2: Modify _detect_communities**

Load ontology scoring weights and apply them as edge weights in the NetworkX graph:

```python
from app.services.ontology_templates import load_ontology

def _detect_communities(entities, relationships, settings):
    ontology = load_ontology()
    scoring_weights = ontology.get("scoring_weights", {})
    default_weight = scoring_weights.get("default", 0.70)

    G = nx.Graph()
    for e in entities:
        G.add_node(e["name"], entity_type=e.get("entity_type", ""))
    for r in relationships:
        rel_type = r.get("type", "")
        weight = scoring_weights.get(rel_type, default_weight)
        G.add_edge(r["source"], r["target"], relationship=rel_type, weight=weight)

    # ... rest of Leiden/Louvain detection (pass weight parameter)
```

**Step 3: Run tests**

```bash
pytest tests/test_graphrag_weighted.py -v
```

**Step 4: Commit**

```bash
git add app/services/graphrag_service.py tests/test_graphrag_weighted.py
git commit -m "feat: ontology-weighted GraphRAG community detection

Use scoring_weights from ontology.yaml to weight edges in Leiden
community detection. Strong relationships (PART_OF, CONTAINS)
cluster equipment hierarchies together."
```

---

### Task 8: GraphRAG Ontology-Aware Report Generation

**Files:**
- Modify: `app/services/graphrag_service.py:277-373` (`_generate_community_reports`)

**Step 1: Write failing test**

```python
# tests/test_graphrag_reports.py
"""Tests for ontology-aware GraphRAG report generation."""

from unittest.mock import patch, MagicMock

from app.services.graphrag_service import _generate_community_reports


def test_report_prompt_includes_ontology_context():
    """The LLM prompt should include entity type and relationship descriptions."""
    communities = [{
        "community_id": "0",
        "entity_names": ["Tombstone", "X-band"],
        "title": "Community 0",
        "level": 0,
    }]
    entities = [
        {"name": "Tombstone", "entity_type": "RADAR_SYSTEM"},
        {"name": "X-band", "entity_type": "FREQUENCY_BAND"},
    ]
    relationships = [
        {"source": "Tombstone", "target": "X-band", "type": "OPERATES_IN_BAND"},
    ]

    captured_prompts = []

    def mock_ollama_call(*args, **kwargs):
        captured_prompts.append(kwargs.get("json", {}).get("messages", []))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "Test report about radar systems."}
        }
        return mock_resp

    with patch("httpx.post", side_effect=mock_ollama_call):
        _generate_community_reports(communities, entities, relationships, MagicMock())

    assert len(captured_prompts) == 1
    system_msg = captured_prompts[0][0]["content"]
    assert "RADAR_SYSTEM" in system_msg
    assert "FREQUENCY_BAND" in system_msg
    assert "OPERATES_IN_BAND" in system_msg
```

**Step 2: Modify _generate_community_reports**

For each community:
1. Collect unique entity types and relationship types present in the community
2. Look up their descriptions from the ontology
3. Inject into the system prompt before the existing member list

**Step 3: Run tests**

```bash
pytest tests/test_graphrag_reports.py -v
```

**Step 4: Commit**

```bash
git add app/services/graphrag_service.py tests/test_graphrag_reports.py
git commit -m "feat: ontology-aware GraphRAG report generation

Inject entity type and relationship type descriptions from the
ontology into the LLM system prompt for community reports."
```

---

### Task 9: GraphRAG Report Refresh + Alembic Migration

**Files:**
- Create: `alembic/versions/0007_graphrag_report_refresh.py`
- Modify: `app/services/graphrag_service.py:376-420` (`_store_communities_and_reports`)

**Step 1: Create Alembic migration**

```bash
cd /home/josh/development/EIP-MMDPP
# Create migration manually (not autogenerate — we're adding a column to an existing table)
```

```python
# alembic/versions/0007_graphrag_report_refresh.py
"""Add generated_at to graphrag_community_reports.

Revision ID: 0007
Revises: 0006
"""

from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"


def upgrade():
    op.add_column(
        "graphrag_community_reports",
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        schema="retrieval",
    )


def downgrade():
    op.drop_column("graphrag_community_reports", "generated_at", schema="retrieval")
```

**Step 2: Modify _store_communities_and_reports**

Change the reports upsert from `ON CONFLICT DO NOTHING` to `ON CONFLICT (community_id) DO UPDATE`:

```python
# For reports:
stmt = pg_insert(GraphragCommunityReport).values(...)
stmt = stmt.on_conflict_do_update(
    index_elements=["community_id"],
    set_={
        "report_text": stmt.excluded.report_text,
        "summary": stmt.excluded.summary,
        "rank": stmt.excluded.rank,
        "generated_at": sa.func.now(),
    },
)
```

**Step 3: Run tests**

```bash
./scripts/run_tests.sh
```

**Step 4: Commit**

```bash
git add alembic/versions/0007_graphrag_report_refresh.py app/services/graphrag_service.py
git commit -m "fix: refresh GraphRAG reports on re-indexing

Change ON CONFLICT DO NOTHING to DO UPDATE for reports. Add
generated_at timestamp column for report freshness tracking."
```

---

### Task 10: Update env.example and Clean Up Dead Config

**Files:**
- Modify: `env.example`
- Modify: `app/config.py` (final cleanup)

**Step 1: Remove old env vars from env.example**

Remove:
- `LLM_PROVIDER`, `OPENAI_API_KEY` (from extraction section — keep if used elsewhere)
- `OLLAMA_NUM_CTX`
- `DOCLING_GRAPH_MODEL` (old, for direct LLM)
- `GRAPH_EXTRACTION_CHUNK_SIZE`, `GRAPH_EXTRACTION_CHUNK_OVERLAP`
- `DOCLING_GRAPH_REQUIRE_LLM`
- `DOCLING_GRAPH_MAX_TOKENS`
- `DOCLING_GRAPH_RETRY_ATTEMPTS`, `DOCLING_GRAPH_RETRY_BACKOFF_SECONDS`
- `DOCLING_REVIEW_CONFIDENCE_THRESHOLD`

Add:
- `DOCLING_GRAPH_LLM_PROVIDER`, `DOCLING_GRAPH_LLM_MODEL`, `DOCLING_GRAPH_PORT`
- `DOCLING_GRAPH_BASE_URL`, `DOCLING_GRAPH_CONCURRENCY`, `DOCLING_GRAPH_TIMEOUT`

Keep: `OLLAMA_BASE_URL` (used by GraphRAG), `GRAPHRAG_MODEL`, all graph confidence gates.

**Step 2: Run full test suite**

```bash
./scripts/run_tests.sh
```

**Step 3: Commit**

```bash
git add env.example app/config.py
git commit -m "chore: clean up env vars for Docling-Graph integration

Remove extraction-specific vars (now in Docling-Graph service).
Add Docling-Graph client configuration vars."
```

---

### Task 11: Integration Test — Full Pipeline Smoke Test

**Files:**
- Modify: `tests/conftest.py` (add Docling-Graph mock)
- Create or modify: `tests/test_pipeline.py` (add integration test)

**Step 1: Add Docling-Graph mock to conftest.py**

Follow the existing pattern for mocking Docling HTTP calls. Add a `mock_docling_graph` fixture that patches `httpx.post` for `docling-graph:8002/extract` calls.

**Step 2: Write integration test**

Test the full `derive_ontology_graph` → Neo4j import path with a mocked Docling-Graph response containing known entities and relationships. Verify:
- Entities above confidence threshold are in Neo4j
- Entities below threshold are filtered
- Relationships are imported
- `document_graph_extractions` row is created

**Step 3: Run tests**

```bash
./scripts/run_tests.sh
```

**Step 4: Commit**

```bash
git add tests/conftest.py tests/test_pipeline.py
git commit -m "test: add integration test for Docling-Graph pipeline path"
```

---

### Task 12: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update architecture section**

Add Docling-Graph service to the architecture diagram / service list. Update the pipeline flow description.

**Step 2: Update Docker Compose instructions**

Add Docling-Graph env vars to the setup guide.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for Docling-Graph integration"
```
