# Docling-Graph Integration & Ontology-Aware GraphRAG

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Replace hand-rolled LLM extraction with the `docling-graph` package as a standalone Docker service, merge ontologies, and make GraphRAG community reports ontology-aware.

## Context

The current extraction pipeline (`app/services/docling_graph_service.py`) duplicates what IBM's `docling-graph` framework provides: LLM-based entity/relationship extraction guided by ontology templates. Meanwhile, GraphRAG community reports operate on the Neo4j graph without any awareness of the ontology schema — community detection is purely topological and report generation uses a generic prompt.

Two ontology files exist (`base.yaml` v2.0.0 and `base_v1.yaml` v1.0.0) with different schemas and overlapping types, and the production code loads `base.yaml` while `base_v1.yaml` goes unused.

## Design

### Section 1: Service Architecture

Three independent Docker Compose services on the shared network:

- **Docling** (existing, unchanged) — `docker/docling/`, granite-docling-258M VLM, GPU, port 8001. Converts documents to structured DoclingDocument JSON.
- **Docling-Graph** (new) — `docker/docling-graph/`, LiteLLM-routed LLM (Ollama or OpenAI), port 8002. Accepts document text, extracts ontology-typed entities and relationships using Pydantic templates generated from volume-mounted ontology YAML.
- **EIP-MMDPP worker** (modified) — `docling_graph_service.py` becomes a thin HTTP client. Keeps confidence gating, Neo4j import, entity-mention detection, and structure links.

Pipeline flow:
```
prepare_document ──HTTP──► Docling (:8001) → DoclingDocument JSON
derive_ontology_graph ──HTTP──► Docling-Graph (:8002) → entities + relationships
    → confidence gate → Neo4j import
derive_structure_links → entity-chunk edges
derive_canonicalization → alias detection
```

### Section 2: Ontology Merge & Cleanup

Starting from `base.yaml` (v2.0.0), which already contains all `base_v1.yaml` types:

**Entity types:** Keep all ~40 types across 5 layers (Reference/Provenance, Military Equipment, EM/RF Signal, Weapon/Missile/AAA, Operational/Capability).

**Relationship type cleanup — remove 5 redundant legacy types:**

| Remove | Covered by |
|---|---|
| `IS_SUBSYSTEM_OF` | `PART_OF` (DoDAF DM2 canonical) |
| `IMPLEMENTS` | `PROVIDES` |
| `MEETS_STANDARD` | `SPECIFIED_BY` |
| `DESCRIBED_IN` | `MENTIONED_IN` |
| `PERFORMED_BY` | `OPERATED_BY` |

Keep `AFFECTS`, `SUPERSEDES`, `TESTED_IN` (no equivalents).

**Validation matrix:** Remove entries referencing deleted relationship types. Ensure primary types cover the same paths.

**Output:** Single `ontology/ontology.yaml` (v3.0.0). Delete `base.yaml` and `base_v1.yaml`.

### Section 3: GraphRAG Ontology Integration

Three changes to `app/services/graphrag_service.py`:

**3a. Ontology-weighted community detection:**
- Load `scoring_weights` from `ontology.yaml` at indexing time
- Set edge weights in the NetworkX graph from `scoring_weights[rel_type]` (default: 0.70)
- Equipment hierarchies (`PART_OF`: 0.90, `CONTAINS`: 0.90) cluster together; weak edges (`MENTIONED_IN`: default 0.70) don't dominate

**3b. Ontology-aware report generation prompt:**
- For each community, collect unique entity types and relationship types present
- Inject their descriptions from `ontology.yaml` into the LLM system prompt
- The LLM understands what `RADAR_SYSTEM`, `USES_WAVEFORM`, etc. mean in the domain

**3c. Report refresh:**
- Change `_store_communities_and_reports` to use `ON CONFLICT (community_id) DO UPDATE` for both communities AND reports
- Add `generated_at TIMESTAMPTZ DEFAULT now()` column to `graphrag_community_reports` (Alembic migration)

### Section 4: Docling-Graph Service Implementation

**Directory structure:**
```
docker/docling-graph/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py          # FastAPI: /extract and /health
│   ├── schemas.py        # Request/response models
│   ├── templates.py      # YAML → Pydantic template generator (runs at startup)
│   └── __init__.py
```

**HTTP API:**

`POST /extract` — accepts `{document_id, text, ontology_version}`, returns `{entities, relationships, ontology_version, model, provider}`.

`GET /health` — confirms LLM reachable and templates loaded.

**Startup:** Read `ONTOLOGY_PATH` env → load YAML → generate and cache Pydantic templates → verify LLM connectivity.

**Docker Compose:**
```yaml
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
```

FIPS bypass: Same multi-stage LD_PRELOAD shim as the Docling Dockerfile.

### Section 5: EIP-MMDPP Pipeline Changes

**`app/services/docling_graph_service.py` — rewrite to thin HTTP client:**
- Delete all LLM extraction logic (~500 lines)
- New `extract_graph(text, document_id) -> dict` — POST to Docling-Graph service
- Redis concurrency gate (`docling-graph:extract` lock)

**`app/services/ontology_templates.py` — simplify:**
- Keep: `load_ontology()`, `build_entity_type_names()`, `build_relationship_type_names()`
- Delete: `build_extraction_prompt()`, `DocumentExtractionResult`, `ExtractedEntity`, `ExtractedRelationship`

**`app/workers/pipeline.py` — `derive_ontology_graph`:**
- Replace LLM extraction with `docling_graph_service.extract_graph()`
- Keep: confidence gating, Neo4j import, entity-mention detection, `document_graph_extractions` storage
- Remove: NER fallback path, chunk-splitting logic

**`app/services/ner.py`:** Keep the module (useful for offline testing) but remove pipeline imports.

**`app/config.py`:**
- Add: `DOCLING_GRAPH_BASE_URL`, `DOCLING_GRAPH_CONCURRENCY`, `DOCLING_GRAPH_TIMEOUT`
- Remove: `docling_graph_require_llm`, `llm_provider`

**New Alembic migration:** Add `generated_at` column to `retrieval.graphrag_community_reports`.

### Section 6: Error Handling & Testing

**Error handling:**
- Docling-Graph service down: `derive_ontology_graph` retries with `self.retry(countdown=30)`
- LLM timeout: service returns 503, worker retries
- Ontology version mismatch: log warning, not a hard failure (allows rolling updates)
- Empty extraction: log and continue, document finalizes successfully

**Testing:**
- Docling-Graph service unit tests: YAML→template generation, mock LLM, template constraint enforcement
- EIP-MMDPP integration tests: mock Docling-Graph HTTP endpoint, test confidence gate + Neo4j import
- GraphRAG tests: weighted vs unweighted community detection, ontology context in prompts
- Docker Compose smoke test: full pipeline with `LLM_PROVIDER=mock`

**`docker-compose.test.yml` override:**
```yaml
docling-graph:
  environment:
    LLM_PROVIDER: mock
    ONTOLOGY_PATH: /app/ontology/ontology.yaml
  deploy: {}
```
