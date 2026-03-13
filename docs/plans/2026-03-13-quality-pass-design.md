# Comprehensive Quality Pass: Extraction & Retrieval

**Date:** 2026-03-13
**Status:** Approved
**Scope:** Improve ontology extraction precision, relationship quality, and retrieval relevance across the full pipeline.

## Context

Analysis of the current pipeline identified these root causes of poor quality:

**Extraction:**
- `llama3.2` (3B) is too weak for constrained structured extraction from technical military text
- Ontology property definitions lack descriptions, examples, and validation patterns — the LLM maps values to wrong fields
- Validation matrix in `base_v1.yaml` is missing, making `_validate_triples` dead code
- Few-shot extraction prompt uses entity/relationship types not in the ontology (`RADAR_SYSTEM`, `OPERATES_IN_BAND`)
- NER regex entity types don't align with ontology types
- No post-extraction enforcement of entity/relationship types against the ontology
- LLM output token cap (1200) causes silent truncation on entity-dense text
- Canonicalization fuzzy threshold uses raw Lucene BM25 scores (not normalized)

**Retrieval:**
- BGE query prefix uses the passage prefix instead of the query prefix (asymmetric model mismatch)
- Structure-aware chunker exists (`chunking.py`) but pipeline uses naive word splitter (`extraction.py`)
- No cross-encoder reranker
- No minimum score threshold — garbage results returned for poor matches
- Ontology-expanded chunks inherit the seed's cosine score instead of their own
- `graphrag_global` ignores query text entirely (returns largest communities)
- `graphrag_local` uses rank-based scores instead of actual BM25 scores
- Image search doesn't oversample for dedup headroom

## Design

### Section 1: Ontology & Extraction Quality

#### 1a. Enrich ontology property definitions

Every property in `base_v1.yaml` gets:
- `description` — unambiguous explanation of what the field holds
- `example` — concrete value for LLM pattern-matching
- `pattern` (where applicable) — regex for post-extraction validation

Example:
```yaml
nsn: {type: string, description: "National Stock Number (NNNN-NN-NNN-NNNN)", example: "5961-01-234-5678", pattern: "^\\d{4}-\\d{2}-\\d{3}-\\d{4}$"}
cage_code: {type: string, description: "5-character Commercial and Government Entity code", example: "1ABC3", pattern: "^[A-Z0-9]{5}$"}
```

Patterns are for **post-extraction validation only**, not extraction. Examples and descriptions go into the LLM prompt to guide correct field placement.

#### 1b. Add validation matrix

```yaml
validation_matrix:
  IS_SUBSYSTEM_OF: [[SUBSYSTEM], [EQUIPMENT_SYSTEM]]
  CONTAINS: [[ASSEMBLY, SUBSYSTEM, EQUIPMENT_SYSTEM], [COMPONENT, ASSEMBLY]]
  IMPLEMENTS: [[EQUIPMENT_SYSTEM, SUBSYSTEM], [CAPABILITY]]
  MEETS_STANDARD: [[COMPONENT, SUBSYSTEM, EQUIPMENT_SYSTEM, ASSEMBLY], [STANDARD]]
  SPECIFIED_BY: [[COMPONENT, SUBSYSTEM, EQUIPMENT_SYSTEM], [SPECIFICATION]]
  DESCRIBED_IN: [[EQUIPMENT_SYSTEM, SUBSYSTEM, COMPONENT, ASSEMBLY, SPECIFICATION, CAPABILITY, STANDARD, ORGANIZATION, PROCEDURE, FAILURE_MODE, TEST_EVENT], [DOCUMENT]]
  PERFORMED_BY: [[PROCEDURE], [ORGANIZATION]]
  AFFECTS: [[FAILURE_MODE], [COMPONENT, SUBSYSTEM, ASSEMBLY]]
  SUPERSEDES: [[DOCUMENT, STANDARD], [DOCUMENT, STANDARD]]
  TESTED_IN: [[EQUIPMENT_SYSTEM, SUBSYSTEM, COMPONENT], [TEST_EVENT]]
```

Activates the existing `_validate_triples` code path in `docling_graph_service.py`.

#### 1c. Fix extraction prompt

- Replace few-shot example with ontology-valid types only
- Include property descriptions and examples in the prompt
- Add explicit instruction: "Only use entity types and relationship types from the provided list"

#### 1d. Post-extraction enforcement

Before Neo4j upsert:
- Reject entities with `entity_type` not in ontology
- Reject relationships with `relationship_type` not in ontology
- Validate property values against `pattern` regexes (log warning, drop invalid value)
- Map NER regex types to ontology types (e.g., `RADAR_SYSTEM` -> `EQUIPMENT_SYSTEM`)

#### 1e. Model upgrade

| Setting | Env Var | Old Default | New Default |
|---|---|---|---|
| `docling_graph_model` | `DOCLING_GRAPH_MODEL` | `llama3.2` | `llama3.1:8b` |
| `docling_graph_max_tokens` | `DOCLING_GRAPH_MAX_TOKENS` | `1200` | `2048` |
| `ollama_num_ctx` | `OLLAMA_NUM_CTX` | `8192` | `16384` |

#### 1f. Canonicalization fixes

- Normalize Lucene BM25 scores to 0-1 range before applying fuzzy threshold
- Add `canonical_name` to the Neo4j fulltext index

### Section 2: Retrieval Quality

#### 2a. Fix BGE query prefix

Add `query=True/False` parameter to `embed_texts()`:
- `query=False` (indexing): `"Represent this sentence: "`
- `query=True` (search): `"Represent this query for searching relevant passages: "`

#### 2b. Wire structure-aware chunker

Replace `chunk_text()` in `derive_text_chunks_and_embeddings` with `structure_aware_chunk()` from `chunking.py`. Preserves section headings, tables, equations as atomic units.

#### 2c. Cross-encoder reranker

- Model: `BAAI/bge-reranker-v2-m3`
- CPU by default, GPU optional via `RERANKER_DEVICE`
- Applied to top-N candidates after fusion scoring
- New service: `app/services/reranker.py`
- Pre-downloaded during `manage.sh --start`

#### 2d. Minimum score threshold

`RETRIEVAL_MIN_SCORE_THRESHOLD=0.25` — Qdrant results below this cosine similarity dropped before expansion.

#### 2e. Re-score expanded chunks

Compute independent cosine similarity for ontology-expanded chunks against the query, instead of inheriting the seed's score.

#### 2f. Fix GraphRAG

- `graphrag_global`: Filter community reports by query text relevance
- `graphrag_local`: Use actual BM25 scores from Neo4j fulltext search

#### 2g. Image search oversample

Apply `retrieval_diversity_oversample_factor` to image search (same as text).

#### 2h. Model download in manage.sh

Pre-download during startup:
- BGE-large-en-v1.5 (existing)
- OpenCLIP ViT-B/32 (existing)
- BGE-reranker-v2-m3 (new)
- `ollama pull llama3.1:8b` (new default)

### Section 3: Configuration

#### New settings

| Setting | Env Var | Default | Purpose |
|---|---|---|---|
| `reranker_model` | `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model |
| `reranker_device` | `RERANKER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `reranker_enabled` | `RERANKER_ENABLED` | `True` | Toggle reranker |
| `reranker_top_n` | `RERANKER_TOP_N` | `20` | Candidates to re-score |
| `retrieval_min_score_threshold` | `RETRIEVAL_MIN_SCORE_THRESHOLD` | `0.25` | Minimum cosine similarity |

### Section 4: Files to modify

| File | Changes |
|---|---|
| `ontology/base_v1.yaml` | Descriptions, examples, patterns, validation_matrix |
| `app/services/reranker.py` | **New** — cross-encoder service |
| `app/services/embedding.py` | Add `query` param for asymmetric prefix |
| `app/services/ontology_templates.py` | Property descriptions/examples in prompt, fix few-shot |
| `app/services/docling_graph_service.py` | Post-extraction type + property validation |
| `app/services/ner.py` | Align entity types with ontology |
| `app/services/canonicalization.py` | Normalize fuzzy scores |
| `app/services/neo4j_graph.py` | Fulltext index includes canonical_name |
| `app/services/qdrant_store.py` | Score threshold parameter |
| `app/services/graphrag_service.py` | Fix global search, BM25 scores for local |
| `app/api/v1/retrieval.py` | Wire reranker, re-score expanded chunks, image oversample |
| `app/api/v1/_retrieval_helpers.py` | Score threshold filtering |
| `app/workers/pipeline.py` | Wire structure-aware chunker |
| `app/config.py` | New settings |
| `manage.sh` | Model pre-download step |
| `env.example` | Document new env vars |

### Section 5: Out of scope

- Multi-pass extraction verification (can add later if single-pass quality is insufficient)
- New entity types in the ontology
- Embedding model changes (BGE-large and CLIP stay)
- Schema migrations (no database changes)
