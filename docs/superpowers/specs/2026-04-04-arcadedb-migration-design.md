# ArcadeDB Migration Design

**Date:** 2026-04-04
**Status:** Approved (Revised)
**Scope:** Remove Microsoft GraphRAG, replace Neo4j with ArcadeDB, eliminate Qdrant (vectors consolidated into ArcadeDB), add community-based global query

## Overview

Major architectural refactor:
1. Completely remove Microsoft GraphRAG (library, services, UI, tests, config)
2. Replace Neo4j with ArcadeDB using server/client model
3. Eliminate Qdrant -- vector search consolidated into ArcadeDB's native LSMVectorIndex (HNSW)
4. Add ArcadeDB-native global query via community detection + LLM summarization
5. Preserve ingest pipeline behavior, multimodal query, query profiles, and API response schemas

**Companion spec:** See `2026-04-04-docling-graph-pipeline-refactor-design.md` for the Docling-Graph service refactor that executes before this migration.

**Migration strategy:** Full reingest of all documents after refactor. No data migration from Neo4j or Qdrant. ArcadeDB starts empty.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Query language | ArcadeDB SQL (primary), Cypher for graph algorithms | SQL for graph operations and DDL; Cypher CALL...YIELD syntax for community detection algorithms (algo.louvain, algo.leiden). HTTP API language parameter set per request. |
| Global query | Community detection (Louvain/Leiden) + LLM reports | Pre-computed, fast query-time, uses ArcadeDB native algorithms |
| Python client | httpx async (HTTP/JSON API) | Matches FastAPI async pattern, no extra driver dependency |
| Schema mode | Schema-full | Catches errors early, enables query optimization, matches structured ontology |
| Abstraction | GraphStore Protocol | Backend-agnostic interface for future replaceability |
| Schema source | Ontology-driven | Schema generated from ontology definition, synced on ontology changes |

## Section 1: Ontology-Driven ArcadeDB Schema

### Core concept

The ArcadeDB schema is generated directly from the ontology definition. Each ontology entity type becomes an ArcadeDB vertex type with schema-full typed properties. Each ontology relationship type becomes an ArcadeDB edge type. When the ontology changes via the UI (new registry activated), a schema sync function diffs against the current schema and applies additive changes.

### Common metadata on entity vertex types (BaseEntity)

Entity vertices are shared across documents via MERGE on identity fields. They do NOT carry a singular source_document_id. Provenance is tracked via EXTRACTED_FROM edges to TextChunk/ImageChunk vertices.

| Property | Type | Required | Purpose |
|----------|------|----------|---------|
| id | STRING | MANDATORY | UUID assigned at creation |
| name | STRING | MANDATORY | Human-readable identifier (populated from primary identity field) |
| entity_type | STRING | MANDATORY | Ontology entity type |
| canonical_name | STRING | | Canonicalized form (set during canonicalization) |
| extraction_confidence | DOUBLE | | Highest extraction confidence seen across all extractions (named to avoid collision with ontology domain properties like ASSERTION.confidence) |
| created_at | DATETIME | | Record creation time |
| updated_at | DATETIME | | Last modification time |

Note: `name` is populated from whichever field is the entity type's primary human-readable identifier. For types where the identity field IS `name` (e.g., RADAR_SYSTEM), it's the same value. For types like DOCUMENT, `name` is populated from `title`. For FIGURE, from `figure_id`. This preserves the generic entity contract that search, dossier, and query profiles depend on.

### Identity fields per entity type

Entity dedup uses the entity type's identity fields (from ontology graph_id_fields), NOT a universal `name`:
- RADAR_SYSTEM: merges on `name`
- DOCUMENT: merges on `document_id`
- FIGURE: merges on `figure_id`
- SPECIFICATION: merges on `(parameter, value)`
- See companion Docling-Graph spec for complete derivation rules.

### Common metadata on structural vertex types (TextChunk, ImageChunk, Document)

These are document-scoped (not shared across documents):

| Property | Type | Required | Purpose |
|----------|------|----------|---------|
| document_id | STRING | MANDATORY | Which document this belongs to |
| page_number | INTEGER | | Page in source document |
| created_at | DATETIME | | Record creation time |

### Common metadata on ontology relationship edge types

Relationship edges can be established by multiple documents:

| Property | Type | Required | Purpose |
|----------|------|----------|---------|
| document_ids | LIST | MANDATORY | All documents that established this relationship |
| extraction_confidence | DOUBLE | | Highest extraction confidence seen |
| created_at | DATETIME | | Record creation time |
| updated_at | DATETIME | | Last modification time |

### Common metadata on structural edge types (CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE, EXTRACTED_FROM)

These are document-scoped:

| Property | Type | Required | Purpose |
|----------|------|----------|---------|
| document_id | STRING | MANDATORY | Which document this edge belongs to |
| extraction_confidence | DOUBLE | | Extraction confidence |
| created_at | DATETIME | | Record creation time |

EXTRACTED_FROM edges additionally carry:
| page_numbers | LIST | | Pages where entity was found in this chunk's document |
| upload_datetime | DATETIME | | When the source document was uploaded |
| document_datetime | DATETIME | | Date extracted from document via LLM |

### Vertex types from ontology (46 types across 5 layers)

**Layer 1 -- Reference & Provenance (7 types):**
- DOCUMENT: title, document_id, document_number, classification, publication_date, source_type, issuing_org, language
- SECTION: heading, page_start, page_end
- FIGURE: figure_id, caption, page, figure_type
- TABLE_REF: table_id, caption, page (TABLE_REF is the ArcadeDB type name; the ontology YAML retains `TABLE` as the entity name. The mapping is handled in `arcadedb_schema.py` via a RESERVED_WORD_MAP: `{"TABLE": "TABLE_REF"}`. All downstream code uses the ontology name `TABLE` -- the rename is transparent, applied only at the ArcadeDB schema layer. Validation matrix lookups, extraction, and query profiles all use `TABLE`; the GraphStore translates to `TABLE_REF` internally.)
- SPREADSHEET: workbook_name, sheet_name
- ASSERTION: assertion_text, confidence, extraction_method, review_status
- STANDARD: designation, title, issuing_org, version, supersedes

**Layer 2 -- Military Equipment / DoDAF (14 types):**
- PLATFORM: name, platform_designation, platform_type, country, service_branch, platform_status
- RADAR_SYSTEM: system_name, nomenclature, ELNOT, DIEQP, radar_type, emitter_function, system_status, responsible_agency, nominal_frequency, frequency_limits, radar_waveform, nominal_PRI, PRI_limits, PRF_limits, nominal_pulse_duration, pulse_duration_limits, ERP, tx_peak_power, duty_cycle, gain, scan_type, scan_period, detection_to_designate_time, designation_to_launch_time
- MISSILE_SYSTEM: system_name, nomenclature, DIEQP, system_status, guidance_type, seeker_nomenclature, seeker_ELNOT, seeker_DIEQP
- AIR_DEFENSE_ARTILLERY_SYSTEM: system_name, DIEQP, caliber, max_tactical_range, max_vertical_range, min_vertical_range, max_horizontal_range, min_horizontal_range, acquisition_delay, handoff_delay, track_delay, muzzle_velocity, maximum_rate_of_fire
- ELECTRONIC_WARFARE_SYSTEM: system_name, nomenclature, ELNOT, ew_role, coverage, power_output
- FIRE_CONTROL_SYSTEM: system_name, nomenclature
- INTEGRATED_AIR_DEFENSE_SYSTEM: name, status, doctrine
- LAUNCHER_SYSTEM: system_name, launcher_type, capacity
- WEAPON_SYSTEM: system_name, nomenclature, weapon_type
- SUBSYSTEM: name, subsystem_role, part_number
- COMPONENT: name, component_type, part_number, nsn, cage_code, manufacturer, material, weight_kg
- ORGANIZATION: name, org_type, cage_code, country, location
- EQUIPMENT_SYSTEM: name, designation, program_office, status, prime_contractor, service_branch
- ASSEMBLY: name, assembly_number

**Layer 3 -- EM/RF Signal & Radar (11 types):**
- FREQUENCY_BAND: band_name, designation, freq_min_mhz, freq_max_mhz, standard_family
- RF_EMISSION: name, emitter_id, center_frequency_mhz, bandwidth_mhz, ERP_kw, tx_peak_power_kw, duty_cycle, polarization, radome_loss_db, total_system_losses_db
- WAVEFORM: waveform_name, waveform_family, nominal_pulse_duration_us, pulse_duration_limits, nominal_PRI_us, PRI_limits, PRF_limits, duty_cycle
- MODULATION: name, intra_pulse_modulation, inter_pulse_modulation, frequency_excursion, code_bits, pulse_compression_ratio, pulse_compression_gain_db, pulse_compression_weighting_function
- RF_SIGNATURE: name, nominal_RF, nominal_PRI, nominal_PD, scan_type, scan_period, modulation_pattern, frequency_agility, beam_characteristics, dwell_time, pulses_per_dwell
- SCAN_PATTERN: scan_type, scan_period_limits, slew_rate, illumination_time, dwell_time, pulses_per_dwell
- ANTENNA: name, antenna_type, tx_polarization, rx_polarization, number_of_beams, beamwidth_az_deg, beamwidth_el_deg, aperture_distribution_az, aperture_distribution_el, gain_dbi, backlobe_level_db, dimension_horizontal_m, dimension_vertical_m, height_m, min_elevation_deg, max_elevation_deg, first_sidelobe_level_az_db, first_sidelobe_level_el_db
- TRANSMITTER: name, peak_power_ERP_kw, peak_power_at_transmitter_kw, duty_cycle, tx_line_loss_db, other_system_losses_db, total_system_losses_db
- RECEIVER: name, radome_loss_db, clutter_improvement_factor_db, noise_figure_db, minimum_discernible_signal_dbm, receive_line_loss_db, peak_power_noise_bandwidth_mhz, average_power_noise_bandwidth_mhz
- IF_AMPLIFIER: stage_number, bandwidth_3db_mhz
- SIGNAL_PROCESSING_CHAIN: name, matched_filter_detection_loss_db, STC, pulse_compression_ratio, pulse_compression_gain_db, pulse_compression_weighting_function, coherent_processing_interval, filter_response_type, doppler_filter_bandwidth_hz, approaching_doppler_coverage, pulses_on_target, predetection_pulses, predetection_integration_gain_db, postdetection_pulses, postdetection_integration_gain_db, effective_number_of_pulses_integrated, minimum_snr_required_db, MTI_improvement_factor_db, drop_track_threshold_improvement_db

**Layer 4 -- Weapon / Missile / AAA (6 types):**
- SEEKER: seeker_nomenclature, seeker_ELNOT, seeker_DIEQP, seeker_type
- GUIDANCE_METHOD: guidance_type, firing_doctrine, track_quality
- MISSILE_PERFORMANCE: maximum_range_km, minimum_range_km, maximum_intercept_range_km, maximum_recommended_intercept_range_km, maximum_altitude_km, minimum_altitude_m, maximum_launch_angle_deg, intercept_assessment_time_s, time_to_go_s, acquisition_delay_s, handoff_delay_s, track_delay_s, launch_delay_s, intra_salvo_time_s, coast_time_s, average_missile_speed_mach, maximum_missile_speed_mach, maximum_flyout_time_s, maximum_offset_deg
- MISSILE_PHYSICAL_CHARACTERISTICS: body_diameter_m, overall_length_m, total_mass_kg
- PROPULSION_STACK: total_burntime_s
- PROPULSION_STAGE: stage_type, burn_time_s, thrust_kn, mass_kg, diameter_m, length_m

**Layer 5 -- Operational / Capability (8 types):**
- CAPABILITY: capability_name, capability_class, trl
- RADAR_PERFORMANCE: max_detection_range_1sqm_km, min_effective_range_km, max_unambiguous_range_km, maximum_scope_limit_km, maximum_processing_range_km, max_unambiguous_velocity_mps, min_range_of_velocity_response_mps, max_range_of_velocity_response_mps, minimum_detectable_velocity_mps, maximum_detectable_velocity_mps
- ENGAGEMENT_TIMELINE: detection_to_designate_time_s, designation_to_launch_time_s, acquisition_delay_s, handoff_delay_s, track_delay_s, launch_delay_s, intercept_assessment_time_s, time_to_go_s
- FORCE_STRUCTURE: name, echelon, service
- SPECIFICATION: parameter, value, unit, condition, source_document
- PROCEDURE: name, type, periodicity, skill_level
- FAILURE_MODE: name, description, fmeca_severity, detection_method
- TEST_EVENT: name, date, location, test_type, outcome

**Structural vertex types (system-internal, not from ontology):**
- TextChunk: chunk_id (FK to PostgreSQL), document_id, page_number, modality, classification, text_embedding (LIST, 1024-dim BGE) -- replaces ChunkRef for text. PostgreSQL remains authoritative for chunk content; ArcadeDB carries embedding + filter metadata.
- ImageChunk: chunk_id (FK to PostgreSQL), document_id, artifact_id, page_number, image_embedding (LIST, 512-dim CLIP) -- replaces ChunkRef for images. Description stays in PostgreSQL.
- TrustedTextChunk: chunk_id, document_id, content_text, text_embedding (LIST, 1024-dim BGE), source, classification -- trusted-data semantic search (replaces Qdrant eip_trusted_text collection).
- Document: document_id, title, upload_datetime, document_datetime
- Alias: alias_name
- CommunityReport: community_id, membership_hash, title, summary, member_count, key_entities (LIST), key_relationships (LIST), report_embedding (LIST, 1024-dim BGE), model_name, generated_at -- authoritative store for community reports (no PostgreSQL table).

Note: ChunkRef is eliminated. TextChunk/ImageChunk vertices serve the same structural role (targets of EXTRACTED_FROM edges, sources of CONTAINS_TEXT/CONTAINS_IMAGE edges) but also carry embeddings for vector search.

### Edge types from ontology (50 types)

Identity/Typing: IS_A, INSTANCE_OF, ALIAS_OF

Whole-Part (DoDAF DM2): PART_OF, CONTAINS, HAS_SUBSYSTEM, HAS_COMPONENT, HAS_STAGE

Installation/Association: INSTALLED_ON, DEPLOYED_ON, ASSOCIATED_WITH, OPERATED_BY, MANUFACTURED_BY

EM/RF Technical: OPERATES_IN_BAND, USES_WAVEFORM, USES_MODULATION, EMITS, RADIATES, RECEIVES, PROCESSES, HAS_SIGNATURE, HAS_SCAN, HAS_RECEIVER, HAS_TRANSMITTER, HAS_ANTENNA, HAS_PROCESSING_CHAIN

Performance: HAS_PERFORMANCE

Weapon/Engagement: CUES, GUIDES, TRACKS, ENGAGES, DEFENDS, DETECTS, DESIGNATES, LAUNCHES, SUPPORTS_ENGAGEMENT_OF, HAS_GUIDANCE, HAS_PROPULSION, HAS_SEEKER

Capability: PROVIDES, HAS_TIMELINE

Provenance: SUPPORTED_BY, MENTIONED_IN, DERIVED_FROM, REVIEWED_BY, ABOUT, SPECIFIED_BY, AFFECTS, SUPERSEDES, TESTED_IN

Structural (system-internal): CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE, EXTRACTED_FROM, HAS_ALIAS

All edge types carry the common metadata properties.

### Dynamic schema sync

`arcadedb_schema.py` reads the ontology definition, diffs against current ArcadeDB schema, and applies additive changes using `CREATE VERTEX TYPE IF NOT EXISTS`, `CREATE PROPERTY IF NOT EXISTS`, and `CREATE INDEX IF NOT EXISTS`.

Triggered by: API startup, ontology registry activation, manual endpoint.

Schema sync is additive only. Removed ontology types stay in ArcadeDB (data may exist) but are no longer used for new extractions. No deletion logic needed.

### Validation matrix enforcement

The ontology's validation_matrix is enforced at the application layer before writing to ArcadeDB. The GraphStore `upsert_relationship()` checks (source_type, rel_type, target_type) against the loaded matrix and rejects invalid triples.

## Section 2: ArcadeDB Client & GraphStore Abstraction

### File structure

```
app/services/graph_store.py          -- GraphStore Protocol (abstract contract)
app/services/arcadedb_client.py      -- httpx async transport layer
app/services/arcadedb_graph.py       -- GraphStore implementation
app/services/arcadedb_schema.py      -- Ontology-driven schema sync
app/services/arcadedb_community.py   -- Community detection + LLM reports
```

### GraphStore Protocol

Everything upstream (pipeline, retrieval, query profiles, API endpoints) depends only on this interface.

Methods:
- Node operations: upsert_node, upsert_nodes_batch, upsert_document_node, create_text_chunk_vertex, create_image_chunk_vertex
- Edge operations: upsert_relationship, upsert_relationships_batch, create_structural_edge, batch_create_entity_chunk_edges
- Query operations: search_nodes, get_neighborhood, get_neighborhood_graph, get_ontology_linked_chunks, get_graph_stats
- Document lifecycle: delete_document_graph
- Schema management: sync_schema, ensure_indexes
- Sync variants of all write operations (for Celery workers)

### ArcadeDBClient (transport layer)

Low-level httpx wrapper for ArcadeDB HTTP API. All endpoints include the database name as a path parameter.

- Token-based auth: POST /api/v1/login with Basic Auth returns bearer token (expires after configurable inactivity, default 30 min). On 401, the client re-authenticates via /api/v1/login (not a refresh -- ArcadeDB has no refresh endpoint). Client stores credentials to re-authenticate automatically.
- query(database, ...) for read-only operations: POST /api/v1/query/{database}
- command(database, ...) for write operations: POST /api/v1/command/{database}
- batch(database, ...) for bulk import: POST /api/v1/batch/{database}
- begin(database)/commit(database, session_id)/rollback(database, session_id) for explicit transactions
- Both async (httpx.AsyncClient for FastAPI) and sync (httpx.Client for Celery) variants
- Singleton lifecycle: created at startup, closed at shutdown

### Connection management

| Context | Client type | Lifecycle |
|---------|------------|-----------|
| FastAPI endpoints | httpx.AsyncClient | Singleton, created at startup |
| Celery workers | httpx.Client (sync) | Singleton per worker process |

### Upstream access

`app/db/session.py` provides `get_graph_store()` returning the singleton GraphStore instance. Pipeline code, retrieval, query profiles all call this -- they never see ArcadeDBClient.

## Section 3: Ingest Pipeline Changes

### Stages unchanged

prepare_document, detect_and_translate, derive_document_metadata, derive_picture_descriptions, finalize_document -- no graph involvement.

### Stages modified

**derive_text_chunks_and_embeddings:** Creates PostgreSQL rows (unchanged) + ArcadeDB TextChunk vertices with text_embedding property (replaces Qdrant upsert). This stage becomes a graph writer.

**derive_image_embeddings:** Creates PostgreSQL rows (unchanged) + ArcadeDB ImageChunk vertices with image_embedding property (replaces Qdrant upsert). This stage becomes a graph writer.

**derive_ontology_graph:** Replace get_neo4j_driver() with get_graph_store(). Replace upsert_nodes_batch/upsert_relationships_batch with GraphStore equivalents. ProvenanceMetadata built from document metadata and passed with every write. Confidence quality gates and validation matrix enforcement remain. Uses ArcadeDB batch endpoint for bulk import (single HTTP call for 50-200 entities + 100-500 relationships per document).

**derive_structure_links:** Creates ArcadeDB Document vertex + structural edges (CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE) connecting Document to existing TextChunk/ImageChunk vertices (which already exist from embedding stages). Also creates EXTRACTED_FROM edges from entities to TextChunk/ImageChunk vertices. No longer creates separate chunk pointer nodes -- TextChunk/ImageChunk vertices serve that role.

**derive_canonicalization:** Replace fulltext_search_entity with graph_store.fulltext_search. Replace create_alias_edge with graph_store.create_alias. Fuzzy matching threshold (0.8) unchanged.

**purge_document / purge_document_derivations:** `graph_store.delete_document_graph(document_id)` is invoked INSIDE the existing document-delete orchestration, not as a standalone replacement. Full delete flow:

1. Delete MinIO artifacts (raw + derived)
2. Delete DocumentElements from PostgreSQL
3. Delete TextChunk/ImageChunk rows from PostgreSQL
4. Delete ChunkLinks from PostgreSQL
5. Delete PipelineRun/StageRun from PostgreSQL
6. Delete DocumentGraphExtraction from PostgreSQL
7. `graph_store.delete_document_graph(document_id)` -- deletes ArcadeDB Document vertex, TextChunk/ImageChunk vertices for that document, structural edges, EXTRACTED_FROM edges for that document. Removes document_id from relationship edge document_ids lists; deletes edges with empty lists. Orphan entity cleanup: delete entity vertices with zero remaining EXTRACTED_FROM edges.
8. Delete Document row from PostgreSQL

### Provenance propagation

Entity provenance tracked via EXTRACTED_FROM edges (document_id, page_numbers, upload_datetime, document_datetime). Structural edges carry document_id. Ontology relationship edges carry document_ids (LIST).

### Post-ingest community detection hook

After finalize_document completes, if COMMUNITY_DETECTION_POST_INGEST_ENABLED is true, increment a Redis counter. When counter reaches COMMUNITY_DETECTION_POST_INGEST_THRESHOLD, trigger community detection task.

## Section 4: Community Detection & Global Query

### Architecture

- arcadedb_community.py: detection + report generation + query
- community_tasks.py: Celery tasks (scheduled, manual, post-ingest)
- app/api/v1/community.py: API endpoints

### Storage

Community reports are CommunityReport vertices in ArcadeDB (authoritative store). No PostgreSQL community_reports table.

community_runs table stays in PostgreSQL (pipeline state): id, status (PENDING|RUNNING|COMPLETE|FAILED), trigger (SCHEDULED|MANUAL|POST_INGEST), total_communities, reports_generated, reports_reused, detection_duration_ms, report_duration_ms, error_message, started_at, completed_at, created_at

### Community detection flow

1. Run community detection algorithm (Louvain or Leiden per COMMUNITY_DETECTION_ALGORITHM setting) on domain-entity projected subgraph (see Addendum: Community Detection Graph Projection)
2. For each community, fetch member entities
3. Compute membership_hash = SHA-256(sorted (entity_type, name) tuples) -- hashes type+name pairs to prevent cross-type collisions and remain stable across alias/canonical-name changes
4. Diff against stored CommunityReport vertices: unchanged hash = skip, changed/new = regenerate, dissolved = delete
5. For changed communities: fetch entities + edges + evidence chunks, build LLM prompt, generate report, embed report, store as CommunityReport vertex with report_embedding
6. Update community_runs record in PostgreSQL

### LLM report prompt

Configurable via COMMUNITY_REPORT_LLM_PROMPT env var. Template supports {entities}, {relationships}, {evidence} placeholders. Default is domain-aware military intelligence summary prompt.

### Global query flow

1. Embed query text (BGE) and search CommunityReport vertices via `vectorNeighbors('CommunityReport[report_embedding]', :query_vector, :top_k)` in ArcadeDB
2. Rank communities by relevance (top_k configurable)
3. Fetch full reports + key entities + key relationships for top communities
4. LLM synthesis: combine reports into comprehensive answer citing source documents
5. Return as UnifiedQueryResponse with strategy="global", modality="community_report"

### Query strategy enum and schema cleanup

QueryStrategy enum becomes: basic, hybrid, global (removed: graphrag_local, graphrag_global, graphrag_drift)

Also removed from app/schemas/retrieval.py:
- GraphRAGJobSubmitResponse, GraphRAGJobStatusResponse classes (only used for async GraphRAG queries)
- _MODE_MAP backward-compat entries for graphrag_local, graphrag_global, graphrag_drift
- Add "global" to _MODE_MAP if backward compat is maintained

### Community report vector search

Community report embeddings are stored on CommunityReport vertices in ArcadeDB with a `report_embedding` property and LSM_VECTOR index (see Addendum: Qdrant Elimination). Searched via `vectorNeighbors('CommunityReport[report_embedding]', :query_vector, :top_k)`.

### Scheduling & triggers

Celery Beat: community-detection task at configurable interval (default 60 min)

API endpoints:
- POST /v1/community/detect {mode: "incremental"|"full"} -- manual trigger, returns run_id
- GET /v1/community/status -- latest run status
- GET /v1/community/status/{run_id} -- specific run
- GET /v1/community/reports -- all reports (paginated)
- GET /v1/community/reports/{community_id} -- single report

Environment variables:
- COMMUNITY_DETECTION_ENABLED, COMMUNITY_DETECTION_INTERVAL_MINUTES
- COMMUNITY_DETECTION_POST_INGEST_ENABLED, COMMUNITY_DETECTION_POST_INGEST_THRESHOLD
- COMMUNITY_DETECTION_RESOLUTION, COMMUNITY_DETECTION_MAX_ITERATIONS
- COMMUNITY_REPORT_LLM_MODEL, COMMUNITY_REPORT_LLM_PROMPT

Redis lock prevents concurrent detection runs.

UI: admin section with last run status, next scheduled run, community count, "Run Now" and "Full Rebuild" buttons, status polling.

## Section 5: Query System Changes

### Retrieval endpoint (app/api/v1/retrieval.py)

Removed: _graphrag_local_query(), _graphrag_global_query(), _graphrag_drift_query(), async job submission/polling.

Modified: expand_ontology_async() and expand_document_structure_cross_modal_async() use GraphStore instead of Neo4j driver.

Added: _global_query() orchestrating community report search + LLM synthesis.

### Query profiles (app/services/query_profiles.py)

Full rewrite: Cypher templates replaced with ArcadeDB SQL. All queries go through GraphStore. Multi-step traversals use ArcadeDB SQL {min,max} syntax on out()/in()/both().

Response schemas QueryProfileSectionResponse, QueryProfileDossierResponse, GraphEntityResult -- unchanged.

### Graph store API (app/api/v1/graph_store.py)

All four endpoints stay, implementation changes from Neo4j driver to GraphStore.

### Dossier service

Renamed from neo4j_dossier_service.py to dossier_service.py. Cypher templates replaced with ArcadeDB SQL.

### Frontend

Removed: GraphRAGLocalDetail, GraphRAGGlobalDetail, graphrag API functions/types, three GraphRAG modes.
Added: GlobalQueryDetail component, global strategy in MODES array.
Unchanged: GraphExplorer.tsx, QueryProfileRegistryPage.tsx.

## Section 6: Docker & Infrastructure

### Services removed

neo4j

### Service added

arcadedb: built from source (git@github.com:ArcadeData/arcadedb.git), multi-stage Dockerfile (JDK build + JRE runtime), healthcheck via GET /api/v1/ready, data persisted to arcadedb_data volume.

### Upstream repo management

manage.sh --start and --start-split pull latest from three GitHub repos before Docker build:
- ArcadeDB: git@github.com:ArcadeData/arcadedb.git -> docker/arcadedb/repo/
- Docling: git@github.com:docling-project/docling.git -> docker/docling/repo/
- Docling-Graph: git@github.com:docling-project/docling-graph.git -> docker/docling-graph/repo/

Docling and Docling-Graph install from local clone during Docker build (pip install /build/repo/) instead of PyPI.

All repo/ directories are gitignored.

### Service dependency updates

All services that depended on `neo4j: condition: service_healthy` must change to `arcadedb: condition: service_healthy`:
- api
- worker (default mode)
- worker-graph (split mode, currently line 366-367 in docker-compose.yml)

### Volume and mount cleanup

Volumes removed: neo4j_data, graphrag_data

Volume mounts removed from api, worker, worker-graph, and beat services:
- `graphrag_data:/app/graphrag_data` (present on api, worker, worker-graph, beat)
- `./docker/neo4j:/docker-entrypoint-initdb.d` (on neo4j service, deleted with service)

Volumes added: arcadedb_data

### Environment variables

Removed: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_BOLT_PORT, NEO4J_HTTP_PORT, and all 20+ GRAPHRAG_* variables.

Added (with defaults):
- ARCADEDB_URL=http://arcadedb:2480
- ARCADEDB_USER=root
- ARCADEDB_PASSWORD=eip_arcadedb_secret
- ARCADEDB_DATABASE=eip_knowledge_graph
- ARCADEDB_HTTP_PORT=2480
- ARCADEDB_GRPC_PORT=2424
- COMMUNITY_DETECTION_ENABLED=true
- COMMUNITY_DETECTION_INTERVAL_MINUTES=60
- COMMUNITY_DETECTION_POST_INGEST_ENABLED=true
- COMMUNITY_DETECTION_POST_INGEST_THRESHOLD=5
- COMMUNITY_DETECTION_ALGORITHM=leiden (options: louvain, leiden)
- COMMUNITY_DETECTION_RESOLUTION=1.0 (Leiden only; Louvain uses tolerance instead)
- COMMUNITY_DETECTION_MAX_ITERATIONS=20
- COMMUNITY_REPORT_LLM_MODEL=llama3.2
- COMMUNITY_REPORT_LLM_PROMPT=(default military domain prompt)

### Celery Beat

Removed: graphrag-indexing, graphrag-auto-tune
Added: community-detection

## Section 7: Files Changed

### Files deleted (35 files + 1 directory)

Services (9): app/services/graphrag_service.py, app/services/graphrag_config.py, app/services/graphrag_bridge.py, app/services/graphrag_prompts.py, app/services/graphrag_runtime_patches.py, app/services/graphrag_provenance.py, app/services/neo4j_graph.py, app/services/neo4j_dossier_service.py, app/services/qdrant_store.py

Workers (1): app/workers/graphrag_tasks.py

Docker (1 directory): docker/neo4j/ (contains init.cypher)

Scripts (1): scripts/migrate_age_to_neo4j.py

Tests (15): tests/unit/test_graphrag_service.py, tests/unit/test_graphrag_config.py, tests/unit/test_graphrag_bridge.py, tests/unit/test_graphrag_provenance.py, tests/unit/test_graphrag_prompts.py, tests/unit/test_graphrag_runtime_patches.py, tests/unit/test_graphrag_query_task.py, tests/unit/test_neo4j_graph_operations.py, tests/unit/test_graph_service.py, tests/unit/test_neighborhood_graph.py, tests/unit/test_extracted_from_edges.py, tests/unit/test_upsert_relationships_batch.py, tests/unit/test_canonicalization.py, tests/integration/test_graph_store_api.py, tests/integration/test_pipeline_graph.py

Docs (8): docs/superpowers/plans/2026-03-17-microsoft-graphrag-integration.md, docs/superpowers/plans/2026-03-18-graphrag-fixes-and-drift-basic.md, docs/superpowers/plans/2026-03-31-async-graphrag-queries.md, docs/superpowers/plans/2026-04-02-graphrag-context-provenance.md, docs/superpowers/plans/2026-04-02-graphrag-citation-provenance.md, docs/superpowers/specs/2026-03-31-async-graphrag-queries-design.md, docs/superpowers/specs/2026-04-02-graphrag-context-provenance-design.md, docs/superpowers/specs/2026-04-02-graphrag-citation-provenance-design.md

### Files created (33 files)

Services (6): app/services/graph_store.py, app/services/arcadedb_client.py, app/services/arcadedb_graph.py, app/services/arcadedb_schema.py, app/services/arcadedb_community.py, app/services/dossier_service.py

Workers (1): app/workers/community_tasks.py

API (1): app/api/v1/community.py

Docker (1): docker/arcadedb/Dockerfile

Migration (1): alembic/versions/XXXX_add_community_tables.py

Unit tests (15): tests/unit/test_arcadedb_client.py, tests/unit/test_arcadedb_graph.py, tests/unit/test_arcadedb_schema.py, tests/unit/test_arcadedb_community.py, tests/unit/test_community_tasks.py, tests/unit/test_graph_store_protocol.py, tests/unit/test_dossier_service.py, tests/unit/test_query_profiles_arcadedb.py, tests/unit/test_pipeline_graph_operations.py, tests/unit/test_pipeline_structure_links.py, tests/unit/test_pipeline_canonicalization.py, tests/unit/test_retrieval_global_query.py, tests/unit/test_retrieval_strategies.py, tests/unit/test_community_api.py, tests/unit/test_provenance_metadata.py

Integration tests (8): tests/integration/test_arcadedb_integration.py, tests/integration/test_arcadedb_schema_sync_integration.py, tests/integration/test_arcadedb_batch_integration.py, tests/integration/test_arcadedb_community_integration.py, tests/integration/test_graph_store_api_integration.py, tests/integration/test_community_api_integration.py, tests/integration/test_query_profiles_integration.py, tests/integration/test_pipeline_graph_integration.py

### Files modified (35+ files)

Config: app/config.py (remove Qdrant/Neo4j settings, remove pgvector settings, add ArcadeDB/community settings), app/main.py (remove Qdrant bootstrap, remove Neo4j bootstrap), app/db/session.py (remove Qdrant client init, remove Neo4j driver init, add GraphStore init)
Schemas: app/schemas/retrieval.py, app/schemas/trusted_data.py (remove qdrant_point_id from response schema)
API: retrieval.py, graph_store.py, sources.py, agent.py, query_profiles.py, governance.py (re-embed path: write to ArcadeDB via graph_store.set_vertex_embedding instead of chunk.embedding), trusted_data.py
Services: query_profiles.py, canonicalization.py
Workers: pipeline.py (derive_ontology_graph, derive_text_chunks_and_embeddings, derive_image_embeddings, derive_structure_links, purge), celery_app.py, trusted_data_tasks.py
Models: app/models/retrieval.py (remove `from pgvector.sqlalchemy import Vector`, drop `embedding` columns, drop `qdrant_point_id` columns), app/models/trusted_data.py (drop `qdrant_point_id`)
Docker: docker-compose.yml (remove neo4j + qdrant services), docker-compose.test.yml, docker/docling/Dockerfile, docker/docling-graph/Dockerfile, docker/postgres/init/01_extensions.sql (remove pgvector extension), docker/postgres/Dockerfile (remove pgvector installation if present)
Env: env.example, .env.test
Frontend: client.ts, QueryPage.tsx, QueryProfileRegistryPage.tsx, GraphExplorer.tsx, App.tsx
Other: manage.sh, .gitignore, example_queries.py, README.md, VERIFICATION_CHECKLIST.md, pyproject.toml, uv.lock
Tests: test_config.py, test_query_coverage.py, test_retrieval_schemas.py, test_startup_bootstrap.py, tests/conftest.py

### New Alembic migrations

1. Drop `embedding` (Vector) column from `retrieval.text_chunks`
2. Drop `embedding` (Vector) column from `retrieval.image_chunks`
3. Drop `qdrant_point_id` column from `retrieval.text_chunks`
4. Drop `qdrant_point_id` column from `retrieval.image_chunks`
5. Drop `qdrant_point_id` column from trusted-data model (if applicable)
6. Add `community_runs` table to `retrieval` schema
7. Remove `pgvector` extension from PostgreSQL (DROP EXTENSION IF EXISTS vector)

### Dependencies

Removed: neo4j>=5.25.0, graphrag>=3.0.0, pyarrow>=14.0.0, qdrant-client>=1.13.0, pgvector>=0.3.6
Added: httpx>=0.27.0

Note: pgvector removed because vector search is handled by ArcadeDB, not PostgreSQL extensions.

## Addendum: Qdrant Elimination

Qdrant is eliminated. ArcadeDB's native LSMVectorIndex (HNSW/Vamana via JVector 4.0.0) replaces all vector search.

### Vector vertex types and indexes

TextChunk: chunk_id, document_id, content_text, page_number, modality, classification, text_embedding (LIST, 1024-dim BGE)
ImageChunk: chunk_id, document_id, artifact_id, page_number, description, image_embedding (LIST, 512-dim CLIP)
CommunityReport: community_id, membership_hash, title, summary, member_count, key_entities, key_relationships, report_embedding (LIST, 1024-dim BGE), model_name, generated_at

```sql
CREATE INDEX ON TextChunk (text_embedding) LSM_VECTOR METADATA {dimensions: 1024, similarity: 'COSINE', quantization: 'INT8', addHierarchy: true}
CREATE INDEX ON ImageChunk (image_embedding) LSM_VECTOR METADATA {dimensions: 512, similarity: 'COSINE', quantization: 'INT8'}
CREATE INDEX ON CommunityReport (report_embedding) LSM_VECTOR METADATA {dimensions: 1024, similarity: 'COSINE', quantization: 'INT8', addHierarchy: true}
```

TextChunk and ImageChunk replace the lightweight ChunkRef pointer vertex. PostgreSQL `retrieval.text_chunks` and `retrieval.image_chunks` tables remain authoritative for chunk content, section hierarchy, translated text, and relational metadata. ArcadeDB TextChunk/ImageChunk vertices carry chunk_id (FK to PostgreSQL), embedding, and minimal filter metadata. `chunk_id` is the stable bridge between stores. `qdrant_point_id` column dropped from PostgreSQL chunk models.

### Trusted-data migration

Trusted-data semantic search moves entirely from Qdrant to ArcadeDB:
- `eip_trusted_text` Qdrant collection -> TrustedTextChunk vertex type with LSM_VECTOR index
- `qdrant_point_id` on trusted-data models -> removed
- `app/api/v1/trusted_data.py`, `app/workers/trusted_data_tasks.py`, `app/models/trusted_data.py` modified to use GraphStore vector_search()

```sql
CREATE INDEX ON TrustedTextChunk (text_embedding) LSM_VECTOR METADATA {
    dimensions: 1024, similarity: 'COSINE', quantization: 'INT8', addHierarchy: true
}
```

### Cross-model queries

Hybrid retrieval uses native ArcadeDB cross-model queries:

```sql
-- Semantic search + graph expansion in one query
SELECT chunk.*, entity.name, entity.entity_type
FROM (
    SELECT expand(vectorNeighbors('TextChunk[text_embedding]', :query_vector, :top_k))
) AS chunk
LET entity = chunk.in('EXTRACTED_FROM')
```

Same API endpoints, same UnifiedQueryResponse contract. Internal implementation uses single-database queries instead of multi-service round trips.

### Docker changes

Removed: qdrant service, qdrant_data volume, qdrant_test_data volume
Removed env vars: QDRANT_URL, QDRANT_HTTP_PORT, QDRANT_GRPC_PORT, QDRANT_API_KEY

### Architecture: 2 stores instead of 4

PostgreSQL: relational data, pipeline state, audit trail
ArcadeDB: graph + vectors + search + community detection
MinIO: object storage (uploaded docs, extracted images)

## Addendum: Provenance Model (Revised)

Entity nodes are shared across documents via MERGE on (entity_type, identity_fields...). The `name` field is a display/search property, NOT the merge key. Entities do NOT carry a mandatory singular source_document_id. Provenance is tracked via EXTRACTED_FROM edges.

### Entity vertex provenance

- `id` STRING (MANDATORY) -- UUID assigned at creation
- `name` STRING (MANDATORY)
- `entity_type` STRING (MANDATORY)
- `canonical_name` STRING
- `confidence` DOUBLE -- highest confidence seen across all extractions
- `created_at` DATETIME
- `updated_at` DATETIME
- No mandatory source_document_id on entity vertices

### Provenance via EXTRACTED_FROM edges

Each EXTRACTED_FROM edge carries:
- `document_id` STRING (MANDATORY) -- which document this extraction came from
- `page_numbers` LIST -- pages where entity was found
- `upload_datetime` DATETIME
- `document_datetime` DATETIME
- `confidence` DOUBLE
- `created_at` DATETIME

An entity mentioned in 3 documents has 3+ EXTRACTED_FROM edges pointing to different TextChunk/ImageChunk vertices.

### Document deletion

1. Delete Document vertex for that document
2. Delete TextChunk/ImageChunk vertices where document_id matches
3. Delete CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE edges for that document
4. Delete EXTRACTED_FROM edges where document_id matches
5. Orphan cleanup: delete entity vertices with zero remaining EXTRACTED_FROM edges

### Relationship edge provenance

Relationship edges (ontology relationships between entities) carry:
- `document_ids` LIST -- all documents that established this relationship
- `confidence` DOUBLE -- highest confidence seen
- `created_at` DATETIME
- `updated_at` DATETIME

Document deletion removes the document_id from the list. If the list becomes empty, the edge is deleted.

## Addendum: Expanded GraphStore Protocol

The GraphStore Protocol includes these additional methods for root resolution and canonicalization:

```python
# Root resolution (full chain: alias -> fulltext -> relationship-count tie-break -> co-extraction fallback)
async def resolve_root_entity(self, query_text: str, root_types: list[str], top_k: int) -> list[GraphEntityResult]: ...

# Alias operations
async def create_alias(self, entity_type: str, entity_name: str, alias_name: str) -> None: ...
async def search_by_alias(self, alias_name: str, entity_types: list[str]) -> list[GraphEntityResult]: ...
async def set_canonical_name(self, entity_type: str, entity_name: str, canonical_name: str) -> None: ...

# Fulltext with scoring
async def fulltext_search(self, query_text: str, entity_types: list[str] | None, top_k: int) -> list[tuple[GraphEntityResult, float]]: ...

# Tie-breaking and fallback
async def get_relationship_count(self, entity_type: str, entity_name: str) -> int: ...
async def get_co_extracted_entities(self, document_id: str, entity_types: list[str]) -> list[GraphEntityResult]: ...

# Vector operations (replaces Qdrant)
async def vector_search(self, vertex_type: str, embedding_property: str, query_vector: list[float], top_k: int, filters: dict | None = None) -> list[dict]: ...
async def set_vertex_embedding(self, vertex_type: str, vertex_id: str, embedding_property: str, embedding: list[float]) -> None: ...
async def cross_model_search(self, query_vector: list[float], top_k: int, expand_edges: list[str] | None = None, filters: dict | None = None) -> list[dict]: ...
```

## Addendum: Non-Pipeline Neo4j Dependencies

Every non-pipeline Neo4j touch point explicitly mapped to GraphStore:

| Current code | Current behavior | GraphStore replacement |
|---|---|---|
| sources.py:494 (document hard-delete) | Cypher DELETE on Document/TextChunk/ImageChunk nodes | graph_store.delete_document_graph(document_id) |
| pipeline.py:1546 (purge_document) | Cypher DELETE on structural subgraph | graph_store.delete_document_graph_sync(document_id) |
| graph_store.py:27 (manual entity ingest) | get_neo4j_async_driver() + upsert_node() | graph_store.upsert_node() |
| graph_store.py:61 (manual relationship ingest) | get_neo4j_async_driver() + upsert_relationship() | graph_store.upsert_relationship() |
| graph_store.py:92 (graph query) | get_neo4j_async_driver() + search_nodes_async() | graph_store.search_nodes() |
| graph_store.py:137 (neighborhood) | get_neo4j_async_driver() + get_neighborhood_graph_async() | graph_store.get_neighborhood_graph() |
| governance.py:351 (comment about AGE) | Comment only, no Neo4j code | Remove comment |

## Addendum: Query Translation Examples

### System components traversal

Current Cypher:
```cypher
MATCH (root:Entity {name: $name, entity_type: 'RADAR_SYSTEM'})
MATCH (root)-[:HAS_COMPONENT]->(component:Entity)
OPTIONAL MATCH (component)-[:EXTRACTED_FROM]->(chunk:TextChunk)
RETURN component, collect(chunk.chunk_id) AS evidence
```

ArcadeDB SQL:
```sql
SELECT component.name, component.entity_type, component.*,
       list(chunk.chunk_id) AS evidence
FROM (
    SELECT expand(out('HAS_COMPONENT')) FROM RADAR_SYSTEM WHERE name = :name
) AS component
LET chunk = component.out('EXTRACTED_FROM')
GROUP BY component.name, component.entity_type
```

### Multi-step traversal (2-hop with evidence)

ArcadeDB SQL:
```sql
SELECT target.name, target.entity_type, target.*,
       target.out('EXTRACTED_FROM').chunk_id AS evidence_chunks
FROM (
    SELECT expand(out('HAS_SUBSYSTEM', 'HAS_COMPONENT'){1,3})
    FROM RADAR_SYSTEM WHERE name = :name
) AS target
```

### Root resolution (fulltext + alias)

ArcadeDB SQL:
```sql
-- Fulltext search on entity names
SELECT *, $score AS score FROM BaseEntity
WHERE name LUCENE :query_text
AND entity_type IN [:root_types]
ORDER BY score DESC LIMIT :top_k
```

## Addendum: Startup Bootstrap and Worker Scheduler

### API startup (main.py lifespan)

1. Create ArcadeDBClient (httpx async)
2. Create/open database if not exists
3. Load active ontology from PostgreSQL
4. `graph_store.sync_schema(ontology)` -- create/update vertex/edge types, properties, indexes, vector indexes
5. `graph_store.ensure_indexes()` -- verify all indexes exist
6. Log "ArcadeDB schema synced, N types, M indexes"

### Celery worker startup

1. Create sync ArcadeDBClient (httpx sync)
2. No full schema sync (API handles that)
3. Graph-writing tasks call `graph_store.ensure_ready_sync()` before first write:
   - Verify database exists (retry with backoff if not)
   - Verify required vertex/edge types exist (retry if not)
   - If types missing after retries, raise and let Celery retry the task
   - Idempotent and safe for concurrent workers

### Celery Beat schedule changes

Removed: graphrag-indexing, graphrag-auto-tune
Added: community-detection (configurable interval, default 60 min)
Kept: directory watcher, other existing schedules

### Worker shutdown

Removed: close_graphrag_loop() cleanup
Added: graph_store.close_sync() to release httpx client

## Addendum: Schema Sync Triggers (Complete List)

Schema sync fires on:
1. API startup
2. Ontology registry activation (POST /registries/{id}/activate)
3. Active registry update (PUT /registries/{id} when registry is_active=true)
4. Manual endpoint (POST /v1/admin/schema/sync)

## Addendum: Transport Decision

httpx (HTTP/JSON) is the canonical transport. One transport, one code path. The PostgreSQL wire protocol is documented in the risk matrix as a contingency but is NOT implemented unless HTTP proves unworkable.

## Addendum: API Breaking Changes (Explicit)

UnifiedQueryResponse schema preserved. Query profile response schemas unchanged.

Intentional breaking changes (per GraphRAG removal requirements):
- QueryStrategy enum: graphrag_local, graphrag_global, graphrag_drift removed. global added.
- GraphRAGJobSubmitResponse, GraphRAGJobStatusResponse removed.
- Async GraphRAG job endpoints removed.
- _MODE_MAP backward-compat entries for graphrag_* removed. "global" added.

## Addendum: Community Detection Graph Projection

Community detection runs on a projected subgraph of domain entities only:

Include vertex types: all ontology entity types (BaseEntity subtypes excluding Document, TextChunk, ImageChunk, Alias, CommunityReport)
Include edge types: all ontology relationship types (50 types)
Exclude: structural vertices (Document, TextChunk, ImageChunk, Alias, CommunityReport) and structural edges (CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE, EXTRACTED_FROM, HAS_ALIAS)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ArcadeDB Cypher/SQL differences cause query bugs | Medium | Medium | Comprehensive integration tests against live ArcadeDB |
| ArcadeDB fulltext search behaves differently than Neo4j | Medium | Medium | Test fuzzy matching thresholds, adjust Lucene query syntax |
| Building ArcadeDB from source fails on Gradle changes | Low | High | Pin to a known-good tag/branch if HEAD breaks |
| Community detection produces poor clusters for small graphs | Medium | Low | Configurable resolution parameter, minimum community size threshold |
| LLM report generation is slow for many communities | Medium | Medium | Incremental reports (only changed communities), parallelizable |
| httpx connection pooling under high concurrency | Low | Medium | Configure pool limits, monitor connection reuse |

## Acceptance Criteria

1. No remaining runtime dependencies on Microsoft GraphRAG
2. No remaining runtime dependencies on Neo4j
3. All graph persistence uses ArcadeDB
4. System runs using ArcadeDB server/client model
5. Ingest pipeline performs same upstream enrichment steps
6. Multimodal query works with ArcadeDB-backed data
7. Query profiles and prescribable ontology work with ArcadeDB
8. Query profile API response schemas unchanged
9. Entity provenance tracked via EXTRACTED_FROM edges with document_id and page_numbers
10. Both upload_datetime and document_datetime persisted on EXTRACTED_FROM edges
11. .env and env.example cleaned of GraphRAG/Neo4j variables
12. Docker configuration cleaned of GraphRAG/Neo4j services
13. Tests pass with full unit and integration coverage
14. No stale imports, dead code, or orphaned config
15. Global query via community detection works end-to-end
16. Community detection is schedulable, manually triggerable, and has post-ingest hook
17. Schema syncs from ontology and updates when ontology changes (including active registry PUT edits)
18. No remaining runtime dependencies on Qdrant
19. All vector search uses ArcadeDB LSMVectorIndex
20. Text and image embeddings stored as ArcadeDB vertex properties
21. Cross-model graph+vector queries used by hybrid retrieval strategy
22. Document deletion correctly handles shared entity vertices via EXTRACTED_FROM edge removal + orphan cleanup
