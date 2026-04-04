# ArcadeDB Migration Design

**Date:** 2026-04-04
**Status:** Approved
**Scope:** Remove Microsoft GraphRAG, replace Neo4j with ArcadeDB, add community-based global query

## Overview

Major architectural refactor:
1. Completely remove Microsoft GraphRAG (library, services, UI, tests, config)
2. Replace Neo4j with ArcadeDB using server/client model
3. Add ArcadeDB-native global query via community detection + LLM summarization
4. Preserve ingest pipeline behavior, multimodal query, query profiles, and API response schemas

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Query language | ArcadeDB SQL | Native, fullest feature coverage |
| Global query | Community detection (Louvain/Leiden) + LLM reports | Pre-computed, fast query-time, uses ArcadeDB native algorithms |
| Python client | httpx async (HTTP/JSON API) | Matches FastAPI async pattern, no extra driver dependency |
| Schema mode | Schema-full | Catches errors early, enables query optimization, matches structured ontology |
| Abstraction | GraphStore Protocol | Backend-agnostic interface for future replaceability |
| Schema source | Ontology-driven | Schema generated from ontology definition, synced on ontology changes |

## Section 1: Ontology-Driven ArcadeDB Schema

### Core concept

The ArcadeDB schema is generated directly from the ontology definition. Each ontology entity type becomes an ArcadeDB vertex type with schema-full typed properties. Each ontology relationship type becomes an ArcadeDB edge type. When the ontology changes via the UI (new registry activated), a schema sync function diffs against the current schema and applies additive changes.

### Common metadata on ALL vertex and edge types

Every vertex and edge gets these properties automatically:

| Property | Type | Required | Purpose |
|----------|------|----------|---------|
| source_document_id | STRING | MANDATORY | Which document this came from |
| page_number | INTEGER | | Page in source document |
| upload_datetime | DATETIME | | When document was uploaded |
| document_datetime | DATETIME | | Date extracted from document via LLM |
| confidence | DOUBLE | | Extraction confidence 0.0-1.0 |
| created_at | DATETIME | | Record creation time |
| updated_at | DATETIME | | Last modification time |

### Vertex types from ontology (46 types across 5 layers)

**Layer 1 -- Reference & Provenance (7 types):**
- DOCUMENT: title, document_id, document_number, classification, publication_date, source_type, issuing_org, language
- SECTION: heading, page_start, page_end
- FIGURE: figure_id, caption, page, figure_type
- TABLE_REF: table_id, caption, page (TABLE_REF avoids ArcadeDB reserved word TABLE)
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
- ChunkRef: chunk_id, chunk_type, document_id, page_number + common metadata
- Alias: alias_name + common metadata

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
- Node operations: upsert_node, upsert_nodes_batch, upsert_document_node, upsert_chunk_ref
- Edge operations: upsert_relationship, upsert_relationships_batch, create_structural_edge, batch_create_entity_chunk_edges
- Query operations: search_nodes, get_neighborhood, get_neighborhood_graph, get_ontology_linked_chunks, get_graph_stats
- Document lifecycle: delete_document_graph
- Schema management: sync_schema, ensure_indexes
- Sync variants of all write operations (for Celery workers)

### ArcadeDBClient (transport layer)

Low-level httpx wrapper for ArcadeDB HTTP API:
- Token-based auth with automatic refresh on 401
- query() for read-only operations (POST /api/v1/query)
- command() for write operations (POST /api/v1/command)
- batch() for bulk import (POST /api/v1/batch)
- begin/commit/rollback for explicit transactions
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

prepare_document, detect_and_translate, derive_document_metadata, derive_picture_descriptions, derive_text_chunks_and_embeddings, derive_image_embeddings, finalize_document -- no graph involvement.

### Stages modified

**derive_ontology_graph:** Replace get_neo4j_driver() with get_graph_store(). Replace upsert_nodes_batch/upsert_relationships_batch with GraphStore equivalents. ProvenanceMetadata built from document metadata and passed with every write. Confidence quality gates and validation matrix enforcement remain. Uses ArcadeDB batch endpoint for bulk import (single HTTP call for 50-200 entities + 100-500 relationships per document).

**derive_structure_links:** Replace all Neo4j node/edge creation with GraphStore equivalents. Document, ChunkRef, CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE, EXTRACTED_FROM edges all through GraphStore. Page number propagated to every edge.

**derive_canonicalization:** Replace fulltext_search_entity with graph_store.search_nodes. Replace create_alias_edge with graph_store.create_structural_edge. Fuzzy matching threshold (0.8) unchanged.

**purge_document / purge_document_derivations:** Replace Cypher deletion with graph_store.delete_document_graph(document_id). Deletes all vertices and edges where source_document_id matches.

### Provenance propagation

Every graph write carries ProvenanceMetadata (source_document_id, page_number, upload_datetime, document_datetime), constructed once per document at the start of derive_ontology_graph.

### Post-ingest community detection hook

After finalize_document completes, if COMMUNITY_DETECTION_POST_INGEST_ENABLED is true, increment a Redis counter. When counter reaches COMMUNITY_DETECTION_POST_INGEST_THRESHOLD, trigger community detection task.

## Section 4: Community Detection & Global Query

### Architecture

- arcadedb_community.py: detection + report generation + query
- community_tasks.py: Celery tasks (scheduled, manual, post-ingest)
- app/api/v1/community.py: API endpoints

### PostgreSQL tables

community_reports: id, community_id, membership_hash, title, summary, member_count, key_entities (JSONB), key_relationships (JSONB), generated_at, model_name, created_at

community_runs: id, status (PENDING|RUNNING|COMPLETE|FAILED), trigger (SCHEDULED|MANUAL|POST_INGEST), total_communities, reports_generated, reports_reused, detection_duration_ms, report_duration_ms, error_message, started_at, completed_at, created_at

### Community detection flow

1. Run Louvain on full graph (ArcadeDB native algo, seconds for 100K+ nodes)
2. For each community, fetch member entities
3. Compute membership_hash = SHA-256(sorted member names)
4. Diff against stored reports: unchanged hash = skip, changed/new = regenerate, dissolved = delete
5. For changed communities: fetch entities + edges + evidence chunks, build LLM prompt, generate report, store
6. Update community_runs record

### LLM report prompt

Configurable via COMMUNITY_REPORT_LLM_PROMPT env var. Template supports {entities}, {relationships}, {evidence} placeholders. Default is domain-aware military intelligence summary prompt.

### Global query flow

1. Embed query text (BGE) and search community report embeddings in Qdrant (eip_community_reports collection)
2. Rank communities by relevance (top_k configurable)
3. Fetch full reports + key entities + key relationships for top communities
4. LLM synthesis: combine reports into comprehensive answer citing source documents
5. Return as UnifiedQueryResponse with strategy="global", modality="community_report"

### Query strategy enum

basic, hybrid, global (removed: graphrag_local, graphrag_global, graphrag_drift)

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

### Volumes

Removed: neo4j_data, graphrag_data
Added: arcadedb_data

### Environment variables

Removed: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_BOLT_PORT, NEO4J_HTTP_PORT, and all 20+ GRAPHRAG_* variables.

Added: ARCADEDB_URL, ARCADEDB_USER, ARCADEDB_PASSWORD, ARCADEDB_DATABASE, ARCADEDB_HTTP_PORT, ARCADEDB_GRPC_PORT, and all COMMUNITY_* variables.

### Celery Beat

Removed: graphrag-indexing, graphrag-auto-tune
Added: community-detection

## Section 7: Files Changed

### Files deleted (34 files)

Services: graphrag_service.py, graphrag_config.py, graphrag_bridge.py, graphrag_prompts.py, graphrag_runtime_patches.py, graphrag_provenance.py, neo4j_graph.py, neo4j_dossier_service.py

Workers: graphrag_tasks.py

Docker: docker/neo4j/init.cypher

Scripts: scripts/migrate_age_to_neo4j.py

Tests (15 files): test_graphrag_service.py, test_graphrag_config.py, test_graphrag_bridge.py, test_graphrag_provenance.py, test_graphrag_prompts.py, test_graphrag_runtime_patches.py, test_graphrag_query_task.py, test_neo4j_graph_operations.py, test_graph_service.py, test_neighborhood_graph.py, test_extracted_from_edges.py, test_upsert_relationships_batch.py, test_canonicalization.py, test_graph_store_api.py (integration), test_pipeline_graph.py (integration)

Docs (8 files): All GraphRAG plans and specs

### Files created (18 files)

Services: graph_store.py, arcadedb_client.py, arcadedb_graph.py, arcadedb_schema.py, arcadedb_community.py, dossier_service.py

Workers: community_tasks.py

API: app/api/v1/community.py

Migration: alembic/versions/XXXX_add_community_tables.py

Unit tests (10): test_arcadedb_client.py, test_arcadedb_graph.py, test_arcadedb_schema.py, test_arcadedb_community.py, test_community_tasks.py, test_graph_store_protocol.py, test_dossier_service.py, test_query_profiles_arcadedb.py, test_pipeline_graph_operations.py, test_pipeline_structure_links.py, test_pipeline_canonicalization.py, test_retrieval_global_query.py, test_retrieval_strategies.py, test_community_api.py, test_provenance_metadata.py

Integration tests (8): test_arcadedb_integration.py, test_arcadedb_schema_sync_integration.py, test_arcadedb_batch_integration.py, test_arcadedb_community_integration.py, test_graph_store_api_integration.py, test_community_api_integration.py, test_query_profiles_integration.py, test_pipeline_graph_integration.py

### Files modified (30+ files)

Config: app/config.py, app/main.py, app/db/session.py
Schemas: app/schemas/retrieval.py
API: retrieval.py, graph_store.py, sources.py, agent.py, query_profiles.py, governance.py
Services: query_profiles.py, canonicalization.py
Workers: pipeline.py, celery_app.py
Models: app/models/retrieval.py
Docker: docker-compose.yml, docker-compose.test.yml, docker/docling/Dockerfile, docker/docling-graph/Dockerfile
Env: env.example, .env.test
Frontend: client.ts, QueryPage.tsx, QueryProfileRegistryPage.tsx, GraphExplorer.tsx, App.tsx
Other: manage.sh, .gitignore, example_queries.py, README.md, VERIFICATION_CHECKLIST.md, pyproject.toml, uv.lock
Tests: test_config.py, test_query_coverage.py, test_retrieval_schemas.py, test_startup_bootstrap.py, tests/conftest.py

### Dependencies

Removed: neo4j>=5.25.0, graphrag>=3.0.0, pyarrow>=14.0.0
Added: httpx>=0.27.0

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
9. Every node and edge contains source_document_id and page_number
10. Both upload_datetime and document_datetime persisted
11. .env and env.example cleaned of GraphRAG/Neo4j variables
12. Docker configuration cleaned of GraphRAG/Neo4j services
13. Tests pass with full unit and integration coverage
14. No stale imports, dead code, or orphaned config
15. Global query via community detection works end-to-end
16. Community detection is schedulable, manually triggerable, and has post-ingest hook
17. Schema syncs from ontology and updates when ontology changes
