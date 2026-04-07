# Multi-Modal Data Platform (MMDP) — Capability Overview

## Platform Summary

The Multi-Modal Data Platform (MMDP) is an enterprise-grade, fully air-gapped intelligence analysis platform purpose-built for ingesting, processing, enriching, and retrieving multi-modal documents across classified and unclassified environments. MMDP operates entirely within local infrastructure — no cloud APIs, no external data egress — making it suitable for deployment in sensitive government networks including SIPR and air-gapped enclaves.

MMDP combines document intelligence, knowledge graph construction, multi-modal vector search, LLM-powered synthesis, and human-governed data curation into a single, containerized platform. All machine learning models run locally via Ollama and purpose-built inference containers, ensuring data sovereignty at every layer.

---

## Core Architecture

MMDP is built on a microservices architecture deployed as a Docker Compose stack comprising nine containerized services:

| Service | Role |
|---------|------|
| **API Server** (FastAPI) | RESTful API gateway; async request handling |
| **Ingest Worker** (Celery) | Asynchronous document processing pipeline |
| **Scheduler** (Celery Beat) | Periodic tasks: directory watching, community detection |
| **PostgreSQL 16** | Metadata, governance, chunk linkage, audit trail |
| **ArcadeDB** | Multi-model database: knowledge graph + native vector search |
| **Ollama** | Local LLM/VLM inference (embeddings, extraction, synthesis) |
| **Redis** | Task broker, result backend, distributed locks |
| **MinIO** | S3-compatible object storage for source documents and artifacts |
| **Docling** | AI-powered document conversion (granite-docling-258M VLM) |

All services communicate over an isolated Docker network with no outbound internet access required.

---

## Ingestion Pipeline

MMDP's ingest pipeline is manifest-first, parallel, and idempotent — designed for high-throughput document processing with full provenance tracking at every stage.

### Supported Document Types
PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, and image formats (PNG, JPG, TIFF, GIF) including technical drawings and schematics.

### Processing Stages

1. **Document Preparation** — MIME detection, AI-powered conversion via Docling (PDF layout analysis, OCR with Tesseract/EasyOCR fallback, table extraction via TableFormer), and persistence of structured document elements.

2. **Metadata Extraction** — Four parallel LLM prompts extract: document summary, date of information (ISO 8601), classification level (UNCLASSIFIED through TOP SECRET), and source characterization.

3. **Image Description** — Multimodal VLM generates natural-language descriptions of embedded images, diagrams, and schematics using document context.

4. **Parallel Derivation** (concurrent):
   - **Text Chunking & Embedding** — Semantic chunking with configurable token limits, embedded via BGE-M3 (1024-dimensional vectors).
   - **Image Embedding** — Cross-modal CLIP embeddings (512-dimensional) enabling text-to-image search.
   - **Ontology Graph Extraction** — Automated entity and relationship extraction across configurable ontology groups using the Docling-Graph service, driven by the active domain ontology.

5. **Structure Linking** — Document structure traversal creates NEXT_CHUNK, SAME_PAGE, and SAME_SECTION relationships for context expansion during retrieval.

6. **Entity Canonicalization** — Alias resolution links variant entity names (e.g., "S-300" = "S-300V" = "TRI-ALPHA") to canonical nodes.

7. **Finalization** — Pipeline completion with status tracking (COMPLETE, PARTIAL_COMPLETE, or FAILED with per-stage diagnostics).

### Pipeline Resilience
- Per-stage tracking with attempt counts, timing metrics, and error capture
- Configurable retry policies with exponential backoff
- Stale document recovery on worker restart
- Redis-based concurrency gates to prevent resource exhaustion
- Re-ingest capability (full, embeddings-only, or graph-only)

---

## Knowledge Graph

MMDP constructs and maintains a richly typed knowledge graph in ArcadeDB, a multi-model database that unifies graph traversal and native vector search in a single engine.

### Domain-Configurable Ontology

A key differentiator of MMDP is its **fully configurable, domain-specific ontology system**. Rather than imposing a fixed data model, the platform is designed to ingest a custom ontology definition that reflects the specific entity types, relationship types, and validation rules relevant to a given mission or organization. The ontology governs all aspects of knowledge graph construction — from entity extraction to relationship validation to query profile design.

**How it works:** The ontology is defined as a structured YAML/JSON specification organized into thematic layers. Each layer declares a set of entity types and the relationship types that connect them. A validation matrix of allowable (source_type, relationship_type, target_type) tuples enforces graph integrity at write time — invalid triples are rejected before they enter the knowledge graph. This ensures that the graph always conforms to the organization's domain model, regardless of what the LLM extraction pipeline produces.

**For an FBI Threat Screening Center deployment**, the ontology would be tailored to the TSC's specific domain — for example:

- **Subjects & Identities** — Person, Known Alias, Identity Document, Biometric Record, Citizenship, Immigration Status
- **Organizations & Networks** — Organization, Cell, Affiliation, Financial Institution, Front Entity
- **Events & Activities** — Encounter, Travel Event, Financial Transaction, Communication, Incident, Investigation, Nomination
- **Locations & Infrastructure** — Address, Port of Entry, Facility, Border Crossing, Jurisdiction
- **Watchlist & Screening** — Screening Record, Nomination Record, Disposition, Derogatory Information, Source Report
- **Documents & Authorities** — SOP, Directive, Policy, Legal Authority, Court Order, Intelligence Report

Relationship types would similarly reflect the TSC domain — ASSOCIATED_WITH, TRAVELED_TO, MEMBER_OF, FUNDED_BY, NOMINATED_BY, IDENTIFIED_AT, SUBJECT_OF, SUPERSEDES, CITED_IN, and so on.

**This ontology is not hard-coded.** It is a configuration artifact that can be developed collaboratively with domain experts, versioned over time, and activated without code changes. The platform's extraction pipeline, graph validation, query profiles, and community detection all automatically adapt to whatever ontology is active.

### Ontology Versioning
A query profile registry system supports multiple concurrent ontology versions with explicit activation. Each registry embeds its full ontology definition, enabling version-controlled evolution of the knowledge model. A validation matrix of 200+ ontology-enforced (source_type, relationship, target_type) tuples ensures graph integrity — invalid triples are rejected before write.

### Community Detection
Louvain community detection runs periodically over the knowledge graph, identifying clusters of related entities. An LLM generates natural-language summaries for each community, enabling strategic-level queries that synthesize insights across document boundaries.

---

## Search and Retrieval

MMDP provides a unified query engine with three retrieval strategies and configurable modality filters, all accessible through a single API endpoint.

### Query Strategies

**Basic (Vector Search)** — Single-pass BGE text vector search returning ranked text chunks. Optimized for fast, keyword-centric queries with 1–3 second latency.

**Hybrid (Multi-Modal Fusion)** — The platform's primary retrieval mode, orchestrating:
1. Parallel vector seeds across text (BGE) and image (CLIP) embedding spaces
2. Document structure expansion via chunk linkage (next chunk, same page, same section)
3. Knowledge graph traversal (1–2 hop neighborhood expansion weighted by relationship type)
4. Independent re-scoring of expanded candidate set
5. Weighted fusion: 65% semantic similarity + 20% document structure + 15% ontology relevance + configurable domain-specific identifier bonuses
6. Cross-encoder reranking (BGE-reranker-v2-m3, GPU-accelerated)
7. Score threshold filtering (configurable minimum confidence)

Hybrid search returns text, image, table, and schematic results with full source attribution. Typical latency: 5–15 seconds.

**Global (LLM Synthesis)** — Leverages community detection summaries to synthesize broad, cross-document responses. The query is matched against community-level representations, and an LLM generates a unified narrative response. Designed for strategic questions and relationship discovery. Latency: 15–60 seconds.

### Modality Filters
All strategies support modality filtering: `all` (text + image + table + schematic), `text` (text chunks only), or `image` (visual content only).

### Source Attribution
Every retrieval result includes:
- `document_id` and `document_name` — originating source document
- `page_number` — specific page reference
- `artifact_id` — direct link to the extracted artifact (image, table, page)
- `classification` — classification level of the source material
- `chunk_text_preview` — excerpt from the source text
- For graph-derived results: `EXTRACTED_FROM` edge traversal linking entities back to the specific text chunk from which they were extracted

---

## Governance and Human-in-the-Loop Curation

MMDP implements a rigorous governance framework ensuring analytic integrity through human oversight at every mutation point.

### Feedback-to-Patch Workflow
1. Users submit structured feedback on any retrieval result (wrong text, incorrect entity, missing relationship, classification error, etc.)
2. The system automatically generates an RFC 6902 JSON Patch representing the proposed correction
3. Patches enter a state machine: DRAFT → UNDER_REVIEW → APPROVED → DUAL_APPROVED → APPLIED

### Dual-Curator Approval
All knowledge graph mutations (entity add/update/delete, relationship add, entity merge) require approval from two distinct curators. Text and classification corrections require single-curator approval. Self-approval is prevented by database constraint.

### Audit Trail
Every governance action generates an immutable `PatchEvent` record capturing: event type, actor identity, timestamp, and context metadata. Snapshots are captured at apply time to support rollback.

### Trusted Data Layer
A separate governed data layer allows analysts to propose high-confidence knowledge assertions with explicit source context and confidence scores. Proposals follow a PROPOSED → APPROVED → INDEXED workflow with curator review gates. Approved trusted data is embedded and indexed separately, queryable via a dedicated endpoint with confidence scores and provenance metadata.

---

## Agent and LLM Integration

MMDP exposes a purpose-built context endpoint (`/v1/agent/context`) designed for integration with LLM agent frameworks such as LangGraph. This endpoint accepts natural language queries and returns markdown-formatted retrieval context with structured source lists, enabling AI agents to ground their responses in authoritative, cited source material.

All LLM inference is performed locally via Ollama with configurable model selection per feature (document analysis, image description, community synthesis). The platform supports provider switching (Ollama for air-gapped, OpenAI for connected environments) via a single configuration variable.

---

## Data Lineage and Provenance

MMDP maintains full data lineage from raw source document through every derived artifact:

```
Source Document (raw file in MinIO)
  → Document Record (metadata, classification, pipeline status)
    → Document Elements (Docling-extracted structural items)
      → Artifacts (pages, images, tables, schematics)
        → Text/Image Chunks (embedded vectors in ArcadeDB)
          → Knowledge Graph Entities (EXTRACTED_FROM edges to source chunks)
```

Every entity in the knowledge graph carries `EXTRACTED_FROM` edges pointing to the specific text chunks from which it was derived. Every chunk carries references to its parent artifact, document, page number, and classification level. This chain is preserved through retrieval — query results always include the complete provenance path.

---

## Query Profiles and Automated Reporting

### Section Profiles
Pre-configured single-axis graph traversals (e.g., "all radar systems installed on naval platforms") that execute structured queries against the knowledge graph and return typed result sets.

### Dossier Profiles
Multi-section compositions that aggregate multiple section profiles into consolidated reports (e.g., a systems report combining radar, missile, and AAA sections). Dossier execution runs all constituent section profiles and merges results.

### Registry Management
Query profiles are organized within versioned registries tied to ontology versions, supporting controlled evolution of analytical templates. Default templates provide starter profiles for common intelligence queries.

---

## Security and Compliance

- **Air-Gapped Deployment** — All ML models pre-loaded at build time; no runtime internet access required
- **Classification Enforcement** — Five-level classification (UNCLASSIFIED through TOP SECRET) tracked at document, artifact, and chunk levels; query results propagate the maximum classification found in results
- **Data Isolation** — Separate storage zones for raw documents (MinIO), metadata (PostgreSQL), embeddings and graph (ArcadeDB), with no cross-zone data leakage
- **FIPS Compliance** — Docker builds include FIPS-compatible shims for deployment on FIPS-enabled kernels
- **Governance Enforcement** — Dual-curator approval gates, immutable audit logs, and revert capability
- **ABAC Framework** — Role-based access control architecture (analyst, curator, admin) with all actions attributed to authenticated user identity

---

## Infrastructure and Scalability

- **Horizontal Scaling** — Celery workers scale independently; multiple ingest workers can process documents concurrently
- **Async I/O** — FastAPI async endpoints with SQLAlchemy AsyncSession for non-blocking database access
- **Connection Pooling** — Persistent HTTP clients for Redis, Ollama, and ArcadeDB
- **Batch Processing** — Embedding batches (up to 64 texts per API call), Celery task grouping for parallel stage execution
- **Health Monitoring** — Liveness and readiness probes checking all dependent services
- **Management CLI** — Unified `manage.sh` for service lifecycle, migrations, seeding, and testing

---

## Alignment with RFI Requirements

The following matrix maps MMDP capabilities to the six requirements specified in the FBI Threat Screening Center AI Enhancement RFI.

### Requirement 1: AI-Enabled Knowledge Base for SOP and Directive Repository

| Requirement Element | MMDP Capability |
|---------------------|-----------------|
| Ingest and index SOPs, directives, policies | Multi-format ingestion pipeline (PDF, DOCX, PPTX, HTML, etc.) with AI-powered document conversion, semantic chunking, and vector indexing |
| Natural language query with real-time responses | Unified query endpoint with three strategies (basic, hybrid, global); hybrid strategy returns results in 5–15 seconds with weighted fusion across semantic, structural, and ontological dimensions |
| Context-aware responses with actionable guidance | Hybrid retrieval expands results via document structure (same-page, same-section context) and knowledge graph traversal; Global strategy synthesizes cross-document narrative responses via LLM |
| Explicit citations (document, section, paragraph) | Every result includes document name, page number, artifact reference, and chunk text preview; graph entities carry EXTRACTED_FROM edges to source text |
| Version control awareness | Ontology versioning via query profile registries with explicit activation; document re-ingest capability tracks processing versions; pipeline version metadata stored per document |

### Requirement 2: Federated Search with Summarization and Source Attribution

| Requirement Element | MMDP Capability |
|---------------------|-----------------|
| Federated search across structured and unstructured repositories | Hybrid strategy simultaneously searches across vector embeddings (unstructured), knowledge graph (structured relationships), document structure links, and trusted data — without requiring data consolidation into a single store |
| Synthesized summary of findings | Global query strategy uses Louvain community detection + LLM synthesis to generate narrative summaries across document boundaries; agent context endpoint provides markdown-formatted synthesized responses |
| Explicit citations identifying originating system, document, and data element | Source attribution chain preserved through every retrieval path: document → artifact → chunk → entity, with classification level propagation |
| Data lineage and transparency | Full provenance from raw document through every derived artifact; EXTRACTED_FROM graph edges trace entities to source text; pipeline stage metrics (timing, confidence scores, extraction versions) recorded per document |

### Requirement 3: Identifier-Driven Federated Reporting with Citation Retention

| Requirement Element | MMDP Capability |
|---------------------|-----------------|
| Report generation based on key identifiers (person, organization, asset, event) | Query profiles support identifier-driven searches: section profiles execute typed graph traversals by entity type (e.g., person, organization, asset, event — as defined in the domain ontology); dossier profiles aggregate multiple sections into consolidated reports |
| Configurable frequency parameters | Celery Beat scheduler supports configurable periodic execution; community detection runs on configurable intervals (default 60 minutes); directory watcher polls at configurable frequencies |
| Federated search across designated systems | Hybrid strategy searches across vector stores, knowledge graph, document structure, and trusted data layer simultaneously |
| Preserved explicit citations for each data element | Every data element in a report retains its full provenance chain: source document, page, artifact, chunk, and extraction confidence score |
| On-demand and scheduled execution | API supports on-demand query execution; Celery Beat supports scheduled task execution; community detection supports both triggered and periodic modes |

### Requirement 4: Natural Language Query and Dataset Synthesis with Source Transparency

| Requirement Element | MMDP Capability |
|---------------------|-----------------|
| Natural language queries across enterprise datasets | All three query strategies accept natural language input; BGE-M3 embeddings capture semantic meaning; the agent context endpoint is designed for LLM agent integration |
| Synthesized or summarized responses | Global strategy delivers LLM-synthesized narrative responses; community summaries aggregate and contextualize information across document collections |
| Aggregate, correlate, and contextualize across diverse data holdings | Hybrid strategy's weighted fusion scoring correlates semantic similarity (65%), document structure (20%), and ontological relationships (15%); cross-modal search correlates text queries with image content via CLIP embeddings |
| Explicit citations and traceability | Every output element — text chunk, image, table, graph entity — carries full source attribution metadata |
| Distinguish between extracted facts and AI-generated synthesis | Result modalities explicitly typed: `text`, `table`, `image`, `schematic` (extracted facts) vs. `community_response` (AI-generated synthesis); trusted data layer separates human-verified assertions from machine-extracted content; governance framework tracks data provenance category |

### Requirement 5: Predictive Modeling Using Enhanced Data with Traceable Lineage

| Requirement Element | MMDP Capability |
|---------------------|-----------------|
| Enriched data elements with documented source attribution | Pipeline enriches documents with LLM-extracted metadata (summary, classification, source characterization), image descriptions, and ontology entities — all with documented provenance |
| Analyze similarity and pattern alignment against existing records | Vector similarity search identifies semantically related content; knowledge graph traversal discovers structural patterns; community detection identifies entity clusters; entity canonicalization resolves aliases to reveal hidden connections |
| Predict where additional relevant information may be derived | Knowledge graph neighborhood expansion (1–2 hop traversal) identifies related entities and documents not directly matched by query; community detection reveals cross-document relationship patterns; weighted fusion scoring surfaces non-obvious connections through ontological proximity |
| Transparent reasoning with traceability to source data | EXTRACTED_FROM edges trace every graph entity to its source text; pipeline stage metrics record extraction confidence scores; fusion scoring weights are configurable and auditable; community summaries explain relationship clusters in natural language |
| Auditable lineage for analytic defensibility | Immutable PatchEvent audit trail; dual-curator approval gates for all graph mutations; trusted data layer with explicit confidence scores and review history; classification level propagation through all derived data |

### Requirement 6: Dynamic Data Visualization with Natural Language Query and Source Traceability

| Requirement Element | MMDP Capability |
|---------------------|-----------------|
| Interactive charts, graphs, dashboards | React frontend with graph topology visualization (GraphView), interactive document viewer (DoclingViewer), pipeline monitoring dashboard, and multi-modal result display |
| Geospatial displays | Platform architecture supports geospatial entity types in the ontology (locations, facilities, jurisdictions); visualization layer is extensible for map-based rendering |
| Natural language query initiation | All visualization-driving queries accept natural language input through the unified query API; agent context endpoint enables LLM-mediated query interpretation |
| Internal data dictionary and metadata catalog | Versioned ontology registry serves as the authoritative data dictionary with domain-configurable entity types, relationship types, and validation rules; query profiles provide pre-configured analytical templates |
| Filtering, drill-down, and temporal analysis | Modality filters (text/image/all), document structure expansion (drill from chunk to page to section), classification-level filtering, and date-of-information metadata support temporal analysis |
| Data sourcing and citation metadata in visualizations | All API responses include full source attribution; graph visualization edges carry relationship types and provenance metadata; every displayed data element is traceable to its authoritative source |

---

## Conclusion

MMDP delivers a production-ready, air-gapped intelligence analysis platform that directly addresses the core requirements of the FBI Threat Screening Center AI Enhancement initiative. Its combination of AI-powered multi-modal ingestion, ontology-governed knowledge graph construction, hybrid retrieval with weighted fusion scoring, LLM synthesis, and rigorous human-in-the-loop governance provides a foundation for trustworthy, auditable, and transparent AI-enhanced threat screening operations.

The platform's fully local deployment model, classification-aware data handling, dual-curator approval workflow, and immutable audit trail are specifically designed for environments where data sovereignty, analytic defensibility, and oversight compliance are non-negotiable requirements.
