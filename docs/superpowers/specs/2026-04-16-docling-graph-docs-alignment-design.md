# Design Spec — docling-graph docs-alignment refactor

**Date:** 2026-04-16
**Status:** Draft for review
**Blocks:** completion of [`2026-04-14-docling-graph-schema-compliance.md`](./2026-04-14-docling-graph-schema-compliance.md) — that plan's Phase 7 Task 53 live-extraction gate cannot pass until this plan lands.
**Authoritative reference:** [`/docling-graph-docs.md`](../../../docling-graph-docs.md) at repo root.

## 1. Scope & approach

This plan realigns the `air_defense_v3` canonical ontology and extraction schemas with the docling-graph library's documented patterns. The motivating finding (2026-04-16 live-extraction investigation) is that the `reference` LLM pass cannot reliably populate its identity fields (`heading`, `page_start`) because the library's docs explicitly call this out as an anti-pattern:

> Prefer short, document-derived ID examples (section numbers, figure/table labels, named items). **Do not use section or chapter titles as entity identities.** — docling-graph-docs.md:18470

### Core theses

1. **Document structure comes from Docling, not the LLM.** SECTION/FIGURE/TABLE are emitted by a new deterministic worker pre-pass that walks `DoclingDocument` structure. The LLM reference pass is deleted. (Ontology DOCUMENT is distinct from the structural `Document` graph vertex — see thesis 2.)
2. **Two DOCUMENT concepts, keep them separate.** The **structural `Document`** vertex (capitalized, ArcadeDB class, identity = UUID) remains the target of `HAS_PROVENANCE`/`CONTAINS_TEXT`/`EXTRACTED_FROM` and is created by `_ensure_structural_document_vertex` + `derive_structure_links` — unchanged by this plan. The **ontology `DOCUMENT`** entity (uppercase, `entities.py`, identity = `document_number` like `"TM 9-1425-386-12"`) is created by the new anchors pre-pass only when `document_number` is extractable; otherwise skipped.
3. **Every canonical entity gets a docs-compliant identity.** No `graph_id_fields=[]`. No long-text identities. No multi-field identities for document structure. Value objects become `is_entity=False` components.
4. **Components are first-class graph vertices per docs.** The existing walker hard-skips components (`extraction_merge.py:564`, `:593`), a divergence from docs where components are shared graph nodes (docs:17500–17509). The plan updates the walker to emit components reached via `edge(label=...)` as proper graph vertices with content-based dedup, matching docs.
5. **ASSERTION is dropped** from the ontology entirely (YAGNI — reintroduce later with real design if a concrete use-case emerges).
6. **Schemas are fault-tolerant by construction.** Strict required identity + lenient coercers on non-identity fields + library-level identity filter and quality gate (R22) handle messy LLM output.
7. **Full corpus re-ingest after canonical changes.** 21 docs re-processed via full pipeline (including Docling reconversion in `prepare_document`) to validate end-to-end on new schemas.

### Out of scope

- Upstream `docling-graph` library patches (we handle everything at the schema level).
- New entity types beyond the audit.
- Retrieval-algorithm changes.
- UI redesign (except the single ASSERTION filter-list deletion).
- Phase 8 provenance from the current plan — that resumes after this plan lands.

### Sequencing relative to the current plan

- **This plan precedes** the current `2026-04-14-docling-graph-schema-compliance.md`.
- Current plan's Phases 1–6 are already done, Phase 7 Task 52 done. Phase 7 Task 53 (live E2E gate) and Phase 8 (provenance) stay paused until this plan lands and merges.
- After this plan merges, the current plan resumes from Task 53 on stable schemas; Phase 8 provenance is additive on top.

## 2. Canonical entity classification

The audit examined all 46 canonical entities in `ontology_bundles/air_defense_v3/entities.py` (verified via `grep -c "^class .*(BaseModel):" entities.py`). Below is the classification table that drives every rewrite in Chunk B. Every canonical class is accounted for in exactly one bucket.

### 2.1 Dropped from ontology (2)

| Entity | Reason |
|---|---|
| ASSERTION | Q3 decision + longstanding KNOWN ANTI-PATTERN (`graph_id_fields=["assertion_text"]` violates R2 "short identities"). Drop entirely; reintroduce later if a concrete use-case emerges with a better design. |
| SPREADSHEET | Redundant with DOCUMENT. Consolidate by adding `SPREADSHEET` to `DocumentEntity.source_type` enum. `workbook_name` and `sheet_name` become optional properties on DocumentEntity. |

### 2.2 Give proper identity (15)

| Entity | Current | New `graph_id_fields` | Scope | Notes |
|---|---|---|---|---|
| DOCUMENT | `[]` | `["document_number"]` required | global | Ontology-level identity = official document designator (e.g. `"TM 9-1425-386-12"`, `"MIL-STD-1553B"`). **Distinct from the structural `Document` vertex** whose identity is the internal UUID and whose lifecycle stays in `derive_structure_links`. The anchors walker emits ontology DOCUMENT only when `document_number` is extractable from the source (e.g. front-matter designator); otherwise no ontology DOCUMENT is created for that doc — the structural vertex still exists. |
| SECTION | `["heading","page_start"]` | `["section_number"]` required | document | Positional enumeration from Docling `section_path` walk (e.g. `"1"`, `"1.1"`, `"2.3.4"`). R17 exempted because Docling-derived (see §2.5 rationale). **Non-identity properties to add to SectionEntity:** `heading: Optional[str] = None` (descriptive — the tail of path_tuple), `section_path: Optional[str] = None` (joined Docling path for TextChunk joins per §3.4), `document_id: Optional[str] = None` (stamped from walker's `document_uuid` arg for graph-side joins per §3.4). |
| FIGURE | `["figure_id","page"]` | `["figure_ref"]` required | document | Docling `self_ref` (e.g. `"#/pictures/3"`). **Non-identity properties to add to FigureEntity:** `figure_label: Optional[str] = None` (human-readable "Figure 3-12" from caption parsing), `document_id: Optional[str] = None` (stamped for joins). |
| TABLE | `["table_id","page"]` | `["table_ref"]` required | document | Docling `self_ref` (e.g. `"#/tables/1"`). **Non-identity properties to add to TableEntity:** `table_label: Optional[str] = None` (descriptive), `document_id: Optional[str] = None` (stamped for joins). |
| ORGANIZATION | `[]` | `["name"]` required | global | `cage_code` stays as an optional secondary property. |
| STANDARD | `[]` | `["designation"]` required | global | `"MIL-STD-1553B"`, `"MIL-DTL-31000G"`. |
| EQUIPMENT_SYSTEM | `[]` | `["name"]` required | global | Named trackable systems. |
| COMPONENT | `[]` | `["part_number"]` required | global | Components without a part_number in the source drop via library identity filter; that's acceptable (R18 "when possible"). |
| ASSEMBLY | `[]` | `["assembly_number"]` required | global | Same pattern as COMPONENT. |
| CAPABILITY | `[]` | `["capability_name"]` required | global | Named operational features. |
| PROCEDURE | `[]` | `["name"]` required | document | Doc-scoped — same-named procedure across unrelated docs ≠ same procedure. |
| FAILURE_MODE | `[]` | `["name"]` required | document | Doc-scoped. |
| TEST_EVENT | `[]` | `["name"]` required | global | Named events, globally unique. |
| FORCE_STRUCTURE | `[]` | `["name"]` required | global | Named military units. |
| SUBSYSTEM | `[]` | `["name"]` required | document | Doc-scoped hierarchical unit (docs Pattern 7 precedent, L16645). "Guidance Section" in Missile A ≠ "Guidance Section" in Missile B. |

### 2.3 Demote to `is_entity=False` (12 components)

Deduplicate by full content per docs L17225+ pattern.

| Entity | Pattern match |
|---|---|
| SPECIFICATION | Measurement — parameter+value+unit+condition is definitionally content (docs Address/Measurement pattern). |
| MODULATION | Value bag: chirp type + bandwidth + code bits. |
| RF_SIGNATURE | Bundle of RF observations. |
| RF_EMISSION | Observation snapshot. |
| SCAN_PATTERN | Value bag: scan type + timing. |
| IF_AMPLIFIER | Electrical stage description. |
| MISSILE_PERFORMANCE | Performance envelope data. |
| MISSILE_PHYSICAL_CHARACTERISTICS | Dimensions + mass. |
| PROPULSION_STACK | Container for stages (KNOWN ANTI-PATTERN in current code). |
| PROPULSION_STAGE | Sub-unit of stack; content-dedup on stage_type+params. |
| RADAR_PERFORMANCE | Performance envelope. |
| ENGAGEMENT_TIMELINE | Timing envelope. |

### 2.4 Unchanged (17 entities, already docs-compliant)

| # | Entity | Identity | Scope |
|---|---|---|---|
| 1 | PlatformEntity | `["name"]` | global |
| 2 | WeaponSystemEntity | `["system_name"]` | global |
| 3 | RadarSystemEntity | `["system_name"]` | global |
| 4 | MissileSystemEntity | `["system_name"]` | global |
| 5 | AirDefenseArtillerySystemEntity | `["system_name"]` | global |
| 6 | ElectronicWarfareSystemEntity | `["system_name"]` | global |
| 7 | FireControlSystemEntity | `["system_name"]` | global |
| 8 | IntegratedAirDefenseSystemEntity | `["name"]` | global |
| 9 | LauncherSystemEntity | `["system_name"]` | global |
| 10 | FrequencyBandEntity | `["band_name"]` | global |
| 11 | WaveformEntity | `["waveform_name"]` | document |
| 12 | AntennaEntity | `["name"]` | document |
| 13 | TransmitterEntity | `["name"]` | document |
| 14 | ReceiverEntity | `["name"]` | document |
| 15 | SignalProcessingChainEntity | `["name"]` | document |
| 16 | GuidanceMethodEntity | `["guidance_type"]` (enum) | global |
| 17 | SeekerEntity | `["seeker_nomenclature"]` | document |

All have single-field short identities on already-extracted named concepts. **These still receive docs-compliance touch-ups in Chunk B**: identity field examples get the §4.3 treatment (distinct, no duplicates, R16/R17-aligned), non-identity fields confirmed `Optional[T] = None` per §4.4, `edge(label=...)` fields gain `description` + `examples` kwargs per §7.1. But the identity shape doesn't change.

**Totals:** 46 canonical classes today → 2 dropped + 15 give-identity + 12 demoted + 17 unchanged = **46 accounted for**. Post-refactor: **32 entities + 12 components = 44 models** in `entities.py`.

### 2.5 Rule-conflict reconciliation for SECTION

Docs rules R14 ("prefer descriptive IDs"), R17 ("avoid examples like `3.1`"), and R21 ("prefer short document-derived IDs like section numbers") appear to conflict for SECTION.

The reconciliation: **R14/R17 apply to LLM-emitted identities** (the intent is to steer model output via examples). **R21 applies to system-constructed identities** (section numbers ARE listed as preferred). SECTION is populated deterministically by the Docling anchor walker; the LLM never sees SECTION's `section_number` field. Therefore R21 governs, and `section_number` with positional-enumeration examples (`["1", "1.1", "2.3.4"]`) is correct.

Captured in a `# Docs rule note:` comment on the SECTION class in entities.py, referencing this section.

### 2.6 Per-entity Chunk B task list

For the implementation plan's Chunk B, this is the per-entity rewrite task list. 27 entities need structural changes (15 give-identity + 12 demote) + 17 touch-ups. Ordered by dependency (entities referenced by others come first):

**Drops (2 tasks):**
1. Delete `AssertionEntity` + remove all references.
2. Delete `SpreadsheetEntity` + merge into `DocumentEntity.source_type` enum.

**Give-identity batch (15 tasks):**
3. `DocumentEntity.graph_id_fields=["document_number"]` required; rename `document_id` field (if present) to avoid collision with structural vertex's `document_id` property.
4. `SectionEntity.graph_id_fields=["section_number"]` required.
5. `FigureEntity.graph_id_fields=["figure_ref"]` required; `figure_label` added as optional.
6. `TableEntity.graph_id_fields=["table_ref"]` required; `table_label` added as optional.
7. `OrganizationEntity.graph_id_fields=["name"]` required.
8. `StandardEntity.graph_id_fields=["designation"]` required.
9. `EquipmentSystemEntity.graph_id_fields=["name"]` required.
10. `ComponentEntity.graph_id_fields=["part_number"]` required.
11. `AssemblyEntity.graph_id_fields=["assembly_number"]` required.
12. `CapabilityEntity.graph_id_fields=["capability_name"]` required.
13. `ProcedureEntity.graph_id_fields=["name"]` required, scope=document.
14. `FailureModeEntity.graph_id_fields=["name"]` required, scope=document.
15. `TestEventEntity.graph_id_fields=["name"]` required.
16. `ForceStructureEntity.graph_id_fields=["name"]` required.
17. `SubsystemEntity.graph_id_fields=["name"]` required, scope=document.

**Demote batch (12 tasks):** Each task flips `is_entity=False` AND drops `graph_id_fields` AND flips any previously-required non-identity fields to `Optional[T] = None`. Note: `SpecificationEntity` currently has required `parameter: str` and `value: str`; both become Optional during demotion (docs R19 — content-dedup components accommodate sparse data). Same pattern applies wherever demoted entities had required fields that weren't identity.

18. `SpecificationEntity`: `is_entity=False`; flip `parameter` + `value` to Optional.
19. `ModulationEntity`: `is_entity=False`.
20. `RfSignatureEntity`: `is_entity=False`.
21. `RfEmissionEntity`: `is_entity=False`.
22. `ScanPatternEntity`: `is_entity=False`.
23. `IfAmplifierEntity`: `is_entity=False`.
24. `MissilePerformanceEntity`: `is_entity=False`.
25. `MissilePhysicalCharacteristicsEntity`: `is_entity=False`.
26. `PropulsionStackEntity`: `is_entity=False`.
27. `PropulsionStageEntity`: `is_entity=False`.
28. `RadarPerformanceEntity`: `is_entity=False`.
29. `EngagementTimelineEntity`: `is_entity=False`.

**Touch-up batch (aggregated, 3 tasks — batched not per-entity):**
30. All 17 unchanged entities: apply R16/R17 example-list cleanup (2-5 distinct, no duplicated examples). One commit.
31. All entities: confirm non-identity fields are `Optional[T] = None`; flip any that aren't. One commit.
32. All entities with `edge(label=...)` fields: extend `edge()` helper to accept `description` + `examples` kwargs, update every call site. One commit.

**Chunk B total: 32 tasks, 32 commits.** (Up from the earlier 18 estimate.)

## 3. Docling-derived anchors — architecture

### 3.1 New Celery task

File: `app/workers/pipeline.py`
Task: `derive_document_anchors(document_id, run_id)` with queue `graph` (co-located with `derive_ontology_graph`).

**Scope clarification:** this task emits ontology-layer entities (SECTION/FIGURE/TABLE + optionally ontology DOCUMENT) and the ontology-layer edges between them (`HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE`/`CHILD_OF`). It does **not** create or modify the structural `Document` vertex — that lifecycle stays with `_ensure_structural_document_vertex` (pipeline.py:999) and `derive_structure_links` (pipeline.py ~line 4433).

### 3.2 Position in Celery chain

Between `derive_image_embeddings` and `derive_ontology_graph`:

```
prepare → translate → metadata → purge → picture_descriptions
  → text_chunks → image_embeddings
  → [NEW] derive_document_anchors
  → derive_ontology_graph → collect → structure_links → canonicalization → finalize
```

Rationale:
- Runs after purge (no concurrency with cleanup).
- Runs before `derive_ontology_graph` so downstream merge can observe anchor nodes.
- Runs before `derive_structure_links` so `EXTRACTED_FROM` edges from TextChunks have live SECTION/FIGURE/TABLE target vertices.

### 3.3 Walker logic

Inputs:
- `docling_document.json` from MinIO (via existing `_build_docling_document_json`).
- `document_uuid: str` — the internal ingest UUID for this document. Used ONLY for (a) `identity_scope="document"` scoping of `_build_logical_identity` calls, and (b) stamping `document_id` as a **property** on SECTION/FIGURE/TABLE vertices (for graph-side joins to TextChunks per §3.4). It is NOT the ontology DOCUMENT's identity — ontology `DocumentEntity.graph_id_fields=["document_number"]` per §2.2.
- `pipeline_run_id: str` — pipeline_run identifier, passed through to `MergedExtraction.pipeline_run_id`.
- `ontology: dict` — the active ontology dict (from `load_ontology()`), passed through to `_build_logical_identity`.

Outputs:
- **Ontology `DocumentEntity`** — emitted **only when** `document_number` is extractable from the Docling document's front matter (heuristic: first `TITLE` + `SECTION_HEADER` items scanned for MIL-STD / TM / similar-pattern designators via regex). If no `document_number` is found, **no ontology DOCUMENT is created** and the `HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE` edges below are skipped. The structural `Document` vertex (separate concept, identity = UUID) is unaffected — it's always created by `derive_structure_links`.
- N `SectionEntity` records — one per unique `section_path` encountered (deduplicated across elements sharing that path).
- M `FigureEntity` records — one per entry in `docling_document.pictures`.
- K `TableEntity` records — one per entry in `docling_document.tables`.
- Edges (only when ontology DOCUMENT was emitted):
  - `(ontology DOCUMENT)-[:HAS_SECTION]->(SECTION)` for each SECTION
  - `(ontology DOCUMENT)-[:HAS_FIGURE]->(FIGURE)` for each FIGURE
  - `(ontology DOCUMENT)-[:HAS_TABLE]->(TABLE)` for each TABLE
  - `(SECTION)-[:CHILD_OF]->(SECTION)` for hierarchical nesting

SECTION/FIGURE/TABLE vertices are still emitted when no ontology DOCUMENT is created; they just don't get doc-level edges. They remain queryable and attach to the structural `Document` vertex via `derive_structure_links`' `CONTAINS_TEXT`/`EXTRACTED_FROM` edges (unchanged structural flow).

#### Design decision: where does `section_path` come from?

Docling's native `ProvenanceItem` (`docling_core.types.doc.ProvenanceItem`) does NOT carry `section_path` — it only has `page_no`, `bbox`, `charspan`. The codebase's `section_path` concept is computed at **conversion time** by `docker/docling/app/converter.py:296–318`, which walks `doc.iterate_items()` and maintains a `section_stack` of `(heading_level, text)` pairs pushed on each `SECTION_HEADER` / `TITLE` item. The resulting `section_path` string (e.g., `"Chapter 3 > Section 3.1"`) is stored on `ingest.document_elements.section_path` (persisted) and on `TextChunk.section_path`.

The anchor walker therefore **replicates the same section-stack logic** when iterating `DoclingDocument.iterate_items()` — mirroring `converter.py:296–318` so anchor generation is self-contained (no DB dependency on `document_elements`, no dependency on converter-internal DTOs).

#### Deterministic algorithm (pseudocode)

```python
def walk(
    docling_doc_json: dict,
    document_uuid: str,
    pipeline_run_id: str,
    ontology: dict,
) -> MergedExtraction:
    """Derive DOCUMENT/SECTION/FIGURE/TABLE entities + edges from a DoclingDocument.

    Args:
      docling_doc_json: the persisted docling_document.json dict (from
          _build_docling_document_json at app/workers/pipeline.py:923).
      document_uuid: internal UUID; used for identity_scope="document" and
          stamped as document_id property on SECTION/FIGURE/TABLE vertices.
      pipeline_run_id: current pipeline_run; propagated to MergedExtraction.
      ontology: active ontology dict (load_ontology()); passed to
          _build_logical_identity for LogicalIdentity construction.
    Returns:
      MergedExtraction carrying entity + edge records.
    """
    from docling_core.types.doc import DoclingDocument, DocItemLabel

    docling_doc = DoclingDocument.model_validate(docling_doc_json)

    # --- Ontology DOCUMENT: conditional on extractable document_number --------
    # Scan the first N items (titles + top-level section headers) for a
    # MIL-STD / TM-number / similar designator via regex. Returns None when
    # no official designator is detectable — in which case NO ontology
    # DOCUMENT is emitted and HAS_* edges below are skipped.
    document_number = _extract_document_number_from_front_matter(docling_doc)
    doc_entity = (
        DocumentEntity(document_number=document_number, ...)
        if document_number is not None
        else None
    )

    # --- section_path construction (mirrors converter.py:296–318) -------------
    # Walk items in document order. Maintain a section_stack of (level, text).
    # For each item encountered, compute its section_path as the tuple of
    # stack entries BEFORE processing that item (for non-heading items) or
    # AFTER the push (for heading items — so a heading appears inside its own
    # section). This matches the chunking contract at converter.py:317.

    # HEADING_LABELS intentionally narrower than converter.py:302–307 —
    # converter includes PAGE_HEADER/PAGE_FOOTER so running headers/footers
    # appear in chunk-level section_path strings. For graph-visible SECTION
    # nodes those crumbs are noise (cosmetic repeats per page), so the
    # walker excludes them. Deliberate divergence; do not add to this tuple
    # without also considering the SECTION-node clutter trade-off.
    HEADING_LABELS = (
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.TITLE,
    )

    section_stack: list[tuple[int, str]] = []
    section_by_path: OrderedDict[tuple[str, ...], SectionEntity] = OrderedDict()
    sibling_counters: dict[tuple[str, ...], int] = defaultdict(int)

    def _register_section(path_tuple: tuple[str, ...]) -> None:
        """Dedup + positional enumerate a SECTION. No-op if path already seen."""
        if not path_tuple or path_tuple in section_by_path:
            return
        parent_tuple = path_tuple[:-1]
        sibling_counters[parent_tuple] += 1
        idx = sibling_counters[parent_tuple]
        parent_number = (
            section_by_path[parent_tuple].section_number
            if parent_tuple in section_by_path
            else None
        )
        section_number = f"{parent_number}.{idx}" if parent_number else str(idx)
        section_by_path[path_tuple] = SectionEntity(
            section_number=section_number,
            heading=path_tuple[-1],
            ...,
        )

    for item, level in docling_doc.iterate_items():
        label = getattr(item, "label", None)
        text = getattr(item, "text", None) or ""

        if label in HEADING_LABELS and text.strip():
            heading_level = level if level and level > 0 else 1
            if label == DocItemLabel.TITLE:
                heading_level = 1
            # Pop deeper-or-equal stack entries, push this heading.
            while section_stack and section_stack[-1][0] >= heading_level:
                section_stack.pop()
            section_stack.append((heading_level, text.strip()))

        # Current section path AFTER the heading push (so a heading appears
        # inside its own section). Non-heading items see the stack as-is.
        path_tuple = tuple(entry[1] for entry in section_stack)
        _register_section(path_tuple)

    # --- Fallback: document with zero headings ---------------------------------
    if not section_by_path:
        fallback_path = ("",)  # sentinel path for the document-body section
        section_by_path[fallback_path] = SectionEntity(
            section_number="0",
            heading=None,
            ...,
        )

    # --- Figures: one per picture ref -----------------------------------------
    def _caption_label(item: dict) -> str | None:
        """Best-effort extract of a human-readable label like "Figure 3-12" from
        the picture/table's captions array. Returns the first caption text that
        starts with "Figure", "Fig.", "Table", or "Tbl." (case-insensitive).
        None if no such caption found. The full caption text is stored on a
        separate `caption` property — this helper returns only the short label.
        """
        captions = item.get("captions") or []
        for cap in captions:
            cap_text = (cap.get("text") or "").strip()
            if re.match(r"^(figure|fig\.?|table|tbl\.?)\s", cap_text, re.IGNORECASE):
                # Take just the label prefix (e.g., "Figure 3-12" from longer caption).
                m = re.match(r"^(figure|fig\.?|table|tbl\.?)\s+[\w.-]+", cap_text, re.IGNORECASE)
                return m.group(0) if m else cap_text.split(".")[0].strip()
        return None

    figures = [
        FigureEntity(
            figure_ref=item["self_ref"],
            figure_label=_caption_label(item),
            ...,
        )
        for item in docling_doc_json.get("pictures", [])
    ]

    tables = [
        TableEntity(
            table_ref=item["self_ref"],
            table_label=_caption_label(item),
            ...,
        )
        for item in docling_doc_json.get("tables", [])
    ]

    # --- Edges ------------------------------------------------------------------
    # MergedEdgeRecord fields per app/services/extraction_merge.py:222 —
    # from_identity, to_identity, rel_type (NOT from/to/label).
    # MergedExtraction fields per :262 — entities, edges (NOT merged_entities/merged_edges).
    edges = []
    if doc_entity is not None:
        doc_identity = _build_logical_identity("DOCUMENT", doc_entity, ontology, document_uuid)
        for section in section_by_path.values():
            section_identity = _build_logical_identity("SECTION", section, ontology, document_uuid)
            edges.append(MergedEdgeRecord(
                from_identity=doc_identity, to_identity=section_identity,
                rel_type="HAS_SECTION", confidence=1.0, pass_origins={"document_anchors"},
            ))
        for fig in figures:
            fig_identity = _build_logical_identity("FIGURE", fig, ontology, document_uuid)
            edges.append(MergedEdgeRecord(
                from_identity=doc_identity, to_identity=fig_identity,
                rel_type="HAS_FIGURE", confidence=1.0, pass_origins={"document_anchors"},
            ))
        for tbl in tables:
            tbl_identity = _build_logical_identity("TABLE", tbl, ontology, document_uuid)
            edges.append(MergedEdgeRecord(
                from_identity=doc_identity, to_identity=tbl_identity,
                rel_type="HAS_TABLE", confidence=1.0, pass_origins={"document_anchors"},
            ))
    # CHILD_OF: hierarchical, independent of ontology DOCUMENT.
    for path_tuple, section in section_by_path.items():
        parent_tuple = path_tuple[:-1]
        if parent_tuple and parent_tuple in section_by_path:
            parent_section = section_by_path[parent_tuple]
            child_identity = _build_logical_identity("SECTION", section, ontology, document_uuid)
            parent_identity = _build_logical_identity("SECTION", parent_section, ontology, document_uuid)
            edges.append(MergedEdgeRecord(
                from_identity=child_identity, to_identity=parent_identity,
                rel_type="CHILD_OF", confidence=1.0, pass_origins={"document_anchors"},
            ))

    entity_models = [*section_by_path.values(), *figures, *tables]
    if doc_entity is not None:
        entity_models.insert(0, doc_entity)
    # Wrap each as MergedEntityRecord (ontology_name, identity, properties, confidence,
    # pass_origins, display_label) — fields per extraction_merge.py:213.
    merged_entities = [_to_merged_entity_record(m, ontology, document_uuid) for m in entity_models]

    return MergedExtraction(
        entities=merged_entities,
        edges=edges,
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id=pipeline_run_id,
        document_id=document_uuid,
    )
```

#### Determinism properties

- `DoclingDocument.iterate_items()` walks body descendants in declared order, resolving `self_ref`/`$ref` pointers. Identical input JSON → identical output order. (Confirmed real Docling API at `docling_core/types/doc/document.py:5387–5396`; used by `docker/docling/app/converter.py:298`.)
- `section_stack` mirrors `converter.py:302–317` exactly — same pop-depth-equal, push-self rule. Headings go inside their own section (e.g., a `"Chapter 3"` heading has `path_tuple = ("Chapter 3",)`, not `()`).
- Dedup key for SECTION: the tuple of heading stems (stack entries). Two elements sharing that tuple attach to the same SECTION.
- `section_number` tie-breaking: document-order first-occurrence. If `("A", "B")` appears at item 10 and `("A", "C")` appears at item 20, then B gets `"1.1"` and C gets `"1.2"` (assuming A is `"1"`).
- Parent-child edge resolution: by `path_tuple[:-1]` lookup on `section_by_path` — no string parsing.
- Fallback (§3.4) is wired into the pseudocode: a single post-loop check creates one SECTION with `section_number="0"` when no headings were encountered.
- `figure_ref` / `table_ref` are Docling's own `self_ref` values — globally unique within a single `DoclingDocument`.
- `_caption_label` is best-effort. When Docling provides no captions or no number-prefixed captions, the `figure_label` / `table_label` property is `None`. Never blocks identity (identity uses `self_ref`).

### 3.4 Fallback for weak structure

If the walker encounters no `SECTION_HEADER` / `TITLE` items at all (a pathological case — Docling usually emits at least a title), the post-loop guard in §3.3 emits one `SectionEntity(section_number="0", heading=None)` representing the whole document body. Prevents zero-SECTION docs.

**Clarification on TextChunk ↔ SECTION linkage:** in the current code, `derive_structure_links` at `app/workers/pipeline.py:4376` creates `SAME_SECTION` edges **chunk-to-chunk** (text_chunk ↔ text_chunk sharing `artifact_element_map[str(tc.artifact_id)].section_path`). It does NOT create TextChunk → SECTION-vertex edges. This plan does not add that attachment either; SECTION vertices are queryable graph nodes but are not directly linked to TextChunks. Retrieval consumers that need the link join by stamped properties:

- SECTION gains a new optional property `section_path: Optional[str]` holding the joined Docling path (e.g. `"Chapter 3 > Section 3.1"`) — same string shape as `TextChunk.section_path`. Non-identity.
- SECTION/FIGURE/TABLE **stamp `document_id` as a non-identity property** on the vertex (value = `document_uuid` param from §3.3). Today `TextChunk` carries `document_id`; this makes SECTION carry it too, enabling a symmetric join. The property is populated by `_to_merged_entity_record` during MergedExtraction construction, flowing to `NodeRecord.properties["document_id"]` on upsert. This is purely a query-affordance property — identity remains single-field (`section_number` / `figure_ref` / `table_ref`), and cross-document dedup via `identity_scope="document"` still relies on the LogicalIdentity machinery, NOT this property.
- Graph-side join works without new edge types: `MATCH (tc:TextChunk), (s:SECTION) WHERE tc.section_path = s.section_path AND tc.document_id = s.document_id`.
- Adding `TextChunk-[CONTAINED_IN]->SECTION` edges is deferred (would be new structural-edge work + a new pass over text_chunks). Out of scope for this plan.

### 3.5 Write path

**Step 1 — Vertex upserts.** Use `graph_store.upsert_nodes_batch_sync`. Identity resolution is automatic via `graph_id_fields` in `model_config`. `upsert_nodes_batch_sync` returns the list of `@rid`s in the same order as the input records.

**Step 2 — Bridge: identity → RID map.** The pseudocode's `MergedEdgeRecord` uses `from_identity: LogicalIdentity` / `to_identity: LogicalIdentity` for edges, but `create_structural_edge_sync` (next step) takes `@rid` strings. A bridging step resolves each `LogicalIdentity` in the edge list to the `@rid` returned by step 1. Concretely:

```python
# After upsert_nodes_batch_sync(records) → rids (same order as records),
# build:
rid_by_identity: dict[LogicalIdentity, str] = {
    record.identity: rid for record, rid in zip(merged_entity_records, rids)
}
```

`LogicalIdentity` is hashable (frozen dataclass per the existing merge code). All edges in `merged_extraction.edges` reference identities of vertices that were just upserted, so every lookup resolves.

**Step 3 — Edge writes use `graph_store.create_structural_edge_sync`**, NOT `upsert_relationships_batch_sync`. The latter (at `arcadedb_graph.py:1796`) enforces the ontology validation-matrix via `_enforce_relationship_triple` at `:1806` and rejects any triple not in `VALIDATION_MATRIX`. Since `HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE`/`CHILD_OF` are deliberately classified as structural (not ontology-domain, per §3.5a), they're not in the validation matrix and must go through the structural path. `create_structural_edge_sync` (at `:1816`) takes source `@rid`, target `@rid`, and label — no matrix check. For each `MergedEdgeRecord` in `merged_extraction.edges`:

```python
for edge in merged_extraction.edges:
    from_rid = rid_by_identity[edge.from_identity]
    to_rid = rid_by_identity[edge.to_identity]
    graph_store.create_structural_edge_sync(from_rid, to_rid, edge.rel_type)
```

This mirrors how `derive_structure_links` already writes `CONTAINS_TEXT`/`SAME_SECTION`/`SAME_PAGE`.

Audit row: the task creates a `StageRun` row with `stage_name="derive_document_anchors"` and `pass_name=NULL` (per the StageRun schema at `app/models/ingest.py:273`; `DocumentGraphExtraction` is one-row-per-document and has no `pass_name` column — it's not a per-pass audit channel). StageRun metrics on the new row record: section_count, figure_count, table_count, document_ontology_emitted (bool: whether `document_number` was extractable), fallback_fired (bool: whether §3.4 sentinel SECTION was needed).

### 3.5a New edge-type declarations (precondition)

The four new labels `HAS_SECTION`, `HAS_FIGURE`, `HAS_TABLE`, `CHILD_OF` must be declared across three registries before the anchor walker can write them. These are **structural/document-layout edges** (they don't participate in the ontology's military-domain semantics), so they go in the structural-edge registry, not the ontology-relationship enum:

- **`app/services/arcadedb_schema.py`**: add to `_STRUCTURAL_EDGE_TYPES` list (currently: `CONTAINS_TEXT`, `CONTAINS_IMAGE`, `SAME_PAGE`, `SAME_SECTION`, `SAME_ARTIFACT`, `NEXT_CHUNK`, `HAS_PROVENANCE`, `EXTRACTED_FROM`, `HAS_ALIAS` — line 79). New entries: `HAS_SECTION`, `HAS_FIGURE`, `HAS_TABLE`, `CHILD_OF`.
- **`ontology_bundles/air_defense_v3/relationships.py`**: these do NOT get added to `RelationshipType` enum — that enum is for ontology-domain relationships (HAS_ANTENNA, USES_WAVEFORM, etc.). Document-structure edges stay out of the ontology relationship set.
- **`ontology_bundles/air_defense_v3/validation_matrix.py`**: same principle — no validation-matrix entries for these (the matrix is ontology-domain triples). Comment added at file top noting the structural-edge exclusion.

This classification preserves the separation of ontology-domain (LLM-extracted, validation-gated) vs structural (deterministic, layout-level) edges.

**Explicit docs-adherence note:** `HAS_SECTION` / `HAS_FIGURE` / `HAS_TABLE` / `CHILD_OF` are **intentionally outside** the ontology relationship validation system. They are NOT registered in `RelationshipType` enum, NOT in `VALIDATION_MATRIX`, and NOT written via `upsert_relationships_batch_sync` (which enforces matrix triples and would reject them). They are written via `create_structural_edge_sync` (§3.5) — the same code path that writes `CONTAINS_TEXT`/`SAME_SECTION`/`HAS_PROVENANCE`/`EXTRACTED_FROM` today. This keeps ontology-domain semantics (radar-HAS_ANTENNA-antenna, missile-HAS_SEEKER-seeker, etc.) cleanly separated from document-layout semantics (doc-HAS_SECTION-section, section-CHILD_OF-section).

**Prerequisite for Chunk D:** these declarations must land as part of Chunk A0 (new prep chunk, see §8) before Chunk D's anchor walker runs, or ArcadeDB will reject edge creation for undeclared types.

### 3.6 Manifest change

Remove the `reference` pass from `ontology_bundles/air_defense_v3/manifest.yaml`. Remaining passes: `radar_domain`, `missile_domain`, `other_systems`, `system_links`.

### 3.7 File deletions

- `ontology_bundles/air_defense_v3/extraction_schemas/reference.py` — deleted.

### 3.8 Testing

Unit tests in `tests/unit/test_docling_anchor_walker.py`:
- Given a fixture `docling_document.json`, verify SECTION/FIGURE/TABLE records emit with exact identities.
- Verify `CHILD_OF` edges wire up for nested sections.
- Verify fallback emits single `section_number="0"` when structure is absent.
- Deterministic; no LLM mocks.

## 4. Schema patterns

### 4.1 Lenient field coercers (R10)

`ontology_bundles/air_defense_v3/validators.py` updates:

```python
import logging
logger = logging.getLogger(__name__)

def coerce_optional_int(value) -> int | None:
    # ... existing coerce attempts ...
    logger.warning("coerce_optional_int: unrecoverable input %r → None", value)
    return None
```

Same update pattern for `coerce_optional_float`, `coerce_optional_confidence`.

New helper: `coerce_identity_str(value) -> str | None` — strips, rejects multi-line, rejects empty-after-strip. Applied via `@field_validator` on every required `graph_id_fields` string field.

### 4.2 Pass-root `mode="after"` dedup (R20)

The existing `_dedupe_entities_by_identity` model_validator is hoisted from `extraction_schemas/reference.py` into `validators.py` as shared code. Converted from the earlier `mode="before"` drop-and-dedup to `mode="after"` pure dedup (per docs L16977).

No pass-root drop-invalid-identity validator is added. Library-level identity_filter + quality_gate handle bad LLM output for us (R22).

### 4.3 Identity field examples (R16, R17) — rule differs by source

**LLM-emitted identities** (every `is_entity=True` class in `ontology_bundles/air_defense_v3/extraction_schemas/*.py`):
- 2–5 distinct values; short; no duplicates.
- **MUST NOT use section/chapter-heading style examples** per docs:18470 and R17. Forbidden patterns: bare numeric (`"1.1"`, `"2.3.4"`), heading titles (`"Chapter 3"`, `"Section 3.1"`), single letters (`"A"`), Roman numerals (`"III"`). These steer the LLM toward inventing heading-style IDs and trigger the library's identity filter.
- Docs-preferred: named items, designators, refs. Examples:
  - `RadarSystemEntity.system_name`: `["Tombstone", "AN/MPQ-65", "AN/TPY-2"]`
  - `StandardEntity.designation`: `["MIL-STD-1553B", "MIL-DTL-31000G", "ANSI/IEEE 802.11"]`
  - `ComponentEntity.part_number`: `["PN-12345-A", "5961-01-234-5678"]`
- Enforced by the new contract test `test_llm_emitted_identity_examples_not_heading_style` (§7.3) which scans `extraction_schemas/*.py` only.

**Docling-derived / system-constructed identities** (SECTION.section_number populated by the anchor walker; FIGURE.figure_ref / TABLE.table_ref populated from Docling `self_ref`):
- Positional numeric is explicitly allowed per docs:18470 ("section numbers, figure/table labels, named items").
- `SectionEntity.section_number`: `["1", "2.3", "4.1.2"]`.
- `FigureEntity.figure_ref`: `["#/pictures/0", "#/pictures/12"]`.
- `TableEntity.table_ref`: `["#/tables/0", "#/tables/3"]`.
- Excluded from the LLM-style contract test scope because `entities.py` carries these — not `extraction_schemas/`.

**Catches current anti-patterns:**
- `examples=["Chapter 3: Maintenance Procedures", "Chapter 3: Maintenance Procedures"]` — heading-style + duplicated.
- `examples=[42, 42]` — duplicated.

### 4.4 Optional non-identity fields (R19)

Every field NOT in `graph_id_fields` becomes `Optional[T]` with `default=None`. Only identity fields are required. Eliminates the docling-graph salvage-bug hit (library fills missing required-int with `""`, which fails coercion) — there are no required non-identity fields to salvage.

### 4.5 Component parent re-attachment

12 entities demoting to `is_entity=False` stay attached to their natural parent entities via existing `edge(label=...)` fields in `entities.py`. No edge rewiring needed; only the `is_entity=False` flip + `graph_id_fields` removal on each demoted class.

Affected parent entities (no edge changes, just receive components):
- RADAR_SYSTEM receives MODULATION, RF_SIGNATURE, RF_EMISSION, SCAN_PATTERN, IF_AMPLIFIER, RADAR_PERFORMANCE, ENGAGEMENT_TIMELINE, SPECIFICATION.
- MISSILE_SYSTEM receives MISSILE_PERFORMANCE, MISSILE_PHYSICAL_CHARACTERISTICS, PROPULSION_STACK, PROPULSION_STAGE, SPECIFICATION.
- EQUIPMENT_SYSTEM, WEAPON_SYSTEM, AIR_DEFENSE_ARTILLERY_SYSTEM etc. receive SPECIFICATION + SUBSYSTEM (SUBSYSTEM stays entity per §2.2).

### 4.6 Library defense tuning (R22)

Pipeline config set:
- `delta_identity_filter_enabled = True` (default, keep on).
- `delta_identity_filter_strict = False` (default, keep off — want real document-derived values, not just allowlisted examples).
- `delta_quality_min_instances` — tune per pass. Docs default varies (docs:2748 shows 20 for older versions; docs:35661 shows 1 for current library versions). Local service currently reads `settings.docling_graph_quality_min_instances` from env (see `docker/docling-graph/app/config_builder.py:108`), default per that settings object. The plan pins this explicitly: add env overrides to `docker-compose.yml` for the docling-graph service: `DOCLING_GRAPH_QUALITY_MIN_INSTANCES=3` as the radar/missile/other default, overridden to `1` for system_links via per-pass config.

Config lives in `docker/docling-graph/app/config_builder.py` (the actual service config, not the vendored library's `cli/config_builder.py`). Modifications land there.

### 4.8 Walker + schema changes to make components first-class vertices

The current walker in `app/services/extraction_merge.py` has two lines that prevent components from becoming graph vertices — both a divergence from docs (docs:17500-17509 "Same address node is shared across multiple people/organizations"):

- **`extraction_merge.py:564`** — `if cfg.get("is_entity") is False: return` skips components entirely during traversal.
- **`extraction_merge.py:593-601`** — when walking `edge(label=...)` fields, rejects any child whose `is_entity is not True` with a "contract violation" warning.

The docs pattern is: components ARE emitted as graph nodes when reached via an `edge(label=...)` field on a parent entity, deduplicated by full-content equality (not by `graph_id_fields`). The plan updates the walker and supporting schema to match:

1. **Walker change 1 — component emission via edge:** When iterating `edge_label` fields on an entity, if the child is `is_entity=False`, emit it via `on_entity` (with a flag indicating "component"), construct a content-based identity (per step 4 below), and do NOT recurse further. The "no recurse" policy is safe only if we also enforce the schema rule that **components cannot carry their own `edge(label=...)` fields** — docs R11 (flatness, docs:16957) implies components should be leaf values, and docs:17517 (staged-extraction considerations) treats components as non-identity paths by default. The plan adds a new contract test `test_components_have_no_edge_label_fields` enforcing this at schema-validation time — any violation is a CI failure, preventing silent drop of content paths. If a future use-case genuinely needs component→component edges, that component should be promoted to `is_entity=True` with a proper identity.

2. **Walker change 2 — embedded component short-circuit:** The existing `return` on line 564 stays for non-edge-reached components (embedded data inside a parent's non-edge fields). This preserves the current embedded-scalar semantics for components that aren't attached via `edge()`.

3. **Contract test relaxation:** Test `test_edge_label_targets_are_is_entity_true` (contract Task 9e per existing plan tracker) is replaced with `test_edge_label_targets_are_is_entity_true_or_is_component`. Either target class is allowed, matching docs.

4. **Content-based identity for components:** `_build_logical_identity` gains a branch for `is_entity=False` types. Per docs R3 (docs:17235 — *"All fields are used for deduplication"*), identity MUST cover **all fields in canonical form, including `None` and `list[primitive]` values**, not just non-None scalars. Concretely: iterate every field in `model_fields` order (deterministic via Pydantic v2), serialize each value canonically (`None` → literal `None`, `list[primitive]` → tuple of primitives), and wrap the full sequence as the identity tuple. Two components with identical field values (even with some fields `None`) produce identical identities → upsert merges them. A wider dedup key than "non-None scalars only" is required for docs alignment: otherwise a component with `{street: "X", city: None}` collides with `{street: "X", city: "Paris"}`, which is wrong.

5. **ArcadeDB vertex classes — no new schema-creation work required.** Code path already creates vertex types for every entry in the `entity_types` output of `ontology_bundles/air_defense_v3/introspect.py:build_entity_types_list` (which iterates `ALL_ENTITIES.items()` at `introspect.py:136`), and `app/services/arcadedb_schema.py:185` iterates `ontology["entity_types"]` unconditionally. Because demoted components REMAIN in `ALL_ENTITIES` (they're still registered model classes, just with `is_entity=False`), they automatically get vertex classes once the canonical change in Chunk B lands. Common-props set is unchanged: every vertex class (entity or component) retains `id`, `name`, `entity_type`, `canonical_name`, `extraction_confidence`, `created_at`, `updated_at`. For component vertices, `name` is populated from content-fingerprint (derived from the content-based identity at step 4) rather than being left null — keeps `_upsert_node_impl_sync` at `arcadedb_graph.py:1722` working uniformly without a component-branch. No schema changes required; no NodeRecord redesign required.

6. **Merge-layer dedup:** `arcadedb_graph.upsert_nodes_batch_sync` already keys by `(entity_type, identity_tuple)`. The content-based identity from step 4 makes this Just Work for components — same content-hash = same vertex.

These changes sequence BEFORE Chunk B (the canonical rewrite) so entity demotions don't strand components mid-plan. See §8 Chunk A0 additions.

### 4.7 Extraction schema rewrites

`extraction_schemas/radar_domain.py`, `missile_domain.py`, `other_systems.py`:
- Match new canonical identity shapes (single short identity field per entity).
- Import shared `_deduplicate_by_identity` helper from `validators.py`.
- Remove local duplicated dedup code.
- All examples lists rewritten per 4.3.
- All non-identity fields become Optional per 4.4.
- Components (demoted entities) carry `is_entity=False` in model_config, no `graph_id_fields`.

`extraction_schemas/system_links.py`: minimal change — Decision-4 DTO pattern preserved. Examples cleaned up.

`extraction_schemas/reference.py`: **deleted**.

## 5. Migration (full purge + re-ingest)

### 5.1 Pre-migration state to wipe

**Postgres — explicit per-table action** (ordered by FK-dependency for truncate):

| Table | Action |
|---|---|
| `ingest.stage_runs` | TRUNCATE |
| `ingest.pipeline_runs` | TRUNCATE |
| `ingest.document_graph_extractions` | TRUNCATE |
| `ingest.document_elements` | TRUNCATE |
| `ingest.artifacts` | TRUNCATE |
| `retrieval.chunk_links` | TRUNCATE |
| `retrieval.text_chunks` | TRUNCATE |
| `retrieval.image_chunks` | TRUNCATE |
| `retrieval.chunks_legacy` | TRUNCATE |
| `retrieval.community_runs` | TRUNCATE |
| `governance.feedback` | TRUNCATE |
| `governance.patch_approvals` | TRUNCATE |
| `governance.patch_events` | TRUNCATE |
| `governance.patches` | TRUNCATE |
| `governance.query_profile_registries` | TRUNCATE |
| `governance.trusted_data_submissions` | TRUNCATE |
| `ontology.entity_types` | TRUNCATE |
| `ontology.relationship_types` | TRUNCATE |
| `ontology.versions` | TRUNCATE |
| `ingest.documents` | UPDATE pipeline_status='PENDING', pipeline_stage=NULL, failed_stages=NULL, error_message=NULL, celery_task_id=NULL (preserve rows) |
| `ingest.watch_logs` | TRUNCATE |
| `ingest.sources` | PRESERVE |
| `ingest.watch_dirs` | PRESERVE |
| `auth.users`, `auth.user_roles` | PRESERVE |
| `public.alembic_version` | PRESERVE |

Any tables added to the schema after this spec is written get TRUNCATE by default; the migration script reads `pg_tables` and truncates anything not in the preserve-list. A dry-run flag (`--dry-run`) prints the list before executing; dry-run output is also included in the migration report (§5.3) so the exact set of truncated tables is audit-able post-hoc.

**ArcadeDB:** DROP + recreate schema.

**MinIO:** empty `derived/*` bucket (includes `docling_document.json` enrichments); preserve `originals/*`.

**Docling reconversion:** because `derived/*` is wiped, `prepare_document` will re-run Docling conversion on each PDF when the pipeline chain restarts. This is necessary (the cached `docling_document.json` is in `derived/`) and accounts for the bulk of migration runtime. Earlier drafts of this spec incorrectly claimed "no Docling reconversion" — that was wrong.

**Redis:** `FLUSHALL` to clear Celery queues.

### 5.2 Migration script

`scripts/full_purge_and_reingest.py`:

1. Safety flag check (`--i-understand-this-deletes-derived-data`).
2. Stop worker + beat containers.
3. Truncate Postgres per 5.1 list.
4. DROP + recreate ArcadeDB schema via `arcadedb_schema.ensure_schema`.
5. Empty MinIO derived bucket.
6. FLUSHALL Redis.
7. Apply `alembic upgrade head`.
8. Restart worker + beat.
9. For each doc in `ingest.documents`: reset `pipeline_status='PENDING'`, clear stage + error fields.
10. Enqueue the full pipeline chain for each document.
11. Poll until all pipelines reach terminal status.
12. Emit `/tmp/migration-report-{timestamp}.md`.

### 5.3 Migration report structure

- Per-doc: UUID, filename, final `pipeline_status`, per-pass `yield_status`, entity counts by type, edge counts by label, `extraction_quality` (ok/degraded/anomaly).
- Summary: total docs, counts by outcome, any FAILED stages with error snippets.

### 5.4 Acceptance gate

- All 21 docs reach `COMPLETE` or `PARTIAL_COMPLETE` (not `FAILED`).
- ≥3 radar/missile-heavy docs produce `extraction_quality="ok"`.
- Every doc produces ≥1 SECTION (or the §3.4 sentinel `section_number="0"`), ≥1 TextChunk, and a **structural `Document` vertex** (identity = UUID; created by `derive_structure_links`, always present for any successfully ingested doc).
- **Ontology DOCUMENT vertex is NOT required per-doc** — it's emitted only when `document_number` is extractable (§3.3). Acceptance gate on ontology DOCUMENT: ≥50% of the 21-doc corpus emits an ontology DOCUMENT (sanity check that the `_extract_document_number_from_front_matter` heuristic actually fires on typical docs). Lower ratio acceptable if manually verifiable that the docs genuinely lack MIL-STD/TM-style designators.
- Zero `page_start: Input should be a valid integer`-style errors in docling-graph container logs. Check via `docker logs eip-mmdpp-docling-graph-1 --since <migration-start> 2>&1 | grep -E "Input should be a valid"` — acceptance requires zero lines.

If the gate fails, root-cause before merging the branch.

### 5.5 Rollback

Single-branch design. Git-revert + re-run the same migration script on reverted code returns to pre-plan state in ~15–25 min.

## 6. Consumer updates

### 6.1 Hardcoded entity-type lists — complete inventory

Grep of `app/`, `frontend/src/` (excluding tests, migrations, `__pycache__`, `node_modules`) for hardcoded entity-type name strings returns the following concrete hits. Each gets an update in Chunk E.

| File:Line | Current string | Action |
|---|---|---|
| `frontend/src/components/GraphExplorer.tsx:27` | `"SUBSYSTEM"` | Keep (SUBSYSTEM stays entity per §2.2). |
| `frontend/src/components/GraphExplorer.tsx:29` | `"SPECIFICATION"` | Per §4.8 revision: SPECIFICATION becomes a first-class component vertex. Keep as filter option. |
| `frontend/src/components/GraphExplorer.tsx:33` | `"ASSERTION"` | Remove — entity dropped. |
| `frontend/src/constants/entityTypes.ts:23` | `"Assertion"` in `REFERENCE_TYPES` array | Remove. |
| `frontend/src/constants/entityTypes.ts` (rest of file) | CamelCase categorized arrays (MILITARY_TYPES, EMRF_TYPES, WEAPON_TYPES, OPERATIONAL_TYPES, REFERENCE_TYPES) | Audit against §2 classification and §4.8 component-vertex rule. Demoted entities (Modulation, RFEmission, RFSignature, ScanPattern, IFAmplifier, Specification, MissilePerformance, MissilePhysicalCharacteristics, PropulsionStack, PropulsionStage, RadarPerformance, EngagementTimeline) all STAY in their respective arrays — they're still renderable vertices, just components now. Only structural changes: drop `"Assertion"` (line 23), drop `"Spreadsheet"` if present. |
| `app/services/dossier_service.py:38–52` (`RF_ENTITY_TYPES` list) | Demoted entries: `RF_EMISSION`, `MODULATION`, `RF_SIGNATURE`, `SCAN_PATTERN`, `IF_AMPLIFIER`, `SPECIFICATION`. Unchanged entries in the same list: `FREQUENCY_BAND`, `WAVEFORM`, `ANTENNA`, `TRANSMITTER`, `RECEIVER`, `SIGNAL_PROCESSING_CHAIN`, `SEEKER`. | Audit only the **demoted entries**: verify filter-by-ontology_name still matches component-kind vertices in ArcadeDB. No entries are deleted from the list — components retain their ontology_name. |
| `app/services/dossier_service.py:54–68` (`PERFORMANCE_ENTITY_TYPES` list) | Demoted entries: `RADAR_PERFORMANCE`, `ENGAGEMENT_TIMELINE`, `MISSILE_PERFORMANCE`, `MISSILE_PHYSICAL_CHARACTERISTICS`, `PROPULSION_STACK`, `PROPULSION_STAGE`, `SPECIFICATION`. Unchanged: `CAPABILITY`, `GUIDANCE_METHOD`, `STANDARD`, `PROCEDURE`, `FAILURE_MODE`, `TEST_EVENT`. | Same audit-only treatment. `SPECIFICATION` correctly appears in both lists (RF and PERFORMANCE categorizations are intentionally cross-indexed — not a duplicate to remove). |
| `app/services/query_profiles.py:49–79` (`_CURRENT_RF_ENTITY_TYPES` + `_CURRENT_PERFORMANCE_ENTITY_TYPES` + related `_CURRENT_*` lists) | Same demoted entries, same pattern as dossier_service. `SPECIFICATION` appears in both RF and PERFORMANCE lists — same intentional cross-indexing. | Same audit-only treatment. |
| `app/services/arcadedb_graph.py:2020–2026` | Docstring listing SECTION/FIGURE/TABLE/ASSERTION/WAVEFORM/... as document-scoped entity classes | Drop ASSERTION, update identity-scope list to match new canonical. |

**Audit note for query_profiles + dossier_service:** these files' entity-type filters are load-bearing for retrieval. Demoting SPECIFICATION/MODULATION/etc. to components changes how they're deduplicated (content-based) but NOT their `ontology_name` or their ArcadeDB vertex class (both remain keyed off `ontology_name` in `arcadedb_schema`). Filters should still match. **Verification step in Chunk E**: write a small integration test that executes the dossier's filtered query against a re-ingested doc and confirms non-zero vertex matches for each of the 13 demoted ontology_names. If any filter fails, the fix is either (a) the filter's entity_type enumeration needs updating, or (b) the component's vertex class name in ArcadeDB differs from expected (unlikely; documented fail-safe only).

### 6.2 Identity-field references

A focused grep for attribute reads `\.heading\b`, `\.figure_id\b`, `\.table_id\b`, `\.page_start\b`, `\.assertion_text\b` across `app/` and `frontend/src/` (excluding tests, migrations, `__pycache__`, `node_modules`) returned **zero hits**. No consumer code currently reads these as entity attributes. The identity fields only exist inside `entities.py`, `extraction_schemas/reference.py`, and test files.

- `app/services/extraction_merge.py:305` — change `_NAME_LIKE_KEYS` tuple's `"heading"` → `"section_number"`.
- `ontology_bundles/air_defense_v3/extraction_schemas/reference.py` — deleted entirely per §3.7.
- Test files updated as part of Chunk B/C schema rewrites.

### 6.3 Retrieval / dossier / query profiles

Most query paths filter by `ontology_name` which is unchanged. The concrete consumer surface is documented in §6.1 above — the two entity-type lists in `dossier_service.py` and `query_profiles.py`.

No `.heading` / `.figure_id` / `.table_id` / `.page_start` identity-field attribute reads exist (confirmed in §6.2 grep). All retrieval consumers query by `ontology_name`.

### 6.4 Derive-rules + structure-links + finalize_document

`ontology_bundles/air_defense_v3/derive_rules.py :: derive_structural_edges` (path corrected — `derive_rules.py` lives in the ontology bundle, not in `app/services/`):
- Lookup SECTION/FIGURE/TABLE vertices by new single-field identities (`section_number`, `figure_ref`, `table_ref`).

`app/workers/pipeline.py :: derive_structure_links`:
- **Structural `Document` vertex creation stays here — unchanged.** The reviewer-flagged earlier draft was wrong: the anchors task targets ontology DOCUMENT (distinct), not the structural vertex.
- **SAME_SECTION edge creation is chunk-to-chunk keyed on `section_path` string match (see `pipeline.py:4376`) — unchanged.** No dependency on SECTION vertex identity because it doesn't target SECTION vertices. The SAME_SECTION logic continues as-is.
- `CONTAINS_TEXT` / `EXTRACTED_FROM` edges continue to target the structural `Document` vertex; these don't change.

`app/workers/pipeline.py :: finalize_document` at `:4799`:
- Has a hardcoded `REQUIRED_STAGES` set used to decide `STATUS_COMPLETE` vs `STATUS_PARTIAL_COMPLETE`. The new `derive_document_anchors` stage must be added to this set or finalize will mark docs `PARTIAL_COMPLETE` because it sees the anchors stage as "missing."
- Update task: add `"derive_document_anchors"` to `REQUIRED_STAGES`. Update `tests/unit/test_derive_ontology_graph_bundle_passes.py` (or the finalize-specific test file) to exercise the new required stage.

### 6.5 Canonicalization

`app/services/canonicalization.py` — verified via grep: contains no `"SPECIFICATION"` or other demoted-entity-name hardcoded references. The canonicalization pathway is ontology-agnostic (it canonicalizes entity names generically), so demotions don't break it. No change required beyond what the re-ingest naturally produces.

### 6.6 Pipeline config audit

Verify `delta_identity_filter_enabled` and `delta_quality_min_instances` are set as §4.6 specifies in whatever config-builder path the worker's `/extract-pass` request uses.

### 6.7 Tests updated

Every test constructing old-identity entities updated to new shape. Parity tests deleted (§7.2).

### 6.8 `_classify_extraction_quality` rewrite

`app/workers/pipeline.py:282–317` currently reads `pass_outcomes.get("reference")` to distinguish `degraded` (reference HIT, no domain HIT) from `anomaly` (no pass HIT at all). Reference is deleted in this plan; the helper needs new semantics.

**New classifier logic** (replaces `_classify_extraction_quality`):

- `ok` — at least one domain pass (radar_domain / missile_domain / other_systems / system_links) achieved `yield_status="HIT"`.
- `degraded` — SECTION vertex count > 0 AND TextChunk count > 0 AND no domain pass HIT. Signal: "real document with readable structure, content off-topic from our ontology."
- `anomaly` — (SECTION vertex count == 0 OR TextChunk count == 0) AND no domain pass HIT. Signal: processing failure or pathological doc.

Source of truth shifts from "reference pass's StageRun row" to graph counts:
- SECTION count: an ArcadeDB query against the SECTION vertex class filtered by `document_id`. **If `graph_store` doesn't already expose a count helper for arbitrary ontology-name vertex counts per document, Chunk E's `_classify_extraction_quality` task includes adding one** (`graph_store.count_ontology_nodes_sync(entity_type: str, document_id: str) -> int` or similar) — check first via grep; add only if missing.
- TextChunk count: `SELECT COUNT(*) FROM retrieval.text_chunks WHERE document_id = :doc_id` via the sync session — plain SQL, no new helpers needed.

The helper becomes:

```python
def _classify_extraction_quality(pass_outcomes: dict, section_count: int, text_chunk_count: int) -> str:
    domain_hit = any(
        v.get("yield_status") == "HIT"
        for k, v in pass_outcomes.items()
        if k in _DOMAIN_PASS_NAMES
    )
    if domain_hit:
        return "ok"
    if section_count > 0 and text_chunk_count > 0:
        return "degraded"
    return "anomaly"
```

Caller `_write_pipeline_run_metrics` is updated to compute `section_count` + `text_chunk_count` from the graph and pass them in.

`_DOMAIN_PASS_NAMES` constant retains the 4 domain passes (reference removed). Matching test updates required:
- `tests/unit/test_classify_extraction_quality.py` — rewrite all test cases to exercise the new three-argument signature + new degraded semantics.
- `tests/unit/test_extraction_schemas.py` at `:8` — update `"5 passes" / extraction_schemas.reference` references.
- `tests/unit/test_ontology_bundles.py` at `:11` — same.
- `tests/unit/test_coverage_checker.py` at `:68` — `module="extraction_schemas.reference"` reference; either delete the test case or re-target to remaining passes.
- Any other test using `extraction_schemas.reference` path — grep during Chunk E.

## 7. Contract tests + xfail resolution

### 7.1 Remaining xfails (both resolved by this plan)

- `test_identity_fields_have_examples` — resolved when `GuidanceMethodEntity.guidance_type` gains `examples=["COMMAND", "SARH", "ARH"]` during the entities.py rewrite.
- `test_descriptions_and_examples_on_extraction_relevant_fields` — resolved when canonical rewrite adds description + examples to every field. `edge()` helper extended with optional `description` + `examples` kwargs that forward to `Field`.

### 7.2 Parity tests deleted

Tests introduced by the YAML→Pydantic migration (current plan) compared Pydantic introspection to the frozen `tests/fixtures/ontology/air_defense_v3_snapshot.yaml`. After canonical rewrite the fixture has no valid oracle. Complete list of parity tests to evaluate for deletion (actual files confirmed via `ls tests/unit/*parity*.py`):

| File | Disposition |
|---|---|
| `tests/unit/test_arcadedb_schema_ontology_source_parity.py` | Delete (YAML vs Pydantic parity). |
| `tests/unit/test_canonicalization_ontology_source_parity.py` | Delete. |
| `tests/unit/test_dossier_service_ontology_source_parity.py` | Delete. |
| `tests/unit/test_extraction_merge_ontology_source_parity.py` | Delete. |
| `tests/unit/test_graph_store_ontology_source_parity.py` | Delete. |
| `tests/unit/test_main_api_ontology_source_parity.py` | Delete. |
| `tests/unit/test_pipeline_ontology_source_parity.py` | Delete. |
| `tests/unit/test_query_profiles_ontology_source_parity.py` | Delete. |
| `tests/unit/test_relationships_parity.py` | Delete. |
| `tests/unit/test_validation_matrix_parity.py` | Delete. |
| `tests/unit/test_introspect_entity_types.py` | Delete. |
| `tests/unit/test_introspect_ontology_dict.py` | Delete. |
| `tests/unit/test_introspect_relationship_types.py` | Delete. |
| `tests/unit/test_introspect_validation_and_weights.py` | Delete. |
| `tests/unit/test_ontology_source_flag.py` | Delete (ONTOLOGY_SOURCE is a no-op post-Task-51). |
| `tests/unit/test_ontology_templates_internals_parity.py` | Delete (YAML-vs-Pydantic internals parity; loses oracle with fixture deletion). |
| `tests/unit/test_arcadedb_schema.py` | Delete. The file is snapshot/YAML-driven throughout — no preservable schema-creation assertions are separable from the YAML comparison. New `tests/unit/test_arcadedb_schema_introspection.py` is added in Chunk A0 to cover schema creation against the Pydantic ontology (no YAML oracle). |
| `tests/fixtures/ontology/air_defense_v3_snapshot.yaml` | Delete. |

**17 full test-file deletions + 1 fixture deletion.** Each deleted file gets its own task in Chunk F so the diff stays reviewable.

### 7.3 New contract tests

Split existing `tests/unit/test_docs_compliance_contracts.py` (~500 lines) into:
- `tests/unit/contracts/test_identity_contract.py`
- `tests/unit/contracts/test_component_contract.py`
- `tests/unit/contracts/test_extraction_schema_contract.py`

Add 12 new tests (R-rule references link to docs):

| Test | Rule(s) | Assertion |
|---|---|---|
| `test_entity_has_identity_or_is_component` | R1, R3 | Every `is_entity=True` model has non-empty `graph_id_fields`; every `is_entity=False` has absent/empty. |
| `test_identity_fields_are_required` | R18 | Every `graph_id_fields` field is `...` required. |
| `test_identity_field_examples_are_short` | R2, R17, R21 | Every identity-field example ≤ 80 chars, no `\n`. |
| `test_identity_fields_not_named_heading_or_title` | R21 | No `graph_id_fields` includes `heading`/`title`/`caption`/`description`. |
| `test_identity_examples_are_distinct` | R16 | Identity examples list has no duplicates. |
| `test_non_identity_fields_are_optional` | R19 | Every non-identity field is `Optional[T]` with `default=None`. |
| `test_edge_fields_have_edge_label` | R4 | Every `List[Entity]`/`Optional[Entity]` field on `is_entity=True` model has `json_schema_extra["edge_label"]`. |
| `test_no_nested_property_dicts` | R11 | Non-edge, non-BaseModel properties are primitive or `list[primitive]`. |
| `test_component_fields_attached_via_edge_helper` | R3 | Components used in catalog paths attach via `edge(label=...)` on parent entities. |
| `test_identity_example_values_populated_for_library_filter` | R22 | Every `is_entity=True` model has ≥2 examples on every identity field. |
| `test_components_have_no_edge_label_fields` | R11, docs:17517 | No `is_entity=False` class has any field with `json_schema_extra["edge_label"]`. Prevents component→component edges (see §4.8 walker change 1). |
| `test_llm_emitted_identity_examples_not_heading_style` | R17, docs:18470 | For every `is_entity=True` model in `extraction_schemas/*.py` (LLM-emitted identities — excludes `entities.py` SECTION because that's Docling-derived per §2.5), every identity-field example string must not match a section/chapter-heading regex (`^\d+(\.\d+)*$`, `^(Chapter|Section|Part) `, `^[A-Z]$` single-letter, `^[IVX]+$` Roman-only). Ensures LLM prompts don't get steered toward invented heading-style IDs. |

### 7.4 New anchor-walker tests

`tests/unit/test_docling_anchor_walker.py`:
- 6–8 tests with fixture `docling_document.json` inputs of varying complexity.
- Assert SECTION/FIGURE/TABLE records with exact identities + hierarchy edges.
- Assert fallback-emit when `section_path` is absent.

### 7.5 Acceptance gate for test suite

- All 21 contract tests pass (**9 existing** in `test_docs_compliance_contracts.py` — count verified via `grep -c "^def test_"` — **+ 12 new**); 0 xfails.
- All 17 parity/schema test files + snapshot fixture deleted; new `test_arcadedb_schema_introspection.py` (Chunk A0) passes.
- Anchor walker tests pass.
- Full `pytest tests/unit/` run passes, zero failures.

## 8. Plan chunks and task count estimate

| Chunk | Theme | Tasks |
|---|---|---|
| A0 | **Prereqs for canonical changes** — 6 explicit tasks below | 6 |
| A | Prep: 12 new contract tests (added xfailed), `edge()` helper extension, lenient-coercer logging, pipeline-config knobs (per §4.6 path) | 6 |
| B | Canonical `entities.py` rewrite — 2 drops + 15 give-identity + 12 demote + 3 batched touch-ups (see §2.6) | 32 |
| C | Extraction schemas rewrite + manifest change + reference.py delete | 5 |
| D | Docling anchor walker + new worker task + fixtures + tests + document_number heuristic + `_to_merged_entity_record` helper (see §8.2) | 6 |
| E | Consumer updates — derive_rules (ontology bundle path), `finalize_document.REQUIRED_STAGES` adds `derive_document_anchors`, structure_links (no behavior change — SAME_SECTION stays chunk-to-chunk), arcadedb_graph docstring, frontend `GraphExplorer.tsx` + `entityTypes.ts`, extraction_merge `_NAME_LIKE_KEYS`, dossier_service filter lists (audit only), query_profiles filter lists (audit only), canonicalization verification (no-op confirmed), `_classify_extraction_quality` rewrite per §6.8, graph_store count helper if missing per §6.8, update `test_coverage_checker.py`+`test_extraction_schemas.py`+`test_ontology_bundles.py` for 4-pass reality | 11 |
| F | Test cleanup — 17 parity/schema test deletions + 1 fixture delete + un-xfail 2 contract tests + verify 21/21 contract tests green | 7 |
| G | Migration — write script, dry-run, execute on 21-doc corpus (accepts 3–6hr runtime including Docling reconversion), produce report, acceptance gate | 4 |

**Total: 6 + 6 + 32 + 5 + 6 + 11 + 7 + 4 = 77 tasks, ~77 commits, ~15–25 days of execution.**

### 8.1 Chunk A0 explicit task list

A0 carries six sequential tasks that land before Chunk B begins (reduced from seven; the "add 12 vertex classes" task was redundant — vertex classes are generated automatically from `ALL_ENTITIES` via `introspect.build_entity_types_list` at `introspect.py:136` + `arcadedb_schema.py:185`):

1. Extend `_build_logical_identity` at `app/services/extraction_merge.py:416` with a content-based branch for `is_entity=False` types. Per §4.8 step 4 and docs:17235 ("All fields are used for deduplication"): identity covers **all** fields in canonical form — iterate `model_fields` in declaration order (deterministic via Pydantic v2), serialize each value canonically (`None` → literal `None`, `list[primitive]` → tuple of primitives), wrap the full sequence as the identity tuple. Also populate `name` on the `NodeRecord` from a content fingerprint (so `_upsert_node_impl_sync` at `arcadedb_graph.py:1722` continues working uniformly without a component-branch).
2. Update the walker at `app/services/extraction_merge.py:564` — remove the unconditional `return` for components reached via `edge_label` fields; keep embedded semantics for components NOT reached via `edge_label` (plain property fields).
3. Update the walker at `app/services/extraction_merge.py:593-601` — change the "contract violation" skip into a component-emit path (calls `on_entity` with a component flag and still emits via `on_edge` when available).
4. Add 4 structural edge types (`HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE`/`CHILD_OF`) to `_STRUCTURAL_EDGE_TYPES` at `app/services/arcadedb_schema.py:79`.
5. Relax contract test 9e — rename `test_edge_label_targets_are_is_entity_true` to `test_edge_label_targets_are_is_entity_true_or_is_component`; update the assertion to allow both.
6. Add new `tests/unit/test_arcadedb_schema_introspection.py` — schema-creation coverage driven by Pydantic introspection (replaces the deleted YAML-snapshot-driven `test_arcadedb_schema.py`).

Sequencing: **A0 + A** (parallel-ok) → **B** → **C** → **D**, **E** after B/C/D, **F** after E, **G** last. A0 MUST land before B or canonical demotions break mid-plan (components won't get vertex classes). A0 and A can be worked in parallel if the team has bandwidth since they're independent.

### 8.2 Chunk D task: `_to_merged_entity_record` helper

The §3.3 pseudocode calls `_to_merged_entity_record(model_instance, ontology, document_uuid)` to convert each Pydantic model (DocumentEntity / SectionEntity / FigureEntity / TableEntity) into a `MergedEntityRecord`. This helper does not exist today in `app/services/extraction_merge.py`; it's new code landed as part of Chunk D. Its contract:

**Signature:**
```python
def _to_merged_entity_record(
    model: BaseModel,
    ontology: dict,
    document_id: str,
    pass_origin: str = "document_anchors",
) -> MergedEntityRecord:
```

**Behavior:**
1. Resolve `ontology_name` from `model.model_config["ontology_name"]`.
2. Build `LogicalIdentity` via `_build_logical_identity(ontology_name, model, ontology, document_id)`.
3. Assemble `properties` dict from `model.model_dump(mode="json")` EXCLUDING `graph_id_fields` entries (those are in identity, not properties) AND EXCLUDING any `edge(label=...)` fields (those materialize as edges, not properties).
4. **Stamp `document_id` into `properties`** (so it lands as a vertex property — enables the §3.4 TextChunk join and the §3.3 determinism property "SECTION/FIGURE/TABLE carry `document_id`"). Similarly stamp `section_path` when `model` is a SectionEntity with non-None `section_path`.
5. Populate `confidence=1.0` (anchors are deterministic), `pass_origins={pass_origin}`, `display_label` via `build_display_label(model, ontology_name)`.
6. Return the assembled `MergedEntityRecord`.

**Placement:** module-private helper in `app/services/extraction_merge.py` (same module as `MergedEntityRecord`). Tests: `tests/unit/test_to_merged_entity_record.py` covering (a) DocumentEntity with document_number → expected identity + properties, (b) SectionEntity with full path_tuple → identity `section_number`, properties include `section_path` + `document_id`, (c) SectionEntity sentinel (`section_number="0"`) → no section_path property, (d) FigureEntity with figure_label from caption → figure_label in properties, figure_ref in identity.

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Chunk B is large: 2 drops + 27 structural rewrites + 3 batched touch-ups = 32 commits. Reviewer fatigue risk. | One-entity-per-commit discipline for the 27 structural rewrites. Each commit is small and individually reviewable. The 3 touch-up commits are deliberate batches. |
| Docling's `section_path` sparse on some docs → zero SECTIONs | §3.4 fallback emits single `section_number="0"` for document body. |
| Migration on 21 docs is slow | MinIO originals preserved so no re-upload, but Docling reconversion runs (derived bucket wiped) + full LLM passes. Realistic runtime 3–6 hours end-to-end on 21 docs (acceptable one-time cost for the canonical reset). |
| Frontend graph visualization shifts due to component demotion | Visually different ≠ broken. Document in migration report. Frontend doesn't need explicit changes beyond ASSERTION filter. |
| Upstream docling-graph salvage bug for required int fields | Avoided by §4.4 (no required non-identity fields) + §4.6 (library identity_filter catches empty-string identities). |
| Canonicalization path for SPECIFICATION breaks with is_entity=False | §6.5 audit during implementation. Components aren't canonicalized; remove SPECIFICATION from any canonicalization registry. |

## 10. Handoff to current plan

After Chunk G's acceptance gate passes and the branch merges:

- The current plan (`2026-04-14-docling-graph-schema-compliance.md`) resumes.
- Its Task 53 (live E2E on b1b0d596) is unblocked — reference pass is deleted, domain passes produce non-zero entities on the new schemas.
- Phase 8 provenance tasks (67–72 in its tracker) execute against the new schemas without conflict; `ExtractionProvenance` is a per-instance additive field, orthogonal to identity redesign.

## 11. Docs-adherence traceability

Every design decision traces to an explicit rule in `docling-graph-docs.md`. A full audit table is in the brainstorm session's memory. Summary:

- **R1, R2, R18** — §2 identity assignments.
- **R3** — §2 component demotions.
- **R4** — §3 anchor edges use explicit `HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE`/`CHILD_OF` labels; §4.5 component attachment via `edge()`.
- **R7, R9** — §3 Docling-derived anchors (no invention).
- **R10, R19** — §4.1 and §4.4 lenient coercers + Optional non-identity.
- **R11** — §4 flat properties verified (all demoted components are primitives).
- **R14, R15, R16, R17, R21** — §2.5 reconciliation note; §4.3 example style.
- **R20** — §4.2 mode="after" dedup.
- **R22** — §4.6 library defense tuning.
