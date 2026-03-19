# Ontology Extraction Quality Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 gaps in the docling-graph LLM extraction prompts and implement chunked entity extraction with global relationship pass to handle context window pressure.

**Architecture:** Fixes 1-6 modify prompt construction and schema generation in `docker/docling-graph/app/`. Fix 7 restructures `_run_full_extraction()` to chunk document text for entity extraction while running relationship extraction on the full text. All changes are in the docling-graph service container.

**Tech Stack:** Python, FastAPI, LiteLLM, YAML ontology, Pydantic

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `docker/docling-graph/app/main.py` | Modify | Fix 1 (pattern), Fix 5 (hierarchy), Fix 7 (chunking) |
| `docker/docling-graph/app/prompts.py` | Modify | Fix 2 (system prompts), Fix 3 (relationships), Fix 4 (dedup guidance), Fix 6 (few-shot) |
| `docker/docling-graph/tests/test_direct_extraction.py` | Modify | Tests for all fixes |
| `ontology/ontology.yaml` | Read-only | Source of truth for relationship types and validation matrix |

---

### Task 1: Add `pattern` fields to schema prompts

**Files:** Modify `docker/docling-graph/app/main.py` — `_build_group_schema_prompts()` (~line 119-162)

- [ ] **Step 1: Add pattern reading to the property loop**

In `_build_group_schema_prompts()`, after the `enum_vals` line (~line 149), add pattern support:

```python
                    pattern = prop_spec.get("pattern", "")
                    # existing code follows...
                    if enum_vals:
                        parts.append(f" [enum: {', '.join(str(v) for v in enum_vals)}]")
                    if pattern:
                        parts.append(f" [format: {pattern}]")
```

- [ ] **Step 2: Commit**

```bash
git add docker/docling-graph/app/main.py
git commit -m "fix(docling-graph): include pattern fields in entity schema prompts"
```

---

### Task 2: Fix system prompts referencing non-existent properties

**Files:** Modify `docker/docling-graph/app/prompts.py`

- [ ] **Step 1: Fix weapon group prompt**

Replace the weapon group prompt to remove "single-shot probability of kill", "wingspan", "warhead mass":

```python
    "weapon": (
        "You are a weapons systems analyst specializing in missile and munitions "
        "technology. Extract weapon and missile subsystem details from the following "
        "text: guidance methods (SARH, ARH, IIR, command guidance, GPS/INS, MMW, "
        "dual-mode), missile performance parameters (max/min range, altitude "
        "envelope, max speed, time-of-flight), missile physical characteristics "
        "(body diameter, length, launch mass), propulsion details (ejector, booster, "
        "sustainer stages, fuel type, burn time), subsystems, and components with "
        "part numbers or NSN identifiers. Return only details explicitly mentioned "
        "in the text."
    ),
```

- [ ] **Step 2: Fix operational group prompt**

Replace the operational group prompt to remove "MTBF", "velocity resolution", "range resolution":

```python
    "operational": (
        "You are a military operations analyst specializing in capability "
        "assessment. Extract operational and capability information from the "
        "following text: functional capabilities (detection, tracking, engagement, "
        "surveillance), radar performance metrics (detection range, ambiguity "
        "limits, clutter rejection), engagement timelines (detection-to-designate, "
        "designation-to-launch, time-to-intercept), force structure elements "
        "(units, echelons, battalions, batteries), equipment system top-level "
        "designations, assemblies, specifications with measurable parameters "
        "(max range, frequency, power, weight), standards (MIL-STD, MIL-DTL, "
        "MIL-PRF references), procedures (maintenance, operational, test), "
        "failure modes (FMECA severity, detection methods), and test events "
        "(DT, OT, IOT results). Return only information explicitly mentioned "
        "in the text."
    ),
```

- [ ] **Step 3: Commit**

```bash
git add docker/docling-graph/app/prompts.py
git commit -m "fix(docling-graph): remove non-existent properties from system prompts"
```

---

### Task 3: Add cross-group dedup guidance to system prompts

**Files:** Modify `docker/docling-graph/app/prompts.py`

- [ ] **Step 1: Append dedup notes to overlapping group prompts**

Add to the end of the `equipment` prompt:
```
" Note: Do not extract detailed RF signal parameters (frequencies, PRIs, "
"waveforms, antenna specs) — those belong to the rf_signal group."
```

Add to the end of the `rf_signal` prompt:
```
" Note: Do not extract top-level system designations or platform names — "
"those belong to the equipment group. Focus on signal-level characteristics."
```

Add to the end of the `operational` prompt:
```
" Note: Do not extract individual equipment system details — those belong "
"to the equipment group. Focus on operational performance and capabilities."
```

- [ ] **Step 2: Commit**

```bash
git add docker/docling-graph/app/prompts.py
git commit -m "fix(docling-graph): add cross-group dedup guidance to system prompts"
```

---

### Task 4: Add type hierarchy to schema prompts

**Files:** Modify `docker/docling-graph/app/main.py` — `_build_group_schema_prompts()`

- [ ] **Step 1: Include parent and label in entity headers**

In `_build_group_schema_prompts()`, update the entity header construction. The function already has `et` (the entity type dict). Change:

```python
            lines.append(f"### {entity_name}")
            if desc:
                lines.append(f"Description: {desc}")
```

To:

```python
            parent = et.get("parent", "")
            label = et.get("label", "")
            header = f"### {entity_name}"
            if parent:
                header += f" ({parent})"
            lines.append(header)
            if label:
                lines.append(f"Label: {label}")
            if desc:
                lines.append(f"Description: {desc}")
```

- [ ] **Step 2: Commit**

```bash
git add docker/docling-graph/app/main.py
git commit -m "fix(docling-graph): include type hierarchy in entity schema prompts"
```

---

### Task 5: Add few-shot examples to entity prompts

**Files:** Modify `docker/docling-graph/app/prompts.py`

- [ ] **Step 1: Add FEW_SHOT_EXAMPLES dict**

Add after the GROUP_PROMPTS dict:

```python
GROUP_FEW_SHOT_EXAMPLES: dict[str, str] = {
    "reference": (
        'Example: Given "See TM 9-1425-386-12, Chapter 3, Table 3-1", extract:\n'
        '{"entities": [\n'
        '  {"name": "TM 9-1425-386-12", "entity_type": "DOCUMENT", "confidence": 0.95, '
        '"properties": {"document_number": "TM 9-1425-386-12"}},\n'
        '  {"name": "Chapter 3", "entity_type": "SECTION", "confidence": 0.9, '
        '"properties": {"heading": "Chapter 3"}},\n'
        '  {"name": "Table 3-1", "entity_type": "TABLE", "confidence": 0.9, '
        '"properties": {"table_id": "3-1"}}\n'
        ']}\n'
    ),
    "equipment": (
        'Example: Given "The AN/MPQ-53 radar is mounted on the M901 launcher station '
        'and operated by Raytheon", extract:\n'
        '{"entities": [\n'
        '  {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM", "confidence": 0.95, '
        '"properties": {"nomenclature": "AN/MPQ-53"}},\n'
        '  {"name": "M901", "entity_type": "PLATFORM", "confidence": 0.9, '
        '"properties": {"platform_type": "launcher station"}},\n'
        '  {"name": "Raytheon", "entity_type": "ORGANIZATION", "confidence": 0.9, '
        '"properties": {"org_name": "Raytheon"}}\n'
        ']}\n'
    ),
    "rf_signal": (
        'Example: Given "The radar operates in C-band (5.4-5.9 GHz) using a linear FM '
        'chirp waveform with 10 us pulse duration", extract:\n'
        '{"entities": [\n'
        '  {"name": "C-band", "entity_type": "FREQUENCY_BAND", "confidence": 0.95, '
        '"properties": {"band_name": "C", "min_freq_ghz": 5.4, "max_freq_ghz": 5.9}},\n'
        '  {"name": "LFM chirp", "entity_type": "WAVEFORM", "confidence": 0.9, '
        '"properties": {"waveform_type": "LFM", "pulse_duration_us": 10}}\n'
        ']}\n'
    ),
    "weapon": (
        'Example: Given "The missile uses semi-active radar homing with a solid-fuel '
        'booster stage providing 3.2s burn time", extract:\n'
        '{"entities": [\n'
        '  {"name": "SARH", "entity_type": "GUIDANCE_METHOD", "confidence": 0.95, '
        '"properties": {"method": "SARH", "seeker_type": "semi-active radar"}},\n'
        '  {"name": "Booster stage", "entity_type": "PROPULSION_STAGE", "confidence": 0.9, '
        '"properties": {"stage_type": "booster", "fuel_type": "solid", "burn_time_s": 3.2}}\n'
        ']}\n'
    ),
    "operational": (
        'Example: Given "The system provides 360-degree surveillance with detection range '
        'of 150 km against 1 m2 RCS targets", extract:\n'
        '{"entities": [\n'
        '  {"name": "360-degree surveillance", "entity_type": "CAPABILITY", "confidence": 0.9, '
        '"properties": {"capability_name": "surveillance", "coverage": "360-degree"}},\n'
        '  {"name": "Detection performance", "entity_type": "RADAR_PERFORMANCE", "confidence": 0.9, '
        '"properties": {"detection_range_km": 150, "reference_rcs_m2": 1.0}}\n'
        ']}\n'
    ),
}
```

- [ ] **Step 2: Commit**

```bash
git add docker/docling-graph/app/prompts.py
git commit -m "fix(docling-graph): add few-shot examples for entity extraction"
```

---

### Task 6: Complete relationship extraction prompt from ontology

**Files:** Modify `docker/docling-graph/app/main.py` and `docker/docling-graph/app/prompts.py`

- [ ] **Step 1: Build relationship prompt from ontology at startup**

In `main.py`, add a new startup function and module-level state. After `_group_schema_prompts`:

```python
_relationship_prompt_context: str = ""
```

In `_build_group_schema_prompts()` (or a new function called from `lifespan`), build the relationship context from the ontology:

```python
def _build_relationship_prompt_context(ontology: dict[str, Any]) -> str:
    """Build relationship type descriptions and validation rules from ontology."""
    rel_types = ontology.get("relationship_types", [])
    validation = ontology.get("validation_matrix", [])

    lines = ["Available relationship types:"]
    for rt in rel_types:
        name = rt["name"]
        desc = rt.get("description", "")
        src = rt.get("source_type") or "any"
        tgt = rt.get("target_type") or "any"
        card = rt.get("cardinality", "")
        line = f"  - {name}: {desc}"
        if src != "any" or tgt != "any":
            line += f" (from: {src}, to: {tgt})"
        if card:
            line += f" [{card}]"
        lines.append(line)

    if validation:
        lines.append("\nAllowed (source, relationship, target) triples:")
        for v in validation:
            lines.append(f"  - {v['source']} → {v['relationship']} → {v['target']}")

    return "\n".join(lines)
```

Call it from `lifespan()` and store in `_relationship_prompt_context`.

- [ ] **Step 2: Update `get_relationship_prompt` to use ontology context**

In `prompts.py`, update `get_relationship_prompt()` to accept the ontology-derived context:

```python
def get_relationship_prompt(entities_context: list[dict], relationship_context: str = "") -> str:
    entity_lines = (
        "\n".join(
            f"  - {e['name']} ({e['entity_type']})" for e in entities_context
        )
        if entities_context
        else "  (no entities extracted)"
    )

    rel_section = relationship_context or _FALLBACK_RELATIONSHIP_TYPES

    return (
        "You are a military systems analyst specializing in relationships between "
        "equipment, capabilities, and organizational elements. "
        "Given the following entities extracted from a military technical document, "
        "identify relationships between them.\n\n"
        f"Known entities:\n{entity_lines}\n\n"
        f"{rel_section}\n\n"
        "Return only relationships supported by the text. "
        "Each relationship must connect two of the known entities listed above."
    )
```

Keep the old hardcoded list as `_FALLBACK_RELATIONSHIP_TYPES` for backward compat.

- [ ] **Step 3: Wire into extraction call**

In `main.py`, update `_extract_relationships()` to pass `_relationship_prompt_context`:

```python
system_prompt = get_relationship_prompt(entities_context, _relationship_prompt_context)
```

- [ ] **Step 4: Commit**

```bash
git add docker/docling-graph/app/main.py docker/docling-graph/app/prompts.py
git commit -m "fix(docling-graph): auto-generate relationship prompt from ontology YAML"
```

---

### Task 7: Chunked entity extraction with global relationship pass

**Files:** Modify `docker/docling-graph/app/main.py`

- [ ] **Step 1: Add text chunking helper**

Add after `_parse_json_from_llm`:

```python
def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks. Returns at least one chunk."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
```

- [ ] **Step 2: Add entity dedup helper**

```python
def _dedup_entities(entities: list[dict]) -> list[dict]:
    """Deduplicate entities by (name, entity_type), keeping highest confidence."""
    best: dict[tuple[str, str], dict] = {}
    for e in entities:
        key = (e.get("name", ""), e.get("entity_type", ""))
        existing = best.get(key)
        if existing is None or e.get("confidence", 0) > existing.get("confidence", 0):
            best[key] = e
    return list(best.values())
```

- [ ] **Step 3: Update `_run_full_extraction` to chunk entities, global relationships**

Read `GRAPH_EXTRACTION_CHUNK_SIZE` and `GRAPH_EXTRACTION_CHUNK_OVERLAP` from env (already in settings, need to pass through or read from env directly in the service).

```python
def _run_full_extraction(text: str) -> tuple[list[dict], list[dict]]:
    chunk_size = int(os.environ.get("GRAPH_EXTRACTION_CHUNK_SIZE", "12000"))
    chunk_overlap = int(os.environ.get("GRAPH_EXTRACTION_CHUNK_OVERLAP", "500"))
    chunks = _chunk_text(text, chunk_size, chunk_overlap)

    # Phase 1: Entity extraction per chunk (all groups in parallel per chunk)
    all_entities: list[dict] = []
    t0 = time.monotonic()

    for chunk_idx, chunk in enumerate(chunks):
        group_names = list(GROUP_MAP.keys())
        with ThreadPoolExecutor(max_workers=len(group_names)) as pool:
            futures = {
                pool.submit(_extract_entities_for_group, chunk, group): group
                for group in group_names
            }
            for future in as_completed(futures):
                group = futures[future]
                try:
                    all_entities.extend(future.result())
                except Exception as exc:
                    logger.warning("Entity extraction group=%s chunk=%d raised: %s", group, chunk_idx, exc)

    # Dedup across chunks
    all_entities = _dedup_entities(all_entities)

    entity_elapsed = time.monotonic() - t0
    logger.info(
        "Phase 1 complete: %d unique entities from %d chunks × %d groups in %.1fs",
        len(all_entities), len(chunks), len(GROUP_MAP), entity_elapsed,
    )

    # Phase 2: Relationship extraction on FULL text with all entities as context
    t1 = time.monotonic()
    entities_context = [
        {"name": e.get("name", ""), "entity_type": e.get("entity_type", "UNKNOWN")}
        for e in all_entities if e.get("name")
    ]
    all_relationships = _extract_relationships(text, entities_context)
    rel_elapsed = time.monotonic() - t1

    logger.info(
        "Phase 2 complete: %d relationships in %.1fs (total: %.1fs)",
        len(all_relationships), rel_elapsed, time.monotonic() - t0,
    )

    return all_entities, all_relationships
```

- [ ] **Step 4: Wire few-shot examples into entity extraction**

In `_extract_entities_for_group()`, add the few-shot example to the user prompt. Import and use `GROUP_FEW_SHOT_EXAMPLES`:

```python
    from app.prompts import GROUP_PROMPTS, GROUP_FEW_SHOT_EXAMPLES

    example = GROUP_FEW_SHOT_EXAMPLES.get(group_name, "")

    user_prompt = (
        f"Extract ALL instances of the following entity types from the text below.\n"
        f"Entity types to extract: {', '.join(entity_type_names)}\n\n"
        f"=== ENTITY TYPE SCHEMAS ===\n{schema_desc}\n"
        f"=== END SCHEMAS ===\n\n"
    )
    if example:
        user_prompt += f"=== EXAMPLE ===\n{example}=== END EXAMPLE ===\n\n"
    user_prompt += (
        f"=== TEXT ===\n{text}\n=== END TEXT ===\n\n"
        f"Return a JSON object with a single key \"entities\" containing an array.\n"
        # ... rest of format instructions unchanged
    )
```

- [ ] **Step 5: Commit**

```bash
git add docker/docling-graph/app/main.py
git commit -m "feat(docling-graph): chunked entity extraction with global relationship pass"
```

---

### Task 8: Add tests for all fixes

**Files:** Modify `docker/docling-graph/tests/test_direct_extraction.py`

- [ ] **Step 1: Test pattern fields in schema prompts**

```python
class TestSchemaPromptImprovements:
    def test_pattern_included_in_schema(self, setup_app):
        """Properties with pattern fields should include format annotation."""
        # component group (weapon) has NSN pattern
        prompt = setup_app._group_schema_prompts.get("weapon", "")
        assert "format:" in prompt or "NNNN" in prompt or "pattern" in prompt.lower()

    def test_parent_hierarchy_in_schema(self, setup_app):
        """Entity headers should include parent type."""
        prompt = setup_app._group_schema_prompts.get("equipment", "")
        assert "MilitarySystem" in prompt or "MilitaryAsset" in prompt
```

- [ ] **Step 2: Test chunking and dedup**

```python
class TestChunkingAndDedup:
    def test_chunk_text_short(self, setup_app):
        result = setup_app._chunk_text("short text", 1000, 100)
        assert len(result) == 1

    def test_chunk_text_splits(self, setup_app):
        text = "a" * 3000
        result = setup_app._chunk_text(text, 1000, 200)
        assert len(result) >= 3

    def test_dedup_keeps_highest_confidence(self, setup_app):
        entities = [
            {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM", "confidence": 0.7},
            {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM", "confidence": 0.95},
        ]
        result = setup_app._dedup_entities(entities)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95
```

- [ ] **Step 3: Test relationship prompt includes ontology context**

```python
class TestRelationshipPrompt:
    def test_relationship_prompt_has_descriptions(self, setup_app):
        """Relationship prompt should include descriptions from ontology."""
        from app.prompts import get_relationship_prompt
        prompt = get_relationship_prompt(
            [{"name": "Test", "entity_type": "RADAR_SYSTEM"}],
            setup_app._relationship_prompt_context,
        )
        assert "LAUNCHES" in prompt  # was missing from old hardcoded list
        assert "HAS_SEEKER" in prompt
        assert "description" in prompt.lower() or "→" in prompt
```

- [ ] **Step 4: Commit**

```bash
git add docker/docling-graph/tests/test_direct_extraction.py
git commit -m "test(docling-graph): add tests for ontology extraction improvements"
```
