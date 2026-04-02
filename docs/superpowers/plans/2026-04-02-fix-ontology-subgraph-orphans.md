# Fix Ontology Subgraph — Orphan Entity Relationships

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the ontology graph so that specification/parameter entities are connected to their parent system entities, and the subgraph viewer shows a real neighborhood instead of a single node.

**Architecture:** Three-layer fix: (1) make the Neo4j relationship writer resilient to entity-type mismatches, (2) strengthen the LLM relationship extraction prompt so it reliably generates SPECIFIED_BY edges, (3) add a co-occurrence fallback in the neighborhood query for any entities that remain orphaned after re-processing.

**Tech Stack:** Python 3.11, Neo4j 2025-community, Cytoscape.js (react-cytoscapejs), FastAPI, Celery

---

## Diagnosis Summary

| Metric | Value |
|--------|-------|
| Total entities | 245 |
| Orphan entities (no Entity↔Entity edge) | 175 (71%) |
| Documents producing 0 entities | 14 / 19 |
| SPECIFIED_BY edges for SNR-75 doc (`55653d21`) | 9 (correct) |
| SPECIFIED_BY edges for SA-2 doc (`34d9606f`) | 0 (broken — 8 specs extracted, 0 linked) |
| Edges lost to entity-type mismatch | 3+ per document |

**Root causes:**
1. `upsert_relationships_batch` uses `MATCH (a:Entity:{from_label} ...)` — if LLM returns `EQUIPMENT_SYSTEM` but entity was created as `MISSILE_SYSTEM`, the MATCH silently fails.
2. The relationship extraction prompt doesn't explicitly instruct the LLM to connect SPECIFICATION entities to systems — it works sometimes (SNR-75) but not always (SA-2).
3. `get_neighborhood_graph_async` uses `OPTIONAL MATCH` + `UNWIND` which drops all rows for orphan entities; the fallback returns only the center node.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `app/services/neo4j_graph.py:178-219` | Fix `upsert_relationships_batch` — name-only MATCH |
| Modify | `app/services/neo4j_graph.py:569-676` | Fix `get_neighborhood_graph_async` — co-occurrence fallback |
| Modify | `docker/docling-graph/app/prompts.py:161-191` | Strengthen `get_relationship_prompt` for SPECIFIED_BY |
| Modify | `docker/docling-graph/app/main.py:496-530` | Add user-prompt instruction for spec→system linking |
| Create | `tests/unit/test_upsert_relationships_batch.py` | Unit tests for name-only matching fix |
| Create | `tests/unit/test_neighborhood_graph.py` | Unit tests for co-occurrence fallback |
| Create | `tests/unit/test_relationship_prompt.py` | Unit test for prompt content |

---

## Chunk 1: Fix relationship writer and neighborhood query

### Task 1: Fix `upsert_relationships_batch` — name-only entity matching

**Files:**
- Modify: `app/services/neo4j_graph.py:178-219`
- Create: `tests/unit/test_upsert_relationships_batch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_upsert_relationships_batch.py`:

```python
"""Unit tests for upsert_relationships_batch name-only matching."""
from unittest.mock import MagicMock
import pytest

pytestmark = pytest.mark.unit


class TestUpsertRelationshipsBatch:
    def test_cypher_matches_by_name_not_type(self, mock_neo4j_driver):
        """MATCH clause should use Entity {name: ...} without type label."""
        from app.services.neo4j_graph import upsert_relationships_batch

        driver, session = mock_neo4j_driver
        session.run.return_value.single.return_value = {"cnt": 1}

        edges = [{
            "from_name": "SA-2 Guideline",
            "from_type": "EQUIPMENT_SYSTEM",  # wrong type — entity is MISSILE_SYSTEM
            "to_name": "Max Range",
            "to_type": "SPECIFICATION",
            "rel_type": "SPECIFIED_BY",
            "artifact_id": "art-1",
            "confidence": 0.9,
            "props": {"artifact_id": "art-1", "confidence": 0.9},
        }]
        upsert_relationships_batch(driver, edges)

        query = session.run.call_args.args[0]
        # Should NOT contain type labels in MATCH
        assert ":EQUIPMENT_SYSTEM" not in query
        assert ":SPECIFICATION" not in query
        # Should contain name-only match
        assert "Entity {name:" in query or "Entity{name:" in query or "{name: edge.from_name}" in query

    def test_groups_by_rel_type_only(self, mock_neo4j_driver):
        """Edges with same rel_type but different entity types go in one batch."""
        from app.services.neo4j_graph import upsert_relationships_batch

        driver, session = mock_neo4j_driver
        session.run.return_value.single.return_value = {"cnt": 2}

        edges = [
            {"from_name": "A", "from_type": "RADAR_SYSTEM", "to_name": "X",
             "to_type": "SPECIFICATION", "rel_type": "SPECIFIED_BY",
             "artifact_id": "a1", "confidence": 0.9, "props": {}},
            {"from_name": "B", "from_type": "MISSILE_SYSTEM", "to_name": "Y",
             "to_type": "SPECIFICATION", "rel_type": "SPECIFIED_BY",
             "artifact_id": "a1", "confidence": 0.9, "props": {}},
        ]
        result = upsert_relationships_batch(driver, edges)

        # One batch call (grouped by rel_type), not two (grouped by label triple)
        assert session.run.call_count == 1

    def test_returns_count(self, mock_neo4j_driver):
        from app.services.neo4j_graph import upsert_relationships_batch
        driver, session = mock_neo4j_driver
        session.run.return_value.single.return_value = {"cnt": 3}
        result = upsert_relationships_batch(driver, [
            {"from_name": "A", "from_type": "T", "to_name": "B", "to_type": "T",
             "rel_type": "REL", "artifact_id": "a", "confidence": 0.5, "props": {}},
        ])
        assert result == 3

    def test_exception_returns_zero(self, mock_neo4j_driver):
        from app.services.neo4j_graph import upsert_relationships_batch
        driver, session = mock_neo4j_driver
        session.run.side_effect = Exception("connection lost")
        result = upsert_relationships_batch(driver, [
            {"from_name": "A", "from_type": "T", "to_name": "B", "to_type": "T",
             "rel_type": "REL", "artifact_id": "a", "confidence": 0.5, "props": {}},
        ])
        assert result == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_upsert_relationships_batch.py -v`
Expected: `test_cypher_matches_by_name_not_type` FAILS (current code includes type labels); `test_groups_by_rel_type_only` FAILS (current code groups by triple)

- [ ] **Step 3: Implement the fix**

Edit `app/services/neo4j_graph.py:178-219`. Replace the function body:

```python
def upsert_relationships_batch(
    driver,
    edges: list[dict[str, Any]],
) -> int:
    """Batch upsert relationships grouped by relationship label. Returns count created.

    Each dict must have: from_name, from_type, to_name, to_type, rel_type,
    artifact_id, confidence, props.

    Nodes are matched by name only (not entity type label) so that
    relationships succeed even when the LLM returns a slightly different
    entity type than the one used during entity creation.

    Note: name-only matching assumes entity names are unique within the
    graph.  If two entities share a name with different types, this may
    create an unintended relationship — acceptable trade-off given the
    domain (military system names are distinct).
    """
    from collections import defaultdict

    by_rel: dict[str, list[dict]] = defaultdict(list)
    for e in edges:
        rel_label = _sanitize_label(e["rel_type"])
        by_rel[rel_label].append(e)

    total = 0
    try:
        with driver.session() as session:
            for rel_label, group in by_rel.items():
                query = f"""
                    UNWIND $edges AS edge
                    MATCH (a:Entity {{name: edge.from_name}})
                    MATCH (b:Entity {{name: edge.to_name}})
                    MERGE (a)-[r:{rel_label} {{artifact_id: edge.artifact_id}}]->(b)
                    ON CREATE SET r += edge.props
                    ON MATCH SET r.confidence = CASE
                        WHEN r.confidence < edge.confidence THEN edge.confidence
                        ELSE r.confidence
                    END
                    RETURN count(r) AS cnt
                """
                result = session.run(query, edges=group)
                total += result.single()["cnt"]
    except Exception as e:
        logger.warning("upsert_relationships_batch failed: %s", e)

    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_upsert_relationships_batch.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `python -m pytest tests/unit/test_neo4j_graph_operations.py tests/integration/test_pipeline_graph.py -v`
Expected: All existing tests PASS (integration tests mock this function, so they are unaffected)

- [ ] **Step 6: Commit**

```bash
git add app/services/neo4j_graph.py tests/unit/test_upsert_relationships_batch.py
git commit -m "fix: match entities by name only in upsert_relationships_batch

Entity type mismatches between extraction and ingestion caused MATCH
failures that silently dropped relationships. Now groups by rel_type
only and matches Entity nodes by name."
```

---

### Task 2: Add co-occurrence fallback in `get_neighborhood_graph_async`

**Files:**
- Modify: `app/services/neo4j_graph.py:569-676`
- Create: `tests/unit/test_neighborhood_graph.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_neighborhood_graph.py`:

```python
"""Unit tests for get_neighborhood_graph_async co-occurrence fallback."""
from unittest.mock import AsyncMock, MagicMock
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_async_driver():
    """Mock async Neo4j driver returning configurable query results."""
    session = AsyncMock()
    driver = AsyncMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return driver, session


class TestGetNeighborhoodGraphAsync:
    @pytest.mark.asyncio
    async def test_orphan_entity_triggers_cooccurrence(self, mock_async_driver):
        """When main query returns 0 rows, co-occurrence query runs."""
        from app.services.neo4j_graph import get_neighborhood_graph_async

        driver, session = mock_async_driver

        call_count = 0
        async def mock_run(query, **kwargs):
            nonlocal call_count
            call_count += 1
            result = AsyncMock()
            if call_count == 1:
                # Main query: no rows (orphan entity)
                result.data = AsyncMock(return_value=[])
            elif call_count == 2:
                # Fallback center node query
                result.data = AsyncMock(return_value=[{
                    "props": {"id": "center-uuid", "name": "Max Range",
                              "entity_type": "SPECIFICATION"},
                    "entity_type": "SPECIFICATION",
                }])
            elif call_count == 3:
                # Co-occurrence query
                result.data = AsyncMock(return_value=[
                    {"other_props": {"id": "sys-uuid", "name": "SA-2 Guideline",
                                     "entity_type": "MISSILE_SYSTEM"},
                     "other_type": "MISSILE_SYSTEM"},
                ])
            return result

        session.run = mock_run

        result = await get_neighborhood_graph_async(driver, "Max Range")
        assert len(result["nodes"]) == 2  # center + co-occurring
        assert len(result["edges"]) == 1
        assert result["edges"][0]["rel_type"] == "CO_OCCURS_WITH"

    @pytest.mark.asyncio
    async def test_connected_entity_skips_cooccurrence(self, mock_async_driver):
        """When main query returns edges, co-occurrence is NOT run."""
        from app.services.neo4j_graph import get_neighborhood_graph_async

        driver, session = mock_async_driver

        call_count = 0
        async def mock_run(query, **kwargs):
            nonlocal call_count
            call_count += 1
            result = AsyncMock()
            if call_count == 1:
                # Main query: has neighbors
                result.data = AsyncMock(return_value=[{
                    "center_props": {"id": "c-id", "name": "SNR-75"},
                    "center_type": "RADAR_SYSTEM",
                    "source": "SNR-75", "source_type": "RADAR_SYSTEM",
                    "source_props": {"id": "c-id", "name": "SNR-75"},
                    "rel_type": "SPECIFIED_BY",
                    "rel_props": {},
                    "target": "Search PRF", "target_type": "SPECIFICATION",
                    "target_props": {"id": "t-id", "name": "Search PRF"},
                }])
            return result

        session.run = mock_run

        result = await get_neighborhood_graph_async(driver, "SNR-75")
        assert len(result["edges"]) == 1
        assert result["edges"][0]["rel_type"] == "SPECIFIED_BY"
        # Only 1 session.run call — no fallback, no co-occurrence
        assert call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_neighborhood_graph.py -v`
Expected: `test_orphan_entity_triggers_cooccurrence` FAILS (no co-occurrence logic exists)

- [ ] **Step 3: Implement the fix**

Edit `app/services/neo4j_graph.py:569-676`. Two changes:

**Change 1** — Replace `OPTIONAL MATCH` with `MATCH` on line 584:
```python
    query = f"""
        MATCH (start:Entity {{name: $name}})
        MATCH path = (start)-[*1..{hop_count}]-(neighbor:Entity)
```
(Change `OPTIONAL MATCH path` to `MATCH path`)

**Change 2** — After the fallback center-node block (after line 670), add the co-occurrence fallback:

```python
    # Co-occurrence fallback: when no Entity-to-Entity edges exist, find
    # entities extracted from the same source chunks via ChunkRef nodes.
    if not edges and center is not None:
        center_id = center.get("id", entity_name)
        try:
            cooccur_q = """
                MATCH (start:Entity {name: $name})-[:EXTRACTED_FROM]->(c:ChunkRef)
                      <-[:EXTRACTED_FROM]-(other:Entity)
                WHERE other.name <> $name
                RETURN DISTINCT
                    properties(other) AS other_props,
                    other.entity_type AS other_type
                LIMIT $limit
            """
            async with driver.session() as session:
                result = await session.run(
                    cooccur_q, name=entity_name, limit=limit,
                )
                records = await result.data()
                for r in records:
                    other_props = r.get("other_props") or {}
                    other_id = other_props.get("id") or other_props.get("name")
                    if not other_id or other_id in nodes_map:
                        continue
                    node = dict(other_props)
                    node["entity_type"] = r["other_type"]
                    nodes_map[other_id] = node
                    edges.append({
                        "source": center_id,
                        "target": other_id,
                        "rel_type": "CO_OCCURS_WITH",
                    })
        except Exception as e:
            logger.warning(
                "get_neighborhood_graph_async co-occurrence query failed for '%s': %s",
                entity_name, e,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_neighborhood_graph.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full unit test suite**

Run: `python -m pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/neo4j_graph.py tests/unit/test_neighborhood_graph.py
git commit -m "fix: add co-occurrence fallback for orphan entities in subgraph view

When an entity has no direct Entity-to-Entity relationships, the
neighborhood query now finds entities extracted from the same source
chunks via ChunkRef nodes and displays them as CO_OCCURS_WITH edges.
Also fixes OPTIONAL MATCH + UNWIND anti-pattern."
```

---

## Chunk 2: Strengthen relationship extraction prompt

### Task 3: Improve relationship extraction prompt for SPECIFIED_BY

**Files:**
- Modify: `docker/docling-graph/app/prompts.py:161-191`
- Modify: `docker/docling-graph/app/main.py:496-530`
- Create: `tests/unit/test_relationship_prompt.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_relationship_prompt.py`:

```python
"""Unit tests for relationship extraction prompt content."""
import sys
from pathlib import Path
import pytest

# docling-graph lives in a separate container; add its source to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker" / "docling-graph"))
from app.prompts import get_relationship_prompt

pytestmark = pytest.mark.unit


class TestRelationshipPrompt:
    def test_prompt_contains_specification_linking_instruction(self):
        """Prompt must explicitly instruct linking SPECIFICATION to systems."""
        # Simulate entities context with specs and a system
        entities = [
            {"name": "SA-2 Guideline", "entity_type": "MISSILE_SYSTEM"},
            {"name": "Maximum missile range", "entity_type": "SPECIFICATION"},
        ]
        prompt = get_relationship_prompt(entities, "")
        lower = prompt.lower()
        assert "specification" in lower
        assert "specified_by" in lower
        # Must instruct to connect specs to their parent system
        assert "parent" in lower or "belongs to" in lower or "connect each" in lower

    def test_user_prompt_contains_specification_instruction(self):
        """The fallback prompt must mention SPECIFIED_BY."""
        prompt = get_relationship_prompt([], "")
        assert "SPECIFIED_BY" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_relationship_prompt.py -v`
Expected: FAIL — current prompt does not contain explicit specification-linking instruction

- [ ] **Step 3: Improve the system prompt**

Edit `docker/docling-graph/app/prompts.py:182-191`. Replace the `return` block in `get_relationship_prompt`:

```python
    return (
        "You are a military systems analyst specializing in relationships between "
        "equipment, capabilities, and organizational elements. "
        "Given the following entities extracted from a military technical document, "
        "identify relationships between them.\n\n"
        f"Known entities:\n{entity_lines}\n\n"
        f"{rel_section}\n\n"
        "IMPORTANT — connect SPECIFICATION entities to their parent systems:\n"
        "For each SPECIFICATION entity in the list above, determine which system "
        "entity it belongs to and create a SPECIFIED_BY relationship "
        "(e.g., MISSILE_SYSTEM → SPECIFIED_BY → SPECIFICATION). "
        "Specifications describe measurable parameters (range, altitude, frequency, "
        "power, speed, time) of a system — connect each one.\n\n"
        "Return only relationships supported by the text. "
        "Each relationship must connect two of the known entities listed above."
    )
```

- [ ] **Step 4: Improve the user prompt**

Edit `docker/docling-graph/app/main.py:501-508`. Add a SPECIFIED_BY reminder to the user prompt:

```python
    user_prompt = (
        f"Analyze this text and extract relationships between the known entities:\n\n"
        f"=== TEXT ===\n{text}\n=== END TEXT ===\n\n"
        "Return a JSON object with a single key 'relationships' containing an array. "
        "Each relationship object must have: from_name, from_type, rel_type, to_name, to_type, confidence (0.0-1.0).\n"
        "IMPORTANT: For every SPECIFICATION entity, create a SPECIFIED_BY edge from "
        "the system it describes to the specification.\n"
        "Example: {\"relationships\": [{\"from_name\": \"AN/MPQ-53\", \"from_type\": \"RADAR_SYSTEM\", "
        "\"rel_type\": \"INSTALLED_ON\", \"to_name\": \"Patriot\", \"to_type\": \"PLATFORM\", \"confidence\": 0.9}]}\n"
        "Return ONLY valid JSON."
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_relationship_prompt.py -v`
Expected: PASS

- [ ] **Step 6: Run docling-graph container tests**

Run: `cd docker/docling-graph && python -m pytest tests/ -v`
Expected: All existing container tests PASS (prompt change does not break template/schema tests)

- [ ] **Step 7: Commit**

```bash
git add docker/docling-graph/app/prompts.py docker/docling-graph/app/main.py tests/unit/test_relationship_prompt.py
git commit -m "fix: strengthen relationship prompt for SPECIFIED_BY edges

The LLM inconsistently generated SPECIFIED_BY relationships between
system entities and their specifications. Add explicit instructions
in both the system and user prompts to connect every SPECIFICATION
entity to its parent system."
```

---

### Task 4: Rebuild containers, re-process documents, and verify

**Files:**
- No code changes — operational verification

- [ ] **Step 1: Rebuild the API and docling-graph containers**

```bash
docker compose build api docling-graph
docker compose up -d api docling-graph
```

Wait for health checks to pass:
```bash
docker compose ps
```

- [ ] **Step 2: Verify the upsert fix with existing data**

Re-process a document that had orphan specifications. Use the API to trigger re-extraction:

```bash
# Find the SA-2 document ID
curl -s http://localhost:8005/v1/documents | python3 -c "
import json, sys
for d in json.load(sys.stdin):
    if 'SA-2' in d.get('title', '') or d['id'] == '34d9606f-95a8-4c9a-84fb-590d7ddef5c3':
        print(d['id'], d.get('title'))
"
```

Trigger re-processing via the pipeline (exact endpoint depends on available admin routes).

- [ ] **Step 3: Verify orphan count decreased**

```bash
docker compose exec neo4j cypher-shell -u neo4j -p eip_neo4j_secret \
  "MATCH (e:Entity) WHERE NOT (e)-[]-(:Entity) RETURN count(e) AS orphan_count"
```

Expected: Significantly fewer than 175 orphans

- [ ] **Step 4: Test the subgraph view via API**

```bash
# Test with a previously-orphan specification
curl -s -X POST http://localhost:8005/v1/graph/neighborhood \
  -H "Content-Type: application/json" \
  -d '{"entity_name": "Maximum missile range", "hop_count": 2}' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Nodes: {len(d[\"nodes\"])}, Edges: {len(d[\"edges\"])}')"
```

Expected: More than 1 node and at least 1 edge (either SPECIFIED_BY or CO_OCCURS_WITH)

- [ ] **Step 5: Test in the browser**

1. Open the UI → Ontology tab → Search
2. Search for "missile"
3. Click the graph view circle on a result
4. Verify: multiple nodes visible, connected by edges

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -v --timeout=120
```

Expected: All tests PASS

- [ ] **Step 7: Run VERIFICATION_CHECKLIST.md**

Review `VERIFICATION_CHECKLIST.md` and confirm all relevant items pass.

- [ ] **Step 8: Update README if needed**

If any user-facing behavior changed beyond the bug fix, update `README.md` to reflect it. Only commit if there are actual doc changes:

```bash
git diff --stat  # check if anything changed
# If README was updated:
git add README.md
git commit -m "docs: update README with ontology subgraph fix notes"
```
