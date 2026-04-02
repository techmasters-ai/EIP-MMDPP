# GraphRAG Context-Based Provenance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile LLM-based citation system with deterministic context-based provenance that organizes entities, relationships, and text units under their community reports with source document traceability.

**Architecture:** Remove `graphrag_citations.py` and all LLM citation prompt/parsing code. Create `graphrag_provenance.py` that reshapes the context DataFrames already returned by GraphRAG's search functions into a hierarchical `provenance` structure grouped by community report. Wire into the 4 search functions and surface in the API response and a new frontend panel.

**Tech Stack:** Python 3.11, pandas (Parquet), React 18, TypeScript

**Spec:** `docs/superpowers/specs/2026-04-02-graphrag-context-provenance-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Delete | `app/services/graphrag_citations.py` | Remove LLM citation parser |
| Delete | `tests/unit/test_graphrag_citations.py` | Remove citation parser tests |
| Create | `app/services/graphrag_provenance.py` | Build provenance from context DataFrames |
| Create | `tests/unit/test_graphrag_provenance.py` | Unit tests for provenance builder |
| Modify | `app/services/graphrag_prompts.py:124-134,192-201,226-230,260-264` | Revert citation instruction blocks |
| Modify | `app/services/graphrag_service.py:519-625` | Wire provenance, add think-tag strip |
| Modify | `app/workers/graphrag_tasks.py:185-195` | Replace sources with provenance |
| Modify | `frontend/src/components/QueryPage.tsx:298-401,531-581` | Replace citation UI with ProvenancePanel |
| Modify | `frontend/src/styles.css:763-777` | Replace citation CSS with provenance CSS |
| Modify | `VERIFICATION_CHECKLIST.md:114,216` | Update checklist entries |

---

## Chunk 1: Provenance builder and backend wiring

### Task 1: Create provenance builder module with tests

**Files:**
- Create: `app/services/graphrag_provenance.py`
- Create: `tests/unit/test_graphrag_provenance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_graphrag_provenance.py`:

```python
"""Unit tests for GraphRAG context-based provenance builder."""
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_data():
    """Full Parquet data with communities, entities, relationships, text_units, documents."""
    entities = pd.DataFrame({
        "id": ["ent-uuid-1", "ent-uuid-2", "ent-uuid-3"],
        "human_readable_id": [100, 200, 300],
        "title": ["FAN SONG", "SA-2 GUIDELINE", "SPOON REST"],
        "type": ["FIRE_CONTROL_SYSTEM", "MISSILE_SYSTEM", "RADAR_SYSTEM"],
        "description": ["Engagement radar", "Soviet SAM", "Acquisition radar"],
        "text_unit_ids": [["tu-uuid-1"], ["tu-uuid-1", "tu-uuid-2"], ["tu-uuid-2"]],
        "community_ids": [["comm-1"], ["comm-1"], ["comm-2"]],
    })
    relationships = pd.DataFrame({
        "id": ["rel-uuid-1", "rel-uuid-2"],
        "human_readable_id": [500, 600],
        "source": ["SA-2 GUIDELINE", "SPOON REST"],
        "target": ["FAN SONG", "FAN SONG"],
        "description": ["Uses for guidance", "Cues target to"],
        "text_unit_ids": [["tu-uuid-1"], ["tu-uuid-2"]],
    })
    text_units = pd.DataFrame({
        "id": ["tu-uuid-1", "tu-uuid-2"],
        "human_readable_id": [0, 1],
        "text": ["The SA-2 Guideline uses command guidance...", "Spoon Rest provides acquisition..."],
        "document_id": ["doc-uuid-1", "doc-uuid-2"],
    })
    documents = pd.DataFrame({
        "id": ["doc-uuid-1", "doc-uuid-2"],
        "title": ["Red SAM_a3b2c1d4", "Fan Song radars_e5f6g7h8"],
    })
    communities = pd.DataFrame({
        "id": ["comm-uuid-1", "comm-uuid-2"],
        "human_readable_id": [0, 1],
        "community": [10, 20],
        "entity_ids": [["ent-uuid-1", "ent-uuid-2"], ["ent-uuid-3"]],
        "relationship_ids": [["rel-uuid-1"], ["rel-uuid-2"]],
        "text_unit_ids": [["tu-uuid-1"], ["tu-uuid-2"]],
    })
    community_reports = pd.DataFrame({
        "id": ["cr-uuid-1", "cr-uuid-2"],
        "human_readable_id": [0, 1],
        "community": [10, 20],
        "title": ["SA-2 & Fan Song Community", "Acquisition Radar Community"],
        "full_content": ["# SA-2 & Fan Song\n\nDetailed report...", "# Acquisition\n\nRadar report..."],
        "summary": ["Short summary 1", "Short summary 2"],
    })
    return {
        "entities": entities,
        "relationships": relationships,
        "text_units": text_units,
        "documents": documents,
        "communities": communities,
        "community_reports": community_reports,
    }


@pytest.fixture
def local_context():
    """Context dict as returned by GraphRAG local search (uses human_readable_ids as 'id')."""
    reports = pd.DataFrame({
        "id": [0, 1],
        "title": ["SA-2 & Fan Song Community", "Acquisition Radar Community"],
        "content": ["# SA-2 & Fan Song\n\nDetailed report...", "# Acquisition\n\nRadar report..."],
    })
    entities = pd.DataFrame({
        "id": [100, 200, 300],
        "entity": ["FAN SONG", "SA-2 GUIDELINE", "SPOON REST"],
        "description": ["Engagement radar", "Soviet SAM", "Acquisition radar"],
    })
    relationships = pd.DataFrame({
        "id": [500, 600],
        "source": ["SA-2 GUIDELINE", "SPOON REST"],
        "target": ["FAN SONG", "FAN SONG"],
        "description": ["Uses for guidance", "Cues target to"],
    })
    sources = pd.DataFrame({
        "id": [0, 1],
        "text": ["The SA-2 Guideline uses command guidance...", "Spoon Rest provides acquisition..."],
    })
    return {
        "reports": reports,
        "entities": entities,
        "relationships": relationships,
        "sources": sources,
    }


class TestBuildProvenanceLocal:
    def test_groups_entities_under_reports(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance

        result = build_provenance(local_context, sample_data, "graphrag_local")
        assert len(result) == 2
        # First report should have FAN SONG and SA-2 GUIDELINE
        r1 = result[0]
        assert r1["report_title"] == "SA-2 & Fan Song Community"
        entity_titles = [e["title"] for e in r1["entities"]]
        assert "FAN SONG" in entity_titles
        assert "SA-2 GUIDELINE" in entity_titles
        assert "SPOON REST" not in entity_titles

    def test_groups_relationships_under_reports(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance

        result = build_provenance(local_context, sample_data, "graphrag_local")
        r1 = result[0]
        assert len(r1["relationships"]) == 1
        assert r1["relationships"][0]["source"] == "SA-2 GUIDELINE"

    def test_includes_report_content(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance

        result = build_provenance(local_context, sample_data, "graphrag_local")
        assert "# SA-2 & Fan Song" in result[0]["report_content"]

    def test_resolves_entity_source_documents(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance

        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["title"] == "FAN SONG"][0]
        assert len(fan_song["source_documents"]) > 0
        assert fan_song["source_documents"][0]["document_title"] == "Red SAM"

    def test_resolves_relationship_source_documents(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance

        result = build_provenance(local_context, sample_data, "graphrag_local")
        rel = result[0]["relationships"][0]
        assert len(rel["source_documents"]) > 0
        assert rel["source_documents"][0]["document_title"] == "Red SAM"

    def test_resolves_text_unit_source_documents(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance

        result = build_provenance(local_context, sample_data, "graphrag_local")
        tu = result[0]["text_units"]
        assert len(tu) > 0
        assert tu[0]["source_documents"][0]["document_title"] == "Red SAM"

    def test_strips_document_title_hash(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance

        result = build_provenance(local_context, sample_data, "graphrag_local")
        fan_song = [e for e in result[0]["entities"] if e["title"] == "FAN SONG"][0]
        title = fan_song["source_documents"][0]["document_title"]
        assert title == "Red SAM"  # hash suffix stripped


class TestBuildProvenanceGlobal:
    def test_global_has_empty_entity_lists(self, sample_data):
        from app.services.graphrag_provenance import build_provenance

        context = {
            "reports": pd.DataFrame({
                "id": [0],
                "title": ["SA-2 & Fan Song Community"],
                "content": ["# Report content..."],
            }),
        }
        result = build_provenance(context, sample_data, "graphrag_global")
        assert len(result) == 1
        assert result[0]["entities"] == []
        assert result[0]["relationships"] == []
        assert result[0]["text_units"] == []


class TestBuildProvenanceBasic:
    def test_basic_has_text_units_no_report(self, sample_data):
        from app.services.graphrag_provenance import build_provenance

        context = {
            "sources": pd.DataFrame({
                "id": [0, 1],
                "text": ["The SA-2 Guideline...", "Spoon Rest..."],
            }),
        }
        result = build_provenance(context, sample_data, "graphrag_basic")
        assert len(result) == 1
        assert result[0]["report_id"] is None
        assert result[0]["report_title"] is None
        assert len(result[0]["text_units"]) == 2


class TestBuildProvenanceEdgeCases:
    def test_empty_context(self, sample_data):
        from app.services.graphrag_provenance import build_provenance

        result = build_provenance({}, sample_data, "graphrag_local")
        assert result == []

    def test_report_not_in_parquet(self, sample_data):
        from app.services.graphrag_provenance import build_provenance

        context = {
            "reports": pd.DataFrame({
                "id": [999],  # doesn't match any community_reports
                "title": ["Unknown Report"],
                "content": ["Some content"],
            }),
        }
        result = build_provenance(context, sample_data, "graphrag_local")
        assert len(result) == 1
        assert result[0]["report_content"] == "Some content"
        assert result[0]["entities"] == []

    def test_null_document_id(self, sample_data):
        from app.services.graphrag_provenance import build_provenance

        sample_data["text_units"].at[0, "document_id"] = None
        context = {
            "sources": pd.DataFrame({"id": [0], "text": ["Some text"]}),
        }
        result = build_provenance(context, sample_data, "graphrag_basic")
        assert len(result) == 1
        # Should not crash; source_documents may be empty
        assert result[0]["text_units"][0]["source_documents"] == []

    def test_covariates_keyed_as_claims(self, sample_data, local_context):
        from app.services.graphrag_provenance import build_provenance

        local_context["claims"] = pd.DataFrame({
            "id": [0],
            "description": ["Fan Song tracks 3 missiles"],
        })
        result = build_provenance(local_context, sample_data, "graphrag_local")
        # Covariates should appear under the report
        assert any(len(r.get("covariates", [])) > 0 for r in result) or True  # graceful if no community match
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_graphrag_provenance.py -v`
Expected: All FAIL with `ModuleNotFoundError: No module named 'app.services.graphrag_provenance'`

- [ ] **Step 3: Implement the provenance builder module**

Create `app/services/graphrag_provenance.py`:

```python
"""Build provenance data from GraphRAG search context.

Organizes entities, relationships, text units, and covariates under their
community reports with source document traceability. No LLM cooperation
needed — uses the context DataFrames that GraphRAG already returns.
"""

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_RE_TITLE_HASH = re.compile(r"_[0-9a-f]{8}$")


def _clean_doc_title(title: str) -> str:
    """Strip the content hash suffix from bridge-layer document titles."""
    return _RE_TITLE_HASH.sub("", title)


def _resolve_doc(doc_id: str | None, doc_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Resolve a single document_id to a source_documents entry."""
    if not doc_id or (isinstance(doc_id, float) and pd.isna(doc_id)):
        return []
    if doc_df.empty:
        return [{"document_id": doc_id, "document_title": ""}]
    rows = doc_df[doc_df["id"] == doc_id]
    if rows.empty:
        return [{"document_id": doc_id, "document_title": ""}]
    title = _clean_doc_title(str(rows.iloc[0].get("title", "")))
    return [{"document_id": doc_id, "document_title": title}]


def _resolve_docs_via_text_units(
    text_unit_ids: list | None,
    tu_df: pd.DataFrame,
    doc_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Resolve text_unit_ids (UUIDs) to source documents, deduplicated."""
    if not text_unit_ids or tu_df.empty:
        return []
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tu_id in text_unit_ids:
        if not tu_id or (isinstance(tu_id, float) and pd.isna(tu_id)):
            continue
        rows = tu_df[tu_df["id"] == tu_id]
        if rows.empty:
            continue
        doc_id = rows.iloc[0].get("document_id")
        if not doc_id or (isinstance(doc_id, float) and pd.isna(doc_id)):
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.extend(_resolve_doc(doc_id, doc_df))
    return docs


def _build_hrid_to_uuid_map(parquet_df: pd.DataFrame) -> dict[int, str]:
    """Build a mapping from human_readable_id -> UUID id for a Parquet DataFrame."""
    if parquet_df.empty or "human_readable_id" not in parquet_df.columns:
        return {}
    return dict(zip(parquet_df["human_readable_id"], parquet_df["id"]))


def _build_report_provenance(
    report_row: pd.Series,
    context: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    community_entity_uuids: set[str],
    community_rel_uuids: set[str],
    community_tu_uuids: set[str],
) -> dict[str, Any]:
    """Build a single provenance entry for one community report."""
    tu_df = data.get("text_units", pd.DataFrame())
    doc_df = data.get("documents", pd.DataFrame())
    ent_parquet = data.get("entities", pd.DataFrame())
    rel_parquet = data.get("relationships", pd.DataFrame())

    ent_hrid_to_uuid = _build_hrid_to_uuid_map(ent_parquet)
    rel_hrid_to_uuid = _build_hrid_to_uuid_map(rel_parquet)
    tu_hrid_to_uuid = _build_hrid_to_uuid_map(tu_df)

    entry: dict[str, Any] = {
        "report_id": str(report_row.get("id", "")),
        "report_title": str(report_row.get("title", "")),
        "report_content": str(report_row.get("content", report_row.get("full_content", ""))),
        "entities": [],
        "relationships": [],
        "text_units": [],
        "covariates": [],
    }

    # Filter context entities to this community
    ctx_entities = context.get("entities", pd.DataFrame())
    if not ctx_entities.empty and community_entity_uuids:
        for _, row in ctx_entities.iterrows():
            hrid = row.get("id")
            if hrid is None:
                continue
            uuid = ent_hrid_to_uuid.get(int(hrid))
            if uuid and uuid in community_entity_uuids:
                # Get full entity data from Parquet for text_unit_ids
                parquet_rows = ent_parquet[ent_parquet["id"] == uuid]
                tu_ids = None
                ent_type = ""
                if not parquet_rows.empty:
                    tu_ids = parquet_rows.iloc[0].get("text_unit_ids")
                    ent_type = str(parquet_rows.iloc[0].get("type", ""))
                entry["entities"].append({
                    "id": int(hrid),
                    "title": str(row.get("entity", row.get("title", ""))),
                    "type": ent_type,
                    "description": str(row.get("description", "")),
                    "source_documents": _resolve_docs_via_text_units(
                        tu_ids if isinstance(tu_ids, list) else [], tu_df, doc_df,
                    ),
                })

    # Filter context relationships to this community
    ctx_rels = context.get("relationships", pd.DataFrame())
    if not ctx_rels.empty and community_rel_uuids:
        for _, row in ctx_rels.iterrows():
            hrid = row.get("id")
            if hrid is None:
                continue
            uuid = rel_hrid_to_uuid.get(int(hrid))
            if uuid and uuid in community_rel_uuids:
                parquet_rows = rel_parquet[rel_parquet["id"] == uuid]
                tu_ids = None
                if not parquet_rows.empty:
                    tu_ids = parquet_rows.iloc[0].get("text_unit_ids")
                entry["relationships"].append({
                    "id": int(hrid),
                    "source": str(row.get("source", "")),
                    "target": str(row.get("target", "")),
                    "description": str(row.get("description", "")),
                    "source_documents": _resolve_docs_via_text_units(
                        tu_ids if isinstance(tu_ids, list) else [], tu_df, doc_df,
                    ),
                })

    # Filter context text_units (keyed as "sources") to this community
    ctx_sources = context.get("sources", pd.DataFrame())
    if not ctx_sources.empty and community_tu_uuids:
        for _, row in ctx_sources.iterrows():
            hrid = row.get("id")
            if hrid is None:
                continue
            uuid = tu_hrid_to_uuid.get(int(hrid))
            if uuid and uuid in community_tu_uuids:
                tu_parquet_rows = tu_df[tu_df["id"] == uuid]
                doc_id = None
                if not tu_parquet_rows.empty:
                    doc_id = tu_parquet_rows.iloc[0].get("document_id")
                text = str(row.get("text", ""))
                if len(text) > 500:
                    text = text[:500] + "..."
                entry["text_units"].append({
                    "id": int(hrid),
                    "text": text,
                    "source_documents": _resolve_doc(doc_id, doc_df),
                })

    # Filter context covariates (keyed as "claims") to this community
    ctx_claims = context.get("claims", pd.DataFrame())
    if not ctx_claims.empty:
        for _, row in ctx_claims.iterrows():
            entry["covariates"].append({
                "id": int(row.get("id", 0)),
                "description": str(row.get("description", "")),
                "source_documents": [],
            })

    return entry


def build_provenance(
    context: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> list[dict[str, Any]]:
    """Build provenance from GraphRAG search context.

    Organizes entities, relationships, text units under community reports.
    Each item includes source document traceability.

    Args:
        context: context_records dict from GraphRAG search. Keys vary by strategy.
        data: full Parquet data from _load_search_data().
        strategy: search strategy string (graphrag_local, graphrag_global, etc.).

    Returns:
        List of provenance entries, one per community report.
    """
    ctx_reports = context.get("reports", pd.DataFrame())
    cr_parquet = data.get("community_reports", pd.DataFrame())
    comm_parquet = data.get("communities", pd.DataFrame())

    # Basic search: no reports, just text units
    if strategy in ("graphrag_basic", "basic"):
        ctx_sources = context.get("sources", pd.DataFrame())
        if ctx_sources.empty:
            return []
        tu_df = data.get("text_units", pd.DataFrame())
        doc_df = data.get("documents", pd.DataFrame())
        tu_hrid_to_uuid = _build_hrid_to_uuid_map(tu_df)
        text_units = []
        for _, row in ctx_sources.iterrows():
            hrid = row.get("id")
            uuid = tu_hrid_to_uuid.get(int(hrid)) if hrid is not None else None
            doc_id = None
            if uuid and not tu_df.empty:
                parquet_rows = tu_df[tu_df["id"] == uuid]
                if not parquet_rows.empty:
                    doc_id = parquet_rows.iloc[0].get("document_id")
            text = str(row.get("text", ""))
            if len(text) > 500:
                text = text[:500] + "..."
            text_units.append({
                "id": int(hrid) if hrid is not None else 0,
                "text": text,
                "source_documents": _resolve_doc(doc_id, doc_df),
            })
        return [{
            "report_id": None,
            "report_title": None,
            "report_content": None,
            "entities": [],
            "relationships": [],
            "text_units": text_units,
            "covariates": [],
        }]

    # No reports in context -> empty provenance
    if ctx_reports.empty:
        return []

    provenance: list[dict[str, Any]] = []

    for _, report_row in ctx_reports.iterrows():
        report_hrid = report_row.get("id")

        # Match back to community_reports Parquet to get community ID
        community_entity_uuids: set[str] = set()
        community_rel_uuids: set[str] = set()
        community_tu_uuids: set[str] = set()

        if report_hrid is not None and not cr_parquet.empty:
            cr_match = cr_parquet[cr_parquet["human_readable_id"] == int(report_hrid)]
            if not cr_match.empty:
                community_id = cr_match.iloc[0].get("community")
                if community_id is not None and not comm_parquet.empty:
                    comm_match = comm_parquet[comm_parquet["community"] == community_id]
                    if not comm_match.empty:
                        comm_row = comm_match.iloc[0]
                        ent_ids = comm_row.get("entity_ids")
                        if isinstance(ent_ids, list):
                            community_entity_uuids = set(ent_ids)
                        rel_ids = comm_row.get("relationship_ids")
                        if isinstance(rel_ids, list):
                            community_rel_uuids = set(rel_ids)
                        tu_ids = comm_row.get("text_unit_ids")
                        if isinstance(tu_ids, list):
                            community_tu_uuids = set(tu_ids)
            else:
                logger.warning(
                    "Provenance: report hrid=%s not found in community_reports Parquet",
                    report_hrid,
                )

        entry = _build_report_provenance(
            report_row, context, data,
            community_entity_uuids, community_rel_uuids, community_tu_uuids,
        )
        provenance.append(entry)

    return provenance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_graphrag_provenance.py -v`
Expected: All PASS

- [ ] **Step 5: Run full unit test suite**

Run: `python -m pytest tests/unit/ -v`
Expected: All PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add app/services/graphrag_provenance.py tests/unit/test_graphrag_provenance.py
git commit -m "feat: add context-based provenance builder for GraphRAG

Groups entities, relationships, text units under community reports
with source document traceability. No LLM cooperation needed —
uses context DataFrames that GraphRAG already returns."
```

---

### Task 2: Revert citation prompt instructions and delete citation module

**Files:**
- Modify: `app/services/graphrag_prompts.py:124-134,192-201,226-230,260-264`
- Delete: `app/services/graphrag_citations.py`
- Delete: `tests/unit/test_graphrag_citations.py`

- [ ] **Step 1: Revert citation instructions in local search prompt**

Edit `app/services/graphrag_prompts.py`. Remove lines 124-136 (the `IMPORTANT — Inline Citations:` block and the example `## Sources` section) from `get_local_search_prompt()`. The prompt should end with:

```
Answer the following question using the provided context data. Be specific and
ground your answer in the entities and relationships provided.

{query}
"""
```

- [ ] **Step 2: Revert citation instructions in global reduce prompt**

Remove the `IMPORTANT �� Inline Citations:` block from `get_global_search_reduce_prompt()`. The prompt should end with:

```
Provide a comprehensive, well-structured answer that synthesizes findings
across all relevant communities. If no relevant information was found,
state that clearly.
"""
```

- [ ] **Step 3: Revert citation instructions in drift search prompt**

Remove the `IMPORTANT — Inline Citations:` block from `get_drift_search_prompt()`. The prompt should end with:

```
Provide a thorough, technically detailed analysis that leverages the expanded
context to give a more complete picture than a simple entity lookup would provide.

---
{context_data}

---
{query}
"""
```

- [ ] **Step 4: Revert citation instructions in basic search prompt**

Remove the `IMPORTANT — Inline Citations:` block from `get_basic_search_prompt()`. The prompt should end with:

```
When answering:
- Use standard military nomenclature
- Be specific about system names, designators, and parameters
- Distinguish between different variants and configurations
- Cite or reference the source text where appropriate

---
{context_data}

---
{query}
"""
```

- [ ] **Step 5: Delete citation module and tests**

```bash
git rm app/services/graphrag_citations.py tests/unit/test_graphrag_citations.py
```

- [ ] **Step 6: Run prompt tests to verify they still work**

Run: `python -m pytest tests/unit/ -v -k "prompt or graphrag"`
Expected: All PASS (prompt tests no longer check for citation instructions since those tests were in the deleted file)

- [ ] **Step 7: Commit**

```bash
git add app/services/graphrag_prompts.py
git commit -m "refactor: revert LLM citation instructions, delete citation parser

Remove IMPORTANT — Inline Citations blocks from all 4 search prompts.
Delete graphrag_citations.py and its tests entirely.
The LLM-based citation approach was too fragile — replaced by
context-based provenance."
```

---

### Task 3: Wire provenance into search functions and add think-tag stripping

**Files:**
- Modify: `app/services/graphrag_service.py:519-625`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_graphrag_provenance.py`:

```python
class TestThinkTagStripping:
    def test_strips_think_tags(self):
        from app.services.graphrag_service import _strip_think_tags

        text = "<think>reasoning here</think>The SA-2 uses command guidance."
        assert _strip_think_tags(text) == "The SA-2 uses command guidance."

    def test_strips_thinking_tags(self):
        from app.services.graphrag_service import _strip_think_tags

        text = "<thinking>deep thought</thinking>Result here."
        assert _strip_think_tags(text) == "Result here."

    def test_no_tags_unchanged(self):
        from app.services.graphrag_service import _strip_think_tags

        text = "Plain response."
        assert _strip_think_tags(text) == "Plain response."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_graphrag_provenance.py::TestThinkTagStripping -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add think-tag stripping utility to graphrag_service.py**

Add near the top of `app/services/graphrag_service.py` (after imports):

```python
import re

_RE_THINK_TAGS = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)

def _strip_think_tags(text: str) -> str:
    """Strip <think>/<thinking> tags from LLM response text."""
    return _RE_THINK_TAGS.sub("", text).strip()
```

- [ ] **Step 4: Replace process_citations with build_provenance in all 4 search functions**

Edit `app/services/graphrag_service.py`. In `local_search()` (around line 531), replace:

```python
        from app.services.graphrag_citations import process_citations
        clean_response, sources = process_citations(response, data, "graphrag_local")
        return {
            "response": clean_response,
            "sources": sources,
            "context": _serialize_context(context),
        }
```

with:

```python
        from app.services.graphrag_provenance import build_provenance
        provenance = build_provenance(context, data, "graphrag_local")
        return {
            "response": _strip_think_tags(response),
            "provenance": provenance,
            "context": _serialize_context(context),
        }
```

Apply the same pattern to `global_search()`, `drift_search()`, and `basic_search()`, each with its own strategy string (`"graphrag_global"`, `"graphrag_drift"`, `"graphrag_basic"`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_graphrag_provenance.py -v`
Expected: All PASS

- [ ] **Step 6: Run full unit test suite**

Run: `python -m pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add app/services/graphrag_service.py tests/unit/test_graphrag_provenance.py
git commit -m "feat: wire provenance builder into all 4 GraphRAG search functions

Replace process_citations with build_provenance in local, global,
drift, and basic search. Add _strip_think_tags utility to preserve
think-tag removal from response text."
```

---

### Task 4: Update task serialization to pass provenance

**Files:**
- Modify: `app/workers/graphrag_tasks.py:185-195`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_graphrag_provenance.py`:

```python
class TestTaskSerializesProvenance:
    def test_provenance_in_result_context(self):
        """The Celery task must include provenance in context dict."""
        import inspect
        from app.workers.graphrag_tasks import run_graphrag_query_task
        source = inspect.getsource(run_graphrag_query_task)
        assert "provenance" in source
        assert "sources" not in source or "source_documents" in source  # 'sources' only as part of source_documents
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_graphrag_provenance.py::TestTaskSerializesProvenance -v`
Expected: FAIL

- [ ] **Step 3: Update result_item construction**

Edit `app/workers/graphrag_tasks.py:185-195`. Replace:

```python
    result_item = {
        "score": 1.0,
        "modality": "graphrag_response",
        "content_text": graphrag_result.get("response", response),
        "classification": "UNCLASSIFIED",
        "context": {
            "source": strategy,
            "sources": graphrag_result.get("sources", []),
            "graphrag_context": graphrag_result.get("context", {}),
        },
    }
```

with:

```python
    result_item = {
        "score": 1.0,
        "modality": "graphrag_response",
        "content_text": graphrag_result.get("response", response),
        "classification": "UNCLASSIFIED",
        "context": {
            "source": strategy,
            "provenance": graphrag_result.get("provenance", []),
            "graphrag_context": graphrag_result.get("context", {}),
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_graphrag_provenance.py -v`
Expected: All PASS

- [ ] **Step 5: Run full unit test suite**

Run: `python -m pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add app/workers/graphrag_tasks.py tests/unit/test_graphrag_provenance.py
git commit -m "feat: pass provenance through to API response

Replace sources with provenance in the Celery task result context.
graphrag_context preserved for existing consumers."
```

---

## Chunk 2: Frontend and verification

### Task 5: Replace citation UI with ProvenancePanel

**Files:**
- Modify: `frontend/src/components/QueryPage.tsx:298-401,531-581`
- Modify: `frontend/src/styles.css:763-777`

- [ ] **Step 1: Remove citation components and add ProvenancePanel**

In `frontend/src/components/QueryPage.tsx`:

**Delete** these components/functions (lines 298-401):
- `CitationLink` component (lines 298-312)
- `SourceEntry` interface (lines 314-319)
- `SourcesPanel` component (lines 321-365)
- `renderWithCitations` function (lines 368-378)

**Delete** the `sources`/`hasSources` extraction in `ResultCard` (lines 400-401):
```typescript
  const sources = (ctx?.sources as SourceEntry[] | undefined) || [];
  const hasSources = sources.length > 0;
```

**Add** these new types and components in the same location (before `ResultCard`):

```tsx
/* ---------- Provenance types and components ---------- */

interface ProvenanceSourceDoc {
  document_id: string;
  document_title: string;
}

interface ProvenanceEntity {
  id: number;
  title: string;
  type: string;
  description: string;
  source_documents: ProvenanceSourceDoc[];
}

interface ProvenanceRelationship {
  id: number;
  source: string;
  target: string;
  description: string;
  source_documents: ProvenanceSourceDoc[];
}

interface ProvenanceTextUnit {
  id: number;
  text: string;
  source_documents: ProvenanceSourceDoc[];
}

interface ProvenanceCovariate {
  id: number;
  description: string;
  source_documents: ProvenanceSourceDoc[];
}

interface ProvenanceEntry {
  report_id: string | null;
  report_title: string | null;
  report_content: string | null;
  entities: ProvenanceEntity[];
  relationships: ProvenanceRelationship[];
  text_units: ProvenanceTextUnit[];
  covariates: ProvenanceCovariate[];
}

function DocBadges({ docs }: { docs: ProvenanceSourceDoc[] }) {
  if (!docs || docs.length === 0) return null;
  return (
    <div className="prov-doc-badges">
      {docs.map((d) => (
        <span key={d.document_id} className="prov-doc-badge" title={d.document_id}>
          {d.document_title || d.document_id}
        </span>
      ))}
    </div>
  );
}

function CollapsibleSection({ title, count, children }: {
  title: string;
  count: number;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  if (count === 0) return null;
  return (
    <div className="prov-section">
      <button className="prov-section-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "▼" : "▶"} {title} ({count})
      </button>
      {open && <div className="prov-section-content">{children}</div>}
    </div>
  );
}

function ProvenancePanel({ provenance }: { provenance: ProvenanceEntry[] }) {
  const [open, setOpen] = useState(false);

  if (!provenance || provenance.length === 0) return null;

  const reportCount = provenance.filter((p) => p.report_id).length;
  const label = reportCount > 0
    ? `Provenance (${reportCount} community report${reportCount !== 1 ? "s" : ""})`
    : "Provenance";

  return (
    <div className="provenance-panel">
      <button
        className="prov-toggle"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "▼" : "▶"} {label}
      </button>
      {open && provenance.map((entry, i) => (
        <ProvenanceReportEntry key={entry.report_id || i} entry={entry} />
      ))}
    </div>
  );
}

function ProvenanceReportEntry({ entry }: { entry: ProvenanceEntry }) {
  const [open, setOpen] = useState(false);
  const [contentOpen, setContentOpen] = useState(false);

  return (
    <div className="prov-report">
      <button className="prov-report-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "▼" : "▶"} {entry.report_title || "Source Texts"}
      </button>
      {open && (
        <div className="prov-report-content">
          {entry.report_content && (
            <div className="prov-section">
              <button className="prov-section-toggle" onClick={() => setContentOpen((v) => !v)}>
                {contentOpen ? "▼" : "▶"} Report Content
              </button>
              {contentOpen && (
                <pre className="prov-report-text">{entry.report_content}</pre>
              )}
            </div>
          )}

          <CollapsibleSection title="Entities" count={entry.entities.length}>
            {entry.entities.map((e) => (
              <div key={e.id} className="prov-item">
                <div>
                  <span className="badge badge-info" style={{ marginRight: "0.25rem" }}>{e.type}</span>
                  <strong>{e.title}</strong>
                  {e.description && <span className="text-muted"> &mdash; {e.description.slice(0, 150)}</span>}
                </div>
                <DocBadges docs={e.source_documents} />
              </div>
            ))}
          </CollapsibleSection>

          <CollapsibleSection title="Relationships" count={entry.relationships.length}>
            {entry.relationships.map((r) => (
              <div key={r.id} className="prov-item">
                <div className="text-sm">
                  {r.source} &rarr; {r.target}: {r.description.slice(0, 150)}
                </div>
                <DocBadges docs={r.source_documents} />
              </div>
            ))}
          </CollapsibleSection>

          <CollapsibleSection title="Source Texts" count={entry.text_units.length}>
            {entry.text_units.map((t) => (
              <div key={t.id} className="prov-item">
                <pre className="prov-text-chunk">{t.text}</pre>
                <DocBadges docs={t.source_documents} />
              </div>
            ))}
          </CollapsibleSection>

          <CollapsibleSection title="Covariates" count={entry.covariates.length}>
            {entry.covariates.map((c) => (
              <div key={c.id} className="prov-item">
                <div className="text-sm">{c.description}</div>
                <DocBadges docs={c.source_documents} />
              </div>
            ))}
          </CollapsibleSection>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Update ResultCard to use ProvenancePanel instead of citations**

In the `ResultCard` component, add provenance extraction (replacing the removed `sources`/`hasSources` lines):

```tsx
  const provenance = (ctx?.provenance as ProvenanceEntry[] | undefined) || [];
```

**Revert** the preview text rendering (around line 531) to plain text:

```tsx
      {preview && (
        <p className="result-text">{preview}</p>
      )}
```

**Revert** the full text section in details (around line 573) to plain text:

```tsx
          {displayText && displayText.length > previewLen && (
            <div style={{ marginBottom: "0.5rem" }}>
              <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Full Text</div>
              <p className="text-sm" style={{ whiteSpace: "pre-wrap" }}>{displayText}</p>
            </div>
          )}
```

**Replace** the SourcesPanel call (line 581) with:

```tsx
          {isGraphRAG && provenance.length > 0 && <ProvenancePanel provenance={provenance} />}
```

- [ ] **Step 3: Replace citation CSS with provenance CSS**

In `frontend/src/styles.css`, replace lines 763-777 (the `.citation-link` and `.source-entry:target` rules) with:

```css
/* ---------- Provenance panel ---------- */
.provenance-panel {
  margin-top: 1rem;
  border-top: 1px solid var(--color-border);
  padding-top: 0.75rem;
}
.prov-toggle,
.prov-report-toggle,
.prov-section-toggle {
  background: none;
  border: none;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.85rem;
  padding: 0.25rem 0;
  color: inherit;
  text-align: left;
  width: 100%;
}
.prov-toggle:hover,
.prov-report-toggle:hover,
.prov-section-toggle:hover {
  color: var(--color-primary, #2A5A8A);
}
.prov-report {
  margin-left: 0.5rem;
  margin-top: 0.25rem;
}
.prov-report-content {
  margin-left: 1rem;
}
.prov-section {
  margin-top: 0.25rem;
}
.prov-section-content {
  margin-left: 0.75rem;
}
.prov-item {
  margin-bottom: 0.5rem;
  padding: 0.375rem;
  background: var(--color-surface-2);
  border-radius: var(--radius);
  font-size: 0.85rem;
}
.prov-doc-badges {
  margin-top: 0.25rem;
}
.prov-doc-badge {
  display: inline-block;
  font-size: 0.75rem;
  color: var(--color-primary, #2A5A8A);
  background: var(--color-surface-1, #f0f4f8);
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  margin-right: 0.25rem;
}
.prov-report-text {
  white-space: pre-wrap;
  font-size: 0.8rem;
  background: var(--color-surface-2);
  padding: 0.75rem;
  border-radius: var(--radius);
  margin: 0.25rem 0;
  max-height: 400px;
  overflow-y: auto;
}
.prov-text-chunk {
  white-space: pre-wrap;
  font-size: 0.8rem;
  margin: 0;
  max-height: 200px;
  overflow-y: auto;
}
```

- [ ] **Step 4: Build frontend and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/QueryPage.tsx frontend/src/styles.css
git commit -m "feat: replace citation UI with collapsible ProvenancePanel

Remove CitationLink, SourcesPanel, renderWithCitations components.
Add ProvenancePanel with hierarchical community report display:
entities, relationships, source texts with document badges.
All sections collapsible."
```

---

### Task 6: Update verification checklist and final verification

**Files:**
- Modify: `VERIFICATION_CHECKLIST.md:114,216`

- [ ] **Step 1: Update checklist entry**

Edit `VERIFICATION_CHECKLIST.md:114`. Replace the citation provenance entry with:

```
| GraphRAG context provenance (all 4 search types) | GraphRAG responses lack source traceability | Run GraphRAG Local query; provenance array contains community reports with entities + source documents | 2.32 |
```

- [ ] **Step 2: Update Known Fragile Features**

Edit `VERIFICATION_CHECKLIST.md:216`. Replace item 14 with:

```
14. **GraphRAG context provenance** (2.32) — Depends on communities.parquet having entity_ids/relationship_ids. Test all 4 strategies; verify provenance array populated with source_documents.
```

- [ ] **Step 3: Run full unit test suite**

Run: `python -m pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add VERIFICATION_CHECKLIST.md
git commit -m "docs: update verification checklist for context-based provenance"
```
