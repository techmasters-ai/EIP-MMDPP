# GraphRAG Citation Provenance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inline `[n]` citations to all four GraphRAG search responses that trace facts back through entities/relationships to source document text, with clickable citation links in the UI.

**Architecture:** Modify 4 search prompts to instruct the LLM to produce `[n]` citations + a `## Sources` block. Post-processing parses the sources block, resolves entity/relationship IDs to full provenance from Parquet data, and returns a uniform `sources` array. Frontend renders citations as clickable links to a sources panel.

**Tech Stack:** Python 3.11, pandas (Parquet), React 18, react-markdown

**Spec:** `docs/superpowers/specs/2026-04-02-graphrag-citation-provenance-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `app/services/graphrag_citations.py` | Citation parser + resolver (new module) |
| Modify | `app/services/graphrag_prompts.py:91-236` | Add citation instructions to 4 search prompts (not map) |
| Modify | `app/services/graphrag_service.py:377-510` | Load `documents.parquet`, wire citation resolver into all 4 search fns |
| Modify | `app/workers/graphrag_tasks.py:182-193` | Pass `sources` through to API response |
| Modify | `frontend/src/components/QueryPage.tsx:290-486` | Render clickable citations + sources panel |
| Create | `tests/unit/test_graphrag_citations.py` | Unit tests for parser + resolver |

---

## Chunk 1: Citation resolver and prompt changes

### Task 1: Create citation parser and resolver module

**Files:**
- Create: `app/services/graphrag_citations.py`
- Create: `tests/unit/test_graphrag_citations.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_graphrag_citations.py`:

```python
"""Unit tests for GraphRAG citation parsing and resolution."""
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class TestStripSourcesBlock:
    def test_extracts_and_strips_sources(self):
        from app.services.graphrag_citations import strip_sources_block

        text = (
            "The SA-2 uses command guidance [1].\n\n"
            "## Sources\n"
            "[1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276\n"
            "[2] Entity: SNR-75 FAN SONG (1494)\n"
        )
        clean, block = strip_sources_block(text)
        assert "## Sources" not in clean
        assert "[1]" in clean  # citation markers preserved
        assert "3349" in block
        assert "1494" in block

    def test_no_sources_block(self):
        from app.services.graphrag_citations import strip_sources_block

        text = "The SA-2 uses command guidance."
        clean, block = strip_sources_block(text)
        assert clean == text
        assert block == ""

    def test_strips_think_tags(self):
        from app.services.graphrag_citations import strip_sources_block

        text = "<think>reasoning</think>The SA-2 [1].\n\n## Sources\n[1] Entity: X (1)\n"
        clean, block = strip_sources_block(text)
        assert "<think>" not in clean
        assert "The SA-2 [1]." in clean


class TestParseCitationBlock:
    def test_parses_id_based_citations(self):
        from app.services.graphrag_citations import parse_citation_block

        block = (
            "[1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276\n"
            "[2] Entity: SNR-75 FAN SONG (1494), Relationship: 3494\n"
        )
        citations = parse_citation_block(block, "local")
        assert len(citations) == 2
        assert citations[1]["entity_ids"] == [3349]
        assert citations[1]["relationship_ids"] == [4276]
        assert citations[2]["entity_ids"] == [1494]

    def test_parses_name_based_citations(self):
        from app.services.graphrag_citations import parse_citation_block

        block = "[1] Entity: SA-2 GUIDELINE\n[2] Entity: SNR-75 FAN SONG\n"
        citations = parse_citation_block(block, "global")
        assert citations[1]["entity_names"] == ["SA-2 GUIDELINE"]
        assert citations[2]["entity_names"] == ["SNR-75 FAN SONG"]

    def test_parses_text_based_citations(self):
        from app.services.graphrag_citations import parse_citation_block

        block = '[1] Source: "The SA-2 Guideline surface-to-air missile..."\n'
        citations = parse_citation_block(block, "basic")
        assert "SA-2 Guideline" in citations[1]["source_text"]

    def test_skips_malformed_lines(self):
        from app.services.graphrag_citations import parse_citation_block

        block = "[1] Entity: GOOD (100)\ngarbage line\n[2] Entity: ALSO GOOD (200)\n"
        citations = parse_citation_block(block, "local")
        assert len(citations) == 2

    def test_duplicate_numbers_keeps_first(self):
        from app.services.graphrag_citations import parse_citation_block

        block = "[1] Entity: FIRST (100)\n[1] Entity: SECOND (200)\n"
        citations = parse_citation_block(block, "local")
        assert citations[1]["entity_ids"] == [100]


class TestResolveCitations:
    @pytest.fixture
    def sample_data(self):
        entities = pd.DataFrame({
            "id": ["uuid-1", "uuid-2"],
            "human_readable_id": [3349, 1494],
            "title": ["SA-2 GUIDELINE", "SNR-75 FAN SONG"],
            "type": ["MISSILE_SYSTEM", "FIRE_CONTROL_SYSTEM"],
            "description": ["Soviet SAM system", "Fire control radar"],
            "text_unit_ids": [["tu-1"], ["tu-1", "tu-2"]],
        })
        relationships = pd.DataFrame({
            "id": ["rel-uuid-1"],
            "human_readable_id": [4276],
            "source": ["S-75 DVINA"],
            "target": ["V-750"],
            "description": ["Interceptor component"],
            "text_unit_ids": [["tu-1"]],
        })
        text_units = pd.DataFrame({
            "id": ["tu-1", "tu-2"],
            "human_readable_id": [0, 1],
            "text": ["The SA-2 Guideline uses command guidance...", "Fan Song radar operates..."],
            "document_ids": [["doc-1"], ["doc-2"]],
        })
        documents = pd.DataFrame({
            "id": ["doc-1", "doc-2"],
            "title": ["Red SAM_a3b2c1d4", "Fan Song radars_e5f6g7h8"],
        })
        return {
            "entities": entities,
            "relationships": relationships,
            "text_units": text_units,
            "documents": documents,
        }

    def test_resolves_id_based(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        parsed = {
            1: {"entity_ids": [3349], "relationship_ids": [4276]},
        }
        sources = resolve_citations(parsed, sample_data, "local")
        assert len(sources) == 1
        assert sources[0]["citation"] == 1
        assert sources[0]["entities"][0]["title"] == "SA-2 GUIDELINE"
        assert sources[0]["relationships"][0]["source"] == "S-75 DVINA"
        assert sources[0]["source_documents"][0]["document_title"] == "Red SAM"

    def test_resolves_name_based(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        parsed = {1: {"entity_names": ["SA-2 GUIDELINE"]}}
        sources = resolve_citations(parsed, sample_data, "global")
        assert sources[0]["entities"][0]["id"] == 3349

    def test_missing_id_skipped(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        parsed = {1: {"entity_ids": [9999], "relationship_ids": []}}
        sources = resolve_citations(parsed, sample_data, "local")
        assert sources[0]["entities"] == []

    def test_null_document_id(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        sample_data["text_units"].at[0, "document_ids"] = None
        parsed = {1: {"entity_ids": [3349], "relationship_ids": []}}
        sources = resolve_citations(parsed, sample_data, "local")
        # Should not crash; source_documents may be empty
        assert len(sources) == 1

    def test_strips_document_title_hash(self, sample_data):
        from app.services.graphrag_citations import resolve_citations

        parsed = {1: {"entity_ids": [3349], "relationship_ids": []}}
        sources = resolve_citations(parsed, sample_data, "local")
        title = sources[0]["source_documents"][0]["document_title"]
        assert title == "Red SAM"  # hash suffix stripped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_graphrag_citations.py -v`
Expected: All FAIL with `ModuleNotFoundError: No module named 'app.services.graphrag_citations'`

- [ ] **Step 3: Implement the citation module**

Create `app/services/graphrag_citations.py`:

```python
"""Parse and resolve inline citations from GraphRAG LLM responses.

Supports three citation formats depending on search strategy:
- ID-based (Local/Drift): [n] Entity: NAME (human_readable_id), Relationship: id
- Name-based (Global): [n] Entity: NAME
- Text-based (Basic): [n] Source: "text excerpt..."
"""

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_RE_THINK_TAGS = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_RE_SOURCES_BLOCK = re.compile(r"\n##\s*Sources\s*\n(.*)", re.DOTALL)
_RE_CITATION_NUM = re.compile(r"^\[(\d+)\]\s*(.*)", re.MULTILINE)

# ID-based: [1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276
_RE_ENTITY_ID = re.compile(r"Entity:\s*(.+?)\s*\((\d+)\)")
_RE_REL_ID = re.compile(r"Relationship:\s*(\d+)")

# Name-based: [1] Entity: SA-2 GUIDELINE
_RE_ENTITY_NAME = re.compile(r"Entity:\s*(.+?)(?:,|$)")

# Text-based: [1] Source: "text excerpt..."
_RE_SOURCE_TEXT = re.compile(r'Source:\s*"(.+?)"')

# Document title hash suffix: "Title_a3b2c1d4" -> "Title"
_RE_TITLE_HASH = re.compile(r"_[0-9a-f]{8}$")


def strip_sources_block(response_text: str) -> tuple[str, str]:
    """Strip the ## Sources block and <think> tags from response text.

    Returns (clean_text, sources_block_text).
    """
    text = _RE_THINK_TAGS.sub("", response_text).strip()
    match = _RE_SOURCES_BLOCK.search(text)
    if not match:
        return text, ""
    clean = text[: match.start()].rstrip()
    block = match.group(1).strip()
    return clean, block


def parse_citation_block(
    block: str, strategy: str,
) -> dict[int, dict[str, Any]]:
    """Parse the sources block into a dict keyed by citation number.

    Strategy determines the parsing format:
    - local/drift: ID-based (entity IDs + relationship IDs)
    - global: name-based (entity names)
    - basic: text-based (source text excerpts)
    """
    citations: dict[int, dict[str, Any]] = {}

    for match in _RE_CITATION_NUM.finditer(block):
        num = int(match.group(1))
        if num in citations:
            continue  # keep first occurrence
        line = match.group(2)

        if strategy in ("graphrag_local", "graphrag_drift", "local", "drift"):
            entity_ids = [int(m.group(2)) for m in _RE_ENTITY_ID.finditer(line)]
            rel_ids = [int(m.group(1)) for m in _RE_REL_ID.finditer(line)]
            citations[num] = {"entity_ids": entity_ids, "relationship_ids": rel_ids}

        elif strategy in ("graphrag_global", "global"):
            names = [m.group(1).strip() for m in _RE_ENTITY_NAME.finditer(line)]
            citations[num] = {"entity_names": names}

        elif strategy in ("graphrag_basic", "basic"):
            text_match = _RE_SOURCE_TEXT.search(line)
            citations[num] = {
                "source_text": text_match.group(1) if text_match else line,
            }
        else:
            logger.debug("Unknown strategy %s for citation line: %s", strategy, line)

    return citations


def _clean_doc_title(title: str) -> str:
    """Strip the content hash suffix from bridge-layer document titles."""
    return _RE_TITLE_HASH.sub("", title)


def _resolve_text_unit_docs(
    text_unit_ids: list[str],
    data: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Resolve text_unit_ids to source documents."""
    tu_df = data.get("text_units", pd.DataFrame())
    doc_df = data.get("documents", pd.DataFrame())
    if tu_df.empty:
        return []

    docs: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    for tu_id in text_unit_ids:
        rows = tu_df[tu_df["id"] == tu_id]
        if rows.empty:
            continue
        row = rows.iloc[0]
        # document_ids is a list column in the Parquet schema
        doc_ids_raw = row.get("document_ids")
        if doc_ids_raw is None or (hasattr(doc_ids_raw, "__len__") and len(doc_ids_raw) == 0):
            continue
        doc_id_list = doc_ids_raw if hasattr(doc_ids_raw, "__iter__") and not isinstance(doc_ids_raw, str) else [doc_ids_raw]

        source_text = str(row.get("text", ""))
        if len(source_text) > 500:
            source_text = source_text[:500] + "..."

        for doc_id in doc_id_list:
            if not doc_id or (isinstance(doc_id, float) and pd.isna(doc_id)):
                continue
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            doc_title = ""
            if not doc_df.empty:
                doc_rows = doc_df[doc_df["id"] == doc_id]
                if not doc_rows.empty:
                    doc_title = _clean_doc_title(str(doc_rows.iloc[0].get("title", "")))

            docs.append({
                "document_id": doc_id,
                "document_title": doc_title,
                "source_text": source_text,
            })

    return docs


def resolve_citations(
    parsed: dict[int, dict[str, Any]],
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> list[dict[str, Any]]:
    """Resolve parsed citations to full provenance data.

    Returns a list of source entries, one per citation number.
    """
    ent_df = data.get("entities", pd.DataFrame())
    rel_df = data.get("relationships", pd.DataFrame())
    sources: list[dict[str, Any]] = []

    for num in sorted(parsed.keys()):
        citation = parsed[num]
        entry: dict[str, Any] = {
            "citation": num,
            "entities": [],
            "relationships": [],
            "source_documents": [],
        }
        all_tu_ids: list[str] = []

        # Resolve entities
        if "entity_ids" in citation:
            for eid in citation["entity_ids"]:
                rows = ent_df[ent_df["human_readable_id"] == eid] if not ent_df.empty else pd.DataFrame()
                if rows.empty:
                    logger.warning("Citation [%d]: entity ID %d not found", num, eid)
                    continue
                row = rows.iloc[0]
                entry["entities"].append({
                    "id": int(eid),
                    "title": str(row.get("title", "")),
                    "type": str(row.get("type", "")),
                    "description": str(row.get("description", "")),
                })
                tu_ids = row.get("text_unit_ids")
                if tu_ids is not None:
                    all_tu_ids.extend(tu_ids if hasattr(tu_ids, "__iter__") and not isinstance(tu_ids, str) else [])

        elif "entity_names" in citation:
            for name in citation["entity_names"]:
                if ent_df.empty:
                    continue
                rows = ent_df[ent_df["title"].str.upper() == name.upper()]
                if rows.empty:
                    logger.warning("Citation [%d]: entity name '%s' not found", num, name)
                    continue
                row = rows.iloc[0]
                entry["entities"].append({
                    "id": int(row.get("human_readable_id", 0)),
                    "title": str(row.get("title", "")),
                    "type": str(row.get("type", "")),
                    "description": str(row.get("description", "")),
                })
                tu_ids = row.get("text_unit_ids")
                if tu_ids is not None:
                    all_tu_ids.extend(tu_ids if hasattr(tu_ids, "__iter__") and not isinstance(tu_ids, str) else [])

        elif "source_text" in citation:
            # Basic search: match text units by substring
            tu_df = data.get("text_units", pd.DataFrame())
            if not tu_df.empty:
                excerpt = citation["source_text"][:100]
                matches = tu_df[tu_df["text"].str.contains(excerpt, case=False, na=False)]
                if not matches.empty:
                    all_tu_ids.extend(matches["id"].tolist())
                else:
                    logger.warning("Citation [%d]: text excerpt not matched", num)

        # Resolve relationships
        if "relationship_ids" in citation:
            for rid in citation["relationship_ids"]:
                rows = rel_df[rel_df["human_readable_id"] == rid] if not rel_df.empty else pd.DataFrame()
                if rows.empty:
                    logger.warning("Citation [%d]: relationship ID %d not found", num, rid)
                    continue
                row = rows.iloc[0]
                entry["relationships"].append({
                    "id": int(rid),
                    "source": str(row.get("source", "")),
                    "target": str(row.get("target", "")),
                    "description": str(row.get("description", "")),
                })
                tu_ids = row.get("text_unit_ids")
                if tu_ids is not None:
                    all_tu_ids.extend(tu_ids if hasattr(tu_ids, "__iter__") and not isinstance(tu_ids, str) else [])

        # Resolve source documents from collected text_unit_ids
        entry["source_documents"] = _resolve_text_unit_docs(all_tu_ids, data)

        sources.append(entry)

    return sources


def process_citations(
    response_text: str,
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Top-level function: strip, parse, resolve citations.

    Returns (clean_response_text, sources_array).
    """
    clean, block = strip_sources_block(response_text)
    if not block:
        return clean, []
    parsed = parse_citation_block(block, strategy)
    if not parsed:
        return clean, []
    sources = resolve_citations(parsed, data, strategy)
    return clean, sources
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_graphrag_citations.py -v`
Expected: All PASS

- [ ] **Step 5: Run full unit test suite**

Run: `python -m pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/graphrag_citations.py tests/unit/test_graphrag_citations.py
git commit -m "feat: add GraphRAG citation parser and resolver module

Supports three citation formats: ID-based (Local/Drift), name-based
(Global), text-based (Basic). Resolves through entities/relationships
to source documents via Parquet data."
```

---

### Task 2: Add citation instructions to search prompts

**Files:**
- Modify: `app/services/graphrag_prompts.py:91-236`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_graphrag_citations.py`:

```python
class TestPromptCitationInstructions:
    def test_local_prompt_has_id_citation_instruction(self):
        from app.services.graphrag_prompts import get_local_search_prompt
        prompt = get_local_search_prompt()
        assert "[n]" in prompt or "inline citation" in prompt.lower()
        assert "## Sources" in prompt
        assert "Entity:" in prompt and "Relationship:" in prompt

    def test_global_reduce_prompt_has_name_citation_instruction(self):
        from app.services.graphrag_prompts import get_global_search_reduce_prompt
        prompt = get_global_search_reduce_prompt()
        assert "[n]" in prompt or "inline citation" in prompt.lower()
        assert "## Sources" in prompt
        assert "Entity:" in prompt

    def test_global_map_prompt_has_no_citation_instruction(self):
        from app.services.graphrag_prompts import get_global_search_map_prompt
        prompt = get_global_search_map_prompt()
        assert "## Sources" not in prompt

    def test_drift_prompt_has_id_citation_instruction(self):
        from app.services.graphrag_prompts import get_drift_search_prompt
        prompt = get_drift_search_prompt()
        assert "## Sources" in prompt

    def test_basic_prompt_has_text_citation_instruction(self):
        from app.services.graphrag_prompts import get_basic_search_prompt
        prompt = get_basic_search_prompt()
        assert "## Sources" in prompt
        assert "Source:" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_graphrag_citations.py::TestPromptCitationInstructions -v`
Expected: All FAIL (current prompts have no citation instructions)

- [ ] **Step 3: Add citation instructions to each prompt**

Edit `app/services/graphrag_prompts.py`.

**Local search** (`get_local_search_prompt`, line 91) — add before the closing `"""`:

```
IMPORTANT — Inline Citations:
Insert [n] citation markers in your response for every factual claim.
After your response, append a "## Sources" section listing each citation.
Format: [n] Entity: ENTITY_NAME (ID), Relationship: REL_ID
Use the numeric IDs from the entity and relationship tables provided above.
Example:
The SA-2 uses command guidance [1] with Fan Song tracking [2].

## Sources
[1] Entity: SA-2 GUIDELINE (3349), Relationship: 4276
[2] Entity: SNR-75 FAN SONG (1494), Relationship: 3494
```

**Global reduce** (`get_global_search_reduce_prompt`, line 154) — add before the closing `"""`:

```
IMPORTANT — Inline Citations:
Insert [n] citation markers in your response for every factual claim.
After your response, append a "## Sources" section listing each citation.
Format: [n] Entity: ENTITY_NAME
Cite the entity names referenced in the analyst reports.
Example:
The SA-2 system provides medium-altitude defense [1].

## Sources
[1] Entity: SA-2 GUIDELINE
```

**Global map** (`get_global_search_map_prompt`, line 128) — NO CHANGES. Map extracts key points; citations belong in the reduce step.

**Drift search** (`get_drift_search_prompt`, line 182) — add before the closing `"""`:

```
IMPORTANT — Inline Citations:
Insert [n] citation markers in your response for every factual claim.
After your response, append a "## Sources" section listing each citation.
Format: [n] Entity: ENTITY_NAME (ID), Relationship: REL_ID
Use the numeric IDs from the entity and relationship data provided above.
```

**Basic search** (`get_basic_search_prompt`, line 211) — add before the closing `"""`:

```
IMPORTANT — Inline Citations:
Insert [n] citation markers in your response for every factual claim.
After your response, append a "## Sources" section listing each citation.
Format: [n] Source: "first 50 characters of the source text excerpt..."
Reference the text excerpts provided above.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_graphrag_citations.py -v`
Expected: All PASS

- [ ] **Step 5: Regenerate prompt files on disk**

The prompts are written to disk by `write_prompt_files()`. They will be regenerated on next container restart when `graphrag_service.py` initializes. No manual step needed, but note that containers must be rebuilt.

- [ ] **Step 6: Commit**

```bash
git add app/services/graphrag_prompts.py tests/unit/test_graphrag_citations.py
git commit -m "feat: add citation instructions to 4 GraphRAG search prompts

Local/Drift: ID-based citations using entity/relationship short_ids.
Global reduce: name-based citations using entity titles.
Basic: text-based citations using source text excerpts.
Global map: no citations (map extracts key points, reduce cites)."
```

---

## Chunk 2: Wire citations into service and API

### Task 3: Load `documents.parquet` and wire citation resolver

**Files:**
- Modify: `app/services/graphrag_service.py:377-510`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_graphrag_citations.py`:

```python
class TestLoadSearchDataIncludesDocuments:
    def test_documents_key_exists(self):
        """_load_search_data must include 'documents' in its output."""
        # We can't easily call _load_search_data without a real filesystem,
        # so test the code path by checking the source.
        import inspect
        from app.services.graphrag_service import _load_search_data
        source = inspect.getsource(_load_search_data)
        assert '"documents"' in source or "'documents'" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_graphrag_citations.py::TestLoadSearchDataIncludesDocuments -v`
Expected: FAIL (current code doesn't load documents)

- [ ] **Step 3: Add `documents` to `_load_search_data()`**

Edit `app/services/graphrag_service.py:382-385`. Change the tuple to include `"documents"`:

```python
    for name in (
        "entities", "communities", "community_reports",
        "text_units", "relationships", "covariates",
        "documents",
    ):
```

- [ ] **Step 4: Wire `process_citations` into all four search functions**

Edit `app/services/graphrag_service.py`. For each of the four search functions (`local_search`, `global_search`, `drift_search`, `basic_search`), change the return to run through the citation resolver.

For example, `local_search` (line 476):

Before:
```python
        return {"response": response, "context": _serialize_context(context)}
```

After:
```python
        from app.services.graphrag_citations import process_citations
        clean_response, sources = process_citations(response, data, "graphrag_local")
        return {
            "response": clean_response,
            "sources": sources,
            "context": _serialize_context(context),
        }
```

Apply the same pattern to `global_search` (line 507), `drift_search`, and `basic_search`. Each passes its own strategy string. Note: `data` is already loaded in each function via `_load_search_data(settings)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_graphrag_citations.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/graphrag_service.py tests/unit/test_graphrag_citations.py
git commit -m "feat: wire citation resolver into all 4 GraphRAG search functions

Load documents.parquet in _load_search_data(). Each search function
now passes the LLM response through process_citations() to extract
and resolve inline citations before returning."
```

---

### Task 4: Pass `sources` through to API response

**Files:**
- Modify: `app/workers/graphrag_tasks.py:182-193`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_graphrag_citations.py`:

```python
class TestTaskSerializesSources:
    def test_sources_in_result_context(self):
        """The Celery task must include sources in context dict."""
        # Verify the code path by checking source
        import inspect
        from app.workers.graphrag_tasks import run_graphrag_query_task
        source = inspect.getsource(run_graphrag_query_task)
        assert "sources" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_graphrag_citations.py::TestTaskSerializesSources -v`
Expected: FAIL

- [ ] **Step 3: Update task serialization**

Edit `app/workers/graphrag_tasks.py:182-193`. Add `sources` to the context dict:

Before:
```python
    result_item = {
        "score": 1.0,
        "modality": "graphrag_response",
        "content_text": response,
        "classification": "UNCLASSIFIED",
        "context": {
            "source": strategy,
            "graphrag_context": graphrag_result.get("context", {}),
        },
    }
```

After:
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

Note: `graphrag_result["response"]` is now the clean text (sources block stripped). The raw `response` variable is the original before citation processing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_graphrag_citations.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add app/workers/graphrag_tasks.py tests/unit/test_graphrag_citations.py
git commit -m "feat: pass citation sources through to API response

The sources array from the citation resolver is included in the
context dict alongside graphrag_context for all 4 search types."
```

---

## Chunk 3: Frontend rendering

### Task 5: Render clickable citations and sources panel

**Files:**
- Modify: `frontend/src/components/QueryPage.tsx:290-486`

- [ ] **Step 1: Add citation rendering for GraphRAG responses**

In `QueryPage.tsx`, add a new component `CitationLink` and `SourcesPanel`:

```tsx
/** Clickable citation marker that scrolls to the matching source entry. */
function CitationLink({ num }: { num: number }) {
  return (
    <a
      href={`#citation-${num}`}
      className="citation-link"
      title={`Source [${num}]`}
      onClick={(e) => {
        e.preventDefault();
        document.getElementById(`citation-${num}`)?.scrollIntoView({ behavior: "smooth" });
      }}
    >
      [{num}]
    </a>
  );
}

interface SourceEntry {
  citation: number;
  entities: Array<{ id: number; title: string; type: string; description: string }>;
  relationships: Array<{ id: number; source: string; target: string; description: string }>;
  source_documents: Array<{ document_id: string; document_title: string; source_text: string }>;
}

function SourcesPanel({ sources }: { sources: SourceEntry[] }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources-panel" style={{
      marginTop: "1rem",
      borderTop: "1px solid var(--color-border)",
      paddingTop: "0.75rem",
    }}>
      <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Sources</div>
      {sources.map((s) => (
        <div key={s.citation} id={`citation-${s.citation}`} className="source-entry" style={{
          marginBottom: "0.75rem",
          padding: "0.5rem",
          background: "var(--color-surface-2)",
          borderRadius: "var(--radius)",
          fontSize: "0.85rem",
        }}>
          <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>[{s.citation}]</div>
          {s.entities.map((e) => (
            <div key={e.id} style={{ marginBottom: "0.25rem" }}>
              <span className="badge badge-info" style={{ marginRight: "0.25rem" }}>{e.type}</span>
              <strong>{e.title}</strong>
              {e.description && <span className="text-muted"> — {e.description.slice(0, 150)}</span>}
            </div>
          ))}
          {s.relationships.map((r) => (
            <div key={r.id} className="text-sm text-muted" style={{ marginBottom: "0.25rem" }}>
              {r.source} → {r.target}: {r.description.slice(0, 150)}
            </div>
          ))}
          {s.source_documents.map((d) => (
            <div key={d.document_id} className="text-xs text-muted" style={{ marginTop: "0.25rem" }}>
              📄 {d.document_title}
              {d.source_text && (
                <pre style={{ margin: "0.25rem 0 0", whiteSpace: "pre-wrap", fontSize: "0.75rem" }}>
                  {d.source_text.slice(0, 300)}{d.source_text.length > 300 ? "..." : ""}
                </pre>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Add helper to replace `[n]` markers with React elements**

```tsx
/** Replace [n] citation markers in text with clickable CitationLink elements. */
function renderWithCitations(text: string, hasSources: boolean): React.ReactNode {
  if (!hasSources) return text;
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      return <CitationLink key={i} num={parseInt(match[1], 10)} />;
    }
    return part;
  });
}
```

- [ ] **Step 3: Update ResultCard to use citation rendering for GraphRAG responses**

In the `ResultCard` component, extract the sources from context and use the citation renderer:

```tsx
  const sources = (ctx?.sources as SourceEntry[] | undefined) || [];
  const hasSources = sources.length > 0;
```

Replace the preview text rendering (around line 375) for GraphRAG results:

```tsx
  {preview && (
    <p className="text-sm" style={{ whiteSpace: "pre-wrap" }}>
      {isGraphRAG && hasSources ? renderWithCitations(preview, true) : preview}
    </p>
  )}
```

And in the full text section (around line 478):

```tsx
  {displayText && displayText.length > previewLen && (
    <div style={{ marginBottom: "0.5rem" }}>
      <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Full Text</div>
      <p className="text-sm" style={{ whiteSpace: "pre-wrap" }}>
        {isGraphRAG && hasSources ? renderWithCitations(displayText, true) : displayText}
      </p>
    </div>
  )}
```

After the full text, render the sources panel:

```tsx
  {isGraphRAG && hasSources && <SourcesPanel sources={sources} />}
```

- [ ] **Step 4: Add minimal CSS for citation links**

Add to `frontend/src/styles.css`:

```css
.citation-link {
  color: var(--color-primary, #2A5A8A);
  font-weight: 600;
  font-size: 0.8em;
  text-decoration: none;
  vertical-align: super;
  cursor: pointer;
}
.citation-link:hover {
  text-decoration: underline;
}
.source-entry:target {
  outline: 2px solid var(--color-primary, #2A5A8A);
  outline-offset: 2px;
}
```

- [ ] **Step 5: Build frontend and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/QueryPage.tsx frontend/src/styles.css
git commit -m "feat: render clickable GraphRAG citations with sources panel

Citation markers [n] in GraphRAG responses become clickable links
that scroll to the corresponding source entry. Sources panel shows
entity details, relationships, and source document excerpts."
```

---

### Task 6: Rebuild, verify end-to-end, update checklist

**Files:**
- No code changes — operational verification
- Modify: `VERIFICATION_CHECKLIST.md`

- [ ] **Step 1: Rebuild containers**

```bash
docker compose build api
docker compose up -d
```

- [ ] **Step 2: Test Local search citations via API**

```bash
curl -s -X POST http://localhost:8005/v1/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "SA-2 guidance system", "strategy": "graphrag_local"}' | \
  python3 -c "
import json, sys
d = json.load(sys.stdin)
for r in d.get('results', []):
    ctx = r.get('context', {})
    sources = ctx.get('sources', [])
    print(f'Citations: {len(sources)}')
    for s in sources[:3]:
        print(f'  [{s[\"citation\"]}] entities={[e[\"title\"] for e in s[\"entities\"]]}')
"
```

Expected: At least some citations with resolved entities

- [ ] **Step 3: Test Global search citations via async flow**

Submit a global query and check the result for sources.

- [ ] **Step 4: Test in the browser**

1. Open UI → Query page
2. Select GraphRAG Local, search for "SA-2 guidance"
3. Verify: `[n]` markers visible as superscript links
4. Click a citation → scrolls to sources panel entry
5. Sources panel shows entity names, types, and document excerpts

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/unit/ -v
```

Expected: All PASS

- [ ] **Step 6: Update VERIFICATION_CHECKLIST.md**

Add to section 3 (Retrieval & Search):

```
| GraphRAG citation provenance (all 4 search types) | Facts in GraphRAG responses have no source attribution | Run GraphRAG Local query; response contains [n] citations; sources array has entities + documents | 2.32 |
```

Add to Known Fragile Features:

```
14. **GraphRAG citation parsing** (2.32) — LLM must produce ## Sources block. Test all 4 strategies; verify sources array populated.
```

- [ ] **Step 7: Commit**

```bash
git add VERIFICATION_CHECKLIST.md
git commit -m "docs: add GraphRAG citation provenance to verification checklist"
```
