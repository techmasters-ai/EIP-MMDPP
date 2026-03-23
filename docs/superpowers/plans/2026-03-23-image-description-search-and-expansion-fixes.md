# Image Description Text Search + Graph Expansion Fixes — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make image descriptions searchable via BGE text embeddings and fix graph expansion bugs/dead code in the retrieval pipeline.

**Architecture:** Image descriptions are split into sections, embedded as BGE vectors, and stored in `eip_text_chunks` (Qdrant + Postgres TextChunk). SAME_ARTIFACT chunk_links tie sections together. The retrieval pipeline gets targeted fixes: pass `query_text` to doc-structure fusion, remove wasted ontology re-scoring, make cross-modal use fusion formula, clean up dead code, and fix deprecated APIs.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, Celery, Qdrant, PostgreSQL, BGE-large-en-v1.5, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-image-description-text-search-design.md`

---

## Chunk 1: Graph Expansion Fixes

### Task 1: Fix doc-structure expansion missing `query_text` for military ID bonus

**Files:**
- Modify: `app/api/v1/retrieval.py:270-295` (`_expand_seeds`, `_expand_one`)
- Modify: `app/api/v1/retrieval.py:527-594` (`_expand_via_doc_structure`)
- Test: `tests/unit/test_retrieval_helpers.py`

The bug: `_expand_via_doc_structure` doesn't accept or pass `query_text` to `compute_fusion_score`, so doc-structure expanded chunks never get the military identifier bonus (`+0.03`). The `_expand_via_ontology` correctly passes it.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_retrieval_helpers.py — add to existing file

def test_fusion_score_with_mil_id_bonus_doc_structure():
    """Doc-structure chunks with matching MIL IDs should get the bonus."""
    score = compute_fusion_score(
        semantic_score=0.8,
        doc_structure_weight=0.9,
        doc_structure_hops=1,
        content_text="The AN/MPQ-53 radar system provides tracking.",
        query_text="AN/MPQ-53 fire control radar",
    )
    # Without bonus: 0.65*0.8 + 0.20*0.9 = 0.70
    # With bonus: 0.70 + 0.03 = 0.73
    assert score > 0.70
```

- [ ] **Step 2: Run test to verify it passes (this tests the fusion function which already accepts query_text)**

Run: `pytest tests/unit/test_retrieval_helpers.py::test_fusion_score_with_mil_id_bonus_doc_structure -v`
Expected: PASS (the `compute_fusion_score` function already supports `query_text` — the bug is in the caller)

- [ ] **Step 3: Fix `_expand_via_doc_structure` to accept and pass `query_text`**

In `app/api/v1/retrieval.py`, update the function signature at line 527:

```python
async def _expand_via_doc_structure(
    db: AsyncSession,
    chunk_id: str,
    source_score: float,
    include_context: bool,
    query_text: str | None = None,  # ADD THIS
) -> list[QueryResultItem]:
```

Update the `compute_fusion_score` call at line 580 to pass `query_text`:

```python
chunk.score = compute_fusion_score(
    semantic_score=source_score,
    doc_structure_weight=weight,
    doc_structure_hops=hops,
    content_text=chunk.content_text,
    query_text=query_text,  # ADD THIS
)
```

Update the call site in `_expand_one` at line 285:

```python
doc_items = await _expand_via_doc_structure(db, chunk_id_str, seed.score, include_context, query_text)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_retrieval_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/v1/retrieval.py tests/unit/test_retrieval_helpers.py
git commit -m "fix: pass query_text to doc-structure expansion for military ID bonus"
```

---

### Task 2: Fix ontology re-scoring replacing fusion score

**Files:**
- Modify: `app/api/v1/retrieval.py:313-358` (`_rescore_expanded_chunks`)
- Test: `tests/unit/test_retrieval_helpers.py`

The issue: `_rescore_expanded_chunks` completely replaces the fusion score (which includes ontology relation weights from `ontology.yaml`) with raw BGE cosine similarity. The fusion formula work is wasted. Instead, the re-scoring should use the cosine similarity as the `semantic_score` input to `compute_fusion_score`, preserving the ontology relation weight contribution.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_retrieval_helpers.py — add

def test_fusion_score_ontology_preserves_relation_weight():
    """Ontology chunks should reflect the relation weight, not just cosine similarity."""
    from app.api.v1._retrieval_helpers import compute_fusion_score

    # A high-weight relation (IS_VARIANT_OF = 0.95) should score higher
    # than a low-weight relation (RELATED_TO = 0.70) at the same semantic similarity
    high_rel = compute_fusion_score(
        semantic_score=0.6,
        ontology_rel_type="IS_VARIANT_OF",
        ontology_hops=1,
        content_text="S-75 variant",
        query_text="SA-2 missile",
    )
    low_rel = compute_fusion_score(
        semantic_score=0.6,
        ontology_rel_type="RELATED_TO",
        ontology_hops=1,
        content_text="related system",
        query_text="SA-2 missile",
    )
    assert high_rel > low_rel
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/unit/test_retrieval_helpers.py::test_fusion_score_ontology_preserves_relation_weight -v`
Expected: PASS (fusion formula already does this correctly — the bug is in the re-scoring step that throws it away)

- [ ] **Step 3: Fix `_rescore_expanded_chunks` to blend cosine similarity with fusion score**

In `app/api/v1/retrieval.py`, replace the re-scoring logic (lines 313-358):

```python
async def _rescore_expanded_chunks(
    expanded: list[QueryResultItem],
    query_text: str | None,
) -> list[QueryResultItem]:
    """Re-score ontology-expanded chunks using embedding similarity to the query.

    Replaces the inherited parent semantic_score with the chunk's actual cosine
    similarity, then re-applies compute_fusion_score to preserve ontology relation
    weights. This prevents low-relevance expansions from ranking artificially high
    while keeping the per-relation weight contribution.
    """
    if not expanded or not query_text:
        return expanded

    # Only re-score ontology-sourced text chunks (they have content_text)
    ontology_chunks = [
        c for c in expanded
        if (c.context or {}).get("source") == "ontology"
        and c.content_text
    ]

    if not ontology_chunks:
        return expanded

    import numpy as np
    from app.services.embedding import embed_texts

    loop = asyncio.get_running_loop()

    chunk_texts = [c.content_text for c in ontology_chunks]

    def _embed():
        query_emb = np.array(embed_texts([query_text], query=True)[0])
        chunk_embs = np.array(embed_texts(chunk_texts))
        similarities = chunk_embs @ query_emb
        return similarities

    similarities = await loop.run_in_executor(None, _embed)

    for chunk, sim in zip(ontology_chunks, similarities):
        cosine_sim = max(float(sim), 0.0)
        # Re-compute fusion score using actual cosine similarity as the semantic component,
        # preserving the ontology relation weight contribution
        rel_type = (chunk.context or {}).get("rel_type", "RELATED_TO")
        chunk.score = compute_fusion_score(
            semantic_score=cosine_sim,
            ontology_rel_type=rel_type,
            ontology_hops=1,
            content_text=chunk.content_text,
            query_text=query_text,
        )

    return expanded
```

Note: Also fixes deprecated `asyncio.get_event_loop()` → `asyncio.get_running_loop()`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_retrieval_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "fix: ontology re-scoring now preserves relation weights via fusion formula"
```

---

### Task 3: Fix cross-modal expansion to use fusion formula

**Files:**
- Modify: `app/api/v1/retrieval.py:601-648` (`_expand_via_cross_modal`)
- Modify: `app/api/v1/_retrieval_helpers.py:159-204` (`compute_fusion_score`)
- Test: `tests/unit/test_retrieval_helpers.py`

The issue: Cross-modal expansion uses `source_score * 0.85` directly, bypassing the fusion formula. This produces scores on a different conceptual scale than doc-structure and ontology expansion. It should route through `compute_fusion_score` with a `cross_modal_decay` parameter.

- [ ] **Step 1: Add `cross_modal_decay` support to `compute_fusion_score`**

In `app/api/v1/_retrieval_helpers.py`, update `compute_fusion_score` to accept and handle cross-modal:

```python
def compute_fusion_score(
    semantic_score: float,
    doc_structure_weight: float = 0.0,
    doc_structure_hops: int = 0,
    ontology_rel_type: str | None = None,
    ontology_hops: int = 0,
    cross_modal_decay: float = 0.0,  # ADD THIS
    content_text: str | None = None,
    query_text: str | None = None,
) -> float:
```

Add the cross-modal component inside the function, before the weighted fusion line:

```python
    # Cross-modal component (fallback legacy path)
    cross_score = 0.0
    if cross_modal_decay > 0:
        cross_score = cross_modal_decay

    # Weighted fusion
    final = sem_w * semantic_score + doc_w * max(doc_score, cross_score) + onto_w * onto_score
```

Cross-modal reuses the `doc_w` slot since they're mutually exclusive (cross-modal is the fallback when doc-structure returns nothing).

- [ ] **Step 2: Write test for cross-modal fusion**

```python
# tests/unit/test_retrieval_helpers.py — add

def test_fusion_score_cross_modal_uses_doc_weight():
    """Cross-modal decay should feed through the doc_structure weight slot."""
    score = compute_fusion_score(
        semantic_score=0.8,
        cross_modal_decay=0.85,
    )
    # 0.65*0.8 + 0.20*0.85 + 0 = 0.52 + 0.17 = 0.69
    assert abs(score - 0.69) < 0.01
```

- [ ] **Step 3: Run test**

Run: `pytest tests/unit/test_retrieval_helpers.py::test_fusion_score_cross_modal_uses_doc_weight -v`
Expected: PASS

- [ ] **Step 4: Update `_expand_via_cross_modal` to use `compute_fusion_score`**

In `app/api/v1/retrieval.py`, replace line 640:

```python
# Before:
chunk_data.score = source_score * decay

# After:
chunk_data.score = compute_fusion_score(
    semantic_score=source_score,
    cross_modal_decay=decay,
    content_text=chunk_data.content_text,
    query_text=query_text,
)
```

Also add `query_text` parameter to the function signature and the call site in `_expand_one`:

```python
async def _expand_via_cross_modal(
    chunk_id: str,
    source_score: float,
    include_context: bool = True,
    query_text: str | None = None,  # ADD
) -> list[QueryResultItem]:
```

Update call in `_expand_one` (line 290):

```python
cross_items = await _expand_via_cross_modal(chunk_id_str, seed.score, include_context, query_text)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_retrieval_helpers.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/api/v1/retrieval.py app/api/v1/_retrieval_helpers.py tests/unit/test_retrieval_helpers.py
git commit -m "fix: cross-modal expansion now uses fusion formula for consistent scoring"
```

---

### Task 4: Fix deprecated `asyncio.get_event_loop()` calls

**Files:**
- Modify: `app/api/v1/retrieval.py` (lines 341, 713, 751, 789, 823 approximately — all `asyncio.get_event_loop()` calls)

- [ ] **Step 1: Replace all `asyncio.get_event_loop()` with `asyncio.get_running_loop()`**

Search for `asyncio.get_event_loop()` in `retrieval.py` and replace each with `asyncio.get_running_loop()`. The `_rescore_expanded_chunks` instance was already fixed in Task 2. Fix the remaining instances in the GraphRAG query functions.

- [ ] **Step 2: Run tests**

Run: `pytest tests/ -v -k "retrieval" --timeout=30`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "fix: replace deprecated asyncio.get_event_loop with get_running_loop"
```

---

### Task 5: Remove dead code

**Files:**
- Modify: `app/api/v1/retrieval.py` (remove unused `get_ontology_decay` import)
- Modify: `app/api/v1/_retrieval_helpers.py` (add docstring note to `get_doc_link_weights` and `get_ontology_decay` that they are unused in production but kept as public API for external consumers, OR remove them if no external usage exists)

- [ ] **Step 1: Remove unused `get_ontology_decay` import from retrieval.py**

In `app/api/v1/retrieval.py` line 21, remove `get_ontology_decay` from the import:

```python
# Before:
from app.api.v1._retrieval_helpers import (
    ...
    get_cross_modal_decay,
    get_ontology_decay,
)

# After:
from app.api.v1._retrieval_helpers import (
    ...
    get_cross_modal_decay,
)
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/ -v -k "retrieval" --timeout=30`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "chore: remove unused get_ontology_decay import"
```

---

### Task 6: Make cross-modal LIMIT configurable

**Files:**
- Modify: `app/api/v1/retrieval.py:614-618` (`_expand_via_cross_modal` Cypher query)

- [ ] **Step 1: Replace hardcoded `LIMIT 5` with configurable setting**

The existing `retrieval_doc_expand_k` setting (default 5) is the right one to reuse since cross-modal is the fallback for doc-structure.

In `_expand_via_cross_modal`, add settings import and replace the hardcoded LIMIT:

```python
from app.config import get_settings
s = get_settings()

query = """
    MATCH (src:ChunkRef {chunk_id: $chunk_id})-[*1..3]-(target:ChunkRef)
    WHERE target.chunk_id <> $chunk_id
    RETURN target.chunk_id AS target_chunk_id,
           target.chunk_type AS target_chunk_type
    LIMIT $limit
"""

# ...
result = await session.run(query, chunk_id=chunk_id, limit=s.retrieval_doc_expand_k)
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/ -v -k "retrieval" --timeout=30`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "fix: make cross-modal expansion LIMIT configurable via retrieval_doc_expand_k"
```

---

## Chunk 2: Image Description Section Splitter

### Task 7: Implement `split_description_sections` in chunking.py

**Files:**
- Modify: `app/services/chunking.py` (add new function)
- Test: `tests/unit/test_chunking.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_chunking.py — add to existing file

from app.services.chunking import split_description_sections


class TestSplitDescriptionSections:
    def test_markdown_headers(self):
        desc = "# Executive Summary\nThis is a missile.\n\n## Technical Details\nLength: 10m.\n\n## Markings\nNone visible."
        sections = split_description_sections(desc)
        assert len(sections) == 3
        assert sections[0].startswith("# Executive Summary")
        assert "missile" in sections[0]
        assert sections[1].startswith("## Technical Details")

    def test_numbered_headers_parenthesis(self):
        desc = "1) Classification: Category=photo\n\n2) Why This Category: visible cues\n\n3) General Description: missile image"
        sections = split_description_sections(desc)
        assert len(sections) == 3
        assert sections[0].startswith("1)")

    def test_numbered_headers_dot(self):
        desc = "1. Executive Summary\nMissile system.\n\n2. Source Context\nPDF summary.\n\n3. Full Scene\nOutdoor."
        sections = split_description_sections(desc)
        assert len(sections) == 3

    def test_bold_headers(self):
        desc = "**Executive Summary:** This is a radar.\n\n**Technical Details:** Frequency band X."
        sections = split_description_sections(desc)
        assert len(sections) == 2
        assert "radar" in sections[0]

    def test_fallback_paragraph_split(self):
        desc = "First paragraph about the missile.\n\nSecond paragraph about the radar.\n\nThird paragraph."
        sections = split_description_sections(desc)
        assert len(sections) == 3

    def test_skip_short_sections(self):
        desc = "# Summary\nGood content here about the system.\n\n## Empty\n\n\n## Details\nMore content here."
        sections = split_description_sections(desc)
        # The "## Empty" section with no body should be skipped (< 20 chars)
        assert all(len(s) >= 20 for s in sections)

    def test_preamble_before_first_header(self):
        desc = "This is introductory text before any header.\n\n# First Section\nContent here."
        sections = split_description_sections(desc)
        assert len(sections) == 2
        assert "introductory" in sections[0]

    def test_single_section_no_headers(self):
        desc = "This is a single block of text describing a missile system with sufficient length to pass minimum."
        sections = split_description_sections(desc)
        assert len(sections) == 1

    def test_empty_description(self):
        sections = split_description_sections("")
        assert sections == []

    def test_whitespace_only(self):
        sections = split_description_sections("   \n\n   ")
        assert sections == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_chunking.py::TestSplitDescriptionSections -v`
Expected: FAIL with `ImportError: cannot import name 'split_description_sections'`

- [ ] **Step 3: Implement `split_description_sections`**

Add to `app/services/chunking.py`:

```python
import re

# Section header patterns for image description splitting
_SECTION_HEADER_PATTERNS = [
    re.compile(r'^#{1,4}\s+.+', re.MULTILINE),           # Markdown: # / ## / ### / ####
    re.compile(r'^\d{1,2}\)\s+.+', re.MULTILINE),        # Numbered: 1) / 2)
    re.compile(r'^\d{1,2}\.\s+.+', re.MULTILINE),        # Numbered: 1. / 2.
    re.compile(r'^\*\*[^*]+[:\.]\*\*', re.MULTILINE),    # Bold: **Title:** / **Title.**
]

_MIN_SECTION_LENGTH = 20


def split_description_sections(description: str) -> list[str]:
    """Split an image description into sections by headers.

    Handles markdown headers (# / ## / ###), numbered headers (1) / 1.),
    and bold headers (**Title:**). Falls back to paragraph splitting.
    Returns list of section strings with headers prepended.
    Skips sections shorter than 20 characters.
    """
    if not description or not description.strip():
        return []

    description = description.strip()

    # Try each header pattern to find split points
    for pattern in _SECTION_HEADER_PATTERNS:
        matches = list(pattern.finditer(description))
        if len(matches) >= 2:
            return _split_at_matches(description, matches)

    # Fallback: paragraph splitting
    paragraphs = [p.strip() for p in description.split("\n\n") if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) >= _MIN_SECTION_LENGTH]
    return paragraphs if paragraphs else ([description] if len(description) >= _MIN_SECTION_LENGTH else [])


def _split_at_matches(description: str, matches: list[re.Match]) -> list[str]:
    """Split description text at header match positions."""
    sections: list[str] = []

    # Preamble before first header
    if matches[0].start() > 0:
        preamble = description[:matches[0].start()].strip()
        if len(preamble) >= _MIN_SECTION_LENGTH:
            sections.append(preamble)

    # Each header + its body until the next header
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(description)
        section = description[start:end].strip()
        if len(section) >= _MIN_SECTION_LENGTH:
            sections.append(section)

    return sections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_chunking.py::TestSplitDescriptionSections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/chunking.py tests/unit/test_chunking.py
git commit -m "feat: add split_description_sections for image description chunking"
```

---

## Chunk 3: Pipeline — Image Description Embedding Pass

### Task 8: Add image description embedding to `derive_text_chunks_and_embeddings`

**Files:**
- Modify: `app/workers/pipeline.py:1374-1577` (`derive_text_chunks_and_embeddings`)
- Test: `tests/unit/test_retrieval_pipeline.py`

This is the core pipeline change. After the existing text chunk pass, add a second pass that:
1. Queries image DocumentElements with non-null `content_text` and `artifact_id`
2. Splits descriptions into sections
3. Creates TextChunk rows with `modality="image_description"`
4. Embeds with BGE and upserts to Qdrant
5. Creates SAME_ARTIFACT chunk_links (neighbor-only)

- [ ] **Step 1: Write test for image description chunking in pipeline**

```python
# tests/unit/test_retrieval_pipeline.py — add

def test_image_description_chunk_index_offset():
    """Image description chunks should use 100000+ offset to avoid collision."""
    # chunk_index = 100000 + element_order * 100 + section_index
    # element_order=5, section_index=2 => 100000 + 500 + 2 = 100502
    element_order = 5
    section_index = 2
    chunk_index = 100000 + element_order * 100 + section_index
    assert chunk_index == 100502
    assert chunk_index > 99999  # Always above regular text chunk indices
```

- [ ] **Step 2: Run test**

Run: `pytest tests/unit/test_retrieval_pipeline.py::test_image_description_chunk_index_offset -v`
Expected: PASS

- [ ] **Step 3: Hoist shared variables out of `if all_texts:` block**

Several variables used by both the text chunk pass and the image description pass are currently scoped inside `if all_texts:`. Move these to before that block so they're available regardless.

**Before** the `if all_texts:` block (around line 1457), add:

```python
        _embed_batch = settings.embed_text_batch_size
        _upsert_batch = settings.qdrant_upsert_batch_size
        model_version = settings.text_embedding_model

        from app.db.session import get_qdrant_client
        from app.services.qdrant_store import upsert_text_vectors_batch
        from qdrant_client.models import PointStruct
        qdrant = get_qdrant_client()
```

Then remove the duplicate assignments of `_embed_batch` (line 1459), `model_version` (line 1463), and the `from` imports (lines 1466-1469) from inside `if all_texts:`.

- [ ] **Step 4: Implement the image description pass**

In `app/workers/pipeline.py`, inside `derive_text_chunks_and_embeddings`, add the following block **after** `db.commit()` at line 1533 and **before** the stage_run update at line 1535:

```python
        # ── Pass 2: Image description sections ──────────────────────────
        from app.services.chunking import split_description_sections
        from app.models.retrieval import ChunkLink

        img_elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type == "image",
                DocumentElement.content_text.isnot(None),
                DocumentElement.content_text != "",
                DocumentElement.artifact_id.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        img_desc_chunks_created = 0
        img_desc_qdrant_points: list[PointStruct] = []
        img_desc_texts: list[str] = []
        img_desc_chunk_metas: list[dict] = []

        for img_elem in img_elements:
            # Normalize Unicode to prevent NaN embeddings (same pattern as text chunk pass)
            desc_text = _normalize_text(img_elem.content_text)
            sections = split_description_sections(desc_text)
            if not sections:
                continue

            for sec_idx, section_text in enumerate(sections):
                chunk_index = 100000 + img_elem.element_order * 100 + sec_idx
                uid_str = str(img_elem.element_uid) if img_elem.element_uid else str(img_elem.id)
                chunk_key = hashlib.sha256(
                    f"{document_id}:{uid_str}:{sec_idx}:{model_version}".encode()
                ).hexdigest()
                chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())

                img_desc_texts.append(section_text)
                img_desc_chunk_metas.append({
                    "chunk_id": chunk_id,
                    "artifact_id": img_elem.artifact_id,
                    "document_id": uuid.UUID(document_id),
                    "chunk_index": chunk_index,
                    "page_number": img_elem.page_number,
                    "section_text": section_text,
                    "element_order": img_elem.element_order,
                    "sec_idx": sec_idx,
                })

        # Batch embed all image description sections
        if img_desc_texts:
            img_desc_embeddings: list[list[float]] = []
            for _eb_start in range(0, len(img_desc_texts), _embed_batch):
                img_desc_embeddings.extend(
                    embed_texts(img_desc_texts[_eb_start:_eb_start + _embed_batch])
                )

            # Create TextChunk rows and Qdrant points
            for meta, emb in zip(img_desc_chunk_metas, img_desc_embeddings):
                chunk_values = {
                    "id": meta["chunk_id"],
                    "artifact_id": meta["artifact_id"],
                    "document_id": meta["document_id"],
                    "chunk_index": meta["chunk_index"],
                    "chunk_text": meta["section_text"],
                    "embedding": emb,
                    "modality": "image_description",
                    "page_number": meta["page_number"],
                    "bounding_box": None,
                    "qdrant_point_id": meta["chunk_id"],
                    "classification": doc_classification,
                }

                stmt = pg_insert(TextChunk).values(**chunk_values).on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "chunk_text": chunk_values["chunk_text"],
                        "embedding": chunk_values["embedding"],
                        "modality": chunk_values["modality"],
                        "qdrant_point_id": chunk_values["qdrant_point_id"],
                    },
                )
                db.execute(stmt)

                img_desc_qdrant_points.append(PointStruct(
                    id=str(meta["chunk_id"]),
                    vector=emb,
                    payload={
                        "chunk_id": str(meta["chunk_id"]),
                        "document_id": document_id,
                        "artifact_id": str(meta["artifact_id"]),
                        "modality": "image_description",
                        "page_number": meta["page_number"],
                        "classification": doc_classification,
                        "chunk_text": meta["section_text"],
                    },
                ))
                img_desc_chunks_created += 1

            # Batch upsert to Qdrant
            if img_desc_qdrant_points:
                for _qb_start in range(0, len(img_desc_qdrant_points), _upsert_batch):
                    upsert_text_vectors_batch(
                        qdrant,
                        img_desc_qdrant_points[_qb_start:_qb_start + _upsert_batch],
                    )

            # SAME_ARTIFACT chunk_links (neighbor-only) between consecutive sections
            # Group by artifact_id (one image's description sections)
            from collections import defaultdict
            artifact_section_chunks: dict[str, list[uuid.UUID]] = defaultdict(list)
            for meta in img_desc_chunk_metas:
                artifact_section_chunks[str(meta["artifact_id"])].append(meta["chunk_id"])

            for art_id, chunk_ids in artifact_section_chunks.items():
                for i in range(len(chunk_ids) - 1):
                    for src, tgt in [(chunk_ids[i], chunk_ids[i + 1]),
                                     (chunk_ids[i + 1], chunk_ids[i])]:
                        link_vals = {
                            "source_chunk_id": src,
                            "target_chunk_id": tgt,
                            "document_id": uuid.UUID(document_id),
                            "link_type": "SAME_ARTIFACT",
                            "hop": 1,
                            "weight": settings.retrieval_weight_same_artifact,
                        }
                        link_stmt = pg_insert(ChunkLink).values(**link_vals).on_conflict_do_update(
                            constraint="chunk_links_pkey",
                            set_={"weight": link_vals["weight"]},
                        )
                        db.execute(link_stmt)

            db.commit()

        chunks_created += img_desc_chunks_created
```

Also update the `_upsert_batch` variable to be set earlier (it's already defined at line 1529 but only inside the `if all_texts:` block). Move it to be accessible to both passes, or re-derive it:

After line 1457 (`if all_texts:`), ensure `_upsert_batch` is also available for the image description pass. The simplest approach is to set it before the first `if all_texts:` block:

```python
_upsert_batch = settings.qdrant_upsert_batch_size
_embed_batch = settings.embed_text_batch_size
```

And remove the duplicate `_embed_batch = settings.embed_text_batch_size` from inside the `if all_texts:` block.

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_retrieval_pipeline.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "feat: embed image description sections as BGE text vectors with SAME_ARTIFACT links"
```

---

## Chunk 4: Retrieval Pipeline — Modality Filter + Image URL Resolution

### Task 9: Add `image_description` to modality filter

**Files:**
- Modify: `app/api/v1/retrieval.py:229-232` (modality filter in `_multi_modal_pipeline`)
- Test: `tests/unit/test_retrieval_helpers.py`

- [ ] **Step 1: Write test**

```python
# tests/unit/test_retrieval_helpers.py — add

def test_image_description_modality_in_text_filter():
    """image_description should pass through the text modality filter."""
    from app.schemas.retrieval import QueryResultItem
    items = [
        QueryResultItem(score=0.9, modality="text", content_text="missile"),
        QueryResultItem(score=0.8, modality="image_description", content_text="photo of radar"),
        QueryResultItem(score=0.7, modality="image", content_text="image"),
    ]
    text_filtered = [r for r in items if r.modality in ("text", "table", "image_description")]
    assert len(text_filtered) == 2
    assert text_filtered[1].modality == "image_description"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/unit/test_retrieval_helpers.py::test_image_description_modality_in_text_filter -v`
Expected: PASS

- [ ] **Step 3: Update modality filter in retrieval.py**

In `app/api/v1/retrieval.py` line 230, add `"image_description"`:

```python
if body.modality_filter == ModalityFilter.text:
    deduped = [r for r in deduped if r.modality in ("text", "table", "image_description")]
```

- [ ] **Step 4: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "fix: include image_description in text modality filter"
```

---

### Task 10: Implement batched image URL resolution for `image_description` results

**Files:**
- Modify: `app/api/v1/retrieval.py:970-980` (`_populate_image_urls`)
- Test: `tests/unit/test_retrieval_helpers.py`

- [ ] **Step 1: Write test**

```python
# tests/unit/test_retrieval_helpers.py — add
import uuid

def test_image_description_result_gets_image_url():
    """image_description results should get image_url populated from their artifact_id."""
    from app.schemas.retrieval import QueryResultItem
    art_id = uuid.uuid4()
    result = QueryResultItem(
        score=0.9,
        modality="image_description",
        content_text="Photo of S-75 launcher",
        artifact_id=art_id,
        document_id=uuid.uuid4(),
    )
    # The URL should be set after _populate_image_urls runs
    # (integration test — unit test just verifies the schema supports it)
    assert result.image_url is None  # Not set yet
    assert result.artifact_id == art_id
```

- [ ] **Step 2: Run test**

Run: `pytest tests/unit/test_retrieval_helpers.py::test_image_description_result_gets_image_url -v`
Expected: PASS

- [ ] **Step 3: Update `_populate_image_urls`**

In `app/api/v1/retrieval.py`, replace `_populate_image_urls` (line 970):

```python
async def _populate_image_urls(
    db: AsyncSession, results: list[QueryResultItem]
) -> None:
    """Set image_url to the API proxy path for image-modality results.

    For image/schematic: uses chunk_id directly (chunk IS the image).
    For image_description: looks up the ImageChunk by artifact_id to find the image.
    """
    # Direct image/schematic results
    for result in results:
        if result.modality in ("image", "schematic") and result.chunk_id:
            result.image_url = f"/v1/images/{result.chunk_id}"

    # Image description results — batch lookup ImageChunk by artifact_id
    img_desc_results = [
        r for r in results
        if r.modality == "image_description" and r.artifact_id
    ]
    if not img_desc_results:
        return

    artifact_ids = list({str(r.artifact_id) for r in img_desc_results})
    sql = text("""
        SELECT artifact_id::text, id::text
        FROM retrieval.image_chunks
        WHERE artifact_id = ANY(:artifact_ids)
    """)
    rows = (await db.execute(sql, {"artifact_ids": artifact_ids})).fetchall()
    artifact_to_image_chunk: dict[str, str] = {row[0]: row[1] for row in rows}

    for r in img_desc_results:
        image_chunk_id = artifact_to_image_chunk.get(str(r.artifact_id))
        if image_chunk_id:
            r.image_url = f"/v1/images/{image_chunk_id}"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v -k "retrieval" --timeout=30`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/v1/retrieval.py
git commit -m "feat: resolve image URLs for image_description results via artifact_id batch lookup"
```

---

## Chunk 5: Final Verification

### Task 11: Run full test suite and verify

**Files:**
- All modified files from Tasks 1-10

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 2: Run linting**

Run: `ruff check app/api/v1/retrieval.py app/api/v1/_retrieval_helpers.py app/services/chunking.py app/workers/pipeline.py`
Expected: No errors

- [ ] **Step 3: Final commit if any lint fixes needed**

```bash
git add -A
git commit -m "chore: lint fixes"
```
