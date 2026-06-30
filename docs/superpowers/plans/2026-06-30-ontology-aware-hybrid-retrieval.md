# Ontology-Aware Hybrid Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the multi-modal **hybrid** search path retrieve chunks related through the air-defense domain ontology (`VARIANT_OF`/`ASSOCIATED_WITH`/`CUES`/`PART_OF`/`CONTAINS`/`USES_COMPONENT`), weight them by the true relation, and carry both the semantic and ontological signal through the `top_k` cap and reranker into the returned context — with a per-query UI control over the reserved-slot count.

**Architecture:** Augment (not replace) the existing `EXTRACTED_FROM` co-mention expansion with a dedicated 1-hop domain-relation traversal (`get_related_entity_chunks` + `_expand_via_domain_relations`). Score domain-expanded chunks via an **env-backed** retrieval relation-weight table (extraction bundle untouched). Guarantee membership via **reserved slots** (per-query UI control) and preserve the signal via a **blended** reranker (`α·rerank + (1−α)·fused`) instead of the current pure-semantic overwrite. A master flag + reserved-slots=0 + α=1.0 gives a byte-identical rollback.

**Tech Stack:** Python 3.11 / FastAPI / pydantic v2 (`APIModel` + `BaseSettings`), ArcadeDB (MATCH traversal via `ArcadeDBGraphStore`), React/TypeScript/Vite frontend, pytest.

**User decisions (already made):**
- Curated relation subset: `VARIANT_OF, ASSOCIATED_WITH, CUES, PART_OF, CONTAINS, USES_COMPONENT`.
- Both soft-weighting + reserved slots; 1 domain hop; undirected (`.both`) traversal.
- Architecture B — dedicated traversal method + expansion function, augmenting co-mention.
- Keep global fusion weights `0.65/0.20/0.15` (no rebalance).
- Env-backed retrieval relation-weight table; extraction bundle `scoring_weights` untouched; no ontology-generator changes.
- Store `context.raw_cosine` + `context.fused_score_pre_rerank` on domain-expanded chunks only.
- Master flag `RETRIEVAL_DOMAIN_EXPANSION_ENABLED`; full byte-identical rollback = flag off + `RESERVED_SLOTS=0` + `ALPHA=1.0`.
- `M` reserved slots is a per-query UI control for hybrid search.
- Explicit `ontology_relation` provenance surfacing in UI + `_agent_helpers` `build_markdown`/`build_sources`.
- New env vars in BOTH `.env` and `.env.example`. Backend deploy = `app/` bind-mount (api force-recreate for new env); frontend deploy = api image rebuild.

**Spec:** `docs/superpowers/specs/2026-06-30-ontology-aware-hybrid-retrieval-design.md`

---

## File structure

| File | Responsibility | Tasks |
|---|---|---|
| `app/config.py` | New `retrieval_*` settings (master flag, reserve knobs, expand_k, weights JSON) | 0 |
| `app/api/v1/_retrieval_helpers.py` | Curated-relation constant, env-backed `get_retrieval_relation_weights()`, optional `relation_weights` param on `compute_fusion_score` | 0 |
| `app/services/arcadedb_graph.py` + `app/services/graph_store.py` | `get_related_entity_chunks` traversal (+ Protocol decl) | 1 |
| `app/api/v1/retrieval.py` | `_expand_via_domain_relations`, wire into `_expand_seeds`, rescore stores fields, reserved-slot logic in `_multi_modal_pipeline`, blended `_apply_reranker`, extend `GET /settings/retrieval` | 2,3,4,5,6 |
| `app/schemas/retrieval.py` | `ontology_reserved_slots` on `UnifiedQueryRequest` | 6 |
| `app/api/v1/agent.py` | `reserved_slots` query param plumbing | 6 |
| `app/api/v1/_agent_helpers.py` | `ontology_relation` formatting in `build_markdown` | 7 |
| `frontend/src/components/QueryPage.tsx`, `frontend/src/api/client.ts` | `ontology_relation` source label; reserved-slots stepper + request param | 7,8 |
| `.env` + `.env.example` | New env vars (both files) | 0 |
| `tests/unit/*` | Unit tests per task | 0–6 |
| E2E + rollback | Acceptance + byte-identical rollback + deploy | 9 |

---

### Task 0: Config + env-backed retrieval relation-weight table

**Goal:** Add all new settings (master flag, reserve knobs, expand_k, weights override) and an env-backed `get_retrieval_relation_weights()` + curated-relation list, plus an optional `relation_weights` param on `compute_fusion_score` so domain chunks use the retrieval table while existing paths are unchanged.

**Files:**
- Modify: `app/config.py:545-572` (add fields in the retrieval block)
- Modify: `app/api/v1/_retrieval_helpers.py` (add constants + getter + `compute_fusion_score` param)
- Modify: `.env`, `.env.example`
- Test: `tests/unit/test_retrieval_relation_weights.py`

**Acceptance Criteria:**
- [ ] `get_retrieval_relation_weights()` returns the code-default table; `VARIANT_OF`→0.95, `ASSOCIATED_WITH`→0.85, `CUES`→0.88, unknown→0.70.
- [ ] Setting `RETRIEVAL_DOMAIN_RELATION_WEIGHTS` to a JSON object overrides/merges onto the default.
- [ ] `get_curated_domain_relations()` returns the 6 curated relations.
- [ ] `compute_fusion_score(..., relation_weights=<dict>)` uses the passed dict; with `relation_weights=None` it behaves exactly as today (bundle weights via `get_ontology_relation_weights`).
- [ ] All new settings present in both `.env` and `.env.example` with comments.

**Verify:** `python3 -m pytest tests/unit/test_retrieval_relation_weights.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests** — `tests/unit/test_retrieval_relation_weights.py`:

```python
from __future__ import annotations
import json
import pytest


def test_default_retrieval_relation_weights():
    from app.api.v1._retrieval_helpers import get_retrieval_relation_weights
    w = get_retrieval_relation_weights()
    assert w["VARIANT_OF"] == 0.95
    assert w["ASSOCIATED_WITH"] == 0.85
    assert w["CUES"] == 0.88
    assert w["USES_COMPONENT"] == 0.92
    assert w["CONTAINS"] == 0.90
    assert w["PART_OF"] == 0.90
    assert w.get("EXTRACTED_FROM", w["default"]) == 0.70  # falls to default
    assert w["default"] == 0.70


def test_curated_domain_relations():
    from app.api.v1._retrieval_helpers import get_curated_domain_relations
    rels = set(get_curated_domain_relations())
    assert rels == {"VARIANT_OF", "ASSOCIATED_WITH", "CUES", "PART_OF", "CONTAINS", "USES_COMPONENT"}


def test_env_override_merges(monkeypatch):
    # Override is parsed from the settings value; reset the cached settings.
    import app.api.v1._retrieval_helpers as h
    monkeypatch.setattr(h, "_settings_cache", None)
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("RETRIEVAL_DOMAIN_RELATION_WEIGHTS", json.dumps({"CUES": 0.99, "NEW_REL": 0.5}))
    w = h.get_retrieval_relation_weights()
    assert w["CUES"] == 0.99          # overridden
    assert w["NEW_REL"] == 0.5         # added
    assert w["VARIANT_OF"] == 0.95     # default preserved
    get_settings.cache_clear()


def test_compute_fusion_score_uses_passed_relation_weights():
    from app.api.v1._retrieval_helpers import compute_fusion_score
    # With an explicit table, ASSOCIATED_WITH (0.85) beats the bundle default behavior.
    s_with = compute_fusion_score(
        semantic_score=0.0, ontology_rel_type="ASSOCIATED_WITH", ontology_hops=1,
        relation_weights={"ASSOCIATED_WITH": 0.85, "default": 0.70},
    )
    s_default = compute_fusion_score(
        semantic_score=0.0, ontology_rel_type="ASSOCIATED_WITH", ontology_hops=1,
        relation_weights={"default": 0.70},
    )
    assert s_with > s_default  # 0.15*0.85 > 0.15*0.70
```

- [ ] **Step 2: Run, expect failure** — `python3 -m pytest tests/unit/test_retrieval_relation_weights.py -q` → ImportError / AttributeError.

- [ ] **Step 3: Add settings to `app/config.py`** — insert after `retrieval_ontology_expand_k: int = 5` (line 549):

```python
    # --- Ontology-aware hybrid retrieval (TODO: ontology-aware-hybrid-retrieval) ---
    retrieval_domain_expansion_enabled: bool = True
    retrieval_domain_expand_k: int = 5
    retrieval_ontology_reserved_slots: int = 3
    retrieval_ontology_reserve_min_rel_weight: float = 0.85
    retrieval_ontology_reserve_min_cosine: float = 0.15
    retrieval_rerank_blend_alpha: float = 0.6
    # JSON object string overriding/merging the code-default retrieval relation
    # weight table; empty = use code default. Env-backed (NOT the ontology bundle).
    retrieval_domain_relation_weights: str = ""
```

- [ ] **Step 4: Add constants + getter + fusion param to `app/api/v1/_retrieval_helpers.py`.** Add near the other getters:

```python
import json as _json

_CURATED_DOMAIN_RELATIONS: tuple[str, ...] = (
    "VARIANT_OF", "ASSOCIATED_WITH", "CUES", "PART_OF", "CONTAINS", "USES_COMPONENT",
)

# Retrieval-specific relation priors. Env-backed (decoupled from the extraction
# bundle's SCORING_WEIGHTS). Keyed on LIVE predicates.
_RETRIEVAL_RELATION_WEIGHTS_DEFAULT: dict[str, float] = {
    "VARIANT_OF": 0.95,
    "USES_COMPONENT": 0.92,
    "CONTAINS": 0.90,
    "PART_OF": 0.90,
    "CUES": 0.88,
    "ASSOCIATED_WITH": 0.85,
    "default": 0.70,
}


def get_curated_domain_relations() -> list[str]:
    return list(_CURATED_DOMAIN_RELATIONS)


def get_retrieval_relation_weights() -> dict[str, float]:
    """Env-backed retrieval relation weights: code default merged with an
    optional RETRIEVAL_DOMAIN_RELATION_WEIGHTS JSON override. Does NOT read the
    ontology bundle (extraction weights stay independent)."""
    weights = dict(_RETRIEVAL_RELATION_WEIGHTS_DEFAULT)
    raw = (_settings().retrieval_domain_relation_weights or "").strip()
    if raw:
        try:
            override = _json.loads(raw)
            if isinstance(override, dict):
                weights.update({str(k): float(v) for k, v in override.items()})
        except (ValueError, TypeError):
            pass  # malformed override → fall back to defaults
    return weights
```

Then modify `compute_fusion_score` (line 166) to accept an optional `relation_weights` and use it for the ontology component. Change its signature to add `relation_weights: dict[str, float] | None = None,` and replace the `rel_weights = get_ontology_relation_weights()` line inside the ontology block with:

```python
        rel_weights = relation_weights if relation_weights is not None else get_ontology_relation_weights()
```

- [ ] **Step 5: Run tests** — `python3 -m pytest tests/unit/test_retrieval_relation_weights.py -q` → PASS.

- [ ] **Step 6: Mirror env vars into `.env` AND `.env.example`** (both files), e.g.:

```
# Ontology-aware hybrid retrieval (domain-relation graph expansion)
RETRIEVAL_DOMAIN_EXPANSION_ENABLED=true     # master flag; false disables domain-relation expansion
RETRIEVAL_DOMAIN_EXPAND_K=5                  # per-seed cap on domain-relation expansions
RETRIEVAL_ONTOLOGY_RESERVED_SLOTS=3          # default reserved top_k slots (per-query UI override)
RETRIEVAL_ONTOLOGY_RESERVE_MIN_REL_WEIGHT=0.85
RETRIEVAL_ONTOLOGY_RESERVE_MIN_COSINE=0.15
RETRIEVAL_RERANK_BLEND_ALPHA=0.6             # final score = alpha*rerank + (1-alpha)*fused
RETRIEVAL_DOMAIN_RELATION_WEIGHTS=           # optional JSON override of retrieval relation weights
```

- [ ] **Step 7: Commit** — `git add app/config.py app/api/v1/_retrieval_helpers.py .env .env.example tests/unit/test_retrieval_relation_weights.py && git commit -m "feat(retrieval): env-backed domain relation weights + ontology-aware settings"`

---

### Task 1: `get_related_entity_chunks` graph traversal

**Goal:** Add a 1-hop domain-relation traversal to `ArcadeDBGraphStore` (and the `GraphStore` Protocol) that returns chunks of entities related to the seed chunk's entities via the curated relations, undirected, with the true relation type captured, deduped keeping the strongest relation, bounded by `limit`.

**Files:**
- Modify: `app/services/arcadedb_graph.py` (new method near `get_ontology_linked_chunks:1494`)
- Modify: `app/services/graph_store.py` (Protocol declaration)
- Test: `tests/unit/test_get_related_entity_chunks.py`

**Acceptance Criteria:**
- [ ] `get_related_entity_chunks(node_id, rel_types, limit)` issues an ArcadeDB MATCH walking `seed_chunk ←EXTRACTED_FROM— entity —[rel in rel_types, both directions]— related_entity —EXTRACTED_FROM→ chunk` (chunk ≠ seed), returning `chunk_id`, `target_chunk_type`, `rel_type` (true relation), `related_entity`.
- [ ] Each returned row carries the relation it was reached by; if a chunk is reachable via multiple relations, the row kept is the one with the highest retrieval weight.
- [ ] Results capped at `limit`; returns `[]` for a non-chunk / unresolvable seed.
- [ ] `GraphStore` Protocol declares the method.

**Verify:** `python3 -m pytest tests/unit/test_get_related_entity_chunks.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests** (mock the client, mirroring `tests/unit/test_graph_store_execute.py` construction). `tests/unit/test_get_related_entity_chunks.py`:

```python
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock


def _store_with_rows(rows, seed_type="TextChunk"):
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    client = MagicMock()
    # _resolve_rid → first query returns the RID; @type lookup returns seed_type;
    # final MATCH returns `rows`. Sequence the query() responses accordingly.
    client.query = AsyncMock(side_effect=[
        [{"@rid": "#12:0"}],                  # _resolve_rid
        [{"node_type": seed_type}],           # @type lookup
        rows,                                  # MATCH
    ])
    return ArcadeDBGraphStore(client, "db"), client


async def test_returns_related_chunks_with_true_rel_type():
    rows = [{
        "chunk_rid": "#20:1", "chunk_type": "TextChunk", "chunk_id": "c-fan-song",
        "document_id": "d1", "text": "Fan Song ...", "modality": "text",
        "related_entity": "Fan Song", "rel_type": "ASSOCIATED_WITH",
    }]
    store, _ = _store_with_rows(rows)
    out = await store.get_related_entity_chunks("c-sa2", ["ASSOCIATED_WITH", "CUES"], 5)
    assert len(out) == 1
    assert out[0]["chunk_id"] == "c-fan-song"
    assert out[0]["rel_type"] == "ASSOCIATED_WITH"
    assert out[0]["related_entity"] == "Fan Song"


async def test_dedup_keeps_strongest_relation():
    rows = [
        {"chunk_rid": "#20:1", "chunk_id": "c1", "rel_type": "ASSOCIATED_WITH", "related_entity": "X"},
        {"chunk_rid": "#20:1", "chunk_id": "c1", "rel_type": "VARIANT_OF", "related_entity": "X"},
    ]
    store, _ = _store_with_rows(rows)
    out = await store.get_related_entity_chunks("c-sa2", ["ASSOCIATED_WITH", "VARIANT_OF"], 5)
    assert len(out) == 1
    assert out[0]["rel_type"] == "VARIANT_OF"  # 0.95 > 0.85


async def test_limit_and_non_chunk_seed():
    store, _ = _store_with_rows([{"chunk_rid": f"#20:{i}", "chunk_id": f"c{i}",
                                  "rel_type": "CUES", "related_entity": "X"} for i in range(10)])
    out = await store.get_related_entity_chunks("c", ["CUES"], 3)
    assert len(out) == 3
    # non-chunk seed
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    c2 = MagicMock(); c2.query = AsyncMock(side_effect=[[{"@rid": "#9:0"}], [{"node_type": "Entity"}]])
    s2 = ArcadeDBGraphStore(c2, "db")
    assert await s2.get_related_entity_chunks("e", ["CUES"], 5) == []
```

- [ ] **Step 2: Run, expect failure** — method does not exist.

- [ ] **Step 3: Implement `get_related_entity_chunks` in `app/services/arcadedb_graph.py`** (after `get_ontology_linked_chunks`). Reuse `_resolve_rid` + the typed-seed lookup pattern; weight comparison uses `get_retrieval_relation_weights`:

```python
    async def get_related_entity_chunks(
        self, node_id: str, rel_types: list[str], limit: int = 5,
    ) -> list[dict[str, Any]]:
        """1-hop domain-relation expansion: chunks of entities related to the
        seed chunk's entities via `rel_types` (undirected). Returns each chunk
        with the true relation it was reached by, deduped keeping the strongest."""
        if not rel_types:
            return []
        rid = await self._resolve_rid(node_id)
        if not rid:
            return []
        type_rows = await self._client.query(
            self._database, "sql", f"SELECT @type AS node_type FROM {rid}",
        )
        seed_type = type_rows[0].get("node_type") if type_rows and isinstance(type_rows[0], dict) else None
        if seed_type not in ("TextChunk", "ImageChunk"):
            return []
        rel_list = ",".join(f"'{r}'" for r in rel_types if r.isidentifier())
        if not rel_list:
            return []
        sql = (
            f"MATCH {{type: {seed_type}, as: seed, where: (@rid = {rid})}}"
            f".in('EXTRACTED_FROM') {{as: entity}}"
            f".bothE({rel_list}) {{as: rel}}.bothV() {{as: related, "
            f"where: ($matched.entity.@rid <> @rid)}}"
            f".out('EXTRACTED_FROM') {{as: chunk, where: (@rid <> {rid})}} "
            f"RETURN chunk.@rid AS chunk_rid, chunk.@type AS chunk_type, "
            f"chunk.chunk_id AS chunk_id, chunk.document_id AS document_id, "
            f"chunk.text AS text, chunk.chunk_text AS chunk_text, "
            f"chunk.modality AS modality, chunk.page_number AS page_number, "
            f"related.name AS related_entity, rel.@type AS rel_type"
        )
        try:
            rows = await self._client.query(self._database, "sql", sql)
        except Exception as exc:  # pragma: no cover - traversal/MATCH edge cases
            logger.debug("get_related_entity_chunks MATCH failed for %s: %s", node_id, exc)
            return []
        from app.api.v1._retrieval_helpers import get_retrieval_relation_weights
        weights = get_retrieval_relation_weights()
        best: dict[str, dict[str, Any]] = {}
        for r in rows:
            crid = str(r.get("chunk_rid", ""))
            if not crid:
                continue
            r = dict(r)
            r["target_chunk_id"] = r.get("chunk_id")
            r["target_chunk_type"] = "image_chunk" if r.get("chunk_type") == "ImageChunk" else "text_chunk"
            w = weights.get(str(r.get("rel_type")), weights.get("default", 0.70))
            prev = best.get(crid)
            if prev is None or w > prev.get("_w", -1.0):
                r["_w"] = w
                best[crid] = r
        out = sorted(best.values(), key=lambda x: x.get("_w", 0.0), reverse=True)[:limit]
        for r in out:
            r.pop("_w", None)
        return out
```

> Implementation note: if `bothE(...).bothV()` proves unreliable on this ArcadeDB build (the codebase documents MATCH quirks), fall back to one `.both('<REL>')` MATCH per relation in `rel_types`, unioned — `rel_type` is then known by construction (the loop variable). Tests use mocked rows so they pass either way; the integration test (Task 9) exercises the real MATCH.

- [ ] **Step 4: Declare in the `GraphStore` Protocol** (`app/services/graph_store.py`, near `get_ontology_linked_chunks`):

```python
    async def get_related_entity_chunks(
        self, node_id: str, rel_types: list[str], limit: int = 5,
    ) -> list[dict[str, Any]]:
        """1-hop domain-relation expansion over the given relation types."""
        ...
```

- [ ] **Step 5: Run tests** → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(graph): get_related_entity_chunks 1-hop domain-relation traversal"`

---

### Task 2: `_expand_via_domain_relations` + wire into `_expand_seeds`

**Goal:** New expansion function that turns `get_related_entity_chunks` rows into `QueryResultItem`s stamped `source="ontology_relation"` with `rel_type`/`related_entity`, gated by the master flag and wired into `_expand_seeds` alongside the existing expansions.

**Files:**
- Modify: `app/api/v1/retrieval.py` (`_expand_via_domain_relations` new; `_expand_seeds:343-379` call site)
- Test: `tests/unit/test_expand_via_domain_relations.py`

**Acceptance Criteria:**
- [ ] `_expand_via_domain_relations(chunk_id, source_score, include_context, query_text)` calls `graph_store.get_related_entity_chunks(chunk_id, get_curated_domain_relations(), settings.retrieval_domain_expand_k)` and returns items with `context["source"]=="ontology_relation"`, `context["rel_type"]`, `context["related_entity"]`, `context["source_chunk_id"]`.
- [ ] `_expand_seeds` invokes it per seed only when `settings.retrieval_domain_expansion_enabled` is true; co-mention (`_expand_via_ontology`) still runs unchanged.
- [ ] When the flag is false, no `ontology_relation` items are produced.

**Verify:** `python3 -m pytest tests/unit/test_expand_via_domain_relations.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests** — mock `get_graph_store` + `_lookup_chunk_by_type`. `tests/unit/test_expand_via_domain_relations.py`:

```python
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import app.api.v1.retrieval as R
from app.schemas.retrieval import QueryResultItem
import uuid


def _item():
    return QueryResultItem(chunk_id=uuid.uuid4(), score=0.0, modality="text", content_text="Fan Song radar")


async def test_domain_expansion_stamps_ontology_relation(monkeypatch):
    gs = MagicMock()
    gs.get_related_entity_chunks = AsyncMock(return_value=[
        {"target_chunk_id": "c1", "target_chunk_type": "text_chunk",
         "rel_type": "ASSOCIATED_WITH", "related_entity": "Fan Song"},
    ])
    monkeypatch.setattr(R, "get_graph_store", lambda: gs)
    monkeypatch.setattr(R, "_lookup_chunk_by_type", AsyncMock(return_value=_item()))
    out = await R._expand_via_domain_relations("c-sa2", 0.7, True, "SA-2")
    assert len(out) == 1
    ctx = out[0].context
    assert ctx["source"] == "ontology_relation"
    assert ctx["rel_type"] == "ASSOCIATED_WITH"
    assert ctx["related_entity"] == "Fan Song"
    assert ctx["source_chunk_id"] == "c-sa2"


async def test_expand_seeds_respects_master_flag(monkeypatch):
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("RETRIEVAL_DOMAIN_EXPANSION_ENABLED", "false")
    get_settings.cache_clear()
    called = AsyncMock(return_value=[])
    monkeypatch.setattr(R, "_expand_via_domain_relations", called)
    monkeypatch.setattr(R, "_expand_via_doc_structure", AsyncMock(return_value=[_item()]))
    monkeypatch.setattr(R, "_expand_via_ontology", AsyncMock(return_value=[]))
    seed = _item()
    await R._expand_seeds(MagicMock(), [seed], True, "SA-2")
    called.assert_not_called()
    get_settings.cache_clear()
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `_expand_via_domain_relations`** in `app/api/v1/retrieval.py` (sibling of `_expand_via_ontology:755`):

```python
async def _expand_via_domain_relations(
    chunk_id: str,
    source_score: float,
    include_context: bool = True,
    query_text: str | None = None,
) -> list[QueryResultItem]:
    """Follow CURATED domain relations (entity -[rel]- related_entity -> chunk)
    one hop to retrieve ontologically-related chunks. Augments co-mention."""
    from app.config import get_settings
    from app.api.v1._retrieval_helpers import get_curated_domain_relations, get_retrieval_relation_weights

    s = get_settings()
    graph_store = get_graph_store()
    try:
        linked = await graph_store.get_related_entity_chunks(
            chunk_id, get_curated_domain_relations(), s.retrieval_domain_expand_k,
        )
    except Exception as e:  # pragma: no cover
        logger.debug("Domain-relation expansion failed for %s: %s", chunk_id, e)
        return []

    weights = get_retrieval_relation_weights()
    items: list[QueryResultItem] = []
    from app.db.session import AsyncSessionFactory
    async with AsyncSessionFactory() as db_session:
        for link in linked:
            target_id = link.get("target_chunk_id") or link.get("chunk_id", "")
            target_type = link.get("target_chunk_type", "text_chunk")
            if not target_id:
                continue
            chunk_data = await _lookup_chunk_by_type(db_session, target_id, target_type, include_context)
            if not chunk_data:
                continue
            rel_type = str(link.get("rel_type", "RELATED_TO"))
            chunk_data.score = compute_fusion_score(
                semantic_score=source_score,
                ontology_rel_type=rel_type,
                ontology_hops=1,
                content_text=chunk_data.content_text,
                query_text=query_text,
                relation_weights=weights,
            )
            chunk_data.context = {
                "source": "ontology_relation",
                "rel_type": rel_type,
                "related_entity": link.get("related_entity", ""),
                "source_chunk_id": chunk_id,
            }
            items.append(chunk_data)
    return items
```

- [ ] **Step 4: Wire into `_expand_seeds`** — inside `_expand_one` (after the `onto_items` lines ~366-367), add:

```python
            if get_settings().retrieval_domain_expansion_enabled:
                domain_items = await _expand_via_domain_relations(chunk_id_str, seed.score, include_context, query_text)
                items.extend(domain_items)
```

(Add `from app.config import get_settings` at the top of `_expand_seeds` or use the module-level import if present.)

- [ ] **Step 5: Run tests** → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(retrieval): _expand_via_domain_relations (master-flag gated)"`

---

### Task 3: Rescore stores `raw_cosine` + `fused_score_pre_rerank` and handles `ontology_relation`

**Goal:** Extend `_rescore_expanded_chunks` to also rescore `ontology_relation` chunks (using their true relation + the retrieval weight table) and to persist `context.raw_cosine` and `context.fused_score_pre_rerank` (instead of discarding the cosine).

**Files:**
- Modify: `app/api/v1/retrieval.py:386-438`
- Test: `tests/unit/test_rescore_stores_fields.py`

**Acceptance Criteria:**
- [ ] Chunks with `context.source` in `{"ontology", "ontology_relation"}` and non-empty `content_text` are rescored.
- [ ] After rescore, each rescored chunk has `context["raw_cosine"]` (the pre-fusion cosine) and `context["fused_score_pre_rerank"]` (== `chunk.score`).
- [ ] `ontology_relation` chunks pass their true `rel_type` + the retrieval weight table to `compute_fusion_score`; `ontology` (co-mention) chunks keep today's behavior (bundle weights, default rel_type).

**Verify:** `python3 -m pytest tests/unit/test_rescore_stores_fields.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write failing test** — `tests/unit/test_rescore_stores_fields.py`:

```python
from __future__ import annotations
from unittest.mock import patch
import numpy as np
import app.api.v1.retrieval as R
from app.schemas.retrieval import QueryResultItem
import uuid


async def test_rescore_stores_raw_cosine_and_fused(monkeypatch):
    item = QueryResultItem(chunk_id=uuid.uuid4(), score=0.99, modality="text",
                           content_text="Fan Song radar associated with SA-2",
                           context={"source": "ontology_relation", "rel_type": "ASSOCIATED_WITH"})
    monkeypatch.setattr(R, "embed_texts", lambda texts, query=False: [np.ones(4) / 2.0 for _ in texts], raising=False)
    # patch the embedding import used inside _embed
    with patch("app.services.embedding.embed_texts", lambda texts, query=False: [np.ones(4) / 2.0 for _ in texts]):
        out = await R._rescore_expanded_chunks([item], "SA-2 guidance radar")
    ctx = out[0].context
    assert "raw_cosine" in ctx
    assert "fused_score_pre_rerank" in ctx
    assert ctx["fused_score_pre_rerank"] == out[0].score
    assert 0.0 <= ctx["raw_cosine"] <= 1.0
```

- [ ] **Step 2: Run, expect failure** (no `raw_cosine` key).

- [ ] **Step 3: Modify `_rescore_expanded_chunks`** (`retrieval.py:386-438`). Change the filter and the loop:

```python
    from app.api.v1._retrieval_helpers import get_retrieval_relation_weights
    retrieval_weights = get_retrieval_relation_weights()

    ontology_chunks = [
        c for c in expanded
        if (c.context or {}).get("source") in ("ontology", "ontology_relation")
        and c.content_text
    ]
    ...
    for chunk, sim in zip(ontology_chunks, similarities):
        cosine_sim = max(float(sim), 0.0)
        ctx = chunk.context or {}
        source = ctx.get("source")
        rel_type = ctx.get("rel_type", "RELATED_TO")
        # ontology_relation chunks use the env-backed retrieval weights; legacy
        # co-mention ("ontology") chunks keep today's bundle-weight behavior.
        rel_weights = retrieval_weights if source == "ontology_relation" else None
        chunk.score = compute_fusion_score(
            semantic_score=cosine_sim,
            ontology_rel_type=rel_type,
            ontology_hops=1,
            content_text=chunk.content_text,
            query_text=query_text,
            relation_weights=rel_weights,
        )
        ctx["raw_cosine"] = cosine_sim
        ctx["fused_score_pre_rerank"] = chunk.score
        chunk.context = ctx
```

- [ ] **Step 4: Run test** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(retrieval): rescore stores raw_cosine + fused_score_pre_rerank; handle ontology_relation"`

---

### Task 4: Reserved slots in `_multi_modal_pipeline`

**Goal:** Before the `top_k` cap, reserve up to `M` slots for qualifying `ontology_relation` chunks (rel weight ≥ gate AND `raw_cosine` ≥ floor), filling the rest by fused score. `M` resolved from `body.ontology_reserved_slots ?? settings`, clamped `[0, top_k]`. `M=0` reverts to pure ranking.

**Files:**
- Modify: `app/api/v1/retrieval.py:264-326` (`_multi_modal_pipeline`, the cap region ~307-309)
- Test: `tests/unit/test_reserved_slots.py`

**Acceptance Criteria:**
- [ ] A helper `_apply_reserved_slots(deduped, top_k, m, min_rel_weight, min_cosine, relation_weights)` returns a `top_k`-length list where up to `m` qualifying `ontology_relation` chunks are guaranteed present, remaining filled by descending fused score, no duplicates.
- [ ] A chunk below `min_cosine` (`context.raw_cosine`) is NOT reserved; a chunk whose relation weight < `min_rel_weight` is NOT reserved.
- [ ] `m=0` → output is identical to `sorted-by-score[:top_k]` (pure ranking).
- [ ] `_multi_modal_pipeline` calls the helper between the modality filter and the rerank, resolving `m` from the request/settings, clamped to `[0, top_k]`.

**Verify:** `python3 -m pytest tests/unit/test_reserved_slots.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests** — `tests/unit/test_reserved_slots.py`:

```python
from __future__ import annotations
import uuid
from app.schemas.retrieval import QueryResultItem
from app.api.v1.retrieval import _apply_reserved_slots

W = {"ASSOCIATED_WITH": 0.85, "VARIANT_OF": 0.95, "RELATED_TO": 0.75, "default": 0.70}


def _seed(score):
    return QueryResultItem(chunk_id=uuid.uuid4(), score=score, modality="text", content_text="x", context={"source": "text"})


def _onto(score, cosine, rel="ASSOCIATED_WITH"):
    return QueryResultItem(chunk_id=uuid.uuid4(), score=score, modality="text", content_text="x",
                           context={"source": "ontology_relation", "rel_type": rel, "raw_cosine": cosine})


def test_qualifying_ontology_chunk_reserved_over_seed():
    seeds = [_seed(0.9), _seed(0.85), _seed(0.8)]
    onto = _onto(score=0.2, cosine=0.3)  # low fused score, but qualifies
    out = _apply_reserved_slots(seeds + [onto], top_k=3, m=1,
                                min_rel_weight=0.85, min_cosine=0.15, relation_weights=W)
    assert onto in out
    assert len(out) == 3


def test_below_floor_rejected():
    seeds = [_seed(0.9), _seed(0.85), _seed(0.8)]
    onto = _onto(score=0.2, cosine=0.10)  # below 0.15 floor
    out = _apply_reserved_slots(seeds + [onto], top_k=3, m=1,
                                min_rel_weight=0.85, min_cosine=0.15, relation_weights=W)
    assert onto not in out


def test_non_tier_relation_rejected():
    seeds = [_seed(0.9), _seed(0.85), _seed(0.8)]
    onto = _onto(score=0.2, cosine=0.5, rel="RELATED_TO")  # 0.75 < 0.85 gate
    out = _apply_reserved_slots(seeds + [onto], top_k=3, m=1,
                                min_rel_weight=0.85, min_cosine=0.15, relation_weights=W)
    assert onto not in out


def test_m_zero_is_pure_ranking():
    seeds = [_seed(0.9), _seed(0.85)]
    onto = _onto(score=0.2, cosine=0.5)
    pool = seeds + [onto]
    out = _apply_reserved_slots(pool, top_k=2, m=0,
                                min_rel_weight=0.85, min_cosine=0.15, relation_weights=W)
    assert out == sorted(pool, key=lambda x: x.score, reverse=True)[:2]
```

- [ ] **Step 2: Run, expect failure** (`_apply_reserved_slots` undefined).

- [ ] **Step 3: Implement the helper** in `app/api/v1/retrieval.py` (module level, near `_multi_modal_pipeline`):

```python
def _apply_reserved_slots(
    deduped: list[QueryResultItem], top_k: int, m: int,
    min_rel_weight: float, min_cosine: float, relation_weights: dict[str, float],
) -> list[QueryResultItem]:
    """Guarantee up to `m` qualifying ontology_relation chunks in the top_k,
    filling the rest by descending fused score. m=0 == pure ranking."""
    ranked = sorted(deduped, key=lambda x: x.score, reverse=True)
    if m <= 0:
        return ranked[:top_k]

    def qualifies(item: QueryResultItem) -> bool:
        ctx = item.context or {}
        if ctx.get("source") != "ontology_relation":
            return False
        w = relation_weights.get(str(ctx.get("rel_type")), relation_weights.get("default", 0.70))
        return w >= min_rel_weight and float(ctx.get("raw_cosine", 0.0)) >= min_cosine

    reserved: list[QueryResultItem] = [it for it in ranked if qualifies(it)][:m]
    reserved_ids = {id(it) for it in reserved}
    for it in reserved:
        (it.context or {}).update({"reserved": True})
    fillers = [it for it in ranked if id(it) not in reserved_ids]
    return (reserved + fillers)[:top_k]
```

- [ ] **Step 4: Wire into `_multi_modal_pipeline`** — replace the `final = deduped[:body.top_k]` line (~309) with:

```python
    # Sort + reserved-slot membership guarantee (TODO ontology-aware retrieval)
    from app.config import get_settings as _gs
    from app.api.v1._retrieval_helpers import get_retrieval_relation_weights as _grw
    _s = _gs()
    _m = body.ontology_reserved_slots
    if _m is None:
        _m = _s.retrieval_ontology_reserved_slots
    _m = max(0, min(int(_m), body.top_k))
    final = _apply_reserved_slots(
        deduped, body.top_k, _m,
        _s.retrieval_ontology_reserve_min_rel_weight,
        _s.retrieval_ontology_reserve_min_cosine,
        _grw(),
    )
```

(Keep the existing `deduped.sort(...)` line; the helper re-sorts internally, which is harmless. Remove the now-redundant separate sort if preferred.)

- [ ] **Step 5: Run tests** → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(retrieval): reserved slots for qualifying ontology_relation chunks"`

---

### Task 5: Blended rerank in `_apply_reranker`

**Goal:** Replace the pure-semantic score overwrite with a blend `α·rerank + (1−α)·fused`, where `fused` is the pre-rerank score the reranker preserves. `α=1.0` reproduces today's behavior. Reserved chunks are never dropped (input pool ≤ top_k already guarantees this).

**Files:**
- Modify: `app/api/v1/retrieval.py:208-257`
- Test: `tests/unit/test_blended_rerank.py`

**Acceptance Criteria:**
- [ ] With `RETRIEVAL_RERANK_BLEND_ALPHA=0.6`, a result's final score equals `0.6*reranker_score + 0.4*pre_rerank_score`.
- [ ] With `alpha=1.0`, final score == reranker_score (today's behavior).
- [ ] All input candidates remain in the output (none dropped) when the input pool size ≤ `top_k`.

**Verify:** `python3 -m pytest tests/unit/test_blended_rerank.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write failing test** — `tests/unit/test_blended_rerank.py`:

```python
from __future__ import annotations
import uuid
from unittest.mock import patch
import app.api.v1.retrieval as R
from app.schemas.retrieval import QueryResultItem, UnifiedQueryRequest


def _items():
    a = QueryResultItem(chunk_id=uuid.uuid4(), score=0.4, modality="text", content_text="a")
    b = QueryResultItem(chunk_id=uuid.uuid4(), score=0.2, modality="text", content_text="b")
    return [a, b]


def test_blend_alpha(monkeypatch):
    from app.config import get_settings
    get_settings.cache_clear(); monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RETRIEVAL_RERANK_BLEND_ALPHA", "0.6"); get_settings.cache_clear()
    items = _items()
    # reranker returns reranker_score=1.0 for first, 0.0 for second, preserving "score"
    def fake_rerank(query, candidates, top_k=10):
        out = []
        for i, c in enumerate(candidates):
            c = dict(c); c["reranker_score"] = 1.0 if i == 0 else 0.0; out.append(c)
        return out
    with patch("app.services.reranker.rerank", fake_rerank):
        body = UnifiedQueryRequest(query_text="q", top_k=2)
        out = R._apply_reranker(items, body)
    by_id = {str(o.chunk_id): o for o in out}
    a = by_id[str(items[0].chunk_id)]
    # 0.6*1.0 + 0.4*0.4 = 0.76
    assert abs(a.score - 0.76) < 1e-6
    get_settings.cache_clear()
```

- [ ] **Step 2: Run, expect failure** (current overwrite gives 1.0, not 0.76).

- [ ] **Step 3: Modify `_apply_reranker`** — read `alpha` and blend. Replace the score-update line (249):

```python
    alpha = _s.retrieval_rerank_blend_alpha
    ...
    for r in reranked:
        key = r["chunk_id"]
        original = by_key.get(key)
        if original:
            rer = r.get("reranker_score", r.get("score", original.score))
            fused = r.get("score", original.score)
            original.score = alpha * rer + (1.0 - alpha) * fused
            output.append(original)
```

- [ ] **Step 4: Run test** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(retrieval): blended rerank (alpha*rerank + (1-alpha)*fused)"`

---

### Task 6: Request param + agent plumbing + settings endpoint

**Goal:** Expose `ontology_reserved_slots` end-to-end on the API: `UnifiedQueryRequest` field, `/agent/context` query param, and `GET /settings/retrieval` default.

**Files:**
- Modify: `app/schemas/retrieval.py:104-137` (`UnifiedQueryRequest`)
- Modify: `app/api/v1/agent.py:36-100`
- Modify: `app/api/v1/retrieval.py:1274-1283` (`get_retrieval_settings`)
- Test: `tests/unit/test_reserved_slots_request.py`

**Acceptance Criteria:**
- [ ] `UnifiedQueryRequest(ontology_reserved_slots=2)` validates; default `None`.
- [ ] `/agent/context` accepts `reserved_slots` and passes it into the `UnifiedQueryRequest` it builds.
- [ ] `GET /settings/retrieval` includes `ontology_reserved_slots`.

**Verify:** `python3 -m pytest tests/unit/test_reserved_slots_request.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests** — `tests/unit/test_reserved_slots_request.py`:

```python
from __future__ import annotations


def test_request_field_default_none():
    from app.schemas.retrieval import UnifiedQueryRequest
    assert UnifiedQueryRequest(query_text="q").ontology_reserved_slots is None
    assert UnifiedQueryRequest(query_text="q", ontology_reserved_slots=2).ontology_reserved_slots == 2


def test_settings_endpoint_includes_reserved_slots(monkeypatch):
    import asyncio
    from app.api.v1.retrieval import get_retrieval_settings
    out = asyncio.get_event_loop().run_until_complete(get_retrieval_settings())
    assert "ontology_reserved_slots" in out
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Add the field to `UnifiedQueryRequest`** (after `include_context: bool = True`):

```python
    ontology_reserved_slots: Optional[int] = Field(
        default=None, ge=0, le=100,
        description="Hybrid only: reserved top_k slots for ontology-related chunks. None = server default.",
    )
```

- [ ] **Step 4: Plumb `/agent/context`** — add a param to the handler signature (after `top_k`):

```python
    reserved_slots: Optional[int] = Query(None, ge=0, le=50, description="Hybrid only: reserved ontology slots"),
```

and pass it into the `UnifiedQueryRequest(...)` constructor:

```python
    body = UnifiedQueryRequest(
        query_text=query,
        strategy=strategy,
        modality_filter=modality_filter,
        top_k=top_k,
        ontology_reserved_slots=reserved_slots,
        include_context=True,
    )
```

(`Optional` is already imported in `agent.py`.)

- [ ] **Step 5: Extend `get_retrieval_settings`** (`retrieval.py:1278-1283`):

```python
    return {
        "top_k": settings.query_default_top_k,
        "reranker_top_n": settings.reranker_top_n,
        "min_confidence": settings.query_default_min_confidence,
        "ontology_reserved_slots": settings.retrieval_ontology_reserved_slots,
    }
```

- [ ] **Step 6: Run tests** → PASS.
- [ ] **Step 7: Commit** — `git commit -m "feat(api): ontology_reserved_slots request param + agent plumbing + settings"`

---

### Task 7: Provenance surfacing (`ontology_relation`)

**Goal:** Render the `ontology_relation` source in the agent markdown and the UI result label (with a "reserved" badge), so the new provenance is visible to humans and LLM consumers — not just raw JSON.

**Files:**
- Modify: `app/api/v1/_agent_helpers.py:28-67` (`build_markdown` source ladder)
- Modify: `frontend/src/components/QueryPage.tsx:588-603` (provenance label ladder)
- Test: `tests/unit/test_agent_markdown_ontology_relation.py`; frontend `tsc` + `vite build`

**Acceptance Criteria:**
- [ ] `build_markdown` emits a line like `**Via ontology**: ASSOCIATED_WITH → Fan Song` for an `ontology_relation` result; adds `(reserved)` when `context.reserved`.
- [ ] `QueryPage.tsx` shows a provenance label for `ctx.source === "ontology_relation"` with the relation + related entity and a reserved badge.
- [ ] `cd frontend && npx tsc --noEmit && npm run build` succeeds.

**Verify:** `python3 -m pytest tests/unit/test_agent_markdown_ontology_relation.py -q` && `cd frontend && npx tsc --noEmit`.

**Steps:**

- [ ] **Step 1: Write failing test** — `tests/unit/test_agent_markdown_ontology_relation.py`:

```python
from __future__ import annotations
import uuid
from app.api.v1._agent_helpers import build_markdown
from app.schemas.retrieval import QueryResultItem


def test_markdown_renders_ontology_relation():
    item = QueryResultItem(chunk_id=uuid.uuid4(), score=0.5, modality="text", content_text="Fan Song",
                           context={"source": "ontology_relation", "rel_type": "ASSOCIATED_WITH",
                                    "related_entity": "Fan Song", "reserved": True})
    md = build_markdown("SA-2", [item])
    assert "ASSOCIATED_WITH" in md and "Fan Song" in md
    assert "reserved" in md.lower()
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Add the branch to `build_markdown`** — insert before the `elif ... "doc_structure"` branch:

```python
        if item.context and item.context.get("source") == "ontology_relation":
            rel_type = item.context.get("rel_type", "")
            related = item.context.get("related_entity", "")
            badge = " (reserved)" if item.context.get("reserved") else ""
            lines.append(f"**Via ontology**: {rel_type} → {related}{badge}")
            lines.append("")
        elif item.context and item.context.get("source") == "ontology":
            ...  # existing branch unchanged
```

- [ ] **Step 4: Add the frontend label** — in `QueryPage.tsx` provenance ladder (~588), add before the `doc_structure` branch:

```tsx
  if (ctx?.source === "ontology_relation") {
    const rel = ctx.rel_type as string | undefined;
    const related = ctx.related_entity as string | undefined;
    const reserved = ctx.reserved ? " (reserved)" : "";
    provenanceLabel = `Via ontology: ${rel || "relation"}${related ? ` → ${related}` : ""}${reserved}`;
  } else if (ctx?.source === "ontology") {
    ... // existing
```

- [ ] **Step 5: Run** — `python3 -m pytest tests/unit/test_agent_markdown_ontology_relation.py -q` → PASS; `cd frontend && npx tsc --noEmit` → exit 0.
- [ ] **Step 6: Commit** — `git commit -m "feat(retrieval): surface ontology_relation provenance in markdown + UI"`

---

### Task 8: Reserved-slots UI control

**Goal:** Add a hybrid-only "Ontology reserved slots" numeric control to `QueryPage`, seeded from `GET /settings/retrieval`, and pass `ontology_reserved_slots` through the `unifiedQuery` client to the request.

**Files:**
- Modify: `frontend/src/api/client.ts:268-296` (`unifiedQuery` params)
- Modify: `frontend/src/components/QueryPage.tsx` (state + hybrid-gated control + pass in request)
- Test: frontend `npx tsc --noEmit` + `npm run build`

**Acceptance Criteria:**
- [ ] `unifiedQuery` accepts `ontology_reserved_slots?: number` and includes it in the POST body.
- [ ] A numeric stepper appears only when `retrievalSelected?.strategy === "hybrid"`, defaulting from the settings endpoint, range `0…topK`.
- [ ] The control's value is sent on the hybrid query.
- [ ] `cd frontend && npx tsc --noEmit && npm run build` succeeds.

**Verify:** `cd frontend && npx tsc --noEmit && npm run build` → exit 0, build OK.

**Steps:**

- [ ] **Step 1: Add the client param** — `client.ts:268`, add to the params type:

```ts
  ontology_reserved_slots?: number;
```

(no other change — `...params` already spreads into the body.)

- [ ] **Step 2: Add state + control to `QueryPage.tsx`.** Add state seeded from settings (the component already fetches `/settings/retrieval` defaults for `topK`/`rerankerTopN`; mirror that):

```tsx
const [reservedSlots, setReservedSlots] = useState<number>(3);
// in the settings-fetch effect, after setTopK(...): setReservedSlots(s.ontology_reserved_slots ?? 3);
```

Add the hybrid-gated control near the existing hybrid-only modality control:

```tsx
{retrievalSelected?.strategy === "hybrid" && (
  <label className="control">
    Ontology reserved slots
    <input type="number" min={0} max={topK} value={reservedSlots}
           onChange={(e) => setReservedSlots(Math.max(0, Math.min(topK, Number(e.target.value) || 0)))} />
  </label>
)}
```

- [ ] **Step 3: Pass it in the request** — in the `unifiedQuery({...})` call (~900-911) add:

```tsx
          ontology_reserved_slots: retrievalSelected.strategy === "hybrid" ? reservedSlots : undefined,
```

- [ ] **Step 4: Verify** — `cd frontend && npx tsc --noEmit && npm run build` → exit 0.
- [ ] **Step 5: Commit** — `git commit -m "feat(ui): hybrid reserved-slots control + client param"`

---

### Task 9: E2E acceptance, full-rollback regression, deploy

**Goal:** Prove end-to-end on the live stack that an ontologically-related chunk reaches the context, that the full rollback is byte-identical, and deploy.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create: `tests/integration/test_ontology_aware_retrieval_e2e.py` (or a documented manual procedure if the integration harness can't reach the live stack)
- Modify: deploy (no code)

**Acceptance Criteria:**
- [ ] **Before/after:** with the feature ON, a hybrid query for SA-2 returns ≥1 chunk with `context.source=="ontology_relation"`, `context.reserved==true`, a curated `rel_type` (e.g. `ASSOCIATED_WITH`/`CUES` → Fan Song); the SAME chunk is **absent** when `RETRIEVAL_DOMAIN_EXPANSION_ENABLED=false`. The test fixture/data must guarantee the SA-2↔Fan Song domain edge exists in the graph.
- [ ] **`raw_cosine` present:** the returned `ontology_relation` results carry `context.raw_cosine` and `context.fused_score_pre_rerank`.
- [ ] **Full rollback byte-identical:** results for a fixed hybrid query with `RETRIEVAL_DOMAIN_EXPANSION_ENABLED=false` + `RETRIEVAL_ONTOLOGY_RESERVED_SLOTS=0` + `RETRIEVAL_RERANK_BLEND_ALPHA=1.0` equal the pre-change baseline for that query (chunk_ids + order).
- [ ] **Deploy:** api image rebuilt (frontend baked) + api force-recreated (new env); `GET /v1/settings/retrieval` returns `ontology_reserved_slots`; `/v1/health` → 200.

**Verify:** `RETRIEVAL_DOMAIN_EXPANSION_ENABLED=true` query shows the reserved Fan Song chunk; flag-off query omits it; rollback combo matches baseline; `curl -s localhost:8005/v1/settings/retrieval` shows the new key.

**Steps:**

- [ ] **Step 1: Capture baseline** — before deploy, run a fixed hybrid query (`POST /v1/retrieval/query` `{"query_text":"SA-2 guidance","strategy":"hybrid","top_k":10}`) and save the ordered `chunk_id` list.
- [ ] **Step 2: Confirm the graph edge exists** — query ArcadeDB for an `ASSOCIATED_WITH`/`CUES` edge between an SA-2 entity and Fan Song; if absent, pick a system/relation pair that does exist and update the assertion target (the proof case is "a curated domain relation surfaces a chunk", not specifically SA-2/Fan Song).
- [ ] **Step 3: Deploy** —
```bash
cd /home/josh/development/EIP-MMDPP
docker compose -p eip-mmdpp build api
docker compose -p eip-mmdpp up -d --force-recreate api
# backend-only callers (workers) don't run retrieval; api carries the new env + frontend
sleep 15 && curl -s -o /dev/null -w "health %{http_code}\n" localhost:8005/v1/health
curl -s localhost:8005/v1/settings/retrieval
```
- [ ] **Step 4: Feature-on assertion** — run the SA-2 hybrid query; assert a result has `context.source=="ontology_relation"`, `reserved==true`, curated `rel_type`, and `raw_cosine` present; record it was absent in the Step-1 baseline.
- [ ] **Step 5: Flag-off + rollback assertion** — set the three rollback env vars (`docker compose -p eip-mmdpp up -d --force-recreate api` with them in `.env`), re-run the fixed query, assert the ordered `chunk_id` list equals the Step-1 baseline.
- [ ] **Step 6: Restore intended defaults** (`ENABLED=true`, `RESERVED_SLOTS=3`, `ALPHA=0.6`), recreate api, confirm health.
- [ ] **Step 7: Commit** test/fixture — `git commit -m "test(e2e): ontology-aware retrieval acceptance + byte-identical rollback"`

```json:metadata
{"files": ["tests/integration/test_ontology_aware_retrieval_e2e.py"], "verifyCommand": "curl -s localhost:8005/v1/settings/retrieval", "acceptanceCriteria": ["feature-on hybrid query returns a reserved ontology_relation chunk (curated rel_type) absent when flag off", "ontology_relation results carry raw_cosine + fused_score_pre_rerank", "full rollback (flag off + reserved 0 + alpha 1.0) reproduces the baseline chunk_id order byte-identically", "api rebuilt + recreated; /v1/settings/retrieval returns ontology_reserved_slots; health 200"], "userGate": true, "tags": ["user-gate"], "requireEvidenceTokens": [["before", "flag-off", "absent", "baseline"], ["after", "flag-on", "reserved", "ontology_relation"]], "modelTier": "standard"}
```

---

## Notes for the implementer

- **Run backend unit tests on the host:** `python3 -m pytest tests/unit/<file> -q`. They mock ArcadeDB/embeddings — no live stack needed for Tasks 0–7.
- **`get_settings()` is `lru_cache`'d** — tests that change env must call `get_settings.cache_clear()` (and reset `_retrieval_helpers._settings_cache = None`) before/after.
- **`context` is schema-free** (`dict[str, Any]`) — `raw_cosine`, `fused_score_pre_rerank`, `source="ontology_relation"`, `reserved` need no schema edits.
- **Deploy:** backend changes are bind-mounted (`app/`); the new env vars require an api force-recreate; the frontend change requires the api image rebuild (frontend is baked in the api image's multi-stage build). Workers don't run retrieval, so the api container is the deploy target.
- **Kill-switch / rollback** is verified in Task 9 and must remain true throughout: with the master flag off + reserved 0 + α 1.0, no domain chunks enter the pool, no slots reserve, and rerank overwrites — reproducing today exactly.
