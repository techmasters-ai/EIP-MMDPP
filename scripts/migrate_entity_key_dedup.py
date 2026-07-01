#!/usr/bin/env python3
"""One-time migration: case-insensitive entity identity (``<field>_key``).

Brings the EXISTING production graph up to the case-insensitive-identity
scheme (feature branch ``feat/case-insensitive-identity``, Tasks 0-2) WITHOUT
a re-ingest. Three ORDERED phases (the order is mandatory):

  1. MERGE case-collision pairs. Two vertices whose identity fields differ
     only by case/whitespace (e.g. ``FAN SONG`` vs ``Fan Song``) would, once
     ``<field>_key = norm(<field>)`` is backfilled, collapse onto the SAME
     ``_key`` and violate the UNIQUE ``<field>_key`` index. So we merge them
     FIRST: pick a survivor, re-point every one of the loser's edges onto the
     survivor (preserving edge properties + direction), then delete the loser
     vertex.
  2. BACKFILL ``<field>_key`` on every existing vertex of each indexed
     domain-entity type. ``norm()`` is computed in PYTHON per row (the exact
     function the write layer uses) — never reimplemented in SQL — so the
     stored keys are byte-identical to what fresh upserts write.
  3. INDEX — ensure the ``<field>_key`` UNIQUE index exists (idempotent,
     ``IF NOT EXISTS``). The always-on schema sync may already have created it
     on a deployment running the new code; creating again is a safe no-op.

Types + identity fields are DERIVED from the ontology the SAME way the schema
builder does (``arcadedb_schema.sync_schema_from_ontology`` Phase 6b +
``arcadedb_graph._key_fields``) so this stays correct if the ontology changes:
every entity type that declares ``identity_fields`` gets a ``<field>_key`` for
each identity field EXCEPT ``document_id`` (opaque UUID, kept raw).

SAFETY
------
* Default behavior (no flags) is DRY-RUN — zero writes. The destructive path
  requires BOTH ``--execute`` AND the explicit ``--yes-i-have-a-backup``
  confirmation flag (works non-interactively under ``docker exec``); with
  ``--execute`` alone the script prints the blast radius + this warning and
  REFUSES to write.
* BEFORE running ``--execute``: (a) take an ArcadeDB backup/snapshot — the
  merge DELETEs loser vertices/edges and those cannot be recovered; and
  (b) PAUSE ingestion so there are no concurrent writers. The zero-edge-loss
  guard assumes a STABLE survivor edge count between the plan read and the
  execute writes; a concurrent writer would invalidate it.
* Re-running after a partial failure is idempotent for the BACKFILL and INDEX
  phases, but already-deleted merge data cannot be restored — which is exactly
  why every loser's vertex- and edge-level lineage is MERGED onto the survivor
  BEFORE the loser is deleted.
* Lineage preservation (this project's "complete data lineage" rule):
  - Vertex merge: every LIST property on the loser (``_evidence_ids``,
    ``_page_numbers``, ``_evidence_texts``, ``source_*``, ...) is UNION'd
    (dedup) into the survivor's same property; SCALAR/other properties fill
    the survivor only when it is null — a non-null survivor value (specs,
    first-seen display casing) is NEVER overwritten.
  - Edge dedup fold: when the survivor already has an equivalent edge, ALL
    list-valued lineage on the loser's duplicate edge (``document_ids``,
    ``source_pages``, ``source_chunk_ids``, ``source_self_refs``, ...) is
    union'd into the survivor's edge and ``extraction_confidence`` is raised
    to the max — not just ``document_ids``.
  - Numeric arrays are written via the same ``_sql_list_set_fragment`` path
    the write layer uses (dodges ArcadeDB's bound-numeric-array nesting bug).
* Merge PRECEDES backfill (else two collision rows backfill to the same key
  and violate the UNIQUE index). Backfill precedes index for the same reason.
* Guards against edge loss: for every merged pair we require
  ``recreated + deduped + self_dropped == loser_edge_count`` (every loser edge
  is re-created on the survivor, folded into an existing duplicate, or dropped
  as a self-reference) and, after the merge,
  ``survivor_edges_after == survivor_edges_before + recreated``.
* Self-referential edges (a loser edge whose OTHER endpoint is the survivor or
  the loser itself) are DROPPED with a logged warning rather than recreated —
  re-pointing one would fabricate a survivor self-loop that never existed.
* Only GLOBAL-scope domain entities are auto-merged. If any case-collision is
  detected in a document-scoped / structural type it is REPORTED and (in
  ``--execute``) the run ABORTS before any write — those have different edge
  topology and are out of this migration's scope.

RUN
---
In-container (recommended — same ArcadeDB hostname + creds the write layer
uses; ``scripts/`` is not bind-mounted, so copy it in first):

    docker cp scripts/migrate_entity_key_dedup.py \
        eip-mmdpp-api-1:/app/scripts/migrate_entity_key_dedup.py
    docker exec eip-mmdpp-api-1 \
        python /app/scripts/migrate_entity_key_dedup.py --dry-run
    # after reviewing the dry-run AND taking a backup + pausing ingestion:
    docker exec eip-mmdpp-api-1 \
        python /app/scripts/migrate_entity_key_dedup.py --execute --yes-i-have-a-backup

From the host (ArcadeDB is on localhost:2480 — the in-container hostname
``arcadedb`` does not resolve here, so override the URL):

    python scripts/migrate_entity_key_dedup.py --dry-run \
        --arcadedb-url http://localhost:2480
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the repo root (parent of scripts/) importable whether this runs from the
# repo root on the host or from /app/scripts in the api container.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.services.arcadedb_client import ArcadeDBClient  # noqa: E402
from app.services.arcadedb_graph import _sql_list_set_fragment  # noqa: E402
from app.services.arcadedb_schema import _safe_type_name  # noqa: E402
from app.services.extraction_merge import norm  # noqa: E402
from ontology_bundles.air_defense_v3.introspect import build_ontology_dict  # noqa: E402

# Property keys that never count toward "data richness" when choosing a merge
# survivor (identity / bookkeeping / normalized-key columns are added per type).
_META_KEYS = frozenset({
    "@rid", "@type", "@cat", "@class", "@version", "rid",
    "name", "id", "entity_type", "canonical_name", "extraction_confidence",
    "created_at", "updated_at", "document_id",
})


# ---------------------------------------------------------------------------
# Ontology-derived type table (mirrors arcadedb_schema Phase 6b + _key_fields)
# ---------------------------------------------------------------------------

def derive_indexed_types() -> list[dict[str, Any]]:
    """Return the indexed entity types + their identity/key fields.

    Derived from ``build_ontology_dict()`` EXACTLY as the schema builder
    (``sync_schema_from_ontology`` Phase 6b) derives them:

      * a type participates iff it declares ``identity_fields`` (component
        classes with none are skipped — they have no upsert/index path);
      * ``key_fields`` = identity fields minus ``document_id`` (mirrors
        ``arcadedb_graph._key_fields``);
      * ``doc_scoped`` iff ``identity_scope == 'document'`` and the type does
        not already carry ``document_id`` in its identity fields — the key
        index then closes on ``document_id`` + ``entity_type``.
    """
    ont = build_ontology_dict()
    out: list[dict[str, Any]] = []
    for e in ont.get("entity_types", []):
        id_fields = list(e.get("identity_fields") or [])
        if not id_fields:
            continue
        scope = e.get("identity_scope", "document")
        out.append({
            "etype": _safe_type_name(e["name"]),
            "id_fields": id_fields,
            # PIN: this `!= "document_id"` filter MUST stay identical to
            # arcadedb_graph._key_fields (the write layer) and the schema
            # builder's Phase 6b key_fields — else backfill keys and the write
            # layer would diverge.
            "key_fields": [f for f in id_fields if f != "document_id"],
            "doc_scoped": scope == "document" and "document_id" not in id_fields,
            "scope": scope,
        })
    return out


def key_index_name(t: dict[str, Any]) -> tuple[str, str]:
    """Return ``(ddl, auto_index_name)`` for a type's ``<field>_key`` UNIQUE index.

    Mirrors the Phase 6b key-index DDL and ArcadeDB's ``Type[f1,f2]``
    auto-naming (no spaces) so we can probe ``schema:indexes`` for existence.
    """
    fields = [f"{f}_key" for f in t["key_fields"]]
    if t["doc_scoped"]:
        fields.append("document_id")
    fields.append("entity_type")
    ddl = f"CREATE INDEX IF NOT EXISTS ON {t['etype']} ({', '.join(fields)}) UNIQUE"
    name = f"{t['etype']}[{','.join(fields)}]"
    return ddl, name


# ---------------------------------------------------------------------------
# Collision detection + survivor selection
# ---------------------------------------------------------------------------

def _rid_sort_key(rid: str) -> tuple[int, int]:
    """Numeric sort key for a ``#bucket:pos`` RID (lowest-@rid tiebreak)."""
    try:
        bucket, pos = rid.lstrip("#").split(":")
        return int(bucket), int(pos)
    except Exception:
        return (1 << 62, 1 << 62)


def detect_collisions(
    client: ArcadeDBClient, db: str, t: dict[str, Any],
) -> list[list[str]]:
    """Return groups of RIDs (len > 1) that collide on the normalized key.

    A group is the set of rows that would map to the SAME ``(<field>_key...,
    document_id?)`` tuple once ``_key`` is backfilled — i.e. exactly what would
    violate the UNIQUE ``<field>_key`` index. ``norm()`` is computed in Python
    (same as the write layer).
    """
    key_fields = t["key_fields"]
    select_cols = ["@rid AS rid"] + list(key_fields)
    if t["doc_scoped"]:
        select_cols.append("document_id")
    rows = client.query_sync(
        db, "sql", f"SELECT {', '.join(select_cols)} FROM {t['etype']}",
    )
    groups: dict[tuple, list[str]] = {}
    for r in rows:
        key = tuple(norm(r.get(f)) for f in key_fields)
        if t["doc_scoped"]:
            key = key + (r.get("document_id"),)
        groups.setdefault(key, []).append(r["rid"])
    return [rids for rids in groups.values() if len(rids) > 1]


def _richness(row: dict[str, Any], t: dict[str, Any]) -> int:
    """Count populated (non-null / non-empty) DOMAIN properties on a row.

    Excludes identity fields, ``name``, bookkeeping columns and ``_key``
    columns — what remains is the entity's actual data payload (specs like
    ``nominal_rf_mhz`` / ``tx_peak_power_kw``). Higher = keep as survivor.
    """
    exclude = set(_META_KEYS) | set(t["id_fields"]) | {f"{f}_key" for f in t["key_fields"]}
    n = 0
    for k, v in row.items():
        if k in exclude or k.startswith("@"):
            continue
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        n += 1
    return n


def choose_survivor(
    client: ArcadeDBClient, db: str, t: dict[str, Any], rids: list[str],
) -> tuple[str, list[str], dict[str, tuple[int, int]]]:
    """Pick the survivor for a collision group.

    Tiebreak order (documented): (1) most populated domain properties
    (non-null specs); (2) more edges; (3) lowest @rid. Returns
    ``(survivor_rid, loser_rids, stats)`` where ``stats[rid] = (richness,
    edge_count)``.
    """
    rid_list = ", ".join(rids)
    rows = client.query_sync(
        db, "sql", f"SELECT *, @rid AS rid FROM {t['etype']} WHERE @rid IN [{rid_list}]",
    )
    by_rid = {r["rid"]: r for r in rows}
    stats: dict[str, tuple[int, int]] = {}
    for rid in rids:
        ec_res = client.query_sync(
            db, "sql", f"SELECT bothE().size() AS ec FROM {rid}",
        )
        edge_count = int(ec_res[0]["ec"]) if ec_res else 0
        stats[rid] = (_richness(by_rid.get(rid, {}), t), edge_count)
    ordered = sorted(
        rids,
        key=lambda r: (-stats[r][0], -stats[r][1], _rid_sort_key(r)),
    )
    return ordered[0], ordered[1:], stats


# ---------------------------------------------------------------------------
# Lineage merge helpers (vertex + edge) — "complete data lineage" rule
# ---------------------------------------------------------------------------

# System/bookkeeping edge columns never folded during a dedup merge.
_EDGE_META = frozenset({
    "@rid", "@type", "@cat", "@class", "@in", "@out", "@version",
    "created_at", "updated_at",
})


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def _union_list(existing: Any, incoming: Any) -> tuple[list[Any], list[Any]]:
    """Order-preserving union of two lists; returns (full_union, added_items).

    Dedup is by canonical JSON so lists-of-scalars AND lists-of-dicts (e.g.
    ``_field_evidence`` rows, provenance records) dedup correctly.
    """
    out = _as_list(existing)
    seen = {json.dumps(x, sort_keys=True, default=str) for x in out}
    added: list[Any] = []
    for x in _as_list(incoming):
        k = json.dumps(x, sort_keys=True, default=str)
        if k not in seen:
            seen.add(k)
            out.append(x)
            added.append(x)
    return out, added


def _vertex_merge_excludes(t: dict[str, Any]) -> set[str]:
    """Columns that must NOT be touched by a vertex lineage merge.

    Identity fields + ``name`` + the ``<field>_key`` columns are write-once
    display/identity (survivor keeps first-seen casing); the rest are
    system/bookkeeping.
    """
    return (
        {"@rid", "@type", "@cat", "@class", "@version", "name",
         "entity_type", "created_at", "updated_at"}
        | set(t["id_fields"])
        | {f"{f}_key" for f in t["key_fields"]}
    )


def plan_vertex_merge(
    surv_row: dict[str, Any], loser_row: dict[str, Any], t: dict[str, Any],
) -> tuple[list[tuple[str, list[Any], list[Any]]], list[tuple[str, Any]], list[tuple[str, Any, Any]]]:
    """Read-only vertex lineage merge plan.

    Returns ``(list_merges, scalar_fills, skipped)``:
      * ``list_merges``  : ``(prop, added_items, full_union)`` — LIST props
        union'd into the survivor (dedup); ``full_union`` is the value to SET.
      * ``scalar_fills`` : ``(prop, value)`` — scalar/other props the survivor
        is MISSING (null) that the loser can fill.
      * ``skipped``      : ``(prop, loser_value, survivor_value)`` — non-null
        survivor scalar/other kept as-is (survivor-wins; loser value logged so
        nothing is silently dropped).
    """
    exclude = _vertex_merge_excludes(t)
    list_merges: list[tuple[str, list[Any], list[Any]]] = []
    scalar_fills: list[tuple[str, Any]] = []
    skipped: list[tuple[str, Any, Any]] = []
    for k, lv in loser_row.items():
        if k in exclude or k.startswith("@") or lv is None:
            continue
        sv = surv_row.get(k)
        if isinstance(lv, list):
            full, added = _union_list(sv, lv)
            if added:
                list_merges.append((k, added, full))
        elif sv is None:
            scalar_fills.append((k, lv))
        elif sv != lv:
            skipped.append((k, lv, sv))
    return list_merges, scalar_fills, skipped


def execute_vertex_merge(
    client: ArcadeDBClient, db: str, surv_rid: str,
    list_merges: list[tuple[str, list[Any], list[Any]]],
    scalar_fills: list[tuple[str, Any]],
) -> None:
    """Apply a vertex lineage merge onto the survivor (WRITES)."""
    for prop, _added, full in list_merges:
        params: dict[str, Any] = {}
        frag = _sql_list_set_fragment(prop, full, params, f"v_{prop}")
        client.command_sync(db, "sql", f"UPDATE {surv_rid} SET {frag}", params or None)
    for prop, val in scalar_fills:
        client.command_sync(db, "sql", f"UPDATE {surv_rid} SET {prop} = :v", {"v": val})


def classify_edge_fold(
    surv_edge: dict[str, Any], loser_edge: dict[str, Any],
) -> tuple[dict[str, list[Any]], dict[str, Any], tuple[Any, Any] | None, dict[str, list[Any]]]:
    """Read-only fold plan for a loser DUPLICATE edge into the survivor's edge.

    Returns ``(list_adds, scalar_fills, conf_change, full_lists)``:
      * ``list_adds``   : ``{prop: [added items]}`` (for display)
      * ``scalar_fills``: ``{prop: value}`` scalar props the survivor edge lacks
      * ``conf_change`` : ``(old, new)`` if extraction_confidence would rise
      * ``full_lists``  : ``{prop: full unioned list}`` (for the SET on execute)
    """
    list_adds: dict[str, list[Any]] = {}
    scalar_fills: dict[str, Any] = {}
    full_lists: dict[str, list[Any]] = {}
    conf_change: tuple[Any, Any] | None = None
    for k, lv in loser_edge.items():
        if k in _EDGE_META or k.startswith("@") or lv is None:
            continue
        sv = surv_edge.get(k)
        if k == "extraction_confidence":
            candidates = [x for x in (sv, lv) if x is not None]
            newv = max(candidates) if candidates else lv
            if sv is None or newv != sv:
                conf_change = (sv, newv)
        elif isinstance(lv, list):
            full, added = _union_list(sv, lv)
            if added:
                list_adds[k] = added
                full_lists[k] = full
        elif sv is None:
            scalar_fills[k] = lv
    return list_adds, scalar_fills, conf_change, full_lists


def execute_edge_fold(
    client: ArcadeDBClient, db: str, surv_edge_rid: str, loser_edge: dict[str, Any],
) -> None:
    """Fold a loser duplicate edge's lineage into the survivor's edge (WRITES)."""
    surv_rows = client.query_sync(db, "sql", f"SELECT FROM {surv_edge_rid}")
    surv_edge = surv_rows[0] if surv_rows else {}
    _adds, scalar_fills, conf_change, full_lists = classify_edge_fold(surv_edge, loser_edge)
    for k, full in full_lists.items():
        params: dict[str, Any] = {}
        frag = _sql_list_set_fragment(k, full, params, f"e_{k}")
        client.command_sync(db, "sql", f"UPDATE {surv_edge_rid} SET {frag}", params or None)
    for k, v in scalar_fills.items():
        client.command_sync(db, "sql", f"UPDATE {surv_edge_rid} SET {k} = :v", {"v": v})
    if conf_change is not None:
        client.command_sync(
            db, "sql",
            f"UPDATE {surv_edge_rid} SET extraction_confidence = :v",
            {"v": conf_change[1]},
        )


# ---------------------------------------------------------------------------
# Edge introspection + re-point
# ---------------------------------------------------------------------------

def _edge_sig(edge: dict[str, Any], anchor_rid: str) -> tuple[str, str, str]:
    """(edge_type, direction-relative-to-anchor, other-endpoint-rid)."""
    if edge.get("@out") == anchor_rid:
        return edge.get("@type"), "OUT", edge.get("@in")
    return edge.get("@type"), "IN", edge.get("@out")


def _survivor_sig_set(
    client: ArcadeDBClient, db: str, surv_rid: str,
) -> set[tuple[str, str, str]]:
    edges = client.query_sync(db, "sql", f"SELECT expand(bothE()) FROM {surv_rid}")
    return {_edge_sig(e, surv_rid) for e in edges}


def plan_merge(
    client: ArcadeDBClient, db: str, t: dict[str, Any], surv_rid: str, loser_rid: str,
) -> dict[str, Any]:
    """Compute (read-only) the full merge plan for one loser → survivor.

    Covers edge re-point (recreate / dedup-fold / self-ref-drop) AND the
    vertex-level lineage merge. No writes.
    """
    loser_edges = client.query_sync(db, "sql", f"SELECT expand(bothE()) FROM {loser_rid}")
    surv_before = client.query_sync(db, "sql", f"SELECT bothE().size() AS ec FROM {surv_rid}")
    surv_before_ct = int(surv_before[0]["ec"]) if surv_before else 0
    surv_sigs = _survivor_sig_set(client, db, surv_rid)

    recreate: list[dict[str, Any]] = []
    dedup: list[dict[str, Any]] = []
    self_ref: list[dict[str, Any]] = []
    by_type_recreate: dict[str, int] = {}
    by_type_dedup: dict[str, int] = {}
    self_ref_info: list[tuple[str, str, str, str]] = []
    dedup_previews: list[dict[str, Any]] = []
    for e in loser_edges:
        etype, direction, other = _edge_sig(e, loser_rid)
        # (M6) Self-referential: the other endpoint IS the survivor or the loser
        # itself. Re-pointing would fabricate a survivor self-loop that never
        # existed; DROP it instead (counted separately for the loss guard).
        if other in (surv_rid, loser_rid):
            self_ref.append(e)
            reason = ("loser self-loop" if other == loser_rid
                      else "loser->survivor edge (would become survivor self-loop)")
            self_ref_info.append((etype, direction, other, reason))
            continue
        if (etype, direction, other) in surv_sigs:
            dedup.append(e)
            by_type_dedup[etype] = by_type_dedup.get(etype, 0) + 1
            new_from, new_to = (surv_rid, other) if direction == "OUT" else (other, surv_rid)
            match = client.query_sync(
                db, "sql",
                f"SELECT FROM {etype} WHERE @out = {new_from} AND @in = {new_to} LIMIT 1",
            )
            surv_edge = match[0] if match else {}
            adds, fills, conf, _full = classify_edge_fold(surv_edge, e)
            dedup_previews.append({
                "etype": etype, "direction": direction, "other": other,
                "surv_edge_rid": surv_edge.get("@rid"),
                "list_adds": adds, "scalar_fills": fills, "conf_change": conf,
            })
        else:
            recreate.append(e)
            by_type_recreate[etype] = by_type_recreate.get(etype, 0) + 1

    surv_row = (client.query_sync(db, "sql", f"SELECT FROM {surv_rid}") or [{}])[0]
    loser_row = (client.query_sync(db, "sql", f"SELECT FROM {loser_rid}") or [{}])[0]
    v_list, v_scalar, v_skipped = plan_vertex_merge(surv_row, loser_row, t)

    return {
        "loser_edge_count": len(loser_edges),
        "surv_before_ct": surv_before_ct,
        "recreate": recreate,
        "dedup": dedup,
        "self_ref": self_ref,
        "by_type_recreate": by_type_recreate,
        "by_type_dedup": by_type_dedup,
        "self_ref_info": self_ref_info,
        "dedup_previews": dedup_previews,
        "vertex_list_merges": v_list,
        "vertex_scalar_fills": v_scalar,
        "vertex_skipped": v_skipped,
    }


def _copy_edge_set(edge: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build a SET clause (+ params) copying a loser edge's properties.

    All-int lists (e.g. ``source_pages`` / ``page_numbers``) are inlined as SQL
    literals to dodge ArcadeDB's bound-numeric-array nesting bug (mirrors
    ``arcadedb_graph._sql_list_set_fragment``); everything else is bound.
    ``@``-prefixed system fields are skipped.
    """
    parts: list[str] = []
    params: dict[str, Any] = {}
    for k, v in edge.items():
        if k.startswith("@"):
            continue
        parts.append(_sql_list_set_fragment(k, v, params, f"p_{k}"))
    return (", ".join(parts) if parts else ""), params


def execute_merge(
    client: ArcadeDBClient, db: str, t: dict[str, Any],
    surv_rid: str, loser_rid: str, plan: dict[str, Any],
) -> None:
    """Perform the lineage merge + edge re-point + delete (WRITES).

    Order: (1) merge the loser's vertex-level lineage onto the survivor
    (list-union additive props; fill survivor-null scalars) BEFORE the vertex
    is deleted; (2) drop self-referential loser edges; (3) fold duplicate
    loser edges' FULL lineage into the survivor's matching edge (list-union of
    document_ids/source_*/…, max extraction_confidence), then drop them;
    (4) recreate the remaining loser edges on the survivor preserving props +
    direction; (5) delete the (now edge-less) loser vertex. Asserts zero edge
    loss (recreate + dedup + self_dropped == loser edges).
    """
    # (1) Vertex lineage merge — MUST precede DELETE VERTEX.
    execute_vertex_merge(
        client, db, surv_rid,
        plan["vertex_list_merges"], plan["vertex_scalar_fills"],
    )

    # (2) Self-referential edges — drop (never recreate a fabricated self-loop).
    for e in plan["self_ref"]:
        etype = e.get("@type")
        client.command_sync(db, "sql", f"DELETE FROM {etype} WHERE @rid = {e['@rid']}")

    # (3) Duplicate edges — fold full lineage into the survivor's edge, then drop.
    for e in plan["dedup"]:
        etype, direction, other = _edge_sig(e, loser_rid)
        new_from, new_to = (surv_rid, other) if direction == "OUT" else (other, surv_rid)
        existing = client.query_sync(
            db, "sql",
            f"SELECT @rid AS rid FROM {etype} WHERE @out = {new_from} AND @in = {new_to} LIMIT 1",
        )
        if existing:
            execute_edge_fold(client, db, existing[0]["rid"], e)
        client.command_sync(db, "sql", f"DELETE FROM {etype} WHERE @rid = {e['@rid']}")

    # (4) Recreate the rest on the survivor, preserving props + direction.
    for e in plan["recreate"]:
        etype, direction, other = _edge_sig(e, loser_rid)
        new_from, new_to = (surv_rid, other) if direction == "OUT" else (other, surv_rid)
        set_clause, params = _copy_edge_set(e)
        set_sql = f" SET {set_clause}" if set_clause else ""
        client.command_sync(
            db, "sql",
            f"CREATE EDGE {etype} FROM {new_from} TO {new_to}{set_sql}",
            params or None,
        )
        client.command_sync(db, "sql", f"DELETE FROM {etype} WHERE @rid = {e['@rid']}")

    # (5) Delete the (now edge-less) loser vertex. ArcadeDB's SQL parser requires
    # the "DELETE VERTEX FROM <target>" form (a bare "DELETE VERTEX <rid>" fails
    # with "missing FROM"); this matches the codebase idiom (arcadedb_graph.py:826).
    # DELETE VERTEX is the graph-safe delete (cleans up any residual edges); the
    # loser is edge-less here by construction (step 3 re-pointed/deleted them all).
    client.command_sync(db, "sql", f"DELETE VERTEX FROM {loser_rid}")

    # Zero-edge-loss assertion.
    surv_after = client.query_sync(db, "sql", f"SELECT bothE().size() AS ec FROM {surv_rid}")
    surv_after_ct = int(surv_after[0]["ec"]) if surv_after else 0
    expected = plan["surv_before_ct"] + len(plan["recreate"])
    if surv_after_ct != expected:
        raise RuntimeError(
            f"EDGE LOSS GUARD tripped for survivor {surv_rid}: "
            f"after={surv_after_ct} expected={expected} "
            f"(before={plan['surv_before_ct']} recreated={len(plan['recreate'])} "
            f"deduped={len(plan['dedup'])} self_dropped={len(plan['self_ref'])})"
        )


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_merge(
    client: ArcadeDBClient, db: str, types: list[dict[str, Any]], dry_run: bool,
) -> int:
    """Phase 1 — detect + merge case-collision groups. Returns merges done/planned."""
    print("\n" + "=" * 78)
    print("PHASE 1 — MERGE case-collision vertices")
    print("=" * 78)

    blocking: list[str] = []
    total = 0
    for t in types:
        groups = detect_collisions(client, db, t)
        if not groups:
            continue
        if t["scope"] != "global":
            for g in groups:
                blocking.append(
                    f"  !! {t['etype']} (scope={t['scope']}) collision on {len(g)} rows: {g}"
                )
            continue
        print(f"\n[{t['etype']}] {len(groups)} collision group(s):")
        for rids in groups:
            surv, losers, stats = choose_survivor(client, db, t, rids)
            surv_name = client.query_sync(
                db, "sql",
                f"SELECT {', '.join(t['key_fields'])} FROM {surv}",
            )
            label = " / ".join(
                str(surv_name[0].get(f)) for f in t["key_fields"]
            ) if surv_name else "?"
            print(f"  collision key -> survivor {surv} ({label})")
            print(
                f"    tiebreak: survivor richness={stats[surv][0]} edges={stats[surv][1]} "
                f"(rule: most populated specs, then most edges, then lowest @rid)"
            )
            for rid in rids:
                mark = "SURVIVOR" if rid == surv else "loser   "
                print(f"      {mark} {rid}: richness={stats[rid][0]} edges={stats[rid][1]}")
            for loser in losers:
                plan = plan_merge(client, db, t, surv, loser)
                total += 1
                print(
                    f"    re-point loser {loser}: {plan['loser_edge_count']} edges "
                    f"-> recreate {sum(plan['by_type_recreate'].values())} "
                    f"{dict(plan['by_type_recreate'])}, "
                    f"dedup {sum(plan['by_type_dedup'].values())} "
                    f"{dict(plan['by_type_dedup'])}, "
                    f"self-ref {len(plan['self_ref'])}"
                )
                # --- vertex-level lineage merge (applied BEFORE loser delete) ---
                if plan["vertex_list_merges"] or plan["vertex_scalar_fills"]:
                    for prop, added, _full in plan["vertex_list_merges"]:
                        print(f"      vertex lineage: survivor {surv} += {prop} {added} (list union)")
                    for prop, val in plan["vertex_scalar_fills"]:
                        print(f"      vertex lineage: survivor {surv} .{prop} = {val!r} (was null; filled)")
                else:
                    print(f"      vertex lineage: nothing additive to merge from {loser}")
                for prop, lval, sval in plan["vertex_skipped"]:
                    print(
                        f"      vertex lineage: KEEP survivor .{prop}={sval!r} "
                        f"(loser had {lval!r}; survivor-wins, not overwritten)"
                    )
                # --- duplicate-edge FULL lineage fold previews ---
                for dp in plan["dedup_previews"]:
                    bits: list[str] = []
                    if dp["list_adds"]:
                        bits.append("lists " + ", ".join(f"{k}+{v}" for k, v in dp["list_adds"].items()))
                    if dp["scalar_fills"]:
                        bits.append("scalars " + ", ".join(f"{k}={v!r}" for k, v in dp["scalar_fills"].items()))
                    if dp["conf_change"]:
                        bits.append(f"confidence {dp['conf_change'][0]}->{dp['conf_change'][1]}")
                    detail = "; ".join(bits) if bits else "no new lineage (subset of survivor edge)"
                    print(
                        f"      edge fold: loser {dp['etype']} {dp['direction']} -> {dp['other']} "
                        f"folds into survivor edge {dp['surv_edge_rid']} [{detail}]"
                    )
                # --- self-referential edge visibility (M6) ---
                for etype, direction, other, reason in plan["self_ref_info"]:
                    print(f"      !! SELF-REF edge DROPPED: {etype} {direction} -> {other} ({reason})")
                # --- edge-loss guard (raise, not assert; safe under python -O) ---
                acct = len(plan["recreate"]) + len(plan["dedup"]) + len(plan["self_ref"])
                if acct != plan["loser_edge_count"]:
                    raise RuntimeError(
                        f"edge accounting mismatch for {loser}: recreate+dedup+self_ref="
                        f"{acct} != loser edges={plan['loser_edge_count']}"
                    )
                print(
                    f"      survivor edges: before={plan['surv_before_ct']} "
                    f"-> after={plan['surv_before_ct'] + len(plan['recreate'])} "
                    f"(loss guard: recreate+dedup+self_ref={acct} == loser edges="
                    f"{plan['loser_edge_count']})"
                )
                print(f"      would DELETE loser vertex {loser} (AFTER its lineage is merged above)")
                if not dry_run:
                    execute_merge(client, db, t, surv, loser, plan)
                    print(f"      [EXECUTED] merged {loser} -> {surv}")

    if blocking:
        print("\nCOLLISIONS IN NON-GLOBAL / STRUCTURAL TYPES (out of scope):")
        for line in blocking:
            print(line)
        if not dry_run:
            raise RuntimeError(
                "Aborting before any backfill: unexpected collisions in "
                "document-scoped/structural types (see above). These are NOT "
                "auto-merged by this migration."
            )
        print("  (dry-run: reported only; --execute would ABORT here)")

    if total == 0:
        print("\n  no global-domain case-collisions found — nothing to merge.")
    return total


def phase_backfill(
    client: ArcadeDBClient, db: str, types: list[dict[str, Any]], dry_run: bool,
) -> None:
    """Phase 2 — backfill ``<field>_key = norm(<field>)`` on all rows."""
    print("\n" + "=" * 78)
    print("PHASE 2 — BACKFILL <field>_key (norm computed in Python)")
    print("=" * 78)
    print(f"{'TYPE':<30} {'FIELD':<18} {'rows':>6} {'would_set':>10} "
          f"{'already':>8} {'null_skip':>9}")
    print("-" * 84)
    for t in types:
        for field in t["key_fields"]:
            rows = client.query_sync(
                db, "sql",
                f"SELECT @rid AS rid, {field} AS raw, {field}_key AS cur FROM {t['etype']}",
            )
            would = already = nullskip = 0
            for r in rows:
                raw = r.get("raw")
                if raw is None:
                    nullskip += 1
                    continue
                desired = norm(raw)
                if r.get("cur") == desired:
                    already += 1
                    continue
                would += 1
                if not dry_run:
                    client.command_sync(
                        db, "sql",
                        f"UPDATE {t['etype']} SET {field}_key = :k WHERE @rid = {r['rid']}",
                        {"k": desired},
                    )
            print(f"{t['etype']:<30} {field:<18} {len(rows):>6} {would:>10} "
                  f"{already:>8} {nullskip:>9}")


def phase_index(
    client: ArcadeDBClient, db: str, types: list[dict[str, Any]], dry_run: bool,
) -> None:
    """Phase 3 — ensure the ``<field>_key`` UNIQUE index exists (idempotent)."""
    print("\n" + "=" * 78)
    print("PHASE 3 — ENSURE <field>_key UNIQUE index (IF NOT EXISTS)")
    print("=" * 78)
    existing = {
        r.get("name")
        for r in client.query_sync(db, "sql", "SELECT name FROM schema:indexes")
    }
    for t in types:
        ddl, name = key_index_name(t)
        present = name in existing
        status = "EXISTS" if present else "MISSING -> would CREATE"
        print(f"  {name:<48} {status}")
        if not dry_run and not present:
            for field in t["key_fields"]:
                client.command_sync(
                    db, "sql",
                    f"CREATE PROPERTY {t['etype']}.{field}_key IF NOT EXISTS STRING",
                )
            client.command_sync(db, "sql", ddl)
            print(f"    [EXECUTED] {ddl}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Inspect + print plan; ZERO writes (default).")
    ap.add_argument("--execute", action="store_true", help="Perform the destructive migration (writes).")
    ap.add_argument("--yes-i-have-a-backup", action="store_true",
                    help="REQUIRED with --execute: confirms you took an ArcadeDB "
                         "backup and paused ingestion. Without it --execute refuses to write.")
    ap.add_argument("--arcadedb-url", default=None,
                    help="Override ArcadeDB base URL (host runs: http://localhost:2480).")
    args = ap.parse_args()

    if args.dry_run and args.execute:
        ap.error("pass at most one of --dry-run / --execute")
    dry_run = not args.execute  # default (no flags) => dry-run

    settings = get_settings()
    base_url = args.arcadedb_url or settings.arcadedb_url
    client = ArcadeDBClient(
        base_url=base_url,
        username=settings.arcadedb_user,
        password=settings.arcadedb_password,
    )
    db = settings.arcadedb_database
    types = derive_indexed_types()

    mode = "DRY-RUN (no writes)" if dry_run else "EXECUTE (WRITES)"
    print("=" * 78)
    print("case-insensitive-identity migration — merge -> backfill -> index")
    print("=" * 78)
    print(f"mode      : {mode}")
    print(f"arcadedb  : {base_url}  db={db}")
    print(f"types     : {len(types)} indexed entity types with identity fields")

    if not dry_run:
        # Blast-radius summary BEFORE touching anything.
        merges = 0
        for t in types:
            if t["scope"] != "global":
                continue
            for g in detect_collisions(client, db, t):
                merges += len(g) - 1
        row_total = 0
        for t in types:
            for _f in t["key_fields"]:
                cnt = client.query_sync(db, "sql", f"SELECT count(*) AS c FROM {t['etype']}")
                row_total += int(cnt[0]["c"]) if cnt else 0
        print("\n" + "!" * 78)
        print("BLAST RADIUS (DESTRUCTIVE — merge DELETEs cannot be undone):")
        print(f"  * merge/delete up to {merges} loser vertex(es); each loser's vertex- and")
        print("    edge-level lineage is UNION-merged onto its survivor FIRST, then deleted")
        print(f"  * backfill <field>_key across ~{row_total} vertex rows")
        print("  * create any missing <field>_key UNIQUE indexes")
        print("REQUIRED BEFORE EXECUTE:")
        print("  1. Take an ArcadeDB backup/snapshot (deleted merge data is unrecoverable).")
        print("  2. Pause ingestion — NO concurrent writers (the edge-loss guard assumes a")
        print("     stable survivor edge count between the plan read and the execute writes).")
        print("!" * 78)
        if not args.yes_i_have_a_backup:
            print("\nREFUSING TO WRITE: --execute requires --yes-i-have-a-backup.")
            print("Re-run once you have taken a backup and paused ingestion:")
            print("  ... migrate_entity_key_dedup.py --execute --yes-i-have-a-backup")
            return 2
        print("\nConfirmation received (--yes-i-have-a-backup). Proceeding with writes.\n")

    phase_merge(client, db, types, dry_run)
    phase_backfill(client, db, types, dry_run)
    phase_index(client, db, types, dry_run)

    print("\n" + "=" * 78)
    print(f"DONE ({mode}).")
    if dry_run:
        print("Review the plan above. After taking a backup + pausing ingestion, apply with:")
        print("  ... migrate_entity_key_dedup.py --execute --yes-i-have-a-backup")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
